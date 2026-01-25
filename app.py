import os
import sys
import subprocess
import time
import requests
import json
import gradio as gr
from PIL import Image
import spaces
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Suppress unnecessary warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# MIG GPU compatibility - disable problematic CUDA optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# Force stable CUDA kernels for MIG compatibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Configuration
REPO_URL = "https://github.com/00quebec/Synthid-Bypass"
COMFYUI_URL = "https://github.com/comfyanonymous/ComfyUI"
PYTHON_EXTENSION_URL = "https://github.com/pydn/ComfyUI-to-Python-Extension"

ROOT_DIR = os.getcwd()
COMFYUI_DIR = os.path.join(ROOT_DIR, "ComfyUI")
BYPASS_REPO_DIR = os.path.join(ROOT_DIR, "reference_repo")

def _patch_qwen_for_mig_gpu():
    """Patch Qwen/Llama text encoder for MIG GPU compatibility"""
    ops_file = os.path.join(COMFYUI_DIR, "comfy/ops.py")
    if not os.path.exists(ops_file):
        return

    with open(ops_file, 'r') as f:
        lines = f.readlines()

    # Check if patch already applied
    content = ''.join(lines)
    if 'MIG GPU CUBLAS fix' in content:
        print("[OK] Qwen MIG GPU patch already applied")
        return

    # Find and patch the return statement in forward_comfy_cast_weights
    patched = False
    for i, line in enumerate(lines):
        if 'return torch.nn.functional.linear(input, weight, bias)' in line and not patched:
            indent = len(line) - len(line.lstrip())
            space = ' ' * indent
            # Replace the single return line with try-except block
            new_lines = [
                f'{space}# MIG GPU CUBLAS fix\n',
                f'{space}try:\n',
                f'{space}    return torch.nn.functional.linear(input, weight, bias)\n',
                f'{space}except RuntimeError as e:\n',
                f'{space}    if "CUBLAS" in str(e):\n',
                f'{space}        input_f32 = input.float()\n',
                f'{space}        weight_f32 = weight.float()\n',
                f'{space}        bias_f32 = bias.float() if bias is not None else None\n',
                f'{space}        result = torch.nn.functional.linear(input_f32, weight_f32, bias_f32)\n',
                f'{space}        return result.to(input.dtype)\n',
                f'{space}    raise\n'
            ]
            lines[i:i+1] = new_lines
            patched = True
            break

    if patched:
        with open(ops_file, 'w') as f:
            f.writelines(lines)
        print("[OK] Applied MIG GPU CUBLAS fix to linear operations")
    else:
        print("[WARN] Could not find patch location in ops.py")

def setup():
    """Environment setup for Hugging Face Space"""
    # Check if a known model file exists to skip setup
    already_initialized = os.path.exists(os.path.join(COMFYUI_DIR, "models/vae/ae.safetensors"))

    if already_initialized:
        print("=== ENVIRONMENT ALREADY INITIALIZED ===")
        # Still apply critical patches even if already initialized
        _patch_qwen_for_mig_gpu()
        return

    print("=== FIRST TIME SETUP ===")

    # 1. Clone Repos
    subprocess.run(["git", "clone", COMFYUI_URL, COMFYUI_DIR], check=True, capture_output=True)
    subprocess.run(["git", "clone", REPO_URL, BYPASS_REPO_DIR], check=True, capture_output=True)
    
    # 2. Setup Custom Nodes
    nodes = [
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
        "https://github.com/wildminder/ComfyUI-dype",
        "https://github.com/rgthree/rgthree-comfy",
        "https://github.com/BadCafeCode/masquerade-nodes-comfyui",
        "https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch",
        PYTHON_EXTENSION_URL
    ]
    custom_nodes_path = os.path.join(COMFYUI_DIR, "custom_nodes")
    os.makedirs(custom_nodes_path, exist_ok=True)

    # Pin Impact packs to exact versions used in reference workflow
    IMPACT_PACK_COMMIT = "61bd8397a18e7e7668e6a24e95168967768c2bed"
    IMPACT_SUBPACK_VERSION = "1.3.4"  # Using 1.3.4 (latest available, ref workflow uses "1.3.5" which doesn't exist as tag)

    for url in nodes:
        name = url.split("/")[-1]
        node_dest = os.path.join(custom_nodes_path, name)
        if not os.path.exists(node_dest):
            subprocess.run(["git", "clone", url, node_dest], check=True, capture_output=True)
            # Checkout specific versions for Impact packs to match reference workflow
            if name == "ComfyUI-Impact-Pack":
                subprocess.run(["git", "checkout", IMPACT_PACK_COMMIT], cwd=node_dest, check=True, capture_output=True)
            elif name == "ComfyUI-Impact-Subpack":
                subprocess.run(["git", "checkout", IMPACT_SUBPACK_VERSION], cwd=node_dest, check=True, capture_output=True)
            elif name == "ComfyUI-dype":
                # changed to v2.1.0 (Nov 29, 2025) before the update introduces new parameters
                subprocess.run(["git", "checkout", "db90fe5012f94c6aef406d743292daf76b4477a6"], cwd=node_dest, check=True, capture_output=True)
    print("[OK] Custom nodes installed")

    # Apply MIG GPU compatibility patches
    _patch_qwen_for_mig_gpu()

    # Install performance optimizations (SageAttention, Flash Attention)
    print("Installing performance optimizations...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "sageattention", "flash-attn", "--no-cache-dir"
    ], capture_output=True, check=False)  # Don't fail if these can't install

    # 3. Models Download logic (Using hf_transfer for speed)
    model_configs = [
        {"repo": "Comfy-Org/z_image_turbo", "file": "split_files/vae/ae.safetensors", "dest": "models/vae/ae.safetensors"},
        {"repo": "Comfy-Org/z_image_turbo", "file": "split_files/diffusion_models/z_image_turbo_bf16.safetensors", "dest": "models/diffusion_models/z_image_turbo_bf16.safetensors"},
        {"repo": "Comfy-Org/z_image_turbo", "file": "split_files/text_encoders/qwen_3_4b.safetensors", "dest": "models/text_encoders/qwen_3_4b.safetensors"},
        {"repo": "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union", "file": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors", "dest": "models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors"},
        {"repo": "deepghs/yolo-face", "file": "yolov8n-face/model.pt", "dest": "models/ultralytics/bbox/yolov8n-face.pt"},
        {"repo": "YouLiXiya/YL-SAM", "file": "sam_vit_b_01ec64.pth", "dest": "models/sams/sam_vit_b_01ec64.pth"}
    ]

    print("Downloading models (fast with HF_TRANSFER)...")
    for i, cfg in enumerate(model_configs, 1):
        out_path = os.path.join(COMFYUI_DIR, cfg['dest'])
        if not os.path.exists(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print(f"  [{i}/{len(model_configs)}] {cfg['file'].split('/')[-1]}")
            hf_hub_download(
                repo_id=cfg['repo'],
                filename=cfg['file'],
                local_dir=COMFYUI_DIR,
                local_dir_use_symlinks=False
            )
            actual_downloaded_path = os.path.join(COMFYUI_DIR, cfg['file'])
            if actual_downloaded_path != out_path and os.path.exists(actual_downloaded_path):
                os.rename(actual_downloaded_path, out_path)

    print("[OK] Setup complete")

def convert_to_api(web_workflow):
    """
    Robustly converts ComfyUI Web JSON (UI format) to API Prompt format.
    Requires mapping links to actual node connections.
    """
    nodes = web_workflow.get("nodes", [])
    links = web_workflow.get("links", [])
    
    # Map link_id -> [origin_node_id, origin_slot_index]
    link_map = {}
    for link in links:
        if link:
            l_id, node_from, slot_from, node_to, slot_to, l_type = link
            link_map[l_id] = [str(node_from), slot_from]
            
    api_prompt = {}
    skipped_nodes = []
    for node in nodes:
        node_id = str(node["id"])
        class_type = node["type"]
        
        # Skip UI-only nodes, provider/loader nodes, and primitive types
        skip_types = ["Note", "Group", "Reroute", "Float", "Int", "String", "Boolean"]
        # Also skip any node with "Provider" or "Loader" in the name (these are config nodes)
        if class_type in skip_types or "Provider" in class_type or "Loader" in class_type and class_type not in ["UNETLoader", "VAELoader", "CLIPLoader"]:
            skipped_nodes.append(f"{node_id}:{class_type}")
            continue
            
        inputs = {}
        
        # 1. Handle Connections (from links)
        for inp in node.get("inputs", []):
            l_id = inp.get("link")
            if l_id and l_id in link_map:
                inputs[inp["name"]] = link_map[l_id]
        
        # 2. Handle Widgets (from widgets_values)
        # This is where it gets tricky since Web format stores values in a list
        # and API format expects them as named keys.
        # We'll use a known mapping for core nodes if possible.
        # For custom nodes, it depends on the node's implementation of 'INPUT_TYPES'.
        
        # Note: If the workflow was saved with 'widgets_values', we inject them.
        # We'll try to guess common input names or just pass them as indices if the server allows.
        # For SynthID-Bypass, we'll hardcode the critical ones if needed.
        
        # Fallback: Many nodes put widgets after connections in their registration.
        # If we don't have names, it might fail. 
        # However, many modern workflows save 'widgets_values' which we need to map.
        
        # For this specific bypass tool, we'll use the pre-known node names for key nodes.
        w_values = node.get("widgets_values", [])
        if class_type == "CLIPTextEncode" and w_values:
            inputs["text"] = w_values[0]
        elif class_type == "KSampler" and len(w_values) >= 7:
            inputs["seed"] = w_values[0]
            inputs["steps"] = w_values[2]
            inputs["cfg"] = w_values[3]
            inputs["sampler_name"] = w_values[4]
            inputs["scheduler"] = w_values[5]
            inputs["denoise"] = w_values[6]
        elif class_type == "VAELoader" and w_values:
            inputs["vae_name"] = w_values[0]
        elif class_type == "UNETLoader" and w_values:
            inputs["unet_name"] = w_values[0]
        elif class_type == "LoadImage" and w_values:
            inputs["image"] = w_values[0]
            inputs["upload"] = w_values[1] if len(w_values) > 1 else "image"
        elif class_type == "ModelSamplingAuraFlow" and w_values:
            inputs["shift"] = w_values[0]
        elif class_type == "DyPE_FLUX" and len(w_values) >= 4:
            inputs["width"] = w_values[0]
            inputs["height"] = w_values[1]
            inputs["preset"] = w_values[2]
            inputs["pe_type"] = w_values[3]
            # Advanced Peet's parameters... usually defaults are okay but we can add more if needed
            
        # Add any other widget values that might be present
        # This is a guestimate, but is usually how API conversion works
        
        api_prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs
        }
    
    print(f"Converted {len(api_prompt)} nodes, skipped {len(skipped_nodes)} nodes: {', '.join(skipped_nodes[:10])}")
    return api_prompt

# Execute setup on boot
setup()

@spaces.GPU(duration=120)
def remove_watermark(input_image):
    if input_image is None:
        return None
        
    # 1. Prepare Paths
    input_dir = os.path.join(COMFYUI_DIR, "input")
    output_dir = os.path.join(COMFYUI_DIR, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input image with a fixed name for the workflow
    input_filename = "input.png"
    input_path = os.path.join(input_dir, input_filename)
    input_image.save(input_path)
    
    # 2. Launch ComfyUI (Headless)
    print("Starting ComfyUI server...")
    # Using the correct CWD is critical for ComfyUI to find its models and custom nodes
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch"]
    proc = subprocess.Popen(cmd, cwd=COMFYUI_DIR)
    
    # Wait for server to be ready (increased timeout and added logging)
    server_ready = False
    for i in range(45): # 90 seconds max
        try:
            resp = requests.get("http://127.0.0.1:8188/history", timeout=2)
            if resp.status_code == 200:
                server_ready = True
                print("Server ready")
                break
        except:
            if i % 5 == 0:
                print(f"Waiting for server... ({i*2}s)")
            time.sleep(2)

    if not server_ready:
        proc.terminate()
        raise RuntimeError("ComfyUI server failed to start within 90 seconds")

    try:
        # 3. Load pre-converted API workflow
        workflow_path = os.path.join(ROOT_DIR, "simple_api_workflow.json")
        with open(workflow_path, 'r') as f:
            api_prompt = json.load(f)
            
        # Update node 11 (LoadImage) to point to our input.png
        if "11" in api_prompt:
            api_prompt["11"]["inputs"]["image"] = input_filename
        
        # Send to ComfyUI
        print(f"Processing workflow ({len(api_prompt)} nodes)...")
        prompt_data = {"prompt": api_prompt}
        resp = requests.post("http://127.0.0.1:8188/prompt", json=prompt_data)

        if resp.status_code != 200:
            error_data = resp.json() if resp.headers.get('content-type') == 'application/json' else {"error": resp.text}
            error_msg = error_data.get('error', {}).get('message', str(error_data))
            raise RuntimeError(f"Workflow validation failed: {error_msg}")

        prompt_id = resp.json().get("prompt_id")
        print(f"Workflow queued (ID: {prompt_id[:8]}...)")

        # 4. Wait for completion
        # We poll the history endpoint until the prompt_id appears
        max_poll = 120 # 120 seconds for processing
        finished = False
        output_filename = None
        
        for p in range(max_poll):
            history_resp = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
            if history_resp.status_code == 200:
                history = history_resp.json()
                if prompt_id in history:
                    # Extract output filename from the SaveImage node (ID 62)
                    output_data = history[prompt_id]['outputs'].get('62')
                    if output_data and 'images' in output_data:
                        output_filename = output_data['images'][0]['filename']
                        finished = True
                        print("Processing complete!")
                        break

            if p % 10 == 0 and p > 0:
                print(f"Still processing... ({p}s)")
            time.sleep(1)
            
        if not finished:
            raise RuntimeError("Processing timed out or failed to save image.")

        # 5. Return result
        output_path = os.path.join(output_dir, output_filename)
        return Image.open(output_path).copy() # Copy to avoid library closing issues
        
    finally:
        print("Shutting down ComfyUI server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        # Cleanup input file
        if os.path.exists(input_path): os.remove(input_path)

# Premium UI with Fixed Height and No Share Buttons
css = """
#container {
    max-width: 1200px;
    margin: 0 auto;
}
.image-preview {
    max-height: 512px !important;
}
footer {display: none !important;}
"""

with gr.Blocks(title="SynthID Remover") as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown("# SynthID Remover")
        gr.Markdown("This tool removes SynthID watermarks by re-rendering images through a high-fidelity diffusion reconstruction pipeline. It is specifically designed to bypass SynthID detection while maintaining the original image structure.")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image", height=512)
            with gr.Column():
                output_img = gr.Image(type="pil", label="Cleaned Image", height=512, interactive=False)
        
        submit_btn = gr.Button("Remove Watermark", variant="primary")
        
        submit_btn.click(
            fn=remove_watermark,
            inputs=[input_img],
            outputs=[output_img]
        )

        with gr.Accordion("How it works", open=False):
            gr.Markdown("""
            ### Technical Breakdown
            This tool removes SynthID watermarks by re-processing images through a specialized diffusion pipeline:

            1. **Image Scaling**: Input is scaled to 2.5 megapixels for optimal processing
            2. **Edge Detection**: Canny edge detection extracts the structural outline
            3. **ControlNet Guidance**: Z-Image-Turbo ControlNet uses edges to maintain composition
            4. **Multi-Pass Denoising**: Three sequential KSampler passes with 0.2 denoise strength:
               - Each pass subtly replaces pixel noise while preserving structure
               - Low denoise (0.2) ensures minimal visual changes
               - 9 sampling steps per pass for efficiency
            5. **Face Refinement**: FaceDetailer with YOLOv8 detection enhances facial details
            6. **Output**: Final image maintains visual fidelity while removing watermark artifacts

            **Models Used:**
            - Z-Image-Turbo (12.3GB) - Main diffusion model
            - Qwen 3 4B (8GB) - Text encoder
            - ControlNet Union (3.1GB) - Structural guidance
            - YOLOv8-Face - Face detection for refinement
            """)

if __name__ == "__main__":
    # In Gradio 6.0+, css moved to launch(), but title remains in Blocks()
    demo.launch(css=css)
