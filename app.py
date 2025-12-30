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

# Configuration
REPO_URL = "https://github.com/00quebec/Synthid-Bypass"
COMFYUI_URL = "https://github.com/comfyanonymous/ComfyUI"
PYTHON_EXTENSION_URL = "https://github.com/pydn/ComfyUI-to-Python-Extension"

ROOT_DIR = os.getcwd()
COMFYUI_DIR = os.path.join(ROOT_DIR, "ComfyUI")
BYPASS_REPO_DIR = os.path.join(ROOT_DIR, "reference_repo")

def setup():
    """Environment setup for Hugging Face Space"""
    # Check if a known model file exists to skip setup
    if os.path.exists(os.path.join(COMFYUI_DIR, "models/vae/ae.safetensors")):
        print("--- ENVIRONMENT ALREADY INITIALIZED ---")
        return

    print("--- FIRST TIME SETUP STARTING ---")

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
        "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
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

    print("✓ Custom nodes installed")

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
        {"repo": "YouLiXiya/YL-SAM", "file": "sam_vit_b_01ec64.pth", "dest": "models/sams/sam_vit_b_01ec64.pth"},
        # SeedVR2 models (15.3GB DiT + 672MB VAE)
        {"repo": "numz/SeedVR2_comfyUI", "file": "seedvr2_ema_7b_sharp_fp16.safetensors", "dest": "models/SEEDVR2/seedvr2_ema_7b_sharp_fp16.safetensors"},
        {"repo": "numz/SeedVR2_comfyUI", "file": "ema_vae_fp16.safetensors", "dest": "models/SEEDVR2/ema_vae_fp16.safetensors"}
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

    print("✓ Setup complete")

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
    print("Launching Headless ComfyUI server...")
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
                print("ComfyUI server is ready!")
                break
        except:
            if i % 5 == 0: print(f"Waiting for server... ({i*2}s)")
            time.sleep(2)
    
    if not server_ready:
        print("Server logs (first 50 lines):")
        # In a real environment, we'd capture stdout/stderr, but for now we'll just fail clearly
        proc.terminate()
        raise RuntimeError("ComfyUI server failed to start. Port 8188 remained closed.")

    try:
        # 3. Load pre-converted API workflow
        workflow_path = os.path.join(ROOT_DIR, "simple_api_workflow.json")
        with open(workflow_path, 'r') as f:
            api_prompt = json.load(f)
            
        # Update node 11 (LoadImage) to point to our input.png
        if "11" in api_prompt:
            api_prompt["11"]["inputs"]["image"] = input_filename
        
        # Send to ComfyUI
        print(f"Queueing workflow to ComfyUI ({len(api_prompt)} nodes)...")
        prompt_data = {"prompt": api_prompt}
        resp = requests.post("http://127.0.0.1:8188/prompt", json=prompt_data)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to queue prompt: {resp.text}")
            
        prompt_id = resp.json().get("prompt_id")
        print(f"Prompt queued successfully (ID: {prompt_id})")

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
                    # Success!
                    print("Processing complete!")
                    # Extract output filename from the SaveImage node (ID 62)
                    output_data = history[prompt_id]['outputs'].get('62')
                    if output_data and 'images' in output_data:
                        output_filename = output_data['images'][0]['filename']
                        finished = True
                        break
            
            if p % 10 == 0: print(f"Still processing... ({p}s)")
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

        with gr.Accordion("How it works & Acknowledgments", open=False):
            gr.Markdown("""
            ### Acknowledgments
            This project is a direct implementation of the research by [00quebec/Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass). All credit for the discovery and the original ComfyUI workflows goes to the original authors.
            
            ### Technical Breakdown
            The removal process works by re-processing the image through a specialized diffusion pipeline:
            1. **Pixel Laundering**: The image is re-rendered using the **Z-Image-Turbo (S3-DiT)** model with a low denoising factor (0.2). This replaces the watermark's subtle noise patterns with new noise from the model.
            2. **Structural Guidance**: To prevent the image from changing, a **Canny ControlNet** locks in the original geometry and composition.
            3. **Multi-Pass Denoising**: The process runs in three iterative stages to gently scrub away the watermark without introducing artifacts.
            4. **Face Restoration**: Using **FaceDetailer (YOLOv8)**, any detected faces are isolated and refined separately to preserve facial identity and high-end detail.
            """)

if __name__ == "__main__":
    # In Gradio 6.0+, css moved to launch(), but title remains in Blocks()
    demo.launch(css=css)
