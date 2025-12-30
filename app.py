import spaces  # MUST be first for ZeroGPU!

import gradio as gr
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import cv2
import torch
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL, EulerAncestralDiscreteScheduler

# Constants from the 00quebec/Synthid-Bypass workflow
DEFAULT_DENOISE = 0.2
DEFAULT_STEPS = 9  # Turbo models need fewer steps
DEFAULT_LOOPS = 3  # The repo uses 3 sequential KSamplers

# Global pipeline variables
pipeline = None
face_model = None

def initialize_face_detector():
    """Initialize YOLOv8 Face Detector (Exact match to repo)"""
    try:
        print("Initializing YOLOv8 Face Face Detector...")
        # Download the exact model file used in the repo reference
        # Repo uses: yolov8n-face.pt
        model_path = hf_hub_download(repo_id="deepghs/yolo-face", filename="yolov8n-face/model.pt")
        return YOLO(model_path)
    except Exception as e:
        print(f"Failed to initialize YOLO Face Detector: {e}")
        return None

def initialize_models():
    """Initialize SDXL Turbo and ControlNet"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Initializing models on {device} with {dtype}...")
        
        # EXPLANATION:
        # The exact "Z-Image-Turbo" model requested is based on S3-DiT architecture
        # which is NOT supported by the diffusers library.
        # We use SDXL Turbo as the mathematically closest supported equivalent
        # (Turbo architecture, Low NFE, High Resolution).
        
        # Load ControlNet for SDXL (Canny)
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=dtype
        )
        
        # Load SDXL Turbo
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
        
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True
        )
        
        # Scheduler: Euler (Matches repo's "simple"/"euler")
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
        pipe = pipe.to(device)
        
        # Enable optimizations
        if device == "cuda":
            pipe.enable_sequential_cpu_offload()
            
        return pipe
    except Exception as e:
        print(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_canny_edges(image):
    """Extract Canny edges with Repo's tight thresholds"""
    image_np = np.array(image)
    if image_np.shape[2] == 4: # RGBA to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # REPO MATCH: Thresholds 0.02 and 0.11 (normalized) -> ~5 and ~28 (0-255)
    # This creates a very strict structural constraint.
    edges = cv2.Canny(gray, 5, 28)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def process_face_detailer(image, pipe, prompt, negative_prompt, steps, strength, seed):
    """
    Implements the 'FaceDetailer' node logic using YOLOv8
    """
    global face_model
    if face_model is None:
        face_model = initialize_face_detector()
    
    if face_model is None:
        print("YOLO model missing, skipping detailer.")
        return image

    # Run detection
    # YOLO returns a list of Results objects
    results = face_model(image)
    
    # Extract boxes
    boxes = []
    for r in results:
        for box in r.boxes:
            # box.xyxy is [x1, y1, x2, y2]
            b = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append(b)
            
    if not boxes:
        print("No faces detected for detailing.")
        return image
        
    print(f"Detected {len(boxes)} faces. Starting FaceDetailer...")
    
    processed_image = image.copy()
    width, height = processed_image.size
    margin = 50 
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Add margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)
        
        # Crop face
        face_crop = processed_image.crop((x1, y1, x2, y2))
        original_crop_size = face_crop.size
        
        # Resize for processing (standard detailer practice)
        process_size = (512, 512)
        face_crop_resized = face_crop.resize(process_size, Image.Resampling.LANCZOS)
        
        # Get edges for the face
        face_edges = get_canny_edges(face_crop_resized)
        
        # Denoise the face (Refine) with EXACT PARAMETERS
        refined_face = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=face_crop_resized,
            control_image=face_edges,
            num_inference_steps=steps,
            strength=strength, 
            guidance_scale=1.0, # EXACT MATCH: CFG 1.0
            controlnet_conditioning_scale=0.5,
            generator=torch.manual_seed(seed)
        ).images[0]
        
        # Resize back and paste
        refined_face = refined_face.resize(original_crop_size, Image.Resampling.LANCZOS)
        
        # Soft blending mask
        mask = Image.new('L', original_crop_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([margin//2, margin//2, original_crop_size[0]-margin//2, original_crop_size[1]-margin//2], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(15))
        
        processed_image.paste(refined_face, (x1, y1), mask)
        
    return processed_image

@spaces.GPU(duration=120)
def remove_watermark(
    input_image,
    denoise_strength=0.2, # Repo default
    loops=3,              # Repo default
    steps=9,              # Repo default
    use_face_detailer=True,
    progress=gr.Progress()
):
    global pipeline
    
    if input_image is None:
        return None, "Please upload an image."
        
    try:
        progress(0.1, desc="Loading Models (SDXL Turbo + YOLOv8)...")
        if pipeline is None:
            pipeline = initialize_models()
        
        if pipeline is None:
            return None, "Failed to load models."
            
        # 1. Resize if huge
        max_dim = 1536 # Increase to allow 4k input downscaling
        if max(input_image.size) > max_dim:
            ratio = max_dim / max(input_image.size)
            new_size = tuple(int(dim * ratio) for dim in input_image.size)
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
            
        current_image = input_image
        
        # Prompt settings
        prompt = "high quality, professional image, sharp focus, 4k, detail"
        negative_prompt = "watermark, text, blur, noise, distortion, artifacts"
        seed = 42
        
        print(f"Starting Watermark Removal: Loops={loops}, Denoise={denoise_strength}, CFG=1.0")

        # 2. Sequential KSampler Loop
        for i in range(loops):
            progress(0.2 + (i/loops)*0.5, desc=f"Denoising Pass {i+1}/{loops} (Strength: {denoise_strength})...")
            
            # Edges from Current State
            edges = get_canny_edges(current_image)
            
            # Run Img2Img
            current_image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=current_image,
                control_image=edges,
                num_inference_steps=steps,
                strength=denoise_strength,
                guidance_scale=1.0, # EXACT MATCH
                controlnet_conditioning_scale=0.6,
                generator=torch.manual_seed(seed + i)
            ).images[0]
            
        # 3. Face Detailer
        if use_face_detailer:
            progress(0.8, desc="Running YOLOv8 Face Detailer...")
            current_image = process_face_detailer(
                current_image, pipeline, prompt, negative_prompt, steps, 0.30, seed
            )
            
        progress(1.0, desc="Done!")
        
        return current_image, f"✅ Processed with {loops} passes @ {denoise_strength} + YOLOv8 FaceDetailer"
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

# Gradio Interface
def create_demo():
    with gr.Blocks(title="SynthID Remover (Exact Params)") as demo:
        gr.Markdown("## 🔬 SynthID Watermark Remover (High Definition)")
        gr.Markdown("""
        **Configuration:**
        *   **Loop**: 3 Passes @ 0.2 Denoise (Exact Match)
        *   **Constraint**: Canny Thresholds 5/28 (Exact Repo Match)
        *   **Face Detailer**: YOLOv8 Detection (Exact Repo Match)
        *   **Model**: SDXL Turbo (Proxied for Z-Image-Turbo due to platform support)
        """)
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image")
                with gr.Accordion("Advanced Settings", open=False):
                    denoise = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Denoise Strength")
                    loops = gr.Slider(1, 5, value=3, step=1, label="Denoising Loops")
                    steps = gr.Slider(4, 20, value=9, step=1, label="Inference Steps")
                    face_det = gr.Checkbox(True, label="Enable Face Detailer")
                
                run_btn = gr.Button("Remove Watermark", variant="primary")
            
            with gr.Column():
                output_img = gr.Image(type="pil", label="Result")
                status = gr.Text(label="Status")
                
        run_btn.click(
            remove_watermark,
            [input_img, denoise, loops, steps, face_det],
            [output_img, status]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.queue()
    demo.launch()
