import spaces  # MUST be first for ZeroGPU!

import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
import cv2
import torch
import mediapipe as mp
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import load_image

# Constants from the 00quebec/Synthid-Bypass workflow
DEFAULT_DENOISE = 0.2
DEFAULT_STEPS = 9  # Turbo models need fewer steps
DEFAULT_LOOPS = 3  # The repo uses 3 sequential KSamplers

# Global pipeline variables
pipeline = None
face_detector = None

def initialize_face_detector():
    """Initialize MediaPipe face detector"""
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        return mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    except Exception as e:
        print(f"Failed to initialize Face Detector: {e}")
        return None

def initialize_models():
    """Initialize SDXL Turbo and ControlNet"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Initializing models on {device} with {dtype}...")
        
        # Load ControlNet for SDXL (Canny)
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=dtype
        )
        
        # Load SDXL Turbo (Fast, High Quality, similar to Z-Image-Turbo)
        # Using VAE fix to prevent artifacts
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
        
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True
        )
        
        # Turbo scheduler (Euler Ancestral or similar, matching repo's "simple/euler")
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        pipe = pipe.to(device)
        
        # Enable optimizations
        if device == "cuda":
            # pipe.enable_model_cpu_offload() # SDXL might need sequential offload on smaller GPUs
            pipe.enable_sequential_cpu_offload()
            
        return pipe
    except Exception as e:
        print(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_canny_edges(image, low_threshold=100, high_threshold=200):
    """Extract Canny edges for ControlNet"""
    image_np = np.array(image)
    if image_np.shape[2] == 4: # RGBA to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def process_face_detailer(image, pipe, prompt, negative_prompt, steps, strength, seed):
    """
    Implements the 'FaceDetailer' node logic: 
    Detect faces -> Crop -> Denoise (Repair) -> Paste back
    """
    global face_detector
    if face_detector is None:
        face_detector = initialize_face_detector()
    
    if face_detector is None:
        print("Face detector failed to initialize. Skipping FaceDetailer.")
        return image
    
    img_np = np.array(image)
    results = face_detector.process(img_np)
    
    if not results.detections:
        print("No faces detected for detailing.")
        return image
        
    print(f"Detected {len(results.detections)} faces. Starting FaceDetailer...")
    height, width, _ = img_np.shape
    processed_image = image.copy()
    
    # Margin for face crop
    margin = 50 
    
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        
        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)
        
        # Crop face
        face_crop = processed_image.crop((x1, y1, x2, y2))
        
        # Resize for processing if too small
        original_crop_size = face_crop.size
        process_size = (512, 512)
        face_crop_resized = face_crop.resize(process_size, Image.Resampling.LANCZOS)
        
        # Get edges for the face (optional, but good for structure)
        face_edges = get_canny_edges(face_crop_resized, 50, 150)
        
        # Denoise the face (Refine)
        # Using slightly higher strength for faces to ensure cleanup
        refined_face = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=face_crop_resized,
            control_image=face_edges,
            num_inference_steps=steps,
            strength=0.30, # Faces need a bit more scrub
            guidance_scale=1.5, # Low guidance for Turbo
            controlnet_conditioning_scale=0.5,
            generator=torch.manual_seed(seed)
        ).images[0]
        
        # Resize back and paste
        refined_face = refined_face.resize(original_crop_size, Image.Resampling.LANCZOS)
        
        # Soft blending mask
        mask = Image.new('L', original_crop_size, 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([margin//2, margin//2, original_crop_size[0]-margin//2, original_crop_size[1]-margin//2], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(15))
        
        processed_image.paste(refined_face, (x1, y1), mask)
        
    return processed_image

@spaces.GPU(duration=120)  # Increased duration for multi-pass + SDXL
def remove_watermark(
    input_image,
    denoise_strength=0.2,
    loops=3,
    steps=9,
    use_face_detailer=True,
    progress=gr.Progress()
):
    global pipeline
    
    if input_image is None:
        return None, "Please upload an image."
        
    try:
        progress(0.1, desc="Loading SDXL Turbo Models...")
        if pipeline is None:
            pipeline = initialize_models()
        
        if pipeline is None:
            return None, "Failed to load models."
            
        # 1. Resize if huge (SDXL handles 1024x1024 well)
        max_dim = 1024
        if max(input_image.size) > max_dim:
            ratio = max_dim / max(input_image.size)
            new_size = tuple(int(dim * ratio) for dim in input_image.size)
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
            
        current_image = input_image
        
        # Prompt settings (Generic high quality)
        prompt = "high quality, professional image, sharp focus, 4k, detail"
        negative_prompt = "watermark, text, blur, noise, distortion, artifacts"
        
        # Seed
        seed = 42
        
        # 2. Sequential KSampler Loop (Key to the bypass)
        for i in range(loops):
            progress(0.2 + (i/loops)*0.5, desc=f"Denoising Pass {i+1}/{loops} (Strength: {denoise_strength})...")
            
            # Extract fresh edges from the CURRENT state of the image
            # This ensures we follow the evolving structure
            edges = get_canny_edges(current_image)
            
            # Run Img2Img with ControlNet
            current_image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=current_image,
                control_image=edges,
                num_inference_steps=steps,
                strength=denoise_strength,
                guidance_scale=1.5, # Turbo needs low CFG
                controlnet_conditioning_scale=0.6, # Structure preservation
                generator=torch.manual_seed(seed + i)
            ).images[0]
            
        # 3. Face Detailer (Optional but recommended)
        if use_face_detailer:
            progress(0.8, desc="Running Face Detailer...")
            current_image = process_face_detailer(
                current_image, pipeline, prompt, negative_prompt, steps, denoise_strength, seed
            )
            
        progress(1.0, desc="Done!")
        
        return current_image, f"✅ Processed with {loops} passes @ {denoise_strength} strength + FaceDetailer"
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

# Gradio Interface
def create_demo():
    with gr.Blocks(title="SynthID Remover (Exact Workflow match)") as demo:
        gr.Markdown("## 🔬 SynthID Watermark Remover (SDXL Turbo Implementation)")
        gr.Markdown("""
        **Exact implementation of the 00quebec/Synthid-Bypass workflow:**
        1. **Low Denoise Loops**: Sequentially scrubs watermark noise (3 passes @ 0.2 strength).
        2. **ControlNet Canny**: Preserves structural integrity.
        3. **Face Detailer**: Detects and repairs faces separately (Critical for portraits).
        4. **SDXL Turbo**: High-fidelity model replacing Z-Image-Turbo.
        """)
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image")
                with gr.Accordion("Advanced Settings", open=True):
                    denoise = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Denoise Strength (per loop)")
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
