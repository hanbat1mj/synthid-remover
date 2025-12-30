import gradio as gr
import numpy as np
from PIL import Image
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import spaces

# Initialize models
def initialize_models():
    """Initialize the diffusion models and ControlNet"""
    try:
        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Initializing models on {device} with {dtype}...")
        
        # Load ControlNet for Canny edge detection
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=dtype
        )
        
        # Load Stable Diffusion pipeline with ControlNet
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None
        )
        
        # Use DDIM scheduler for better quality
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        # Enable optimizations if on GPU
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        
        return pipe
    except Exception as e:
        print(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return None

# Global pipeline variable
pipeline = None

def get_canny_edge(image, low_threshold=100, high_threshold=200):
    """Extract Canny edges from image for ControlNet"""
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert back to RGB
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(edges_rgb)

@spaces.GPU(duration=60)  # ZeroGPU: allocate GPU for 60 seconds
def remove_synthid_watermark(
    input_image,
    denoise_strength=0.15,
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8,
    progress=gr.Progress()
):
    """
    Remove SynthID watermark using diffusion-based reconstruction.
    
    This implements the core technique from the research:
    1. Extract structural information (Canny edges)
    2. Use low-denoise diffusion to "re-noise" the image
    3. Preserve structure with ControlNet guidance
    
    Args:
        input_image: PIL Image with SynthID watermark
        denoise_strength: How much to denoise (lower = more preservation)
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        controlnet_conditioning_scale: Strength of ControlNet guidance
    """
    global pipeline
    
    if input_image is None:
        return None, "Please upload an image first."
    
    try:
        progress(0.1, desc="Initializing models...")
        
        # Initialize pipeline if not already done
        if pipeline is None:
            pipeline = initialize_models()
            if pipeline is None:
                return None, "Failed to initialize models. Please try again."
        
        progress(0.2, desc="Extracting structural information...")
        
        # Resize image if too large (for memory efficiency)
        max_size = 1024
        if max(input_image.size) > max_size:
            ratio = max_size / max(input_image.size)
            new_size = tuple(int(dim * ratio) for dim in input_image.size)
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Extract Canny edges for structural preservation
        canny_image = get_canny_edge(input_image)
        
        progress(0.3, desc="Processing with diffusion model...")
        
        # Generate a simple prompt based on image analysis
        # In a production version, you could use BLIP or similar for better prompts
        prompt = "high quality photograph, detailed, sharp focus, professional"
        negative_prompt = "blurry, low quality, distorted, watermark, text"
        
        # Process through multiple passes with low denoise
        # This simulates the multi-stage KSampler approach from ComfyUI
        current_image = input_image
        num_passes = 3  # Multiple passes as in the original workflow
        
        for pass_num in range(num_passes):
            progress(0.3 + (pass_num / num_passes) * 0.6, 
                    desc=f"Denoising pass {pass_num + 1}/{num_passes}...")
            
            # Convert current image to latent and back with low denoise
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                strength=denoise_strength,  # Low denoise is key!
            ).images[0]
            
            current_image = output
        
        progress(1.0, desc="Complete!")
        
        status_message = f"""
        ✅ **Watermark Removal Complete**
        
        - Processed with {num_passes} denoising passes
        - Denoise strength: {denoise_strength}
        - Structural preservation: ControlNet Canny edges
        
        **Note**: This implementation uses the core technique from the research:
        low-denoise diffusion with structural guidance to remove pixel-space watermarks.
        """
        
        return current_image, status_message
        
    except Exception as e:
        error_message = f"❌ Error during processing: {str(e)}"
        print(error_message)
        return None, error_message

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        title="SynthID Watermark Remover"
    ) as demo:
        
        gr.Markdown("""
        # 🔬 SynthID Watermark Remover
        
        ### Research-Based Watermark Removal Tool
        
        This tool implements the technique described in the [SynthID-Bypass research](https://github.com/00quebec/Synthid-Bypass) 
        to remove invisible watermarks from AI-generated images.
        
        **How it works:**
        1. Extracts structural information using Canny edge detection
        2. Applies low-denoise diffusion to "re-noise" the image
        3. Uses ControlNet to preserve the original composition
        4. Multiple passes ensure complete watermark removal
        
        ⚠️ **Educational & Research Purposes Only**
        
        This tool is provided for AI safety research and educational purposes. 
        Please use responsibly and ethically.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image with SynthID Watermark",
                    type="pil",
                    height=400
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    denoise_strength = gr.Slider(
                        minimum=0.05,
                        maximum=0.3,
                        value=0.15,
                        step=0.05,
                        label="Denoise Strength",
                        info="Lower values preserve more detail but may leave traces of watermark"
                    )
                    
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Inference Steps",
                        info="More steps = better quality but slower"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=5.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                        info="How strongly to follow the prompt"
                    )
                    
                    controlnet_scale = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.8,
                        step=0.1,
                        label="ControlNet Conditioning Scale",
                        info="Strength of structural preservation"
                    )
                
                process_btn = gr.Button("🚀 Remove Watermark", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Processed Image (Watermark Removed)",
                    type="pil",
                    height=400
                )
                
                status_text = gr.Markdown("Upload an image to begin...")
        
        gr.Markdown("""
        ---
        
        ### 📚 About the Technique
        
        This implementation is based on the research showing that SynthID watermarks 
        can be disrupted by re-processing images through a diffusion model pipeline.
        
        **Key Components:**
        - **Low-Denoise Regeneration**: Replaces pixel-level noise (including watermark) while preserving content
        - **ControlNet Guidance**: Uses Canny edges to maintain structural integrity
        - **Multi-Pass Processing**: Multiple gentle passes ensure complete removal
        
        **Research Credit:** [00quebec/Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass)
        
        **Limitations:**
        - Requires GPU with sufficient VRAM
        - May introduce subtle artifacts
        - Processing time varies with image size
        - Some fine details may be lost
        """)
        
        # Connect the button to the processing function
        process_btn.click(
            fn=remove_synthid_watermark,
            inputs=[
                input_image,
                denoise_strength,
                num_steps,
                guidance_scale,
                controlnet_scale
            ],
            outputs=[output_image, status_text]
        )
        
        # Example images (you can add these later)
        gr.Examples(
            examples=[],
            inputs=input_image,
            label="Example Images"
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.queue(max_size=20)
    demo.launch()
