# SynthID Bypass: Headless ComfyUI Implementation

This application has been transitioned to a **Headless ComfyUI backend** to ensure a 100% exact replication of the original research.

## Why this change?
While `diffusers` is a powerful library, the `00quebec/Synthid-Bypass` research relies on a highly specialized stack of technologies:
1.  **Z-Image-Turbo (S3-DiT)**: A specific architecture that differs from standard Stable Diffusion.
2.  **Union ControlNet**: A multi-mode ControlNet that handles structural guidance in a unique way.
3.  **ComfyUI Custom Nodes**: Specifically `Impact Pack` for face restoration and `SeedVR2` for upscaling.

By running the actual ComfyUI engine in the background of the Hugging Face Space, we guarantee:
- **Identical Model Loading**: Using the exact `.safetensors` files from the research.
- **Identical Logic**: Processing images through the exact same node graph.

## Architecture
- **Backend**: Headless ComfyUI server.
- **Frontend**: Gradio UI acting as a client.
- **Environment**: Hugging Face ZeroGPU (with model offloading to CPU when idle).

## Deployment Note
The first run on a fresh Hugging Face Space will involve:
1. Cloning ComfyUI and 6+ custom node repositories.
2. Downloading approximately 10GB of model weights.
This may lead to a long initial "Building" phase, but ensures the most faithful output possible.
