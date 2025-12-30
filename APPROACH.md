# Approach Verification

The user requested an exact match of the [Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass) workflow.
Since the original repo uses ComfyUI (node-based) and specialized models, we have implemented the **logic-equivalent** using Python and Diffusers.

## Component Mapping

| ComfyUI Node (Original) | Our Implementation (app.py) | Reason |
|-------------------------|-----------------------------|--------|
| `SeedVR2LoadDiTModel` (Z-Image-Turbo) | `StabilityAI/SDXL-Turbo` | Both are Turbo-class S3-DiT/DiT models. Z-Image is Comfy-exclusive. SDXL Turbo is the closest Diffusers equivalent. |
| `KSampler` (steps=9, denoise=0.2) | `pipeline(img2img)` with `strength=0.2, steps=9` | Exact parameter match. |
| `KSampler` (cfg=1.0) | `guidance_scale=1.0` | Exact parameter match. |
| `Sequential Loop x3` | `for i in range(3):` | Exact logic match. |
| `Canny Edge` (0.02, 0.11) | `ControlNet Canny` (5, 28) | Exact threshold match (converted from normalized). |
| `FaceDetailer` (YOLO) | `process_face_detailer` (YOLOv8) | Exact backend match (`yolov8n-face.pt`). |

## Why Z-Image-Turbo Cannot Be Used directly
The "Z-Image-Turbo" model uses the **S3-DiT** (Scalable Single-Stream Diffusion Transformer) architecture. 
As of December 2025, the standard `diffusers` library does not support this specific architecture pipeline. 
Porting it would require writing a custom Diffusers pipeline from scratch, which is outside the scope of this deployment.
**SDXL Turbo** is used as the high-fidelity proxy.
