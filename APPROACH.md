# Approach Verification

The user requested an exact match of the [Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass) workflow.
Since the original repo uses ComfyUI (node-based) and specialized models, we have implemented the **logic-equivalent** using Python and Diffusers.

## Component Mapping

| ComfyUI Node (Original) | Our Implementation (app.py) | Reason |
|-------------------------|-----------------------------|--------|
| `SeedVR2LoadDiTModel` (Z-Image-Turbo) | `StabilityAI/SDXL-Turbo` | Both are Turbo-class models. Z-Image is available (`Tongyi-MAI/Z-Image-Turbo`) but lacks **ControlNet** support in Diffusers, which is required for this workflow. SDXL Turbo is the closest equivalent *with* ControlNet support. |
| `KSampler` (steps=9, denoise=0.2) | `pipeline(img2img)` with `strength=0.2, steps=9` | Exact parameter match. |
| `KSampler` (cfg=1.0) | `guidance_scale=1.0` | Exact parameter match. |
| `Sequential Loop x3` | `for i in range(3):` | Exact logic match. |
| `Canny Edge` (0.02, 0.11) | `ControlNet Canny` (5, 28) | Exact threshold match (converted from normalized). |
| `FaceDetailer` (YOLO) | `process_face_detailer` (YOLOv8) | Exact backend match (`yolov8n-face.pt`). |

## Why Z-Image-Turbo Cannot Be Used
While `Tongyi-MAI/Z-Image-Turbo` is supported in Diffusers for generation, the **ControlNet** implementation (used in the research as `Z-Image-Turbo-Fun-Controlnet-Union`) has not been ported to the Diffusers library yet.
Without ControlNet, the watermark removal process fails to preserve image structure (creating hallucinations).
**SDXL Turbo** + **SDXL ControlNet** is the only viable combination to replicate the *behavior* of the research.
