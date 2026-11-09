---
title: SynthID Watermark Remover
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
startup_duration_timeout: 1h
tags:
  - research
  - ai-safety
  - watermark-removal
  - diffusion
  - controlnet
---

# 🔬 SynthID Watermark Remover

A research tool demonstrating the removal of invisible SynthID watermarks from
AI-generated images using diffusion-based reconstruction.

This Space is a faithful port of the **V2** ComfyUI workflow from
[cebeuq/Synthid-Bypass](https://github.com/cebeuq/Synthid-Bypass)
(`Synthid-Bypass-v2.0.json`) to a headless, in-process Hugging Face Space.

## 🎯 Overview

The core hypothesis, from the original research: if the watermark lives in
low-level image noise, a tightly constrained diffusion reconstruction can keep
the composition while discarding the watermark-carrying pixels.

The upstream author arrived at this by rendering a blank black image through
Nano Banana Pro and then pushing exposure and shadows until the watermark
pattern became visible. A key observation was that the pattern is **not
deterministic** — regenerating the same prompt yields a different pattern each
time — which is what makes a re-noising attack plausible in the first place.

## 🔧 How It Works (V2 pipeline)

V2 combines two model paths in a single graph:

1. **Input normalization and adaptive denoise** — the image is loaded and a
   denoise target is computed from its resolution by the bundled
   `Synthid-Bypass-AdaptiveDenoise` node, so the strength scales predictably
   across input sizes.
2. **Global redraw pass** — a single low-denoise `KSampler` redraws the image
   through the **Qwen-Image** path, while a Canny edge map drives
   `QwenImageDiffsynthControlnet` to lock the composition in place.
3. **Strict face segmentation** — faces are detected with Ultralytics, refined
   with MediaPipe face mesh and a SAM-assisted path, and converted to SEGS.
4. **Adaptive face reconstruction** — the bundled
   `Synthid-Bypass-Facedetailer` (`SEGSDetailerModelSwap`) swaps to the
   **Z-Image Turbo** path for a face-only cleanup pass with adaptive denoise
   scaling, keeping renoised faces close to the originals.
5. **Composite and save** — the updated SEGS regions are pasted back onto the
   main image.

Input images are processed at their **native resolution**, matching the
reference workflow — which ships with its `ImageScaleToTotalPixels` node
bypassed, enabling it only for inputs above roughly 4K.

## 🚀 Usage

1. Upload an AI-generated image carrying a SynthID watermark.
2. Click **Remove Watermark**.
3. Download the result.

The first request after a cold start is slower, because the Space fetches the
model weights before the first run.

## ⚠️ Ethical Considerations & Disclaimer

**This tool is provided for educational and AI safety research purposes only.**

- ❌ Do NOT use for malicious purposes
- ❌ Do NOT use to circumvent copyright
- ❌ Do NOT use to misrepresent content origin
- ✅ DO use for research into watermark robustness
- ✅ DO use to help develop more robust watermarking

This proof-of-concept is presented "as-is" and without warranty.

## 🛠️ Technical Details

**Qwen-Image path (global redraw)**

| File | Directory |
| --- | --- |
| `qwen-image-2512-Q4_K_M.gguf` | `models/diffusion_models/` |
| `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `models/clip/` |
| `qwen_image_vae.safetensors` | `models/vae/` |
| `qwen_image_canny_diffsynth_controlnet.safetensors` | `models/model_patches/` |
| `Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors` | `models/loras/` |

**Z-Image Turbo path (face cleanup)**

| File | Directory |
| --- | --- |
| `z_image_turbo-Q4_K_M.gguf` | `models/diffusion_models/` |
| `Qwen_3_4b-imatrix-IQ4_XS.gguf` | `models/clip/` |
| `ae.safetensors` | `models/vae/` |

**Detection**

| File | Directory |
| --- | --- |
| `yolov8n-face.pt` | `models/ultralytics/bbox/` |
| `sam_vit_b_01ec64.pth` | `models/sams/` |

V2 is GGUF-first: both the model and CLIP paths use GGUF loaders, which is what
keeps two full model stacks resident at once.

## 📚 Credits & References

- **Original research**: [cebeuq/Synthid-Bypass](https://github.com/cebeuq/Synthid-Bypass)
  (formerly `00quebec/Synthid-Bypass`) — all workflow design and parameters are theirs.
- **Related paper**: Hu, Y., et al. (2024). "Stable signature is unstable:
  Removing image watermark from diffusion models."
  [arXiv:2405.07145](https://arxiv.org/abs/2405.07145)
- **SynthID**: [Google DeepMind](https://deepmind.google/models/synthid/)

Built on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and the
ComfyUI-GGUF, Impact Pack, Impact Subpack, KJNodes, controlnet_aux,
Inpaint-CropAndStitch, rgthree and RES4LYF node packs.

## 🤝 Contributing

This is a research tool. If you develop techniques that defeat this bypass,
produce more robust watermarking, or improve the removal process, please
contribute to the broader AI safety dialogue — upstream asks that you open an
issue or PR on the GitHub repo with your results.

## 📄 License

MIT License

---

**Remember**: the goal of this research is to improve AI safety, not to
undermine it. Use responsibly and ethically.
