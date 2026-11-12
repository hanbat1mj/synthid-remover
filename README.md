---
title: SynthID Watermark Remover
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
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

It runs a two-stage diffusion reconstruction pipeline headless and in-process,
on ZeroGPU or anywhere with a compatible GPU (CUDA / Apple Silicon MPS).

## 🎯 Overview

The core hypothesis, from the original research: if the watermark lives in
low-level image noise, a tightly constrained diffusion reconstruction can keep
the composition while discarding the watermark-carrying pixels.

The pattern is **not deterministic** — regenerating the same prompt yields a
different pattern each time — which is what makes a re-noising attack plausible
in the first place.

## 🔧 How It Works (V2 pipeline)

V2 combines two model paths in a single graph:

1. **Input normalization and adaptive denoise** — the image is loaded and a
   denoise target is computed from its resolution by the bundled
   `Synthid-Bypass-AdaptiveDenoise` node, so the strength scales predictably
   across input sizes.
2. **Global redraw pass** — a single low-denoise `KSampler` redraws the image
   through the **Qwen-Image** path, while a Canny edge map drives
   `QwenImageDiffsynthControlnet` to lock the composition in place.
3. **Strict face segmentation** — faces are detected with Ultralytics and
   refined with SAM, then converted to SEGS regions.
4. **Adaptive face reconstruction** — the bundled
   `Synthid-Bypass-Facedetailer` (`SEGSDetailerModelSwap`) swaps to the
   **Z-Image Turbo** path for a face-only cleanup pass with adaptive denoise
   scaling, keeping renoised faces close to the originals.
5. **Composite and save** — the updated SEGS regions are pasted back onto the
   main image.

Input images are processed at their **native resolution**, matching the
reference workflow — which ships with its `ImageScaleToTotalPixels` node
bypassed, enabling it only for inputs above roughly 4K.

## 🧭 This fork: changes & fixes

This repository is a fork of the original
[SynthID Remover](https://huggingface.co/spaces/dennny123/synth-id-remover) by
[dennny123](https://huggingface.co/dennny123). The V2 pipeline is unchanged;
these are the additions on top.

### ✨ Features

- **Transparent (RGBA) input/output** — Gradio's `gr.Image` defaults to
  `image_mode="RGB"`, which silently drops the alpha channel and exposes the
  black pixels stored under transparency. Input/output now use
  `image_mode="RGBA"`; the image is flattened onto white for the diffusion pass,
  then the original alpha is re-attached at the end.
- **Runs anywhere (backend-agnostic)** — the hard ZeroGPU/CUDA dependency was
  removed. The app now runs on Hugging Face ZeroGPU, Apple Silicon (MPS),
  Google Colab (T4+), RunPod, or any CUDA box, auto-detecting the backend.
- **Memory-aware loading** — model stacks load lazily and, in low-memory mode,
  evict each other so the two large stacks are never resident at once. This lets
  the pipeline fit in ~16 GB VRAM or a 36 GB unified-memory Mac.
- **Fidelity blend** — a UI slider (0–1) blends the redraw toward the original
  to reduce visible change at the cost of removal strength.
- **Smart skips** — the face pass (and its ~7 GiB model stack) is skipped when
  no face is detected, and redundant resizes are skipped when sizes already
  match.
- **Colab / RunPod / local-Mac tooling** — `colab.ipynb`, `runpod_setup.sh`, and
  `RUN_MAC.md` are included.

### 🐛 Fixes

- Fixed the "black background / lost transparency" bug caused by Gradio's
  `image_mode="RGB"` default.
- Fixed a crash in the fidelity blend when the VAE output was a pixel or two off
  the input size (KJNodes `ImageResizeKJv2` `keep_proportion="resize"` is
  aspect-fit, not a stretch; switched to `"stretch"`).
- Removed a white halo on semi-transparent edges (un-premultiply the white
  flatten before re-applying alpha).
- Removed a faint white rim on the cutout silhouette (a 2 px edge-preserving
  band keeps the original pixels there).
- Fixed a threshold bug that could revert near-opaque pixels and disable the
  watermark removal entirely.
- Added a TLS fallback for the SAM checkpoint download.

### ⚙️ Recommended settings

For transparent PNGs, `Adaptive level = 2` and `Fidelity blend = 0.9` give a
good balance: the result passes the detector while staying visually close to the
original.

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

- **Original author**: [dennny123](https://huggingface.co/dennny123) —
  [SynthID Remover (upstream)](https://huggingface.co/spaces/dennny123/synth-id-remover)
- **Related paper**: Hu, Y., et al. (2024). "Stable signature is unstable:
  Removing image watermark from diffusion models."
  [arXiv:2405.07145](https://arxiv.org/abs/2405.07145)
- **SynthID**: [Google DeepMind](https://deepmind.google/models/synthid/)

Built on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (pinned to
v0.7.0) and the ComfyUI-GGUF, Impact Pack, Impact Subpack, KJNodes and RES4LYF
node packs, plus the `Comfyui-SynthidBypass` node pack.

Note on fidelity: of the 46 nodes in the source graph, 30 actually feed the
output. The MediaPipe face-mesh branch, both `InpaintCropImproved` nodes and the
`MaskToSEGS` chains are disconnected, so they are not reproduced here — which
means `comfyui_controlnet_aux` and `ComfyUI-Inpaint-CropAndStitch` are not
needed. The rgthree Power Lora Loader (one enabled LoRA, null `strengthTwo`) is
exactly core `LoraLoader` with `strength_clip == strength_model`, and
masquerade's `Get Image Size` is an image-shape read, so both are inlined.

## 🤝 Contributing

This is a research tool. If you develop techniques that defeat this bypass,
produce more robust watermarking, or improve the removal process, please
contribute to the broader AI safety dialogue.

## 📄 License

MIT License

---

**Remember**: the goal of this research is to improve AI safety, not to
undermine it. Use responsibly and ethically.
