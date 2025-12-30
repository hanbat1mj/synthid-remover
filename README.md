---
title: SynthID Watermark Remover
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
tags:
  - research
  - ai-safety
  - watermark-removal
  - diffusion
  - controlnet
---

# 🔬 SynthID Watermark Remover

A research tool demonstrating the removal of invisible SynthID watermarks from AI-generated images using diffusion-based reconstruction techniques.

## 🎯 Overview

This application implements the technique described in the [SynthID-Bypass research](https://github.com/00quebec/Synthid-Bypass) by 00quebec. It demonstrates that pixel-space watermarks embedded by Google's SynthID technology can be disrupted through careful re-processing with diffusion models.

## 🔧 How It Works

The core technique involves three key steps:

1. **Structural Extraction**: Uses Canny edge detection to create a structural map of the image
2. **Low-Denoise Diffusion**: Applies multiple passes of low-strength denoising to "re-noise" the image, replacing the watermark-carrying pixels
3. **ControlNet Guidance**: Preserves the original composition and structure using ControlNet conditioning

This process effectively "launders" the pixels - keeping semantic and structural information while replacing the low-level noise that carries the watermark.

## 🚀 Usage

1. Upload an AI-generated image with a SynthID watermark
2. Adjust settings if needed (default values work well for most images)
3. Click "Remove Watermark" and wait for processing
4. Download the processed image

### Advanced Settings

- **Denoise Strength** (0.05-0.3): Lower values preserve more detail but may leave watermark traces
- **Inference Steps** (10-50): More steps = better quality but slower processing
- **Guidance Scale** (5.0-15.0): Controls how strongly the model follows the prompt
- **ControlNet Scale** (0.5-1.0): Strength of structural preservation

## ⚠️ Ethical Considerations & Disclaimer

**This tool is provided for educational and AI safety research purposes only.**

- ❌ Do NOT use for malicious purposes
- ❌ Do NOT use to circumvent copyright
- ❌ Do NOT use to misrepresent content origin
- ✅ DO use for research and understanding watermark robustness
- ✅ DO use to develop better watermarking techniques

This proof-of-concept is presented "as-is" and without warranty.

## 🔬 Research Background

This implementation demonstrates a fundamental challenge in synthetic media detection: watermarks embedded in pixel space are vulnerable to reconstruction-style attacks. The research shows that:

- SynthID watermarks are not deterministic (different noise patterns each time)
- Low-denoise diffusion can replace watermark-carrying noise
- Structural guidance (ControlNet) prevents content degradation
- Multiple passes ensure complete watermark removal

## 🛠️ Technical Details

**Models Used:**
- Stable Diffusion v1.5 (base diffusion model)
- ControlNet Canny (structural preservation)
- DDIM Scheduler (quality optimization)

**Processing Pipeline:**
1. Image preprocessing and resizing
2. Canny edge extraction
3. 3-pass low-denoise diffusion
4. ControlNet-guided reconstruction

## 📚 Credits & References

- **Original Research**: [00quebec/Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass)
- **Related Paper**: Hu, Y., et al. (2024). "Stable signature is unstable: Removing image watermark from diffusion models." [arXiv:2405.07145](https://arxiv.org/abs/2405.07145)
- **SynthID**: [Google DeepMind](https://deepmind.google/models/synthid/)

## 🤝 Contributing

This is a research tool. If you develop techniques that:
- Defeat this bypass method
- Create more robust watermarking
- Improve the removal process

Please contribute to the broader AI safety dialogue!

## 📄 License

MIT License - See LICENSE file for details

---

**Remember**: The goal of this research is to improve AI safety, not to undermine it. Use responsibly and ethically.
