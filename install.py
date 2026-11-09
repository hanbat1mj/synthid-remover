"""
Environment preparation for the SynthID Remover Space.

Responsibilities, in order:
  1. Clone ComfyUI at a *pinned* commit.
  2. Clone each custom node pack at the *exact* ref stamped in the reference
     workflow JSON (`cnr_id` / `ver` on each node).
  3. Vendor the reference repo's own bundled node pack
     (`Comfyui-SynthidBypass`) into `custom_nodes/`.
  4. Symlink model weights out of the Hugging Face cache into ComfyUI's
     `models/` tree.

Why pinning matters: the previous revision of this Space cloned ComfyUI at
unpinned `master` and never installed ComfyUI's own requirements. When upstream
added `comfy-aimdo` as a hard dependency, every run died with
`ModuleNotFoundError: No module named 'comfy_aimdo'`. All pip dependencies now
live in this Space's `requirements.txt` (installed at build time) and nothing is
pip-installed at runtime, so a dependency can never be silently missing again.

No pip install happens here on purpose -- a runtime `pip install` inside a Space
can resolve a different torch and break the ZeroGPU patching.
"""

from __future__ import annotations

import os
import shutil
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_DIR = os.path.join(ROOT_DIR, "ComfyUI")
REFERENCE_DIR = os.path.join(ROOT_DIR, "reference_repo")
CUSTOM_NODES_DIR = os.path.join(COMFYUI_DIR, "custom_nodes")
MODELS_DIR = os.path.join(COMFYUI_DIR, "models")

# --------------------------------------------------------------------------
# Pins
# --------------------------------------------------------------------------
# ComfyUI: pinned to v0.7.0 (2025-12-31), chosen for three reasons:
#
#  1. It is the last release BEFORE `comfy-kitchen` (v0.8.0), `comfy-aimdo`
#     (v0.12.0) and `comfy-angle` (v0.28.0) entered requirements.txt. Those are
#     compiled extensions, and a missing `comfy_aimdo` is exactly what broke the
#     previous revision of this Space. Pinning here removes them entirely rather
#     than depending on them loading correctly under ZeroGPU.
#  2. It satisfies the workflow's highest `comfy-core` node stamp (0.7.0) and
#     registers every core node V2 needs.
#  3. It is contemporaneous with the custom node packs pinned below (all
#     Dec 2025 / Jan 2026), so their ComfyUI import surface matches.
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI"
COMFYUI_REF = "v0.7.0"

# The reference workflow for this port.
REFERENCE_REPO = "https://github.com/cebeuq/Synthid-Bypass"
REFERENCE_REF = "main"

# Custom node packs. Each ref is the one stamped on the nodes in
# Synthid-Bypass-v2.0.json, so these are the versions the workflow was authored
# against -- not "latest", which is what rotted last time.
# Only the packs whose nodes are actually reachable from SaveImage in the
# reference graph. Dropped after resolving the graph:
#   - comfyui_controlnet_aux  -> its only use, MediaPipe-FaceMeshPreprocessor,
#     feeds MediaPipeFaceMeshToSEGS whose SEGS output goes nowhere. This also
#     removes the mediapipe/onnxruntime dependency stack entirely. (The Canny
#     the README mentions is core ComfyUI's Canny, not controlnet_aux's.)
#   - ComfyUI-Inpaint-CropAndStitch -> both InpaintCropImproved nodes feed only
#     a preview and a terminal VAEEncodeTiled.
#   - rgthree-comfy -> "Power Lora Loader" with one enabled LoRA and a null
#     strengthTwo is exactly core LoraLoader(strength_model=strength_clip);
#     "Image Comparer" is preview-only.
#   - masquerade-nodes-comfyui -> "Get Image Size" is image.shape, inlined.
CUSTOM_NODES: list[dict[str, str]] = [
    {
        # UnetLoaderGGUF, CLIPLoaderGGUF -- V2 is GGUF-first.
        "name": "ComfyUI-GGUF",
        "url": "https://github.com/city96/ComfyUI-GGUF",
        "ref": "795e45156ece99afbc3efef911e63fcb46e6a20d",
    },
    {
        # BboxDetectorCombined_v2, MaskToSEGS, ImpactSEGSToMaskBatch, SEGSPaste,
        # ImpactSimpleDetectorSEGS, MediaPipeFaceMeshToSEGS, SAMLoader.
        "name": "ComfyUI-Impact-Pack",
        "url": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "ref": "6a517ebe06fea2b74fc41b3bd089c0d7173eeced",
    },
    {
        # UltralyticsDetectorProvider lives in the Subpack, not the Pack.
        "name": "ComfyUI-Impact-Subpack",
        "url": "https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
        "ref": "1.3.4",
    },
    {
        # ImageResizeKJv2
        "name": "ComfyUI-KJNodes",
        "url": "https://github.com/kijai/ComfyUI-KJNodes",
        "ref": "eaa9dbf239996ad575b5b276b745a2410c9e1ffc",
    },
    {
        # Registers the res_2s sampler / bong_tangent scheduler that the V2 face
        # detail pass selects.
        "name": "RES4LYF",
        "url": "https://github.com/ClownsharkBatwing/RES4LYF",
        # No tags published; pinned to a known commit rather than a branch.
        "ref": "6780724de0ddb98e7454d54bb80387836a911ece",
    },
]

# The pack bundled inside the reference repo. Not pip-installable and not on the
# ComfyUI registry, so it is copied out of the reference clone.
BUNDLED_PACK = "Comfyui-SynthidBypass"

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# (repo_id, filename_in_repo, destination relative to ComfyUI/models/)
#
# Sourced from the reference repo README section 8, with one correction: the
# README points `ae.safetensors` at `Comfy-Org/Z-Image-ComfyUI`, which does not
# exist (404). The live repo is `Comfy-Org/Z-Image-Turbo`.
MODELS: list[tuple[str, str, str]] = [
    # --- Qwen-Image path: global redraw ---
    (
        "unsloth/Qwen-Image-2512-GGUF",
        "qwen-image-2512-Q4_K_M.gguf",
        "diffusion_models/qwen-image-2512-Q4_K_M.gguf",
    ),
    (
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "clip/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    ),
    (
        "Comfy-Org/Qwen-Image_ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
        "vae/qwen_image_vae.safetensors",
    ),
    (
        "Comfy-Org/Qwen-Image-DiffSynth-ControlNets",
        "split_files/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors",
        "model_patches/qwen_image_canny_diffsynth_controlnet.safetensors",
    ),
    (
        "lightx2v/Qwen-Image-2512-Lightning",
        "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
        "loras/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
    ),
    # --- Z-Image Turbo path: face cleanup ---
    (
        "jayn7/Z-Image-Turbo-GGUF",
        "z_image_turbo-Q4_K_M.gguf",
        "diffusion_models/z_image_turbo-Q4_K_M.gguf",
    ),
    (
        "worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF",
        "Qwen_3_4b-imatrix-IQ4_XS.gguf",
        "clip/Qwen_3_4b-imatrix-IQ4_XS.gguf",
    ),
    (
        "Comfy-Org/Z-Image-Turbo",
        "split_files/vae/ae.safetensors",
        "vae/ae.safetensors",
    ),
    # --- Detection ---
    (
        "Bingsu/adetailer",
        "face_yolov8n.pt",
        "ultralytics/bbox/yolov8n-face.pt",
    ),
]

# SAM checkpoint is not on the Hub under a stable repo; fetched directly.
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_DEST = "sams/sam_vit_b_01ec64.pth"


def _run(cmd: list[str], cwd: str | None = None) -> None:
    """Run a command, surfacing stderr on failure instead of swallowing it."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _clone(url: str, dest: str, ref: str) -> None:
    """Clone `url` at `ref` into `dest`, skipping if already present."""
    if os.path.isdir(os.path.join(dest, ".git")):
        return
    if os.path.exists(dest):
        shutil.rmtree(dest)
    # Full clone: a shallow clone cannot check out an arbitrary commit SHA.
    _run(["git", "clone", "--filter=blob:none", url, dest])
    _run(["git", "checkout", "--detach", ref], cwd=dest)


def _link(src: str, dest: str) -> None:
    """Symlink `src` -> `dest`, replacing a stale or broken link."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.islink(dest) or os.path.exists(dest):
        if os.path.realpath(dest) == os.path.realpath(src):
            return
        os.remove(dest)
    os.symlink(src, dest)


def install_repos() -> None:
    """Clone ComfyUI, the reference repo, and every pinned custom node pack."""
    print(f"[setup] ComfyUI @ {COMFYUI_REF}", flush=True)
    _clone(COMFYUI_REPO, COMFYUI_DIR, COMFYUI_REF)

    print(f"[setup] reference workflow repo @ {REFERENCE_REF}", flush=True)
    _clone(REFERENCE_REPO, REFERENCE_DIR, REFERENCE_REF)

    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)
    for pack in CUSTOM_NODES:
        dest = os.path.join(CUSTOM_NODES_DIR, pack["name"])
        print(f"[setup] {pack['name']} @ {pack['ref']}", flush=True)
        _clone(pack["url"], dest, pack["ref"])

    # Vendor the reference repo's bundled pack. Upstream warns against nesting
    # it twice, so copy the inner directory to a flat destination.
    bundled_src = os.path.join(REFERENCE_DIR, "custom_nodes", BUNDLED_PACK)
    bundled_dest = os.path.join(CUSTOM_NODES_DIR, BUNDLED_PACK)
    if not os.path.isdir(bundled_src):
        raise RuntimeError(
            f"bundled node pack missing from reference repo: {bundled_src}"
        )
    if not os.path.isdir(bundled_dest):
        shutil.copytree(bundled_src, bundled_dest)
    print(f"[setup] {BUNDLED_PACK} (vendored from reference repo)", flush=True)


def install_models() -> None:
    """Fetch weights into the HF cache and symlink them into ComfyUI/models."""
    from huggingface_hub import hf_hub_download

    for index, (repo_id, filename, dest_rel) in enumerate(MODELS, start=1):
        dest = os.path.join(MODELS_DIR, dest_rel)
        if os.path.exists(dest):
            continue
        print(
            f"[models] ({index}/{len(MODELS)}) {os.path.basename(dest_rel)}",
            flush=True,
        )
        # Download into the shared HF cache, then symlink. Note: passing
        # `local_dir` would nest the repo's subdirectories under it, and the old
        # `local_dir_use_symlinks` argument was REMOVED in huggingface_hub 1.0
        # (it is a hard TypeError now, not a warning).
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        _link(cached, dest)

    sam_dest = os.path.join(MODELS_DIR, SAM_DEST)
    if not os.path.exists(sam_dest):
        print("[models] sam_vit_b_01ec64.pth", flush=True)
        os.makedirs(os.path.dirname(sam_dest), exist_ok=True)
        _run(["curl", "-fsSL", "-o", sam_dest, SAM_URL])


def main() -> None:
    install_repos()
    install_models()
    print("[setup] complete", flush=True)


if __name__ == "__main__":
    main()
