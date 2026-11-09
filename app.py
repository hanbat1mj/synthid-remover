"""
SynthID Watermark Remover -- Hugging Face Space.

A headless port of the V2 ComfyUI workflow from cebeuq/Synthid-Bypass
(`Synthid-Bypass-v2.0.json`) to a ZeroGPU Gradio Space.

Why this runs ComfyUI *in-process* rather than as a subprocess
--------------------------------------------------------------
The previous revision started `ComfyUI/main.py` with `subprocess.Popen` inside a
`@spaces.GPU` function and talked to it over HTTP. That cannot work on ZeroGPU:
GPU access is granted per-PID to the forked worker that runs the decorated
function, so an exec'd child process gets no CUDA at all. It also failed earlier
than that -- ComfyUI was cloned at unpinned `master`, whose requirements had
grown a hard `comfy_aimdo` import that was never installed, so the server died
on startup and every request timed out after ~90s.

Both problems are addressed here: ComfyUI is imported into this process and its
node classes are called directly, and every dependency is pinned (see
`install.py` and `requirements.txt`).
"""

from __future__ import annotations

# `spaces` must be imported before torch or anything that can initialize CUDA,
# or it raises "CUDA has been initialized before importing the `spaces` package".
import spaces  # noqa: F401  # isort: skip

import asyncio
import os
import random
import sys
import threading
import traceback

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

import install

# --------------------------------------------------------------------------
# 1. Environment: clone pinned ComfyUI + node packs, fetch weights.
#    Must happen before ComfyUI can be imported.
# --------------------------------------------------------------------------
install.main()

COMFYUI_DIR = install.COMFYUI_DIR
sys.path.insert(0, COMFYUI_DIR)

# ComfyUI only parses sys.argv if comfy.options.enable_args_parsing() is called,
# and the sole caller is main.py -- which this module deliberately never imports
# (importing it would also re-trigger `import comfy_aimdo.control`). Clearing
# argv anyway guards against a custom node pack enabling arg parsing itself.
sys.argv = [sys.argv[0]]

from comfy.cli_args import args  # noqa: E402

args.base_directory = COMFYUI_DIR
# Load only the packs this workflow needs, by directory name, so an unrelated
# pack on disk can never break startup.
args.disable_all_custom_nodes = True
args.whitelist_custom_nodes = [
    "ComfyUI-GGUF",
    "ComfyUI-Impact-Pack",
    "ComfyUI-Impact-Subpack",
    "ComfyUI-KJNodes",
    "RES4LYF",
    install.BUNDLED_PACK,
]

import folder_paths  # noqa: E402

for _directory in (
    folder_paths.get_input_directory(),
    folder_paths.get_output_directory(),
    folder_paths.get_temp_directory(),
):
    os.makedirs(_directory, exist_ok=True)


# --------------------------------------------------------------------------
# 2. Register nodes.
# --------------------------------------------------------------------------
def _force_comfy_utils_package():
    """
    Make the top-level name `utils` resolve to ComfyUI/utils.

    ComfyUI claims several very generic top-level module names (`utils`, `app`,
    `server`, `nodes`). Some other installed distribution also provides a
    top-level `utils` module, and whichever one lands in sys.modules first
    wins. When the other one won, ComfyUI/app/frontend_management.py's
    `from utils.install_util import ...` failed with:

        ModuleNotFoundError: No module named 'utils.install_util';
        'utils' is not a package

    Path order alone is not enough to prevent this -- ComfyUI's directory is
    first on sys.path, but an already-imported module is never re-resolved. So
    purge any cached `utils` and load ComfyUI's package explicitly by path.
    """
    import importlib.util

    comfy_utils_init = os.path.join(COMFYUI_DIR, "utils", "__init__.py")
    existing = sys.modules.get("utils")
    if existing is not None:
        existing_file = getattr(existing, "__file__", "") or ""
        if existing_file and os.path.realpath(existing_file) == os.path.realpath(
            comfy_utils_init
        ):
            return
        print(
            f"[setup] top-level 'utils' resolved to {existing_file or '<unknown>'}; "
            "replacing it with ComfyUI's utils package",
            flush=True,
        )

    for name in [n for n in list(sys.modules) if n == "utils" or n.startswith("utils.")]:
        del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        "utils",
        comfy_utils_init,
        submodule_search_locations=[os.path.join(COMFYUI_DIR, "utils")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = module
    spec.loader.exec_module(module)


def _bootstrap_nodes():
    """Register comfy_extras + the whitelisted custom node packs."""
    _force_comfy_utils_package()

    import server
    from nodes import init_extra_nodes

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Impact Pack registers aiohttp routes against PromptServer.instance at
    # import time, so a real PromptServer must exist before node init. A stub
    # will not do. Constructing it opens no socket -- PromptServer.start() does
    # that, and is never called here.
    #
    # The constructor gained a required `asset_manager` argument after v0.7.0,
    # so try the newer signature first and fall back. Assets default to
    # disabled, which makes default_asset_manager() a no-op (no DB, no scan).
    try:
        from app.assets.manager import default_asset_manager

        server.PromptServer(loop, default_asset_manager())
    except ImportError:
        server.PromptServer(loop)

    # init_extra_nodes is a coroutine on modern ComfyUI. Calling it without
    # awaiting silently registers nothing and every custom node lookup then
    # raises KeyError. init_api_nodes=False skips the cloud/partner nodes.
    loop.run_until_complete(
        init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)
    )
    return loop


EVENT_LOOP = _bootstrap_nodes()

from nodes import NODE_CLASS_MAPPINGS  # noqa: E402

_NODE_INSTANCES: dict[str, object] = {}


def node(class_type: str, **inputs):
    """
    Call a ComfyUI node by its registered class name with keyword inputs.

    Looks up the node's declared FUNCTION rather than hardcoding method names,
    supplies any `hidden` inputs it asks for, and validates input names against
    INPUT_TYPES so a typo surfaces as a precise error instead of a TypeError
    from deep inside the node.
    """
    try:
        cls = NODE_CLASS_MAPPINGS[class_type]
    except KeyError:
        raise RuntimeError(
            f"node {class_type!r} is not registered -- its custom node pack "
            f"failed to load. Registered packs: {args.whitelist_custom_nodes}"
        ) from None

    schema = cls.INPUT_TYPES()
    accepted = set(schema.get("required", {})) | set(schema.get("optional", {}))
    unknown = set(inputs) - accepted
    if unknown:
        raise RuntimeError(
            f"{class_type}: unknown input(s) {sorted(unknown)}; "
            f"accepted: {sorted(accepted)}"
        )
    missing = set(schema.get("required", {})) - set(inputs)
    if missing:
        raise RuntimeError(f"{class_type}: missing required input(s) {sorted(missing)}")

    for name, kind in schema.get("hidden", {}).items():
        if kind == "UNIQUE_ID":
            inputs.setdefault(name, 0)
        elif kind in ("PROMPT", "EXTRA_PNGINFO"):
            inputs.setdefault(name, {})

    instance = _NODE_INSTANCES.setdefault(class_type, cls())
    result = getattr(instance, cls.FUNCTION)(**inputs)
    # Nodes may return a bare tuple or {"ui":..., "result":...}.
    if isinstance(result, dict):
        return result.get("result", result)
    return result


# --------------------------------------------------------------------------
# 3. Tensor helpers. ComfyUI IMAGE == float32 NHWC in [0, 1].
# --------------------------------------------------------------------------
def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).numpy()
    return Image.fromarray(array[0])


def size_of(tensor: torch.Tensor) -> tuple[int, int]:
    """(width, height) -- the masquerade `Get Image Size` node, inlined."""
    return int(tensor.shape[2]), int(tensor.shape[1])


# --------------------------------------------------------------------------
# 4. Workflow constants, lifted verbatim from Synthid-Bypass-v2.0.json.
#    Node ids from that file are kept in comments for traceability.
# --------------------------------------------------------------------------
QWEN_UNET_FILE = "qwen-image-2512-Q4_K_M.gguf"                  # 605
QWEN_CLIP_FILE = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"           # 608
QWEN_VAE_FILE = "qwen_image_vae.safetensors"                    # 12
QWEN_CONTROLNET_FILE = "qwen_image_canny_diffsynth_controlnet.safetensors"  # 401
QWEN_LORA_FILE = "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"   # 564

ZIMAGE_UNET_FILE = "z_image_turbo-Q4_K_M.gguf"                  # 606
ZIMAGE_CLIP_FILE = "Qwen_3_4b-imatrix-IQ4_XS.gguf"              # 607
ZIMAGE_VAE_FILE = "ae.safetensors"                              # 582

FACE_DETECTOR_FILE = "bbox/yolov8n-face.pt"                     # 516
SAM_FILE = "sam_vit_b_01ec64.pth"                               # 612

# Both CLIP loaders use type "lumina2", exactly as the reference workflow does.
#
# That looks wrong on node 608, which loads a Qwen2.5-VL encoder for the
# Qwen-Image path where "qwen_image" would be the obvious choice. It is in fact
# a no-op: comfy/sd.py picks the text-encoder implementation by *state-dict
# detection*, not by this argument. Qwen2.5-VL-7B is detected as
# TEModel.QWEN25_7B and routed to comfy.text_encoders.qwen_image.te unless the
# type is specifically HUNYUAN_IMAGE; Qwen3-4B is detected as TEModel.QWEN3_4B
# and routed to comfy.text_encoders.z_image.te. So both files load through the
# correct encoder either way, and the reference value is kept verbatim.
ZIMAGE_CLIP_TYPE = "lumina2"
QWEN_CLIP_TYPE = "lumina2"

LORA_STRENGTH = 0.8          # 564: strength 0.8, strengthTwo null -> clip 0.8
AURA_SHIFT = 3               # 6
CANNY_LOW = 0.05             # 551
CANNY_HIGH = 0.25            # 551
CONTROLNET_STRENGTH = 1      # 552

POSITIVE_PROMPT = "ultra clear and smoothe skin, spotless skin"   # 9
NEGATIVE_PROMPT = "moles, freckes, high detail skin"              # 10
FACE_POSITIVE_PROMPT = ""                                         # 583
FACE_NEGATIVE_PROMPT = "blurry, ugly, bad quality,"               # 581

# 3: KSampler. `denoise` is driven by the adaptive-denoise node, not a literal.
SAMPLER_STEPS = 4
SAMPLER_CFG = 1
SAMPLER_NAME = "dpmpp_2m"
SAMPLER_SCHEDULER = "sgm_uniform"

# 595: Synthid-Bypass-AdaptiveDenoise
DENOISE_MIN = 0.08
DENOISE_MAX = 0.15
DEFAULT_ADAPTIVE_LEVEL = 6

# A cold worker pays the model mmap + host->VRAM transfer on its first call, so
# this has to cover that plus a full run. ZeroGPU rejects a declared duration
# above the caller's quota ("ZeroGPU illegal duration"), and anonymous visitors
# have the smallest quota of all -- so this is deliberately conservative rather
# than generous. Raise it if cold-start runs get aborted.
GPU_DURATION = 120


# --------------------------------------------------------------------------
# 5. Lazy model loading -- MUST happen inside the @spaces.GPU worker.
# --------------------------------------------------------------------------
# This is the one structural constraint ZeroGPU imposes on a GGUF workflow, and
# getting it wrong is fatal rather than slow.
#
# At module scope `spaces` has torch patched, which includes a TorchDispatchMode
# that routes `aten::alias` through Python. `GGMLTensor(t, ...)` goes
# Tensor.__new__ -> base_tensor_ctor -> t.alias(), and that Python round-trip
# binds a plain `torch.Tensor` pyobj to the fresh TensorImpl. The C++ guard in
# THPVariable_NewWithVar then refuses to rebind it as a subclass:
#
#   RuntimeError: Creating a new Tensor subclass GGMLTensor but the raw Tensor
#   object is already associated to a python object of type Tensor which is not
#   a subclass of the requested type
#
# Inside a @spaces.GPU function `spaces` has called torch.unpatch() and real
# CUDA is attached, so the construction succeeds normally. Defining the
# GGMLTensor class at import is fine; instantiating one is not.
#
# (A `GGMLTensor.__new__` shim routing through `Tensor._make_subclass` does fix
# the constructor -- that call disables dispatch modes in C++ -- but it only
# moves the failure: GGMLTensor.shape is the logical shape while its storage is
# packed uint8, so ZeroGPU's startup packing then dies in pack_to_cuda's
# `set_(storage, 0, tensor.shape, tensor.stride())`. Loading in the worker is
# the approach that actually works.)
#
# Worker reuse is what makes this cheap: `spaces.zero.wrappers` keeps a
# `workers` dict in the main process and hands an idle worker back for
# subsequent calls, and the worker body is a long-lived loop that initializes
# once. So this cache survives across requests until the scheduler reclaims the
# GPU slot. A cold worker re-pays only an mmap over warm page cache --
# ComfyUI-GGUF's loader is `torch.from_numpy(tensor.data)` over a memory-mapped
# file, not a 29GB read.
_MODEL_LOCK = threading.Lock()
_MODELS: dict[str, object] = {}


def models() -> dict[str, object]:
    """Load every model once per worker. Only call inside @spaces.GPU."""
    with _MODEL_LOCK:
        if _MODELS:
            return _MODELS

        print("[models] loading Qwen-Image path", flush=True)
        qwen_unet = node("UnetLoaderGGUF", unet_name=QWEN_UNET_FILE)
        qwen_clip = node("CLIPLoaderGGUF", clip_name=QWEN_CLIP_FILE, type=QWEN_CLIP_TYPE)

        # 564: "Power Lora Loader (rgthree)" with a single enabled LoRA and a
        # null strengthTwo is exactly core LoraLoader with
        # strength_clip == strength_model, so rgthree is not needed.
        qwen_model, qwen_clip_patched = node(
            "LoraLoader",
            model=qwen_unet[0],
            clip=qwen_clip[0],
            lora_name=QWEN_LORA_FILE,
            strength_model=LORA_STRENGTH,
            strength_clip=LORA_STRENGTH,
        )

        print("[models] loading Z-Image Turbo path", flush=True)
        zimage_unet = node("UnetLoaderGGUF", unet_name=ZIMAGE_UNET_FILE)
        zimage_clip = node(
            "CLIPLoaderGGUF", clip_name=ZIMAGE_CLIP_FILE, type=ZIMAGE_CLIP_TYPE
        )

        print("[models] loading VAEs, controlnet and detectors", flush=True)
        _MODELS.update(
            qwen_sampling=node(
                "ModelSamplingAuraFlow", model=qwen_model, shift=AURA_SHIFT
            )[0],
            qwen_clip=qwen_clip_patched,
            qwen_vae=node("VAELoader", vae_name=QWEN_VAE_FILE)[0],
            qwen_controlnet=node("ModelPatchLoader", name=QWEN_CONTROLNET_FILE)[0],
            zimage_unet=zimage_unet[0],
            zimage_clip=zimage_clip[0],
            zimage_vae=node("VAELoader", vae_name=ZIMAGE_VAE_FILE)[0],
            bbox_detector=node(
                "UltralyticsDetectorProvider", model_name=FACE_DETECTOR_FILE
            )[0],
            sam=node("SAMLoader", model_name=SAM_FILE, device_mode="AUTO")[0],
        )
        print("[models] ready", flush=True)
        return _MODELS

# RES4LYF contributes no nodes -- it appends `res_2s` to
# comfy.samplers.KSampler.SAMPLERS and `bong_tangent` to Impact Pack's scheduler
# list. Those are combo *values*, so nothing in the workflow JSON references the
# pack by name and a silent load failure would only surface deep inside the face
# pass. Check it up front instead.
def _assert_res4lyf():
    import comfy.samplers

    detailer_schema = NODE_CLASS_MAPPINGS["SEGSDetailerModelSwap"].INPUT_TYPES()
    samplers = list(comfy.samplers.KSampler.SAMPLERS)
    schedulers = list(detailer_schema["required"]["scheduler"][0])
    missing = []
    if "res_2s" not in samplers:
        missing.append("sampler 'res_2s'")
    if "bong_tangent" not in schedulers:
        missing.append("scheduler 'bong_tangent'")
    if missing:
        raise RuntimeError(
            "RES4LYF did not register: " + ", ".join(missing) + ". The face "
            "reconstruction pass requires both. Check that the RES4LYF pack "
            "cloned and imported cleanly."
        )


_assert_res4lyf()
print("[setup] nodes registered; models load on the first request", flush=True)


# --------------------------------------------------------------------------
# 6. The pipeline.
# --------------------------------------------------------------------------
@spaces.GPU(duration=GPU_DURATION)
def remove_watermark(
    input_image: Image.Image | None,
    adaptive_level: int = DEFAULT_ADAPTIVE_LEVEL,
    seed: int = -1,
):
    # No gr.Progress() parameter: ZeroGPU runs this in a forked worker and the
    # arguments cross a pickling boundary, which a live Progress handle does not
    # survive. Stage logging goes to the Space logs instead.
    if input_image is None:
        raise gr.Error("Upload an image first.")

    if seed is None or int(seed) < 0:
        global_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    else:
        global_seed = int(seed)
    face_seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)

    try:
        m = models()
        with torch.inference_mode():
            # 11 LoadImage. The image is used at native resolution: node 609
            # (ImageScaleToTotalPixels) is bypassed in the reference workflow,
            # so it is a pass-through and is not reproduced here.
            image = pil_to_tensor(input_image)

            # 595 Synthid-Bypass-AdaptiveDenoise -> denoise for the redraw pass.
            print("[run] Computing adaptive denoise", flush=True)
            denoise = node(
                "Synthid-Bypass-AdaptiveDenoise",
                image=image,
                adaptive_level=int(adaptive_level),
                denoise_min=DENOISE_MIN,
                denoise_max=DENOISE_MAX,
            )[0]

            # 9 / 10 CLIP Text Encode for the Qwen-Image path.
            print("[run] Encoding prompts", flush=True)
            positive = node(
                "CLIPTextEncode", clip=m["qwen_clip"], text=POSITIVE_PROMPT
            )[0]
            negative = node(
                "CLIPTextEncode", clip=m["qwen_clip"], text=NEGATIVE_PROMPT
            )[0]

            # 551 Canny -> 552 QwenImageDiffsynthControlnet: lock composition.
            print("[run] Extracting structure", flush=True)
            edges = node(
                "Canny",
                image=image,
                low_threshold=CANNY_LOW,
                high_threshold=CANNY_HIGH,
            )[0]
            controlled_model = node(
                "QwenImageDiffsynthControlnet",
                model=m["qwen_sampling"],
                model_patch=m["qwen_controlnet"],
                vae=m["qwen_vae"],
                image=edges,
                strength=CONTROLNET_STRENGTH,
            )[0]

            # 8 VAEEncode -> 3 KSampler -> 61 VAEDecode: the global redraw.
            print("[run] Redrawing image", flush=True)
            latent = node("VAEEncode", pixels=image, vae=m["qwen_vae"])[0]
            sampled = node(
                "KSampler",
                model=controlled_model,
                positive=positive,
                negative=negative,
                latent_image=latent,
                seed=global_seed,
                steps=SAMPLER_STEPS,
                cfg=SAMPLER_CFG,
                sampler_name=SAMPLER_NAME,
                scheduler=SAMPLER_SCHEDULER,
                denoise=denoise,
            )[0]
            decoded = node("VAEDecode", samples=sampled, vae=m["qwen_vae"])[0]

            # 520 Get Image Size -> 568 ImageResizeKJv2: restore the redraw to
            # the input's exact dimensions (the VAE round-trip pads to a
            # multiple of the latent stride).
            print("[run] Restoring resolution", flush=True)
            width, height = size_of(image)
            redrawn = node(
                "ImageResizeKJv2",
                image=decoded,
                width=width,
                height=height,
                upscale_method="lanczos",
                keep_proportion="resize",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=1,
                device="cpu",
            )[0]

            # 575 Get Image Size -> 576 ImageResizeKJv2: the untouched input,
            # matched to the redraw's geometry. Faces are detected and
            # re-rendered from *this* image, then pasted onto the redraw.
            redrawn_width, redrawn_height = size_of(redrawn)
            reference = node(
                "ImageResizeKJv2",
                image=image,
                width=redrawn_width,
                height=redrawn_height,
                upscale_method="lanczos",
                keep_proportion="crop",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=1,
                device="cpu",
            )[0]

            # 613 Simple Detector (SEGS), SAM-refined.
            print("[run] Detecting faces", flush=True)
            segs = node(
                "ImpactSimpleDetectorSEGS",
                bbox_detector=m["bbox_detector"],
                image=reference,
                bbox_threshold=0.2,
                bbox_dilation=0,
                crop_factor=2.5,
                drop_size=1,
                sub_threshold=0.93,
                sub_dilation=0,
                sub_bbox_expansion=0,
                sam_mask_hint_threshold=0.7,
                post_dilation=0,
                sam_model_opt=m["sam"],
            )[0]

            # 583 / 581 CLIP Text Encode for the Z-Image face pass.
            face_positive = node(
                "CLIPTextEncode", clip=m["zimage_clip"], text=FACE_POSITIVE_PROMPT
            )[0]
            face_negative = node(
                "CLIPTextEncode", clip=m["zimage_clip"], text=FACE_NEGATIVE_PROMPT
            )[0]

            # 592 Synthid-Bypass-Facedetailer: swap to the Z-Image Turbo path
            # for a face-only cleanup with adaptive denoise.
            print("[run] Reconstructing faces", flush=True)
            detailed = node(
                "SEGSDetailerModelSwap",
                image=reference,
                segs=segs,
                model=m["zimage_unet"],
                clip=m["zimage_clip"],
                vae=m["zimage_vae"],
                positive=face_positive,
                negative=face_negative,
                guide_size=768,
                guide_size_for=True,
                max_size=1024,
                seed=face_seed,
                steps=8,
                cfg=1,
                sampler_name="res_2s",
                scheduler="bong_tangent",
                denoise=0.1,
                noise_mask=True,
                force_inpaint=True,
                refiner_ratio=0.2,
                batch_size=1,
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                adaptive_denoise=True,
                adaptive_mode="largest_face",
                adaptive_ratio=0.03,
                adaptive_denoise_min=0.05,
                adaptive_denoise_max=0.28,
            )[0]

            # 574 SEGSPaste: composite the reconstructed faces onto the redraw.
            print("[run] Compositing", flush=True)
            result = node(
                "SEGSPaste",
                image=redrawn,
                segs=detailed,
                feather=10,
                alpha=255,
            )[0]

            return tensor_to_pil(result)

    except gr.Error:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        raise gr.Error(
            "Ran out of GPU memory. This workflow processes images at their "
            "native resolution, so try a smaller input image."
        ) from exc
    except Exception as exc:
        # Surface the real failure. The previous revision swallowed every error
        # into a generic timeout after ~3 minutes, which told users nothing.
        traceback.print_exc()
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------
# 7. UI
# --------------------------------------------------------------------------
CSS = """
#container { max-width: 1200px; margin: 0 auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="SynthID Remover") as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown("# SynthID Remover")
        gr.Markdown(
            "Removes SynthID watermarks by re-rendering an image through a "
            "constrained diffusion reconstruction: a ControlNet-locked global "
            "redraw, followed by a dedicated face reconstruction pass. A port "
            "of the V2 workflow from "
            "[cebeuq/Synthid-Bypass](https://github.com/cebeuq/Synthid-Bypass). "
            "**Research tool** — please read the ethical use notes in the "
            "Space README."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image", height=512)
            with gr.Column():
                output_image = gr.Image(
                    type="pil", label="Cleaned Image", height=512, interactive=False
                )

        submit = gr.Button("Remove Watermark", variant="primary")

        with gr.Accordion("Settings", open=False):
            adaptive_level = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=DEFAULT_ADAPTIVE_LEVEL,
                label="Adaptive level",
                info=(
                    "Drives the redraw denoise strength from the input "
                    "resolution. 5 is neutral. Higher resolutions are harder "
                    "to bypass and may need 7–8, at the cost of fidelity."
                ),
            )
            seed = gr.Number(
                value=-1,
                precision=0,
                label="Seed",
                info="-1 picks a random seed each run.",
            )

        submit.click(
            fn=remove_watermark,
            inputs=[input_image, adaptive_level, seed],
            outputs=[output_image],
        )

        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                """
### Technical breakdown

The watermark is treated as a layer of low-level image noise, so the image is
re-rendered while its composition is held in place.

1. **Adaptive denoise** — `Synthid-Bypass-AdaptiveDenoise` derives the redraw
   strength from the input's megapixel count, mapping ~0.30 MP to 0.08 and
   ~3.70 MP to 0.15, then offsetting by the adaptive level.
2. **Global redraw** — a 4-step `dpmpp_2m` / `sgm_uniform` pass through
   **Qwen-Image 2512** (Q4_K_M) with a 4-step Lightning LoRA, while a Canny
   edge map drives `QwenImageDiffsynthControlnet` to lock the composition.
3. **Face segmentation** — YOLOv8 face detection refined with SAM, converted
   to SEGS regions.
4. **Face reconstruction** — `Synthid-Bypass-Facedetailer` swaps to
   **Z-Image Turbo** (Q4_K_M) for a face-only pass using the `res_2s` sampler
   and `bong_tangent` scheduler, with adaptive per-face denoise.
5. **Composite** — the reconstructed faces are pasted back onto the redraw.

Images are processed at their **native resolution**, matching the reference
workflow, which ships with its rescale node bypassed.
"""
            )

if __name__ == "__main__":
    # In Gradio 6, css/theme/js moved from Blocks() to launch(). Passing css to
    # Blocks() there does not raise -- it is silently ignored.
    demo.queue().launch(css=CSS)
