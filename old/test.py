# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Example adapted from: https://huggingface.co/facebook/sam3

import os
# Unified-memory friendly allocator on ROCm; ignored on NVIDIA/CPU.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HSA_XNACK", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/cache/inductor")

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import numpy as np
import matplotlib
import hashlib
from nicegui import ui
import base64
import io
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
COMPILED_MODEL_PATH = "/cache/sam3_compiled.pt"

torch._dynamo.config.capture_scalar_outputs = True
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = Sam3Model.from_pretrained(
    "facebook/sam3",
    torch_dtype=dtype,
    attn_implementation="sdpa",
).to(device)
model.eval()
model = model.to(memory_format=torch.channels_last)
# model = torch.compile(model, mode="max-autotune", fullgraph=True)
processor = Sam3Processor.from_pretrained("facebook/sam3")

if os.path.exists(COMPILED_MODEL_PATH):
    print(f"Loading compiled model state from {COMPILED_MODEL_PATH}")
    model.load_state_dict(torch.load(COMPILED_MODEL_PATH, map_location=device))
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    print("Skipping warmup compile — inductor cache should populate from /cache/inductor")
else:
    print("Compiling model (first run, this will take a few minutes)...")
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    # SAM3 always processes at 1024x1024
    _dummy_image = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))
    _warmup_probe = processor(images=_dummy_image, text="warmup", return_tensors="pt")
    _warmup_all = {k: v.to(device) if torch.is_tensor(v) else v
                   for k, v in _warmup_probe.items()}
    with torch.inference_mode():
        _ = model(**_warmup_all)
    if device == "cuda":
        torch.cuda.synchronize()
    os.makedirs("/cache", exist_ok=True)
    torch.save(model._orig_mod.state_dict(), COMPILED_MODEL_PATH)
    print(f"Saved model state to {COMPILED_MODEL_PATH}")
    
# Load image, keep original for final-resolution output, and build a downscaled
# copy (longest edge = 1024) for inference. Masks produced at 1024 will be
# upsampled back to the original size via target_sizes in post-processing.
image = Image.open("/ryzers/data/toucan.jpg").convert("RGB")
original_size = image.size  # (w, h)

TARGET_LONGEST_EDGE = 1024
w, h = original_size
scale = TARGET_LONGEST_EDGE / max(w, h)
if scale < 1.0:
    infer_image = image.resize(
        (int(round(w * scale)), int(round(h * scale))),
        Image.BILINEAR,
    )
else:
    infer_image = image

# Preallocate device-resident input buffers once. On Ryzen AI Max the iGPU shares
# LPDDR5X with the CPU, so keeping these persistent avoids per-frame allocation
# churn and the staging copy done by pin_memory().to(device).
_probe = processor(images=infer_image, text="bird", return_tensors="pt")
_device_buffers = {
    k: torch.empty_like(v, device=device, dtype=(dtype if v.is_floating_point() else v.dtype))
    for k, v in _probe.items() if torch.is_tensor(v)
}
_non_tensor = {k: v for k, v in _probe.items() if not torch.is_tensor(v)}

# Segment using text prompt
for i in range(20):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    cpu_inputs = processor(images=infer_image, text="bird", return_tensors="pt")
    # Copy straight into the preallocated device buffers — no pin_memory staging,
    # no per-frame device allocation.
    for k, buf in _device_buffers.items():
        src = cpu_inputs[k]
        if src.is_floating_point():
            src = src.to(dtype)
        buf.copy_(src, non_blocking=True)
    inputs = {**_non_tensor, **_device_buffers}
    # Upsample masks back to the *original* image size, not the downscaled one.
    target_sizes = [[original_size[1], original_size[0]]]
    if device == "cuda":
        torch.cuda.synchronize()
    t_proc = time.time() - start

    t_model_start = time.time()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    t_model = time.time() - t_model_start

    t_post_start = time.time()
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=target_sizes,
    )[0]
    if device == "cuda":
        torch.cuda.synchronize()
    t_post = time.time() - t_post_start

    print(f"Iteration {i + 1}: proc={t_proc:.3f}s  model={t_model:.3f}s  post={t_post:.3f}s  total={t_proc + t_model + t_post:.3f}s")

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores

# Function to overlay masks on the image
def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]
    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

image_with_mask = overlay_masks(image, results["masks"])

# Convert PIL images to base64 data URIs for NiceGUI
def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

# Create NiceGUI page to display results
@ui.page('/')
def index():
    ui.label('SAM3 segmentation results').classes('text-2xl font-bold mb-4')
    with ui.row().classes('w-full gap-4'):
        with ui.column().classes('w-[45%]'):
            ui.label('Original image')
            ui.image(pil_to_base64(image.convert("RGB"))).classes('w-full')
        with ui.column().classes('w-[45%]'):
            ui.label('Segmentation result')
            ui.image(pil_to_base64(image_with_mask.convert("RGB"))).classes('w-full')

ui.run()