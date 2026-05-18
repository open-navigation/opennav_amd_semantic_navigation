from .tracker import SAM3OnnxTracker, preprocess_image, retarget_resolution

# Apply ROCm compatibility patches at import time (scipy fill_holes + PyTorch NMS).
# Idempotent — safe to import tracker multiple times.
from .rocm_patches import apply as _apply_rocm_patches
_apply_rocm_patches()
