"""ROCm compatibility patches for Sam3VideoModel post-processing.

Applied at import time by calling apply(). Replaces functions in
transformers.models.sam3_video.modeling_sam3_video that silently skip
post-processing when the CUDA-only cv_utils kernel is unavailable.

Fixes:
  - fill_holes / sprinkle_removal: scipy.ndimage.label connected components
  - NMS: pure-PyTorch greedy NMS on the already-computed IoU matrix (GPU)
"""
from __future__ import annotations

import torch


# ─── replacement implementations ───────────────────────────────────────────

def _cc_scipy(mask: torch.Tensor):
    """Connected components via scipy.ndimage — CPU, ROCm-compatible."""
    import numpy as np
    import scipy.ndimage

    mask = mask.to(torch.uint8)
    _, _, H, W = mask.shape
    np_mask = mask.cpu().numpy()
    B, C = np_mask.shape[:2]
    labels = torch.zeros_like(mask, dtype=torch.int32)
    counts = torch.zeros_like(mask, dtype=torch.int32)
    for b in range(B):
        for c in range(C):
            lbl, n = scipy.ndimage.label(np_mask[b, c])
            lbl_t = torch.from_numpy(lbl.astype(np.int32))
            labels[b, c] = lbl_t
            if n > 0:
                areas = torch.bincount(lbl_t.flatten(), minlength=n + 1)
                counts[b, c] = areas[lbl_t]
            else:
                counts[b, c] = H * W + 1
    return labels.to(mask.device), counts.to(mask.device)


def _greedy_nms(ious: torch.Tensor, probs: torch.Tensor,
                iou_threshold: float) -> torch.Tensor:
    """Greedy NMS on a precomputed (N, N) IoU matrix — pure PyTorch, GPU."""
    order = probs.argsort(descending=True)
    suppressed = torch.zeros(len(probs), dtype=torch.bool, device=probs.device)
    kept = []
    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(idx)
        suppressed = suppressed | (ious[idx] > iou_threshold)
    if not kept:
        return torch.zeros(0, dtype=torch.long, device=probs.device)
    return torch.stack(kept)


# ─── apply ──────────────────────────────────────────────────────────────────

def apply() -> None:
    """Monkey-patch Sam3VideoModel post-processing for ROCm compatibility.

    Safe to call multiple times (idempotent via sentinel attribute).
    No-op if transformers is not importable.
    """
    try:
        import transformers.models.sam3_video.modeling_sam3_video as mod
    except ImportError:
        return

    if getattr(mod, "_rocm_patches_applied", False):
        return

    def _patched_cc(mask):
        # cv_utils kernel has no ROCm build variant; skip the HF Hub
        # lookup and always use the scipy fallback on this platform.
        _, _, H, W = mask.shape
        try:
            return _cc_scipy(mask)
        except Exception:
            m = mask.to(torch.uint8)
            return (torch.zeros_like(m, dtype=torch.int32),
                    torch.full_like(m, fill_value=H * W + 1, dtype=torch.int32))

    def _patched_nms(pred_probs, pred_masks, prob_threshold, iou_threshold):
        is_valid = pred_probs > prob_threshold
        probs = pred_probs[is_valid]
        masks_binary = pred_masks[is_valid] > 0
        if probs.numel() == 0:
            return is_valid

        ious = mod.mask_iou(masks_binary, masks_binary)

        # cv_utils kernel has no ROCm build variant; use the pure-PyTorch
        # greedy NMS directly to avoid the HF Hub lookup at startup.
        kept_inds = _greedy_nms(ious, probs, iou_threshold)

        valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1,
                                 torch.tensor(-1, device=is_valid.device))
        return torch.isin(valid_inds, kept_inds)

    mod._get_connected_components_with_padding = _patched_cc
    mod.nms_masks = _patched_nms
    mod._rocm_patches_applied = True
