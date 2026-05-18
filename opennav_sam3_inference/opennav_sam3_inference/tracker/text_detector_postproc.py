"""
Detector head postprocessing for SAM3 text-prompt path.

Ports the host-side logic from `Sam3VideoModel.run_detection`
(transformers.models.sam3_video.modeling_sam3_video) so it can run on top
of a `Sam3Model.forward` call (single-image text→mask) without depending
on the full `Sam3VideoModel` machinery.

Two pieces:
  1. presence-aware score: pred_probs = sigmoid(pred_logits) * sigmoid(presence_logits)
  2. mask-IoU NMS over the N queries (greedy, pure Python — no CUDA kernel)

Why a Python NMS?
  The official `nms_masks` calls `cv_utils_kernel.generic_nms`, a CUDA
  kernel that isn't available on ROCm in this stack. Greedy Python NMS
  on N=200 candidates × 288×288 masks runs in single-digit ms on GPU.

Mask resize follows `Sam3ImageProcessor.post_process_masks`:
  bilinear-interpolate logits to original (H, W), then threshold at 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _mask_iou(masks: torch.Tensor) -> torch.Tensor:
    """Pair-wise IoU over a stack of binary masks.

    masks: (N, H, W) bool or {0, 1}
    returns: (N, N) IoU
    """
    flat = masks.flatten(1).float()           # (N, HW)
    inter = flat @ flat.T                     # (N, N)
    area = flat.sum(dim=1, keepdim=True)      # (N, 1)
    union = area + area.T - inter
    return inter / union.clamp(min=1.0)


def greedy_mask_nms(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Greedy NMS by mask IoU.

    pred_probs: (N,) probabilities
    pred_masks: (N, H, W) logits (binarized at >0 internally)
    Returns LongTensor of indices to keep, sorted by score descending.
    """
    if pred_probs.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=pred_probs.device)

    masks_bin = pred_masks > 0
    ious = _mask_iou(masks_bin)
    order = pred_probs.argsort(descending=True)

    suppressed = torch.zeros(pred_probs.numel(), dtype=torch.bool, device=pred_probs.device)
    kept: list[int] = []
    for idx_t in order.tolist():
        if suppressed[idx_t]:
            continue
        kept.append(idx_t)
        suppressed |= ious[idx_t] > iou_threshold
    return torch.tensor(kept, dtype=torch.long, device=pred_probs.device)


@dataclass
class Detection:
    score: float                      # presence-aware probability in [0, 1]
    mask_logits: torch.Tensor         # (H_mask, W_mask) logits at decoder resolution
    bbox_xyxy: torch.Tensor           # (4,) — model-space xyxy from detector head


def postprocess_detector_outputs(
    pred_logits: torch.Tensor,        # (1, N)
    presence_logits: torch.Tensor,    # (1, 1)
    pred_masks: torch.Tensor,         # (1, N, H_mask, W_mask)
    pred_boxes: torch.Tensor,         # (1, N, 4) xyxy
    score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.1,
) -> list[Detection]:
    """Apply the official `Sam3VideoModel.run_detection` postproc.

    Returns a list of Detection sorted by score descending.
    """
    pred_probs = pred_logits.sigmoid() * presence_logits.sigmoid()  # (1, N)

    if nms_iou_threshold > 0.0:
        valid = pred_probs[0] > score_threshold
        valid_idx = torch.where(valid)[0]
        if valid_idx.numel() > 0:
            kept_local = greedy_mask_nms(
                pred_probs[0, valid_idx],
                pred_masks[0, valid_idx],
                iou_threshold=nms_iou_threshold,
            )
            kept = valid_idx[kept_local]
        else:
            kept = torch.empty(0, dtype=torch.long, device=pred_probs.device)
    else:
        kept = torch.where(pred_probs[0] > score_threshold)[0]

    detections = [
        Detection(
            score=float(pred_probs[0, i]),
            mask_logits=pred_masks[0, i],
            bbox_xyxy=pred_boxes[0, i],
        )
        for i in kept.tolist()
    ]
    detections.sort(key=lambda d: -d.score)
    return detections


def mask_logits_to_image(
    mask_logits: torch.Tensor,
    original_hw: tuple[int, int],
    mask_threshold: float = 0.0,
) -> np.ndarray:
    """Resize decoder-resolution mask logits to original (H, W) and binarize.

    Matches `Sam3ImageProcessor.post_process_masks`.
    """
    H, W = original_hw
    m = mask_logits.unsqueeze(0).unsqueeze(0).float()
    m = F.interpolate(m, (H, W), mode="bilinear", align_corners=False)
    return (m[0, 0] > mask_threshold).cpu().numpy()


def bbox_from_mask(mask_bool: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tightest axis-aligned xyxy bbox around a True region. None if empty."""
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
