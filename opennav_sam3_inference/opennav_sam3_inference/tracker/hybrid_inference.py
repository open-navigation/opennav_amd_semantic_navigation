"""Hybrid SAM3 detect + SAM2-style tracker propagation pipeline.

Why:
  ``SAM3Live`` with ``redetect_every > 1`` is broken for stuff-class prompts:
  HF ``Sam3VideoModel``'s presence head requires fresh DETR detection input
  every frame, so propagation-only frames return empty masks. ``rd=1`` is the
  only stable config, capping us at ~3 FPS on 2-3 prompt scenes.

  Meanwhile ``SAM3OnnxTracker`` (SAM2-style box-prompt tracker) propagates
  fine for many frames from a single box prompt (12.21 FPS @504 on DAVIS).

How:
  ``SAM3HybridLive`` runs SAM3 detection every ``keyframe_every`` frames to
  refresh per-prompt detections, then runs SAM2 trackers on the K-1 frames
  in between. At each keyframe we re-associate new SAM3 detections to
  existing trackers by mask IoU to preserve obj_id continuity.

Public API matches ``SAM3Live`` so ``demo_live.py``'s ``overlay`` /
``filter_result`` / output schema are reusable as-is.
"""
from __future__ import annotations

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")

import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from .live_inference import SAM3Live
from .tracker import SAM3OnnxTracker, SharedTrackerResources, preprocess_image


def _upscale_mask(mask_imgsz: np.ndarray, H: int, W: int) -> np.ndarray:
    """imgsz×imgsz bool → H×W bool, nearest-neighbour."""
    return cv2.resize(
        mask_imgsz.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def _scale_box_to_imgsz(box, src_w: int, src_h: int, imgsz: int) -> list[float]:
    """(x1,y1,x2,y2) in src px → (x1,y1,x2,y2) in imgsz px."""
    sx = imgsz / src_w
    sy = imgsz / src_h
    x1, y1, x2, y2 = box
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two bool masks of the same shape."""
    if a.shape != b.shape or not a.any() or not b.any():
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / max(1, union)


def _mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Bool HxW → (x1, y1, x2, y2) tight bounding box. Empty mask → (0,0,0,0)."""
    if not mask.any():
        return (0.0, 0.0, 0.0, 0.0)
    ys, xs = np.where(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


class SAM3HybridLive:
    """SAM3 text-prompt detection at keyframes + SAM2-tracker propagation in between.

    Matches ``SAM3Live`` public API (constructor kwargs subset, ``infer``,
    ``reset_prompts``, ``reset_tracking``) and ``infer`` return-dict schema
    so ``demo_live.py`` callers can swap classes transparently.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        prompts: Sequence[str],
        *,
        onnx_dir: str | Path,
        imgsz: int = 504,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        mig: bool = True,
        keyframe_every_ms: float = 1000.0,
        max_objects_per_prompt: int | dict[str, int] | None = 5,
        iou_assoc_threshold: float = 0.3,
        # SAM3Live passthrough
        max_vision_features_cache_size: int = 1,
        bootstrap_frames: int = 0,
        bootstrap_min_score: float = 0.3,
    ):
        """Args:
            keyframe_every_ms: Wall-clock interval between SAM3 keyframe detections
                (milliseconds). Decoupled from camera frame rate — use a value that
                reflects how fast the scene changes for your robot (e.g. 1000 ms for
                indoor navigation at walking speed). 0 < keyframe_every_ms.
                The first frame is always a keyframe.
            iou_assoc_threshold: mask-IoU floor for re-using an existing obj_id
                when a new SAM3 detection matches an existing tracker. Below
                this, the detection spawns a new obj_id and the unmatched
                tracker is dropped.
            max_objects_per_prompt: forwarded to the underlying SAM3Live for
                its own per-prompt cap. The hybrid then mirrors the cap on the
                tracker pool.
            bootstrap_frames: forwarded to the underlying SAM3Live keyframe
                detector. When > 0, the first N keyframes use text prompts
                normally and capture high-confidence boxes; subsequent
                keyframes inject those boxes as input_boxes (see SAM3Live
                docstring for the full text-bootstrap → box-prompt flow).
                Default 0 = pure text-prompt keyframes (original behavior).
            bootstrap_min_score: passthrough to underlying SAM3Live.
        """
        self.imgsz = imgsz
        self.onnx_dir = Path(onnx_dir)
        self.keyframe_interval_s = max(0.001, float(keyframe_every_ms) / 1000.0)
        self.iou_thresh = float(iou_assoc_threshold)
        self.max_per_prompt = max_objects_per_prompt

        # 1) SAM3Live for keyframe detection. Use rd=1 (no internal skip).
        t = time.perf_counter()
        self.live = SAM3Live(
            checkpoint=checkpoint,
            prompts=prompts,
            onnx_dir=onnx_dir,
            imgsz=imgsz,
            dtype=dtype,
            device=device,
            mig=mig,
            redetect_every=1,
            max_objects_per_prompt=max_objects_per_prompt,
            max_vision_features_cache_size=max_vision_features_cache_size,
            bootstrap_frames=bootstrap_frames,
            bootstrap_min_score=bootstrap_min_score,
        )
        print(f"[SAM3HybridLive] SAM3Live ready in {time.perf_counter()-t:.1f}s "
              f"(bootstrap_frames={bootstrap_frames})")

        # 2) Shared tracker modules (backbone + dec_init + dec_prop + mem_enc + mem_attn)
        t = time.perf_counter()
        self.shared = SharedTrackerResources(
            onnx_dir=onnx_dir,
            imgsz=imgsz,
            backbone="auto",
        )
        print(f"[SAM3HybridLive] tracker resources ready in {time.perf_counter()-t:.1f}s")

        # 3) Per-obj tracker pool
        self.trackers: dict[int, SAM3OnnxTracker] = {}
        self.tracker_to_prompt: dict[int, str] = {}
        self.tracker_to_score: dict[int, float] = {}

        # 4) State
        self._call_count = 0
        self._next_obj_id = 0
        self._force_keyframe_next = True  # f=0 + after any reset
        self._last_was_keyframe = False
        self._last_keyframe_time: float = 0.0  # 0 → first frame always keyframe
        print(f"[SAM3HybridLive] ready. keyframe_every_ms={keyframe_every_ms:.0f} "
              f"iou_thresh={self.iou_thresh}")

    # ------------------------------------------------------------------
    # Public API (matches SAM3Live)
    # ------------------------------------------------------------------

    def reset_prompts(self, prompts: Sequence[str]) -> None:
        self.live.reset_prompts(prompts)
        self.trackers.clear()
        self.tracker_to_prompt.clear()
        self.tracker_to_score.clear()
        self._force_keyframe_next = True

    def reset_tracking(self) -> None:
        self.live.reset_tracking()
        self.trackers.clear()
        self.tracker_to_prompt.clear()
        self.tracker_to_score.clear()
        self._force_keyframe_next = True

    def infer(self, frame_bgr: np.ndarray, *, full_detection: bool | None = None) -> dict:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR, got shape {frame_bgr.shape}")
        H, W = frame_bgr.shape[:2]

        _now = time.perf_counter()
        is_keyframe = (
            self._force_keyframe_next
            or (_now - self._last_keyframe_time) >= self.keyframe_interval_s
        )
        self._force_keyframe_next = False

        # Tracker backbone input (preprocessed to imgsz). Cheap; ~1ms.
        img_np = preprocess_image(frame_bgr, self.imgsz)

        if is_keyframe:
            result = self._keyframe_infer(frame_bgr, img_np, H, W)
        else:
            result = self._propagation_infer(img_np, H, W)

        if is_keyframe:
            self._last_keyframe_time = time.perf_counter()
        self._last_was_keyframe = is_keyframe
        result["keyframe"] = is_keyframe
        result["frame_idx"] = self._call_count
        self._call_count += 1
        return result

    # ------------------------------------------------------------------
    # Keyframe + propagation
    # ------------------------------------------------------------------

    def _keyframe_infer(self, frame_bgr, img_np, H, W) -> dict:
        # 1. SAM3 detection (force full detection regardless of internal schedule)
        sam3_r = self.live.infer(frame_bgr, full_detection=True)

        # 2. Shared tracker backbone (used for both IoU propagation of existing
        #    trackers and init of new trackers).
        bbf = self.shared.run_backbone(img_np)

        # 3. Propagate each existing tracker to current frame for IoU matching.
        existing_masks: dict[int, np.ndarray] = {}
        for oid, trk in list(self.trackers.items()):
            mask_imgsz, _ = trk.propagate_with_features(*bbf)
            existing_masks[oid] = _upscale_mask(mask_imgsz, H, W)

        # 4. Greedy per-prompt IoU assignment: each detection → best
        #    available existing tracker of the same prompt (above threshold).
        assignments: dict[int, int | None] = {}
        for prompt, det_oids in sam3_r["prompt_to_obj_ids"].items():
            available = [o for o, p in self.tracker_to_prompt.items() if p == prompt]
            for det_oid in det_oids:
                det_mask = sam3_r["masks"].get(det_oid)
                if det_mask is None or not det_mask.any():
                    assignments[det_oid] = None
                    continue
                best_iou, best_ex = 0.0, None
                for ex_oid in available:
                    iou = _mask_iou(det_mask, existing_masks[ex_oid])
                    if iou > best_iou:
                        best_iou, best_ex = iou, ex_oid
                if best_ex is not None and best_iou >= self.iou_thresh:
                    assignments[det_oid] = best_ex
                    available.remove(best_ex)
                else:
                    assignments[det_oid] = None

        # 5. Rebuild tracker pool: reuse matched obj_ids, spawn new for unmatched
        #    detections, drop any existing trackers that weren't claimed.
        new_trackers: dict[int, SAM3OnnxTracker] = {}
        new_tracker_to_prompt: dict[int, str] = {}
        new_tracker_to_score: dict[int, float] = {}

        out_obj_ids: list[int] = []
        out_scores: dict[int, float] = {}
        out_masks: dict[int, np.ndarray] = {}
        out_boxes: dict[int, tuple] = {}
        out_prompt_to_obj_ids: dict[str, list[int]] = {
            p: [] for p in sam3_r["prompt_to_obj_ids"]
        }

        for prompt, det_oids in sam3_r["prompt_to_obj_ids"].items():
            for det_oid in det_oids:
                bbox = sam3_r["boxes"].get(det_oid)
                mask = sam3_r["masks"].get(det_oid)
                score = float(sam3_r["scores"].get(det_oid, 0.0))
                if bbox is None or mask is None or not mask.any():
                    continue
                bbox_imgsz = _scale_box_to_imgsz(bbox, W, H, self.imgsz)

                reuse = assignments.get(det_oid)
                if reuse is not None:
                    obj_id = reuse
                else:
                    obj_id = self._next_obj_id
                    self._next_obj_id += 1

                # Spawn fresh tracker (MemoryBank reset implicit)
                trk = SAM3OnnxTracker(
                    checkpoint=None, onnx_dir=self.onnx_dir,
                    imgsz=self.imgsz, shared=self.shared,
                )
                trk.init_with_features(*bbf, box=bbox_imgsz)

                new_trackers[obj_id] = trk
                new_tracker_to_prompt[obj_id] = prompt
                new_tracker_to_score[obj_id] = score

                out_obj_ids.append(obj_id)
                out_scores[obj_id] = score
                out_masks[obj_id] = mask
                out_boxes[obj_id] = tuple(float(v) for v in bbox)
                out_prompt_to_obj_ids[prompt].append(obj_id)

        # 6. Commit pool swap
        self.trackers = new_trackers
        self.tracker_to_prompt = new_tracker_to_prompt
        self.tracker_to_score = new_tracker_to_score

        return {
            "object_ids": out_obj_ids,
            "scores": out_scores,
            "masks": out_masks,
            "boxes": out_boxes,
            "prompt_to_obj_ids": out_prompt_to_obj_ids,
            "detected": True,
        }

    def _propagation_infer(self, img_np, H, W) -> dict:
        # 1. Shared backbone once
        bbf = self.shared.run_backbone(img_np)

        out_obj_ids: list[int] = []
        out_scores: dict[int, float] = {}
        out_masks: dict[int, np.ndarray] = {}
        out_boxes: dict[int, tuple] = {}
        # Build prompt list from current pool — empty pool yields empty groups
        all_prompts = set(self.tracker_to_prompt.values())
        out_prompt_to_obj_ids: dict[str, list[int]] = {p: [] for p in all_prompts}

        # 2. Propagate each tracker
        for oid, trk in list(self.trackers.items()):
            mask_imgsz, trk_score = trk.propagate_with_features(*bbf)
            mask = _upscale_mask(mask_imgsz, H, W)
            if not mask.any():
                # Mask collapsed → drop tracker
                del self.trackers[oid]
                self.tracker_to_prompt.pop(oid, None)
                self.tracker_to_score.pop(oid, None)
                continue
            prompt = self.tracker_to_prompt[oid]
            out_obj_ids.append(oid)
            # Use last-keyframe SAM3 detection score (more meaningful than the
            # propagation-internal tracker_score for downstream filtering).
            out_scores[oid] = self.tracker_to_score.get(oid, float(trk_score))
            out_masks[oid] = mask
            out_boxes[oid] = _mask_to_bbox(mask)
            out_prompt_to_obj_ids[prompt].append(oid)

        return {
            "object_ids": out_obj_ids,
            "scores": out_scores,
            "masks": out_masks,
            "boxes": out_boxes,
            "prompt_to_obj_ids": out_prompt_to_obj_ids,
            "detected": False,
        }


__all__ = ["SAM3HybridLive"]
