"""Skip the detector on selected frames for higher steady-state FPS.

Sam3VideoModel runs detection (DETR encoder + decoder + CLIP cross-attention)
on every frame. The vision encoder + tracker propagation still need to run
each frame to keep the memory bank fresh, but re-running detection every
frame is the dominant cost in steady state for live-streaming workloads.

This patch wraps ``model.run_detection``:
  - When ``model._skip_detection`` is False (default): behaves identically to
    the original — full detection runs.
  - When ``model._skip_detection`` is True: returns an empty detection dict.
    ``_merge_detections_from_prompts`` already handles the empty case (it
    builds a zero-row merged_det_out), so the rest of _det_track_one_frame
    proceeds with tracker propagation only — no new objects are added, no
    re-detection of existing ones.

Caller is responsible for setting ``_skip_detection`` per frame. Typical
schedule: detect on frame 0 and every Nth frame, propagate-only in between.

Usage:
    from tracker.redetect_schedule import patch_redetect_schedule
    patch_redetect_schedule(model)

    # Frame 0 — must detect (no tracked objects yet)
    model._skip_detection = False
    out = model(session, frame=...)

    # Frames 1..4 — propagate only
    model._skip_detection = True
    for _ in range(4):
        out = model(session, frame=...)

    # Frame 5 — re-detect
    model._skip_detection = False
    out = model(session, frame=...)
"""
from __future__ import annotations

import types


_PATCH_FLAG = "_redetect_schedule_patch_applied"


def patch_redetect_schedule(model) -> None:
    """Monkey-patch model.run_detection to skip when _skip_detection is True.

    Idempotent. Initializes ``model._skip_detection = False``.
    """
    if getattr(model, _PATCH_FLAG, False):
        return

    orig_run_detection = model.run_detection

    def patched_run_detection(self, inference_session, vision_embeds):
        if getattr(self, "_skip_detection", False):
            # Empty all_detections — downstream _merge_detections_from_prompts
            # already builds a zero-row merged_det_out for this case.
            return {}
        return orig_run_detection(inference_session, vision_embeds)

    model.run_detection = types.MethodType(patched_run_detection, model)
    model._skip_detection = False
    setattr(model, _PATCH_FLAG, True)


__all__ = ["patch_redetect_schedule"]
