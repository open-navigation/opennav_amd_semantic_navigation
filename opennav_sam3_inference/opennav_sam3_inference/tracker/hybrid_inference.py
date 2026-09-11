# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Scheduled detection and propagation on one SAM3 live inference session.

Before each scheduled keyframe, the wrapper replaces the tracking session
while preserving its prompt cache and then runs fresh text detection. Between
keyframes, it uses the same model's detector-skip tracker path. No second
tracker backend or second vision backbone is constructed.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Sequence

from .rocm_env import apply as _apply_rocm_env


_apply_rocm_env()

import numpy as np  # noqa: E402, I100
import torch  # noqa: E402

from .live_inference import SAM3Live  # noqa: E402


class SAM3HybridLive:
    """
    Schedule full detection and propagation on one SAM3Live session.

    The public constructor remains compatible with the former dedicated
    hybrid implementation. Public object IDs are maintained across clean
    keyframe sessions by same-prompt mask-IoU association.
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
        parallel_tail: bool | None = None,
        fixed_detr_decoder: bool = False,
        redetect_interval_ms: float = 1000.0,
        max_objects_per_prompt: int | dict[str, int] | None = 5,
        iou_assoc_threshold: float = 0.3,
        max_vision_features_cache_size: int = 1,
        bootstrap_frames: int = 0,
        bootstrap_min_score: float = 0.3,
        periodic_rebootstrap_seconds: float | None = None,
    ) -> None:
        """Args:
            redetect_interval_ms: Wall-clock interval between SAM3 keyframe detections
                (milliseconds). Decoupled from camera frame rate — use a value that
                reflects how fast the scene changes for your robot (e.g. 1000 ms for
                indoor navigation at walking speed). 0 < redetect_interval_ms.
                The first frame is always a keyframe.
            iou_assoc_threshold: Mask-IoU floor for re-using an existing public obj_id
                when a fresh same-prompt detection matches a previous public mask.
                Below this, the detection receives a new public obj_id.
            max_objects_per_prompt: forwarded to the underlying SAM3Live for
                its own per-prompt cap.
            bootstrap_frames: forwarded to the underlying SAM3Live keyframe
                detector. When > 0, the first N keyframes use text prompts
                normally and capture high-confidence boxes; subsequent
                keyframes inject those boxes as input_boxes (see SAM3Live
                docstring for the full text-bootstrap → box-prompt flow).
                Default 0 = pure text-prompt keyframes (original behavior).
            bootstrap_min_score: passthrough to underlying SAM3Live.
            parallel_tail: Overlap detector/tracker work on SAM3 keyframes.
                ``None`` (default) enables it with MIG; pass ``False`` for a
                serial diagnostic fallback.
            fixed_detr_decoder: Use the direct-MXR fixed 504px DETR decoder.
                Disabled by default because it requires ``bootstrap_frames=0``.
        """
        self.imgsz = imgsz
        self.onnx_dir = Path(onnx_dir)
        self.keyframe_interval_s = max(
            0.001,
            float(redetect_interval_ms) / 1000.0,
        )
        self.iou_thresh = float(iou_assoc_threshold)
        self.max_per_prompt = max_objects_per_prompt

        started_at = time.perf_counter()
        self.live = SAM3Live(
            checkpoint=checkpoint,
            prompts=prompts,
            onnx_dir=onnx_dir,
            imgsz=imgsz,
            dtype=dtype,
            device=device,
            mig=mig,
            parallel_tail=parallel_tail,
            fixed_detr_decoder=fixed_detr_decoder,
            redetect_every=1,
            max_objects_per_prompt=max_objects_per_prompt,
            max_vision_features_cache_size=max_vision_features_cache_size,
            bootstrap_frames=bootstrap_frames,
            bootstrap_min_score=bootstrap_min_score,
            periodic_rebootstrap_seconds=periodic_rebootstrap_seconds,
        )
        self.device = self.live.device

        self._call_count = 0
        self._last_keyframe_time = 0.0
        self._last_was_keyframe = False
        self._force_keyframe_next = True
        self._pending_redetect_reason: str | None = 'first_frame'
        self._previous_output_object_ids: set[int] = set()
        self._previous_public_masks: dict[int, np.ndarray] = {}
        self._previous_public_prompts: dict[int, str] = {}
        self._inner_to_public: dict[int, int] = {}
        self._next_public_object_id = 0
        self._inner_session_fresh = True
        print(
            '[SAM3HybridLive] unified SAM3Live ready in '
            f'{time.perf_counter() - started_at:.1f}s '
            f'(redetect_interval_ms={redetect_interval_ms:.0f}, '
            'detector-skip propagation)'
        )

    def _reset_scheduler(self, reason: str) -> None:
        """Reset scheduling and public object-association state."""
        self._force_keyframe_next = True
        self._pending_redetect_reason = reason
        self._last_keyframe_time = 0.0
        self._previous_output_object_ids.clear()
        self._previous_public_masks.clear()
        self._previous_public_prompts.clear()
        self._inner_to_public.clear()
        self._inner_session_fresh = True

    def reset_prompts(self, prompts: Sequence[str]) -> None:
        """Replace prompts and force fresh detection on the next frame."""
        self.live.reset_prompts(prompts)
        self._reset_scheduler('reset_prompts')

    def reset_tracking(self) -> None:
        """Clear tracking state and force fresh detection on the next frame."""
        self.live.reset_tracking()
        self._reset_scheduler('reset_tracking')

    def close(self) -> None:
        """Release worker resources owned by the live model."""
        self.live.close()

    @staticmethod
    def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
        """Calculate IoU between two Boolean masks."""
        if first.shape != second.shape or not first.any() or not second.any():
            return 0.0
        intersection = int(np.logical_and(first, second).sum())
        if intersection == 0:
            return 0.0
        return intersection / int(np.logical_or(first, second).sum())

    @staticmethod
    def _object_prompts(result: dict) -> dict[int, str]:
        """Map every result object ID to its prompt text."""
        return {
            int(object_id): prompt
            for prompt, object_ids in result.get('prompt_to_obj_ids', {}).items()
            for object_id in object_ids
        }

    def _allocate_public_id(self) -> int:
        """Allocate a monotonically increasing public object ID."""
        object_id = self._next_public_object_id
        self._next_public_object_id += 1
        return object_id

    def _fresh_keyframe_mapping(self, result: dict) -> dict[int, int]:
        """Associate fresh inner detections with the previous public masks."""
        inner_ids = [int(value) for value in result.get('object_ids', [])]
        inner_prompts = self._object_prompts(result)
        candidates = []
        for inner_id in inner_ids:
            prompt = inner_prompts.get(inner_id)
            mask = result.get('masks', {}).get(inner_id)
            if prompt is None or mask is None:
                continue
            for public_id, previous_mask in self._previous_public_masks.items():
                if self._previous_public_prompts.get(public_id) != prompt:
                    continue
                iou = self._mask_iou(np.asarray(mask), previous_mask)
                if iou >= self.iou_thresh:
                    candidates.append((iou, inner_id, public_id))

        mapping: dict[int, int] = {}
        used_public_ids: set[int] = set()
        for _iou, inner_id, public_id in sorted(
            candidates,
            key=lambda item: (-item[0], item[1], item[2]),
        ):
            if inner_id in mapping or public_id in used_public_ids:
                continue
            mapping[inner_id] = public_id
            used_public_ids.add(public_id)

        for inner_id in inner_ids:
            if inner_id not in mapping:
                mapping[inner_id] = self._allocate_public_id()
        return mapping

    def _continuing_mapping(self, result: dict) -> dict[int, int]:
        """Extend the active inner-to-public mapping for new objects."""
        mapping = dict(self._inner_to_public)
        for inner_id in result.get('object_ids', []):
            inner_id = int(inner_id)
            if inner_id not in mapping:
                mapping[inner_id] = self._allocate_public_id()
        return mapping

    @staticmethod
    def _translate_result_ids(result: dict, mapping: dict[int, int]) -> dict:
        """Translate session-local IDs to stable public IDs."""
        translated = dict(result)
        translated['object_ids'] = [
            mapping[int(object_id)]
            for object_id in result.get('object_ids', [])
            if int(object_id) in mapping
        ]
        for field in ('scores', 'masks', 'boxes'):
            translated[field] = {
                mapping[int(object_id)]: value
                for object_id, value in result.get(field, {}).items()
                if int(object_id) in mapping
            }
        translated['prompt_to_obj_ids'] = {
            prompt: [
                mapping[int(object_id)]
                for object_id in object_ids
                if int(object_id) in mapping
            ]
            for prompt, object_ids in result.get('prompt_to_obj_ids', {}).items()
        }
        return translated

    def _remember_public_output(self, result: dict) -> None:
        """Retain masks and prompts needed for the next keyframe association."""
        prompts = self._object_prompts(result)
        self._previous_public_masks = {
            int(object_id): np.asarray(
                result['masks'][object_id],
                dtype=bool,
            ).copy()
            for object_id in result.get('object_ids', [])
            if object_id in result.get('masks', {})
        }
        self._previous_public_prompts = {
            object_id: prompts[object_id]
            for object_id in self._previous_public_masks
            if object_id in prompts
        }

    def infer(
        self,
        frame_bgr: np.ndarray,
        *,
        full_detection: bool | None = None,
    ) -> dict:
        """Run scheduled detection or tracker-only propagation for one frame."""
        started_at = time.perf_counter()
        if self._force_keyframe_next:
            run_detection = True
            redetect_reason = self._pending_redetect_reason or 'forced'
        elif full_detection is True:
            run_detection = True
            redetect_reason = 'caller_override'
        elif full_detection is False:
            run_detection = False
            redetect_reason = None
        elif started_at - self._last_keyframe_time >= self.keyframe_interval_s:
            run_detection = True
            redetect_reason = 'interval'
        else:
            run_detection = False
            redetect_reason = None

        clean_keyframe = False
        if run_detection:
            if not self._inner_session_fresh:
                infer_calls = self.live._infer_calls
                self.live._replace_tracking_session_preserving_prompts()
                # Session-local indices restart, but the long-running bootstrap
                # and drift cadence must remain monotonic.
                self.live._infer_calls = infer_calls
                self._inner_session_fresh = True
            clean_keyframe = True

        try:
            inner_result = self.live.infer(
                frame_bgr,
                full_detection=run_detection,
            )
        finally:
            self._inner_session_fresh = False

        actual_detected = bool(
            inner_result.get('detected', run_detection)
        )
        if actual_detected and clean_keyframe:
            mapping = self._fresh_keyframe_mapping(inner_result)
        else:
            mapping = self._continuing_mapping(inner_result)
        self._inner_to_public = mapping
        result = self._translate_result_ids(inner_result, mapping)

        if actual_detected:
            self._last_keyframe_time = started_at
            self._force_keyframe_next = False
            self._pending_redetect_reason = None
            if not run_detection:
                redetect_reason = 'inner_forced'
        elif run_detection:
            self._force_keyframe_next = True
            self._pending_redetect_reason = 'detection_retry'
            redetect_reason = None

        current_ids = {
            int(object_id) for object_id in result.get('object_ids', [])
        }
        lost_object_ids = (
            sorted(self._previous_output_object_ids - current_ids)
            if not actual_detected
            else []
        )
        if lost_object_ids:
            self._force_keyframe_next = True
            self._pending_redetect_reason = 'object_loss'

        self._previous_output_object_ids = current_ids
        self._last_was_keyframe = actual_detected
        result['detected'] = actual_detected
        result['keyframe'] = actual_detected
        result['frame_idx'] = self._call_count
        result['lost_object_ids'] = lost_object_ids
        result['redetect_reason'] = redetect_reason
        result['negative_evidence_valid'] = actual_detected
        self._remember_public_output(result)
        self._call_count += 1
        return result


__all__ = ['SAM3HybridLive']
