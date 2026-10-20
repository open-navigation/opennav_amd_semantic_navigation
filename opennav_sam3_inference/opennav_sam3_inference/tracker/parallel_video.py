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
Overlap the independent detector and tracker tails of ``Sam3VideoModel``.

The vision encoder is shared by both branches and remains serial.  Once its
output is available, detection and tracker propagation do not exchange data
until the association/update phase.  Running those branches on separate HIP
streams hides part of the detector tail behind tracker propagation without
changing the model outputs.

MIG live inference enables this patch by default; callers can disable it for
diagnosis.  It is meant for one ordered frame owner at a time, and the model
and its inference session must not be shared by concurrent callers.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, wait
import inspect
import threading
import types
import weakref

from packaging.version import Version
import torch
import transformers

from .ort_gpu_io import fence_ort_inputs


class _ParallelTailRuntime:
    def __init__(self, model) -> None:
        device = next(model.parameters()).device
        if device.type != 'cuda':
            raise ValueError('parallel video tails require a CUDA/HIP model')

        self.device = device
        self.original = model._det_track_one_frame.__func__
        self.executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix='sam3-tail',
        )
        with torch.cuda.device(device):
            self.detector_stream = torch.cuda.Stream(device=device)
            self.tracker_stream = torch.cuda.Stream(device=device)
        self.call_lock = threading.Lock()
        self._closed = False

    def close(self, wait_for_workers: bool = True) -> None:
        with self.call_lock:
            if not self._closed:
                self.executor.shutdown(
                    wait=wait_for_workers,
                    cancel_futures=True,
                )
                self._closed = True

    def _detect(self, model, inference_session, vision_embeds):
        with (
            torch.inference_mode(),
            torch.cuda.device(self.device),
            torch.cuda.stream(self.detector_stream),
            fence_ort_inputs(),
        ):
            try:
                return model.run_detection(
                    inference_session=inference_session,
                    vision_embeds=vision_embeds,
                )
            finally:
                # Drain work even when Python-side post-processing raises, so
                # callers cannot free storage that a side stream still uses.
                self.detector_stream.synchronize()

    def _track(
        self,
        model,
        inference_session,
        vision_embeds,
        frame_idx: int,
        reverse: bool,
    ):
        with (
            torch.inference_mode(),
            torch.cuda.device(self.device),
            torch.cuda.stream(self.tracker_stream),
            fence_ort_inputs(),
        ):
            try:
                vision_feats, vision_pos_embeds = model.get_vision_features_for_tracker(
                    vision_embeds=vision_embeds,
                )
                inference_session.cache.cache_vision_features(
                    frame_idx,
                    {
                        'vision_feats': vision_feats,
                        'vision_pos_embeds': vision_pos_embeds,
                    },
                )
                output = model.run_tracker_propagation(
                    inference_session=inference_session,
                    frame_idx=frame_idx,
                    reverse=reverse,
                )
                return output, vision_feats, vision_pos_embeds
            finally:
                self.tracker_stream.synchronize()

    def run(
        self,
        model,
        inference_session,
        frame_idx: int,
        reverse: bool,
        streaming: bool = False,
    ):
        if self._closed:
            raise RuntimeError('parallel-tail runtime is closed')
        if getattr(inference_session, '_parallel_tail_failed', False):
            raise RuntimeError(
                'parallel-tail session is invalid after a worker failure; '
                'create a new inference session before continuing'
            )

        # Frame 0 has no tracker branch to overlap.  The fallback also caches
        # prompt embeddings before two worker threads can access the session.
        if (
            frame_idx == 0
            or not inference_session.obj_ids
            or getattr(model, '_skip_detection', False)
        ):
            return self.original(
                model,
                inference_session=inference_session,
                frame_idx=frame_idx,
                reverse=reverse,
                streaming=streaming,
            )

        pixel_values = inference_session.get_frame(frame_idx).unsqueeze(0)
        vision_embeds = model.detector_model.get_vision_features(
            pixel_values=pixel_values,
        )

        # The current backbone implementation synchronizes before returning.
        # Keep explicit stream dependencies so this remains correct if that
        # synchronization is later narrowed to an event.
        producer_stream = torch.cuda.current_stream(self.device)
        self.detector_stream.wait_stream(producer_stream)
        self.tracker_stream.wait_stream(producer_stream)

        detection_future = self.executor.submit(
            self._detect,
            model,
            inference_session,
            vision_embeds,
        )
        tracker_future = self.executor.submit(
            self._track,
            model,
            inference_session,
            vision_embeds,
            frame_idx,
            reverse,
        )
        try:
            wait((detection_future, tracker_future))
            all_detections = detection_future.result()
            (
                tracker_outputs,
                tracker_vision_feats,
                tracker_vision_pos_embeds,
            ) = tracker_future.result()
            tracker_low_res_masks_global, tracker_obj_scores_global = tracker_outputs
        except BaseException as exc:
            # A KeyboardInterrupt can arrive while wait() is blocked. Ensure
            # both workers reach their finally blocks before caller-owned
            # model/session storage can be released.
            wait((detection_future, tracker_future))
            # Tracker propagation writes per-frame state before it returns.  A
            # failure in the other branch can therefore leave a half-advanced
            # session. Refuse a silent retry on that session.
            inference_session._parallel_tail_failed = True
            if isinstance(exc, Exception):
                raise RuntimeError(
                    'parallel detector/tracker branch failed; create a new '
                    'inference session before retrying'
                ) from exc
            raise

        main_stream = torch.cuda.current_stream(self.device)
        _record_stream(all_detections, main_stream)
        _record_stream(tracker_vision_feats, main_stream)
        _record_stream(tracker_vision_pos_embeds, main_stream)
        _record_stream(tracker_low_res_masks_global, main_stream)
        _record_stream(tracker_obj_scores_global, main_stream)

        # The branches meet here.  The remainder mirrors the upstream
        # Sam3VideoModel._det_track_one_frame implementation.
        det_out, det_idx_to_prompt_id = model._merge_detections_from_prompts(
            all_detections,
            inference_session,
        )
        tracker_update_plan, tracker_metadata_new = model.run_tracker_update_planning_phase(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_obj_scores_global=tracker_obj_scores_global,
            det_idx_to_prompt_id=det_idx_to_prompt_id,
            streaming=streaming,
        )
        model.run_tracker_update_execution_phase(
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_update_plan=tracker_update_plan,
        )
        obj_id_to_mask = model.build_outputs(
            inference_session=inference_session,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_update_plan=tracker_update_plan,
            reconditioned_obj_ids=tracker_update_plan['reconditioned_obj_ids'],
        )
        obj_id_to_score = tracker_metadata_new['obj_id_to_score']
        if tracker_obj_scores_global.shape[0] > 0:
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_metadata_new['obj_id_to_tracker_score_frame_wise'][frame_idx].update(
                dict(zip(inference_session.obj_ids, tracker_obj_scores_global))
            )

        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_metadata_new,
            tracker_obj_scores_global,
        )


def _parallel_det_track_one_frame(
    model,
    inference_session,
    frame_idx: int,
    reverse: bool,
    streaming: bool = False,
):
    runtime = model._parallel_tail_runtime
    if not runtime.call_lock.acquire(blocking=False):
        raise RuntimeError('parallel-tail model does not support concurrent callers')
    try:
        return runtime.run(
            model=model,
            inference_session=inference_session,
            frame_idx=frame_idx,
            reverse=reverse,
            streaming=streaming,
        )
    finally:
        runtime.call_lock.release()


def _record_stream(value, stream) -> None:
    """Tell Torch's allocator that branch outputs are consumed on ``stream``."""
    if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
        value.record_stream(stream)
    elif isinstance(value, Mapping):
        for child in value.values():
            _record_stream(child, stream)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _record_stream(child, stream)


def patch_parallel_video_tail(model) -> None:
    """Run detection and tracker propagation concurrently after the backbone."""
    if hasattr(model, '_parallel_tail_runtime'):
        return
    if model.training:
        raise ValueError('parallel-tail requires model.eval()')

    version = Version(transformers.__version__)
    if not (Version('5.8') <= version < Version('5.9')):
        raise RuntimeError(
            'parallel-tail mirrors transformers 5.8.x internals; got '
            f'transformers {transformers.__version__}'
        )
    parameters = tuple(inspect.signature(model._det_track_one_frame).parameters)
    expected = ('inference_session', 'frame_idx', 'reverse', 'streaming')
    if parameters != expected:
        raise RuntimeError(
            'unsupported Sam3VideoModel._det_track_one_frame signature: '
            f'{parameters!r}'
        )

    runtime = _ParallelTailRuntime(model)
    model._parallel_tail_runtime = runtime
    model._parallel_tail_finalizer = weakref.finalize(model, runtime.close, False)
    model._det_track_one_frame = types.MethodType(
        _parallel_det_track_one_frame,
        model,
    )


def close_parallel_video_tail(model) -> None:
    """Restore serial execution and release worker threads."""
    runtime = getattr(model, '_parallel_tail_runtime', None)
    if runtime is not None:
        runtime.close()
        model._det_track_one_frame = types.MethodType(runtime.original, model)
        finalizer = getattr(model, '_parallel_tail_finalizer', None)
        if finalizer is not None:
            finalizer.detach()
            delattr(model, '_parallel_tail_finalizer')
        delattr(model, '_parallel_tail_runtime')


__all__ = ['close_parallel_video_tail', 'patch_parallel_video_tail']
