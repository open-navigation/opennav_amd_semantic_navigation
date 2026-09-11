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

"""Focused tests for the single-model hybrid inference scheduler."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def hybrid_module():
    """Import the runtime only when its pinned dependencies are available."""
    pytest.importorskip('torch')
    return pytest.importorskip(
        'opennav_sam3_inference.tracker.hybrid_inference'
    )


class _FakeLive:
    """Record the hybrid wrapper's calls without loading a model."""

    def __init__(self, outputs):
        self.device = 'cpu'
        self.outputs = list(outputs)
        self.infer_calls = []
        self.reset_prompts_calls = []
        self.reset_tracking_calls = 0
        self.replace_session_calls = 0
        self.close_calls = 0
        self._infer_calls = 0

    def infer(self, _frame, *, full_detection):
        """Return the next scripted inference result."""
        self.infer_calls.append(full_detection)
        self._infer_calls += 1
        result = dict(self.outputs.pop(0))
        result['object_ids'] = list(result.get('object_ids', []))
        result['scores'] = dict(result.get('scores', {}))
        result['masks'] = dict(result.get('masks', {}))
        result['boxes'] = dict(result.get('boxes', {}))
        result['prompt_to_obj_ids'] = {
            prompt: list(object_ids)
            for prompt, object_ids in result.get(
                'prompt_to_obj_ids',
                {},
            ).items()
        }
        result.setdefault('detected', bool(full_detection))
        return result

    def reset_prompts(self, prompts):
        """Record a prompt reset."""
        self.reset_prompts_calls.append(list(prompts))

    def reset_tracking(self):
        """Record a tracking reset."""
        self.reset_tracking_calls += 1
        self._infer_calls = 0

    def _replace_tracking_session_preserving_prompts(self):
        """Record a clean-keyframe session replacement."""
        self.replace_session_calls += 1
        self._infer_calls = 0

    def close(self):
        """Record runtime shutdown."""
        self.close_calls += 1


def _output(object_id=4, *, detected=True):
    """Create one complete result with a stable synthetic mask."""
    mask = np.array([[True, True], [False, False]])
    return {
        'object_ids': [object_id],
        'scores': {object_id: 0.9},
        'masks': {object_id: mask},
        'boxes': {object_id: (0.0, 0.0, 1.0, 1.0)},
        'prompt_to_obj_ids': {'floor': [object_id]},
        'frame_idx': 0,
        'detected': detected,
    }


def _make_hybrid(module, outputs):
    """Construct the scheduler around a fake SAM3Live instance."""
    hybrid = module.SAM3HybridLive.__new__(module.SAM3HybridLive)
    hybrid.live = _FakeLive(outputs)
    hybrid.device = hybrid.live.device
    hybrid.imgsz = 504
    hybrid.onnx_dir = None
    hybrid.keyframe_interval_s = 1.0
    hybrid.iou_thresh = 0.3
    hybrid.max_per_prompt = 5
    hybrid._call_count = 0
    hybrid._last_keyframe_time = 0.0
    hybrid._last_was_keyframe = False
    hybrid._force_keyframe_next = True
    hybrid._pending_redetect_reason = 'first_frame'
    hybrid._previous_output_object_ids = set()
    hybrid._previous_public_masks = {}
    hybrid._previous_public_prompts = {}
    hybrid._inner_to_public = {}
    hybrid._next_public_object_id = 0
    hybrid._inner_session_fresh = True
    return hybrid


def test_live_session_replacement_preserves_only_prompt_state(hybrid_module):
    """Replace tracking state atomically while retaining cached prompts."""
    created = []

    class FakeProcessor:
        """Construct empty synthetic inference sessions."""

        def init_video_session(self, **kwargs):
            session = SimpleNamespace(
                prompts={},
                prompt_input_ids={},
                prompt_embeddings={},
                prompt_attention_masks={},
                tracking_marker='empty',
            )
            created.append((session, kwargs))
            return session

    prompt_embedding = object()
    old_session = SimpleNamespace(
        prompts={3: 'floor'},
        prompt_input_ids={3: object()},
        prompt_embeddings={3: prompt_embedding},
        prompt_attention_masks={3: object()},
        tracking_marker='populated',
    )
    live = hybrid_module.SAM3Live.__new__(hybrid_module.SAM3Live)
    live.processor = FakeProcessor()
    live.device = SimpleNamespace(type='cpu')
    live.dtype = object()
    live.max_vision_features_cache_size = 2
    live.model = SimpleNamespace(_skip_detection=True)
    live.session = old_session
    live._next_frame_idx = 8
    live._infer_calls = 9
    live._force_detect_next = False
    live._detector_call_counter = 4

    live._replace_tracking_session_preserving_prompts()

    assert live.session is created[0][0]
    assert live.session is not old_session
    assert live.session.tracking_marker == 'empty'
    assert live.session.prompts == old_session.prompts
    assert live.session.prompts is not old_session.prompts
    assert live.session.prompt_embeddings[3] is prompt_embedding
    assert created[0][1]['max_vision_features_cache_size'] == 2
    assert live._next_frame_idx == 0
    assert live._infer_calls == 0
    assert live._force_detect_next is True
    assert live.model._skip_detection is False
    assert live.session._parallel_tail_failed is False


def test_constructor_uses_only_sam3_live(hybrid_module, monkeypatch):
    """Avoid the legacy tracker backend and forward optimized options."""
    captured = {}

    class FakeConstructedLive:
        """Capture constructor arguments."""

        device = 'cpu'

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(hybrid_module, 'SAM3Live', FakeConstructedLive)

    hybrid = hybrid_module.SAM3HybridLive(
        checkpoint='checkpoint',
        prompts=['floor', 'wall'],
        onnx_dir='onnx',
        parallel_tail=False,
        fixed_detr_decoder=True,
        bootstrap_frames=0,
    )

    assert type(hybrid.live) is FakeConstructedLive
    assert not hasattr(hybrid, 'shared')
    assert not hasattr(hybrid, 'trackers')
    assert captured['prompts'] == ['floor', 'wall']
    assert captured['parallel_tail'] is False
    assert captured['fixed_detr_decoder'] is True


def test_full_detection_override_and_first_frame_semantics(
    hybrid_module,
    monkeypatch,
):
    """Honor caller overrides while keeping the first frame detectable."""
    hybrid = _make_hybrid(
        hybrid_module,
        [_output(7), _output(7), _output(7, detected=False)],
    )
    clock = iter((10.0, 10.2, 12.0))
    monkeypatch.setattr(
        hybrid_module.time,
        'perf_counter',
        lambda: next(clock),
    )

    first = hybrid.infer(np.zeros((2, 2, 3), dtype=np.uint8),
                         full_detection=False)
    forced = hybrid.infer(np.zeros((2, 2, 3), dtype=np.uint8),
                          full_detection=True)
    skipped = hybrid.infer(np.zeros((2, 2, 3), dtype=np.uint8),
                           full_detection=False)

    assert hybrid.live.infer_calls == [True, True, False]
    assert first['redetect_reason'] == 'first_frame'
    assert forced['redetect_reason'] == 'caller_override'
    assert skipped['redetect_reason'] is None
    assert skipped['negative_evidence_valid'] is False


def test_scheduled_keyframe_replaces_only_tracking_session(
    hybrid_module,
    monkeypatch,
):
    """Start each later keyframe clean while preserving cadence and public ID."""
    hybrid = _make_hybrid(
        hybrid_module,
        [_output(9), _output(0)],
    )
    clock = iter((10.0, 11.1))
    monkeypatch.setattr(
        hybrid_module.time,
        'perf_counter',
        lambda: next(clock),
    )

    first = hybrid.infer(np.zeros((2, 2, 3), dtype=np.uint8))
    second = hybrid.infer(np.zeros((2, 2, 3), dtype=np.uint8))

    assert first['object_ids'] == [0]
    assert second['object_ids'] == [0]
    assert hybrid.live.replace_session_calls == 1
    assert hybrid.live._infer_calls == 2


@pytest.mark.parametrize(
    ('method_name', 'reason', 'counter'),
    [
        ('reset_prompts', 'reset_prompts', 'reset_prompts_calls'),
        ('reset_tracking', 'reset_tracking', 'reset_tracking_calls'),
    ],
)
def test_reset_delegates_and_forces_fresh_detection(
    hybrid_module,
    monkeypatch,
    method_name,
    reason,
    counter,
):
    """Reset the live session and force a keyframe despite a false override."""
    hybrid = _make_hybrid(
        hybrid_module,
        [_output(1), _output(2)],
    )
    clock = iter((10.0, 10.1))
    monkeypatch.setattr(
        hybrid_module.time,
        'perf_counter',
        lambda: next(clock),
    )
    hybrid.infer(np.zeros((2, 2, 3), dtype=np.uint8))

    if method_name == 'reset_prompts':
        hybrid.reset_prompts(['wall'])
        assert getattr(hybrid.live, counter) == [['wall']]
    else:
        hybrid.reset_tracking()
        assert getattr(hybrid.live, counter) == 1

    result = hybrid.infer(
        np.zeros((2, 2, 3), dtype=np.uint8),
        full_detection=False,
    )
    assert result['keyframe'] is True
    assert result['redetect_reason'] == reason


def test_close_delegates_to_live_runtime(hybrid_module):
    """Close the parallel runtime through the hybrid API."""
    hybrid = _make_hybrid(hybrid_module, [])
    hybrid.close()
    assert hybrid.live.close_calls == 1
