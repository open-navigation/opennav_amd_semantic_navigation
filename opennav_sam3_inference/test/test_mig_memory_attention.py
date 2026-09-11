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

from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def memory_runtime():
    """Import the runtime only in the Conda GPU test environment."""
    torch = pytest.importorskip('torch')
    module = pytest.importorskip(
        'opennav_sam3_inference.tracker.mig_memory_attention'
    )
    return SimpleNamespace(module=module, torch=torch)


class _FakeSession:
    def __init__(self, path, loaded, hw=4):
        name = Path(path).name
        match = re.search(r'_S(\d+)_P(\d+)\.onnx$', name)
        assert match is not None
        self.slots, self.ptr_tokens = map(int, match.groups())
        self.hw = hw
        self.last_inputs = None
        loaded.append(name)

    def get_providers(self):
        return ['CPUExecutionProvider']

    def get_inputs(self):
        return [
            SimpleNamespace(
                name='current_vision_features',
                shape=(self.hw, 1, 256),
            ),
            SimpleNamespace(
                name='memory',
                shape=(self.slots * self.hw + self.ptr_tokens, 1, 64),
            ),
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name='conditioned_features',
                shape=(1, 256, 1, self.hw),
            )
        ]

    def run(self, _output_names, inputs):
        self.last_inputs = inputs
        return [np.zeros((1, 256, 1, self.hw), dtype=np.float32)]


def _write_artifact_family(directory, ptr_tokens):
    paths = []
    for slots in range(1, 11):
        path = (
            directory
            / f'memory_attention_fixed_S{slots}_P{ptr_tokens}.onnx'
        )
        path.touch()
        paths.append(path)
    return paths


def _mock_sessions(monkeypatch, module):
    loaded = []

    def make_session(path, sess_options, providers):
        assert sess_options is not None
        assert providers[0][0] == 'MIGraphXExecutionProvider'
        return _FakeSession(path, loaded)

    monkeypatch.setattr(module.ort, 'InferenceSession', make_session)
    return loaded


@pytest.mark.parametrize('ptr_tokens', [64, 48])
def test_discovers_only_the_matching_s1_to_s10_family(
    memory_runtime, monkeypatch, tmp_path, ptr_tokens
):
    module = memory_runtime.module
    loaded = _mock_sessions(monkeypatch, module)
    expected = _write_artifact_family(tmp_path, ptr_tokens)
    other_ptr_tokens = 48 if ptr_tokens == 64 else 64
    _write_artifact_family(tmp_path, other_ptr_tokens)

    shim = module.MIGMemoryAttention(
        expected[6],
        original_forward=lambda **_kwargs: None,
    )

    assert set(shim._sessions) == set(range(1, 11))
    assert shim.ptr_tokens == ptr_tokens
    assert set(loaded) == {path.name for path in expected}
    assert all(f'_P{ptr_tokens}.onnx' in name for name in loaded)


def test_selects_the_exact_spatial_shape_and_pads_pointers(
    memory_runtime, monkeypatch, tmp_path
):
    module = memory_runtime.module
    torch = memory_runtime.torch
    _mock_sessions(monkeypatch, module)
    paths = _write_artifact_family(tmp_path, 64)
    fallbacks = []
    fallback_result = object()

    def original_forward(**kwargs):
        fallbacks.append(kwargs)
        return fallback_result

    shim = module.MIGMemoryAttention(paths[6], original_forward)
    current = torch.zeros((4, 1, 256), dtype=torch.float32)
    current_pos = torch.zeros_like(current)
    memory = torch.zeros((3 * 4 + 2, 1, 64), dtype=torch.float32)
    memory_pos = torch.zeros_like(memory)

    output = shim(
        current_vision_features=current,
        memory=memory,
        current_vision_position_embeddings=current_pos,
        memory_posision_embeddings=memory_pos,
        num_object_pointer_tokens=2,
    )

    selected = shim._sessions[3][0]
    assert selected.last_inputs is not None
    assert selected.last_inputs['memory'].shape == (3 * 4 + 64, 1, 64)
    assert selected.last_inputs['memory_pos_embed'].shape == (
        3 * 4 + 64, 1, 64,
    )
    assert output.shape == (1, 1, 4, 256)
    assert shim._mig_calls == 1
    assert shim._pt_fallback_calls == 0
    assert not fallbacks


def test_missing_spatial_shape_preserves_pytorch_fallback(
    memory_runtime, monkeypatch, tmp_path
):
    module = memory_runtime.module
    torch = memory_runtime.torch
    _mock_sessions(monkeypatch, module)
    paths = _write_artifact_family(tmp_path, 48)
    fallback_result = object()
    fallbacks = []

    def original_forward(**kwargs):
        fallbacks.append(kwargs)
        return fallback_result

    shim = module.MIGMemoryAttention(paths[6], original_forward)
    current = torch.zeros((4, 1, 256), dtype=torch.float32)
    memory = torch.zeros((11 * 4 + 3, 1, 64), dtype=torch.float32)

    result = shim(
        current_vision_features=current,
        memory=memory,
        current_vision_position_embeddings=torch.zeros_like(current),
        memory_posision_embeddings=torch.zeros_like(memory),
        num_object_pointer_tokens=3,
    )

    assert result is fallback_result
    assert len(fallbacks) == 1
    assert shim._mig_calls == 0
    assert shim._pt_fallback_calls == 1


def test_over_capacity_keeps_conditioning_and_newest_pointer_tokens(
    memory_runtime, monkeypatch, tmp_path
):
    module = memory_runtime.module
    torch = memory_runtime.torch
    _mock_sessions(monkeypatch, module)
    paths = _write_artifact_family(tmp_path, 4)

    def unexpected_fallback(**_kwargs):
        pytest.fail('exact S2 shape should not use the PyTorch fallback')

    shim = module.MIGMemoryAttention(paths[6], unexpected_fallback)
    current = torch.zeros((4, 1, 256), dtype=torch.float32)
    spatial = torch.full((2 * 4, 1, 64), -1.0)
    spatial_pos = torch.full_like(spatial, -2.0)

    # HF orders conditioning pointers first, then non-conditioning pointers
    # from newest to oldest. Values make the retained end observable.
    pointer_order = torch.tensor(
        [100, 90, 80, 70, 60, 50], dtype=torch.float32
    ).view(-1, 1, 1).expand(-1, 1, 64)
    pointer_pos_order = torch.tensor(
        [1100, 1090, 1080, 1070, 1060, 1050], dtype=torch.float32
    ).view(-1, 1, 1).expand(-1, 1, 64)
    memory = torch.cat([spatial, pointer_order], dim=0)
    memory_pos = torch.cat([spatial_pos, pointer_pos_order], dim=0)

    shim(
        current_vision_features=current,
        memory=memory,
        current_vision_position_embeddings=torch.zeros_like(current),
        memory_posision_embeddings=memory_pos,
        num_object_pointer_tokens=6,
    )

    selected = shim._sessions[2][0]
    kept = selected.last_inputs['memory'][-4:, 0, 0].tolist()
    kept_pos = selected.last_inputs['memory_pos_embed'][-4:, 0, 0].tolist()
    assert kept == [100, 90, 80, 70]
    assert kept_pos == [1100, 1090, 1080, 1070]
