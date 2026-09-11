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

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
CONFIG = PACKAGE_ROOT / 'config' / 'sam3_inference.yaml'
NODE = PACKAGE_ROOT / 'opennav_sam3_inference' / 'sam3_node.py'


def test_ros_profile_keeps_fixed_decoder_internal():
    """Keep hybrid defaults while hiding fixed-decoder implementation details."""
    config = CONFIG.read_text()
    node = NODE.read_text()

    assert 'redetect_interval_ms: 1000.0' in config
    assert 'bootstrap_frames: 0' in config
    assert 'fixed_detr_decoder:' not in config
    assert "declare_parameter('fixed_detr_decoder'" not in node
    assert 'imgsz == 504 and _boot == 0' in node


@pytest.fixture(scope='module')
def gpu_runtime():
    """Import GPU dependencies only when these tests are executed."""
    nn = pytest.importorskip('torch.nn')
    pytest.importorskip('migraphx')
    fixed_decoder = pytest.importorskip(
        'opennav_sam3_inference.tracker.mig_detr_decoder'
    )
    hybrid_inference = pytest.importorskip(
        'opennav_sam3_inference.tracker.hybrid_inference'
    )
    live_inference = pytest.importorskip(
        'opennav_sam3_inference.tracker.live_inference'
    )

    class OriginalDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.box_head = nn.Identity()

    return SimpleNamespace(
        fixed_decoder=fixed_decoder,
        OriginalDecoder=OriginalDecoder,
        SAM3HybridLive=hybrid_inference.SAM3HybridLive,
        SAM3Live=live_inference.SAM3Live,
    )


class _Shape:
    def __init__(self, lens, type_name='float_type'):
        self._lens = tuple(lens)
        self._type_name = type_name

    def lens(self):
        return self._lens

    def strides(self):
        stride = 1
        result = []
        for length in reversed(self._lens):
            result.append(stride)
            stride *= length
        return list(reversed(result))

    def type_string(self):
        return self._type_name


class _Program:
    def __init__(self, shapes):
        self._shapes = shapes

    def get_parameter_shapes(self):
        return self._shapes


def _valid_shapes():
    return {
        'vision_features': _Shape((1, 1296, 256)),
        'text_features': _Shape((1, 32, 256)),
        'vision_pos_encoding': _Shape((1, 1296, 256)),
        'text_cross_attn_mask': _Shape((1, 1, 1, 32)),
        'main:#output_0': _Shape((6, 1, 200, 256)),
        'main:#output_1': _Shape((6, 1, 200, 4)),
        'main:#output_2': _Shape((6, 1, 1)),
    }


def test_fixed_decoder_is_opt_in_for_opennav(gpu_runtime):
    assert (
        inspect.signature(gpu_runtime.SAM3Live).parameters[
            'fixed_detr_decoder'
        ].default
        is False
    )
    assert (
        inspect.signature(gpu_runtime.SAM3HybridLive).parameters[
            'fixed_detr_decoder'
        ].default
        is False
    )


def test_fixed_decoder_requires_mig(gpu_runtime):
    with pytest.raises(ValueError, match='requires mig=True'):
        gpu_runtime.SAM3Live(
            checkpoint='unused',
            prompts=['obstacle'],
            mig=False,
            fixed_detr_decoder=True,
        )


def test_fixed_decoder_requires_504px(gpu_runtime):
    with pytest.raises(ValueError, match='requires imgsz=504'):
        gpu_runtime.SAM3Live(
            checkpoint='unused',
            prompts=['obstacle'],
            imgsz=1008,
            mig=True,
            fixed_detr_decoder=True,
        )


def test_fixed_decoder_rejects_bootstrap_mode(gpu_runtime):
    with pytest.raises(ValueError, match='cannot be combined with bootstrap_frames'):
        gpu_runtime.SAM3Live(
            checkpoint='unused',
            prompts=['obstacle'],
            imgsz=504,
            mig=True,
            fixed_detr_decoder=True,
            bootstrap_frames=1,
        )


def test_fixed_decoder_rejects_wrong_artifact_hash(gpu_runtime, tmp_path):
    artifact = tmp_path / 'wrong.mxr'
    artifact.write_bytes(b'not the accepted artifact')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        gpu_runtime.fixed_decoder.MIGFixedDetrDecoder(
            artifact, gpu_runtime.OriginalDecoder()
        )


def test_fixed_decoder_accepts_local_build_checksum(
    gpu_runtime, monkeypatch, tmp_path
):
    artifact = tmp_path / 'direct_gpuio.mxr'
    artifact.write_bytes(b'locally compiled fixture')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact.with_suffix('.mxr.sha256').write_text(
        f'{digest}  {artifact.name}\n', encoding='ascii'
    )
    monkeypatch.setattr(
        gpu_runtime.fixed_decoder.migraphx,
        'load',
        lambda path: _Program(_valid_shapes()),
    )
    decoder = gpu_runtime.fixed_decoder.MIGFixedDetrDecoder(
        artifact, gpu_runtime.OriginalDecoder()
    )
    assert decoder.output_names == (
        'main:#output_0', 'main:#output_1', 'main:#output_2',
    )


def test_fixed_decoder_rejects_malformed_local_checksum(gpu_runtime, tmp_path):
    artifact = tmp_path / 'direct_gpuio.mxr'
    artifact.write_bytes(b'fixture')
    artifact.with_suffix('.mxr.sha256').write_text('not-a-checksum\n')
    with pytest.raises(RuntimeError, match='invalid fixed decoder checksum'):
        gpu_runtime.fixed_decoder.MIGFixedDetrDecoder(
            artifact, gpu_runtime.OriginalDecoder()
        )


def test_fixed_decoder_validates_parameter_contract(
    gpu_runtime, monkeypatch, tmp_path
):
    artifact = tmp_path / 'fixed.mxr'
    artifact.write_bytes(b'mock')
    monkeypatch.setattr(
        gpu_runtime.fixed_decoder,
        '_sha256',
        lambda path: gpu_runtime.fixed_decoder.FIXED_DETR_DECODER_SHA256,
    )
    monkeypatch.setattr(
        gpu_runtime.fixed_decoder.migraphx,
        'load',
        lambda path: _Program(_valid_shapes()),
    )
    original = gpu_runtime.OriginalDecoder()
    decoder = gpu_runtime.fixed_decoder.MIGFixedDetrDecoder(artifact, original)
    assert decoder.box_head is original.box_head
    assert decoder._original_decoder_keepalive is original
    assert decoder.output_names == (
        'main:#output_0',
        'main:#output_1',
        'main:#output_2',
    )


def test_fixed_decoder_rejects_wrong_shape(gpu_runtime, monkeypatch, tmp_path):
    artifact = tmp_path / 'fixed.mxr'
    artifact.write_bytes(b'mock')
    shapes = _valid_shapes()
    shapes['text_features'] = _Shape((1, 31, 256))
    monkeypatch.setattr(
        gpu_runtime.fixed_decoder,
        '_sha256',
        lambda path: gpu_runtime.fixed_decoder.FIXED_DETR_DECODER_SHA256,
    )
    monkeypatch.setattr(
        gpu_runtime.fixed_decoder.migraphx,
        'load',
        lambda path: _Program(shapes),
    )
    with pytest.raises(RuntimeError, match='unexpected fixed decoder parameter'):
        gpu_runtime.fixed_decoder.MIGFixedDetrDecoder(
            artifact, gpu_runtime.OriginalDecoder()
        )


def test_fixed_decoder_selects_outputs_by_contract_shape(
    gpu_runtime, monkeypatch, tmp_path
):
    artifact = tmp_path / 'fixed.mxr'
    artifact.write_bytes(b'mock')
    shapes = {
        **{name: shape for name, shape in _valid_shapes().items() if '#output' not in name},
        **{f'main:#output_{index}': _Shape((1, 200, 256)) for index in range(5)},
        'main:#output_5': _Shape((6, 1, 200, 256)),
        'main:#output_6': _Shape((6, 1, 200, 4)),
        'main:#output_7': _Shape((6, 1, 1)),
    }
    monkeypatch.setattr(
        gpu_runtime.fixed_decoder,
        '_sha256',
        lambda path: gpu_runtime.fixed_decoder.FIXED_DETR_DECODER_SHA256,
    )
    monkeypatch.setattr(
        gpu_runtime.fixed_decoder.migraphx,
        'load',
        lambda path: _Program(shapes),
    )
    decoder = gpu_runtime.fixed_decoder.MIGFixedDetrDecoder(
        artifact, gpu_runtime.OriginalDecoder()
    )
    assert decoder.output_names == (
        'main:#output_5', 'main:#output_6', 'main:#output_7',
    )
