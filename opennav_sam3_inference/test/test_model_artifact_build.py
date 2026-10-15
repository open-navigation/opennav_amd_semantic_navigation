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

"""CPU-only tests for optimized model-artifact build orchestration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / 'opennav_sam3_setup/export/build_text_prompt_mig.py'
SPEC = importlib.util.spec_from_file_location('build_text_prompt_mig', BUILD_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def _args(tmp_path, steps, *, force=False, max_spatial_slots=10):
    return SimpleNamespace(
        onnx_root=tmp_path,
        checkpoint=tmp_path / 'sam3',
        force=force,
        steps=list(steps),
        ptr_tokens=None,
        max_spatial_slots=max_spatial_slots,
    )


def _capture_runs(monkeypatch):
    calls = []

    def fake_run(command, label, *, env=None):
        calls.append((command, label, env))
        return True

    monkeypatch.setattr(build, 'run', fake_run)
    return calls


def test_backbone_build_keeps_host_artifact_and_adds_gpu_io(monkeypatch, tmp_path):
    probes = []

    def fake_exists(path, label, force):
        probes.append((path, label, force))
        return path.name != 'tuned_gpuio.mxr'

    monkeypatch.delenv('ROCMLIR_SINK_FINAL_ERF', raising=False)
    monkeypatch.setattr(build, 'exists', fake_exists)
    calls = _capture_runs(monkeypatch)

    assert build.build_for_imgsz(504, _args(tmp_path, ['backbone']))
    assert [item[0].name for item in probes] == [
        'single_fp32.onnx',
        'single_simplified.onnx',
        'tuned.mxr',
        'tuned_gpuio.mxr',
    ]
    assert len(calls) == 1
    command, _, env = calls[0]
    assert '--gpu-io' in command
    assert env['ROCMLIR_SINK_FINAL_ERF'] == '1'
    assert 'ROCMLIR_SINK_FINAL_ERF' not in os.environ


def test_memory_attention_builds_each_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'exists', lambda path, label, force: False)
    calls = _capture_runs(monkeypatch)

    args = _args(tmp_path, ['memory_attention'], max_spatial_slots=10)
    assert build.build_for_imgsz(504, args)
    assert len(calls) == 10
    spatial_slots = []
    for command, _, env in calls:
        assert env is None
        index = command.index('--spatial-slots')
        spatial_slots.append(int(command[index + 1]))
    assert spatial_slots == list(range(1, 11))


def test_fixed_decoder_build_invokes_export_then_compile(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'exists', lambda path, label, force: False)
    calls = _capture_runs(monkeypatch)

    assert build.build_for_imgsz(
        504,
        _args(tmp_path, ['fixed_decoder'], force=True),
    )
    assert len(calls) == 2
    assert calls[0][0][1] == 'export/detector/export_fixed_detr_decoder.py'
    assert calls[1][0][1] == 'export/detector/compile_fixed_detr_decoder.py'
    assert calls[1][0][-1] == '--force'


def test_fixed_decoder_1008_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'exists', lambda path, label, force: True)
    calls = _capture_runs(monkeypatch)

    assert build.build_for_imgsz(1008, _args(tmp_path, ['all']))
    assert calls == []
    assert not build.build_for_imgsz(1008, _args(tmp_path, ['fixed_decoder']))
    assert calls == []


def test_detr_encoder_uses_32_token_default(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'exists', lambda path, label, force: False)
    calls = _capture_runs(monkeypatch)

    assert build.build_for_imgsz(504, _args(tmp_path, ['detr_encoder']))
    assert len(calls) == 1
    command = calls[0][0]
    assert '--onnx-dir' in command
    assert '--text-seq-len' not in command
