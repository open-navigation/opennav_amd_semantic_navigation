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


def _assert_full_autotune(env):
    assert env is not None
    assert 'MIGRAPHX_SKIP_BENCHMARKING' not in env
    assert 'MIGRAPHX_MLIR_USE_SPECIFIC_OPS' not in env


def test_backbone_build_keeps_host_artifact_and_adds_gpu_io(monkeypatch, tmp_path):
    probes = []

    def fake_exists(path, label, force):
        probes.append((path, label, force))
        return path.name != 'tuned_gpuio.mxr'

    monkeypatch.setenv('MIGRAPHX_SKIP_BENCHMARKING', '1')
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
    _assert_full_autotune(env)
    assert os.environ['MIGRAPHX_SKIP_BENCHMARKING'] == '1'


def test_build_does_not_set_custom_compiler_flags():
    # ROCMLIR_SINK_FINAL_ERF only affected the earlier custom MIGraphX build.
    assert 'ROCMLIR_SINK_FINAL_ERF' not in BUILD_SCRIPT.read_text()


def test_memory_attention_exports_each_shape_then_autotunes(monkeypatch, tmp_path):
    monkeypatch.setenv('MIGRAPHX_SKIP_BENCHMARKING', '1')
    monkeypatch.setattr(build, 'exists', lambda path, label, force: False)
    calls = _capture_runs(monkeypatch)

    args = _args(tmp_path, ['memory_attention'], max_spatial_slots=10)
    assert build.build_for_imgsz(504, args)
    assert len(calls) == 11
    spatial_slots = []
    for command, _, env in calls[:10]:
        assert env is None
        index = command.index('--spatial-slots')
        spatial_slots.append(int(command[index + 1]))
    assert spatial_slots == list(range(1, 11))
    command, _, env = calls[10]
    assert command[1] == 'export/tracker_modules/compile_memory_attention.py'
    assert command[command.index('--ptr-tokens') + 1] == '64'
    assert command[command.index('--max-spatial-slots') + 1] == '10'
    _assert_full_autotune(env)


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
    _assert_full_autotune(calls[1][2])


def test_prewarm_runs_the_runtime_once_with_autotuning(monkeypatch, tmp_path):
    calls = _capture_runs(monkeypatch)

    assert build.build_for_imgsz(504, _args(tmp_path, ['prewarm']))
    assert len(calls) == 1
    command, _, env = calls[0]
    assert command[1] == 'export/prewarm_runtime_caches.py'
    assert command[command.index('--onnx-dir') + 1] == str(tmp_path / 'onnx_files_504')
    _assert_full_autotune(env)


def test_only_504px_is_built(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'exists', lambda path, label, force: False)
    calls = _capture_runs(monkeypatch)

    assert not build.build_for_imgsz(1008, _args(tmp_path, ['all']))
    assert calls == []


def test_detr_encoder_uses_32_token_default(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'exists', lambda path, label, force: False)
    calls = _capture_runs(monkeypatch)

    assert build.build_for_imgsz(504, _args(tmp_path, ['detr_encoder']))
    assert len(calls) == 1
    command = calls[0][0]
    assert '--onnx-dir' in command
    assert '--text-seq-len' not in command
