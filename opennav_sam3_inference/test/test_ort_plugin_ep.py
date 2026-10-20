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

"""CPU-only guards for the ROCm 10.1 plugin-EP session factory and stack checks."""

import os
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRACKER_DIR = PACKAGE_ROOT / 'opennav_sam3_inference' / 'tracker'
EXPORT_DIR = PACKAGE_ROOT.parent / 'opennav_sam3_setup' / 'export'
GOOD = {
    'ep': 'plugin', 'onnxruntime': '1.29.0',
    'migraphx': '2.18.0+rocm10.1.0', 'ep_package': '1.0.0+rocm10.1.0',
}


@pytest.fixture(scope='module')
def ep():
    """Import the factory only where ONNX Runtime is installed."""
    pytest.importorskip('onnxruntime')
    return pytest.importorskip('opennav_sam3_inference.tracker.ort_plugin_ep')


def test_supported_stack_is_accepted(ep):
    assert ep.stack_errors(GOOD) == []


@pytest.mark.parametrize('override', [
    {'migraphx': '2.17.0.dev+2e9924db6'},
    {'migraphx': None},
    {'onnxruntime': '1.24.2'},
    {'ep_package': None},
    {'ep_package': '0.9.0'},
])
def test_unsupported_stack_is_rejected(ep, override):
    assert ep.stack_errors({**GOOD, **override})


class _Options:
    def __init__(self):
        self.entries, self.provider = {}, None

    def add_session_config_entry(self, key, value):
        self.entries[key] = value

    def add_provider_for_devices(self, devices, options):
        self.provider = (devices, options)


class _Assignment:
    def __init__(self, name, nodes):
        self.ep_name, self._nodes = name, nodes

    def get_nodes(self):
        return self._nodes


def _fake_session(ep, monkeypatch, assignment):
    session = types.SimpleNamespace(get_provider_graph_assignment_info=lambda: assignment)
    monkeypatch.setattr(ep, '_devices', lambda: ['device0'])
    monkeypatch.setattr(ep.ort, 'InferenceSession', lambda path, sess_options: session)
    return session


def test_plugin_session_uses_new_option_keys_and_is_tagged(ep, monkeypatch, tmp_path):
    session = _fake_session(
        ep, monkeypatch, [_Assignment('MIGraphXExecutionProvider', [1, 2])])
    options = _Options()
    result = ep.create_mig_session('m.onnx', options, tmp_path, fp16=True)
    assert result is session and ep.is_plugin_session(result)
    devices, provider_options = options.provider
    assert devices == ['device0']
    assert provider_options == {'cache_dir': str(tmp_path), 'migraphx_fp16_enable': '1'}
    assert options.entries['session.record_ep_graph_assignment_info'] == '1'


def test_fp32_session_omits_fp16_option(ep, monkeypatch, tmp_path):
    _fake_session(ep, monkeypatch, [_Assignment('MIGraphXExecutionProvider', [1])])
    options = _Options()
    ep.create_mig_session('m.onnx', options, tmp_path, fp16=False)
    assert options.provider[1] == {'cache_dir': str(tmp_path)}


@pytest.mark.parametrize('assignment', [
    [],
    [_Assignment('CPUExecutionProvider', [1])],
    [_Assignment('MIGraphXExecutionProvider', [1]), _Assignment('CPUExecutionProvider', [2])],
])
def test_any_cpu_node_raises_instead_of_falling_back(ep, monkeypatch, tmp_path, assignment):
    _fake_session(ep, monkeypatch, assignment)
    with pytest.raises(RuntimeError, match='refusing silent CPU fallback'):
        ep.create_mig_session('m.onnx', _Options(), tmp_path)


def test_no_device_raises(ep, monkeypatch, tmp_path):
    monkeypatch.setattr(ep, '_devices', lambda: [])
    with pytest.raises(RuntimeError, match='no device'):
        ep.create_mig_session('m.onnx', _Options(), tmp_path)


def test_no_legacy_provider_options_remain():
    # The plugin EP rejects these keys and ORT then silently runs on CPU.
    paths = list(TRACKER_DIR.rglob('*.py')) + list(EXPORT_DIR.rglob('*.py'))
    assert paths
    for path in paths:
        source = path.read_text()
        assert 'migraphx_model_cache_dir' not in source, path
        assert not re.search(r"providers=\[['\"]MIGraphXExecutionProvider", source), path
        assert not re.search(r"\(['\"]MIGraphXExecutionProvider['\"],\s*\{", source), path


def test_importing_tracker_does_not_change_benchmarking_env():
    # Build scripts pop MIGRAPHX_SKIP_BENCHMARKING, import the tracker, then
    # compile caches; an import-time setdefault would silently disable autotuning.
    pytest.importorskip('migraphx')
    env = {k: v for k, v in os.environ.items() if k != 'MIGRAPHX_SKIP_BENCHMARKING'}
    env['PYTHONPATH'] = os.pathsep.join(
        [str(PACKAGE_ROOT)] + [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p])
    code = ('import os, opennav_sam3_inference.tracker.migraphx_runtime; '
            "assert 'MIGRAPHX_SKIP_BENCHMARKING' not in os.environ")
    result = subprocess.run([sys.executable, '-c', code], env=env, cwd=PACKAGE_ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
