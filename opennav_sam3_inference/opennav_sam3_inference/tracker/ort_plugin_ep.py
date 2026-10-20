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
ONNX Runtime MIGraphX session factory (ROCm 10.1 official stack).

The supported runtime uses AMD's ``onnxruntime_ep_migraphx`` plugin execution
provider with ONNX Runtime 1.29. It differs from the old built-in
``MIGraphXExecutionProvider`` in three ways that matter here:

* it is registered from a library and attached with ``add_provider_for_devices``;
  ``providers=[("MIGraphXExecutionProvider", ...)]`` silently falls back to CPU;
* it rejects unknown option keys (the old built-in cache key is now
  ``cache_dir``), and ONNX Runtime then also falls back to CPU instead of raising;
* its device memory carries the AMD vendor id, so ``bind_input("cuda", ...)``
  fails; Torch tensors are bound through DLPack ``OrtValue`` instead.

Every session created here asserts that all graph nodes were assigned to the
MIGraphX EP, so a CPU fallback raises instead of running slowly.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import onnxruntime as ort

_REGISTERED = False
_MIG = 'MIGraphXExecutionProvider'

# The only supported combination. Checked by the image smoke test.
SUPPORTED_STACK = {
    'migraphx': '2.18.0+rocm10.1.0',
    'onnxruntime_prefix': '1.29.',
    'ep_package': '1.0.0+rocm10.1.0',
}


def _dist_version(*names: str) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def describe_ort_migraphx_stack() -> dict:
    """
    Return the versions of the MIGraphX / ONNX Runtime stack in this interpreter.

    Does not register the EP, so it is safe to call before any session exists.
    """
    try:
        import migraphx
        mgx = str(migraphx.__version__)
    except ImportError:
        mgx = None
    return {
        'ep': 'plugin',
        'onnxruntime': ort.__version__,
        'migraphx': mgx,
        'ep_package': _dist_version('onnxruntime-ep-migraphx', 'onnxruntime_ep_migraphx'),
    }


def stack_errors(stack: dict | None = None) -> list[str]:
    """Return human-readable reasons ``stack`` is not the supported stack."""
    stack = describe_ort_migraphx_stack() if stack is None else stack
    errors = []
    if stack.get('migraphx') != SUPPORTED_STACK['migraphx']:
        errors.append(
            f"MIGraphX {SUPPORTED_STACK['migraphx']} is required, got {stack.get('migraphx')}"
        )
    if not str(stack.get('onnxruntime', '')).startswith(SUPPORTED_STACK['onnxruntime_prefix']):
        errors.append(
            f"ONNX Runtime {SUPPORTED_STACK['onnxruntime_prefix']}x is required, "
            f"got {stack.get('onnxruntime')}"
        )
    if stack.get('ep_package') != SUPPORTED_STACK['ep_package']:
        errors.append(
            f"onnxruntime-ep-migraphx {SUPPORTED_STACK['ep_package']} is required, "
            f"got {stack.get('ep_package')}"
        )
    return errors


def is_plugin_session(session) -> bool:
    return bool(getattr(session, '_sam3_plugin_ep', False))


def _devices():
    global _REGISTERED
    import onnxruntime_ep_migraphx as ep

    if not _REGISTERED:
        ort.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        _REGISTERED = True
    return [d for d in ort.get_ep_devices() if d.ep_name == ep.get_ep_name()]


def create_mig_session(path, sess_options, cache_dir, *, fp16: bool = True):
    """Create an ORT session whose graph is fully assigned to the MIGraphX plugin EP."""
    # The plugin EP does not create its cache directory; a missing one fails at run time.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    devices = _devices()
    if not devices:
        raise RuntimeError('plugin MIGraphX EP registered but no device enumerated')
    sess_options.add_session_config_entry('session.record_ep_graph_assignment_info', '1')
    options = {'cache_dir': str(cache_dir)}
    if fp16:
        options['migraphx_fp16_enable'] = '1'
    sess_options.add_provider_for_devices(devices, options)
    session = ort.InferenceSession(str(path), sess_options=sess_options)
    assigned = [(a.ep_name, len(a.get_nodes()))
                for a in session.get_provider_graph_assignment_info()]
    if not assigned or any(name != _MIG for name, _ in assigned):
        raise RuntimeError(f'refusing silent CPU fallback for {path}: assignment={assigned}')
    session._sam3_plugin_ep = True
    return session
