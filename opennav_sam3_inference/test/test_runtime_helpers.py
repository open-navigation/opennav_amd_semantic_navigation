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

from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def gpu_runtime():
    """Import GPU dependencies only when these tests are executed."""
    pytest.importorskip('torch')
    pytest.importorskip('migraphx')
    return SimpleNamespace(
        migraphx_runtime=pytest.importorskip(
            'opennav_sam3_inference.tracker.migraphx_runtime'
        ),
        ort_gpu_io=pytest.importorskip(
            'opennav_sam3_inference.tracker.ort_gpu_io'
        ),
    )


class _FakeProgram:
    def get_parameter_names(self):
        return ["pixel_values", "#output_1", "#output_0"]


class _FakeMigraphx:
    def __init__(self):
        self.loaded = []

    def load(self, path):
        self.loaded.append(path)
        return _FakeProgram()


def test_backbone_prefers_gpu_io_cache(gpu_runtime, monkeypatch, tmp_path):
    migraphx_runtime = gpu_runtime.migraphx_runtime
    fake_migraphx = _FakeMigraphx()
    monkeypatch.setattr(
        migraphx_runtime, "_load_migraphx_module", lambda: fake_migraphx
    )
    host_cache = tmp_path / "tuned.mxr"
    gpu_cache = tmp_path / "tuned_gpuio.mxr"
    host_cache.touch()
    gpu_cache.touch()

    backbone = migraphx_runtime.MIGraphXBackbone(
        tmp_path / "model.onnx", host_cache, gpu_cache
    )

    assert backbone.gpu_io
    assert fake_migraphx.loaded == [str(gpu_cache)]
    assert backbone._gpu_output_names == ["#output_0", "#output_1"]


def test_backbone_keeps_host_io_when_gpu_cache_is_missing(
    gpu_runtime, monkeypatch, tmp_path
):
    migraphx_runtime = gpu_runtime.migraphx_runtime
    fake_migraphx = _FakeMigraphx()
    monkeypatch.setattr(
        migraphx_runtime, "_load_migraphx_module", lambda: fake_migraphx
    )
    host_cache = tmp_path / "tuned.mxr"
    host_cache.touch()

    backbone = migraphx_runtime.MIGraphXBackbone(
        tmp_path / "model.onnx", host_cache, tmp_path / "missing.mxr"
    )

    assert not backbone.gpu_io
    assert fake_migraphx.loaded == [str(host_cache)]


class _FakeTensor:
    def __init__(self):
        self.device = SimpleNamespace(type="cuda", index=0)
        self.shape = (1,)

    def detach(self):
        return self

    def to(self, **_kwargs):
        return self

    def contiguous(self):
        return self

    def data_ptr(self):
        return 1


class _FakeBinding:
    def bind_input(self, *_args):
        pass

    def bind_output(self, *_args):
        pass

    def synchronize_outputs(self):
        pass


def test_gpu_io_execution_failure_is_not_safe_fallback(gpu_runtime, monkeypatch):
    ort_gpu_io = gpu_runtime.ort_gpu_io
    synchronized = []
    fake_torch = SimpleNamespace(
        float32=object(),
        empty=lambda *_args, **_kwargs: _FakeTensor(),
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            synchronize=lambda **kwargs: synchronized.append(kwargs["device"]),
        ),
    )
    monkeypatch.setattr(ort_gpu_io, "torch", fake_torch)
    session = SimpleNamespace(
        io_binding=lambda: _FakeBinding(),
        run_with_iobinding=lambda _binding: (_ for _ in ()).throw(
            RuntimeError("submitted")
        ),
    )

    with pytest.raises(ort_gpu_io.GpuIoExecutionError):
        ort_gpu_io.run_float32_gpu(
            session, {"input": _FakeTensor()}, "output", (1,)
        )

    assert len(synchronized) == 1
