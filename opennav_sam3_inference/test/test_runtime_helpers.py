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

import inspect
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
        parallel_video=pytest.importorskip(
            'opennav_sam3_inference.tracker.parallel_video'
        ),
        live_inference=pytest.importorskip(
            'opennav_sam3_inference.tracker.live_inference'
        ),
        hybrid_inference=pytest.importorskip(
            'opennav_sam3_inference.tracker.hybrid_inference'
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


def test_gpu_io_input_fence_is_scoped(gpu_runtime, monkeypatch):
    ort_gpu_io = gpu_runtime.ort_gpu_io
    synchronized = []
    fake_stream = SimpleNamespace(
        synchronize=lambda: synchronized.append('input')
    )
    fake_torch = SimpleNamespace(
        float32=object(),
        empty=lambda *_args, **_kwargs: _FakeTensor(),
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            current_stream=lambda **_kwargs: fake_stream,
            synchronize=lambda **_kwargs: synchronized.append('device'),
        ),
    )
    monkeypatch.setattr(ort_gpu_io, 'torch', fake_torch)
    session = SimpleNamespace(
        io_binding=lambda: _FakeBinding(),
        run_with_iobinding=lambda _binding: None,
    )

    ort_gpu_io.run_float32_gpu(
        session, {'input': _FakeTensor()}, 'output', (1,)
    )
    with ort_gpu_io.fence_ort_inputs():
        ort_gpu_io.run_float32_gpu(
            session, {'input': _FakeTensor()}, 'output', (1,)
        )
    ort_gpu_io.run_float32_gpu(
        session, {'input': _FakeTensor()}, 'output', (1,)
    )

    assert synchronized == ['input']


def test_parallel_tail_policy_defaults_to_mig_auto_mode(gpu_runtime):
    live_inference = gpu_runtime.live_inference
    hybrid_inference = gpu_runtime.hybrid_inference
    assert (
        inspect.signature(live_inference.SAM3Live).parameters[
            'parallel_tail'
        ].default
        is None
    )
    assert (
        inspect.signature(hybrid_inference.SAM3HybridLive).parameters[
            'parallel_tail'
        ].default
        is None
    )


def test_parallel_tail_reset_clears_failed_session(gpu_runtime):
    live_inference = gpu_runtime.live_inference
    calls = []
    session = SimpleNamespace(
        _parallel_tail_failed=True,
        processed_frames={3: object()},
        reset_inference_session=lambda: calls.append('reset'),
    )
    live = live_inference.SAM3Live.__new__(live_inference.SAM3Live)
    live.session = session
    live.parallel_tail = True
    live._next_frame_idx = 3
    live._force_detect_next = False

    live.reset_tracking()

    assert calls == ['reset']
    assert session._parallel_tail_failed is False
    assert session.processed_frames == {}
    assert live._next_frame_idx == 0
    assert live._force_detect_next is True


def test_parallel_tail_close_is_forwarded(gpu_runtime, monkeypatch):
    parallel_video = gpu_runtime.parallel_video
    live_inference = gpu_runtime.live_inference
    hybrid_inference = gpu_runtime.hybrid_inference
    closed = []
    model = SimpleNamespace(_parallel_tail_runtime=object())
    monkeypatch.setattr(
        parallel_video,
        'close_parallel_video_tail',
        lambda value: closed.append(value),
    )
    live = live_inference.SAM3Live.__new__(live_inference.SAM3Live)
    live.model = model

    live.close()
    hybrid = hybrid_inference.SAM3HybridLive.__new__(
        hybrid_inference.SAM3HybridLive
    )
    hybrid.live = SimpleNamespace(close=lambda: closed.append('hybrid'))
    hybrid.close()

    assert closed == [model, 'hybrid']
