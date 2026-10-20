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
def vision_runtime():
    """Import the vision shim only in an environment with its ML dependencies."""
    torch = pytest.importorskip('torch')
    module = pytest.importorskip(
        'opennav_sam3_inference.tracker.mig_vision_encoder'
    )

    class CountingPositionEncoding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            # Ensure state-dict checks detect accidental cache registration.
            self.register_buffer('state_marker', torch.tensor(1.0))

        def forward(self, shape, device, dtype, mask=None):
            assert mask is None
            self.calls += 1
            return torch.full(
                (shape[0], 256, shape[2], shape[3]),
                self.calls,
                device=device,
                dtype=dtype,
            )

    return SimpleNamespace(
        MIGVisionEncoder=module.MIGVisionEncoder,
        CountingPositionEncoding=CountingPositionEncoding,
        torch=torch,
    )


class _FakeBackbone:
    gpu_io = True

    def __init__(self, outputs=None):
        self.outputs = outputs

    def run_torch(self, _pixel_values):
        if self.outputs is None:
            raise AssertionError('fake backbone has no configured outputs')
        return self.outputs


def _outputs(torch, dtype=None, device='cpu'):
    if dtype is None:
        dtype = torch.float16
    fpn = [
        torch.empty((1, 256, side, side), dtype=dtype, device=device)
        for side in (16, 8, 4, 2)
    ]
    return [*fpn, torch.empty((1, 16, 1024), dtype=dtype, device=device)]


def _position_encodings(encoder, outputs):
    return tuple(
        encoder._get_position_encoding(tensor)
        for tensor in outputs[:4]
    )


def test_position_encoding_cache_is_per_encoder_and_reuses_tensors(
    vision_runtime,
):
    """Keep cached position encodings local to each encoder instance."""
    torch = vision_runtime.torch
    position_encoding = vision_runtime.CountingPositionEncoding()
    input_tensor = torch.empty((1, 3, 4, 4), dtype=torch.float16)
    first_encoder = vision_runtime.MIGVisionEncoder(
        _FakeBackbone(_outputs(torch)), position_encoding
    )
    second_encoder = vision_runtime.MIGVisionEncoder(
        _FakeBackbone(_outputs(torch)), position_encoding
    )

    first = first_encoder(input_tensor).fpn_position_encoding
    second = second_encoder(input_tensor).fpn_position_encoding
    first_again = first_encoder(input_tensor).fpn_position_encoding
    second_again = second_encoder(input_tensor).fpn_position_encoding

    assert position_encoding.calls == 8
    assert first is not None
    assert second is not None
    assert first_again is not None
    assert second_again is not None
    assert (
        first_encoder._position_encoding_cache
        is not second_encoder._position_encoding_cache
    )
    assert len(first_encoder._position_encoding_cache) == 4
    assert len(second_encoder._position_encoding_cache) == 4
    assert all(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(first, first_again, strict=True)
    )
    assert all(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(second, second_again, strict=True)
    )
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(first, second, strict=True)
    )


def test_position_encoding_cache_does_not_change_state_dict_keys(
    vision_runtime,
):
    """Keep cached tensors out of the module state dict."""
    torch = vision_runtime.torch
    encoder = vision_runtime.MIGVisionEncoder(
        _FakeBackbone(),
        vision_runtime.CountingPositionEncoding(),
    )
    keys_before = tuple(encoder.state_dict().keys())

    _position_encodings(encoder, _outputs(torch))

    keys_after = tuple(encoder.state_dict().keys())
    assert keys_before == ('position_encoding.state_marker',)
    assert keys_after == keys_before
    assert all(
        'position_encoding_cache' not in key
        for key in keys_after
    )


def test_position_encoding_cache_is_bounded_and_cleared_by_apply(
    vision_runtime,
):
    """Clear stale cached allocations when the module changes dtype or device."""
    torch = vision_runtime.torch
    position_encoding = vision_runtime.CountingPositionEncoding()
    encoder = vision_runtime.MIGVisionEncoder(
        _FakeBackbone(), position_encoding
    )

    for side in (2, 4, 8, 16, 32):
        encoder._get_position_encoding(
            torch.empty((1, 256, side, side), dtype=torch.float16)
        )

    assert position_encoding.calls == 5
    assert len(encoder._position_encoding_cache) == 4

    cached_values = [
        value
        for value, _ in encoder._position_encoding_cache.values()
    ]
    assert all(value.dtype == torch.float16 for value in cached_values)
    encoder.float()
    assert not encoder._position_encoding_cache

    rebuilt = _position_encodings(
        encoder,
        _outputs(torch, dtype=torch.float32),
    )
    assert position_encoding.calls == 9
    assert len(encoder._position_encoding_cache) == 4
    assert all(value.dtype == torch.float32 for value in rebuilt)
    old_ptrs = {value.data_ptr() for value in cached_values}
    new_ptrs = {value.data_ptr() for value in rebuilt}
    assert old_ptrs.isdisjoint(new_ptrs)


def test_position_encoding_cache_cross_stream_eviction_and_clear_lifetime(
    vision_runtime,
):
    """Keep cached GPU allocations valid across stream use and eviction."""
    torch = vision_runtime.torch
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA or ROCm')

    device = torch.device('cuda', torch.cuda.current_device())
    position_encoding = vision_runtime.CountingPositionEncoding().to(device)
    encoder = vision_runtime.MIGVisionEncoder(
        _FakeBackbone(), position_encoding
    )
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    source = torch.empty(
        (1, 256, 64, 64),
        device=device,
        dtype=torch.float16,
    )
    source_key = (tuple(source.shape), device.type, device.index, source.dtype)

    with torch.cuda.stream(producer):
        torch.cuda._sleep(100_000_000)
        first = encoder._get_position_encoding(source)
    ready_event = encoder._position_encoding_cache[source_key][1]
    assert ready_event is not None
    assert not ready_event.query()

    with torch.cuda.stream(consumer):
        cross_stream = encoder._get_position_encoding(source)
        consumed = cross_stream.clone()
        first_consume_done = torch.cuda.Event()
        first_consume_done.record(consumer)
    assert cross_stream.data_ptr() == first.data_ptr()
    assert not first_consume_done.query()

    first_consume_done.synchronize()
    assert torch.count_nonzero(consumed != 1).item() == 0
    del consumed, cross_stream, first
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    with torch.cuda.stream(consumer):
        cross_stream = encoder._get_position_encoding(source)
        cached_ptr = cross_stream.data_ptr()
        torch.cuda._sleep(100_000_000)
        consumed_after_clear = cross_stream.clone()
        second_consume_done = torch.cuda.Event()
        second_consume_done.record(consumer)
    assert not second_consume_done.query()

    for side in (2, 4, 8, 16):
        encoder._get_position_encoding(
            torch.empty(
                (1, 256, side, side),
                device=device,
                dtype=torch.float16,
            )
        )
    assert source_key not in encoder._position_encoding_cache
    del cross_stream

    encoder.float()
    assert not encoder._position_encoding_cache
    with torch.cuda.stream(producer):
        trash = [
            torch.full(
                (1, 256, 64, 64),
                -123.0,
                device=device,
                dtype=torch.float16,
            )
            for _ in range(4)
        ]
    assert not second_consume_done.query()
    assert all(tensor.data_ptr() != cached_ptr for tensor in trash)

    second_consume_done.synchronize()
    assert torch.count_nonzero(consumed_after_clear != 1).item() == 0
    assert all(
        tensor[0, 0, 0, 0].item() == -123.0
        for tensor in trash
    )
