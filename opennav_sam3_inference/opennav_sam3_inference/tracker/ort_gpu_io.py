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

"""Torch GPU tensor bindings for fixed-shape ONNX Runtime sessions.

ONNX Runtime's ROCm/MIGraphX builds expose the HIP device through the
``"cuda"`` I/O-binding device name. Binding Torch allocations directly avoids
the otherwise implicit GPU -> NumPy -> ORT -> NumPy -> GPU bridge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


class GpuIoExecutionError(RuntimeError):
    """ORT failed after a GPU-bound run may already have been submitted."""


def run_float32_gpu(
    session,
    inputs: Mapping[str, torch.Tensor],
    output_name: str,
    output_shape: Sequence[int],
) -> torch.Tensor:
    """Run one fixed-shape ORT output directly into a Torch GPU allocation."""
    if not inputs:
        raise ValueError("GPU I/O binding requires at least one input")

    first = next(iter(inputs.values()))
    if first.device.type != "cuda":
        raise ValueError(
            f"GPU I/O binding requires CUDA/HIP tensors, got {first.device}"
        )
    device = first.device
    device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )

    # The exported ONNX boundaries are FP32 even when MIGraphX quantizes the
    # graph internally. Keep these tensors alive until run_with_iobinding
    # returns because ORT holds only their raw addresses.
    bound_inputs = {
        name: tensor.detach().to(
            device=device, dtype=torch.float32
        ).contiguous()
        for name, tensor in inputs.items()
    }
    output = torch.empty(tuple(output_shape), device=device, dtype=torch.float32)

    # Binding failures happen before execution is submitted and callers may
    # safely disable GPU I/O and retry through the host path.
    binding = session.io_binding()
    for name, tensor in bound_inputs.items():
        binding.bind_input(
            name,
            "cuda",
            device_id,
            np.float32,
            tuple(tensor.shape),
            tensor.data_ptr(),
        )
    binding.bind_output(
        output_name,
        "cuda",
        device_id,
        np.float32,
        tuple(output.shape),
        output.data_ptr(),
    )
    try:
        session.run_with_iobinding(binding)
        # MIGraphX may enqueue work on an ORT-owned asynchronous stream. Wait
        # before returning the Torch allocation to its caller.
        binding.synchronize_outputs()
    except Exception as exc:
        # Execution may already have been submitted. Drain the device before
        # pointer-backed tensors leave scope and do not let callers retry the
        # same inference through another backend.
        try:
            torch.cuda.synchronize(device=device)
        except Exception:
            pass
        raise GpuIoExecutionError("ORT GPU-I/O execution failed") from exc
    return output


__all__ = ["GpuIoExecutionError", "run_float32_gpu"]
