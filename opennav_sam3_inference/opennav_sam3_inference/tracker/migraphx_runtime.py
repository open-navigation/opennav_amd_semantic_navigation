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

"""Shared MIGraphX runtime helpers for SAM3 inference."""

from __future__ import annotations

import glob
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .rocm_env import apply as _apply_rocm_env


_apply_rocm_env()
os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")

# Keep the existing OpenNav development build as a compatibility fallback.
# Packaged runtimes can select a different binding through PYTHONPATH or
# MIGRAPHX_BUILD_LIB.
_MXR_BUILD_LIB = os.environ.get(
    "MIGRAPHX_BUILD_LIB",
    "/home/amd/project/tools/AMDMIGraphX/build_docker/lib",
)


def _load_migraphx_module():
    """Import the MIGraphX Python binding selected by the host environment."""
    rocm_path = os.environ.get("ROCM_PATH", "").rstrip("/")
    rocm_lib = (
        f"{rocm_path}/lib"
        if rocm_path and os.path.isdir(f"{rocm_path}/lib")
        else next(
            (
                path
                for path in sorted(glob.glob("/opt/rocm-7.2.*/lib"), reverse=True)
                if os.path.isdir(path)
            ),
            "/opt/rocm-7.2.0/lib",
        )
    )
    if rocm_lib not in sys.path:
        sys.path.insert(0, rocm_lib)
    if (
        _MXR_BUILD_LIB
        and os.path.isdir(_MXR_BUILD_LIB)
        and _MXR_BUILD_LIB not in sys.path
    ):
        sys.path.append(_MXR_BUILD_LIB)
    import migraphx

    return migraphx


class MIGraphXSession:
    """Expose a compiled MIGraphX program through an ORT-like ``run`` API."""

    def __init__(
        self,
        onnx_path: str | Path,
        cache_path: str | Path,
        fp16: bool = True,
        label: str = "MIGraphX session",
    ) -> None:
        mxr = _load_migraphx_module()
        self._mxr = mxr

        cache_path = Path(cache_path)
        onnx_path = Path(onnx_path)

        if cache_path.exists():
            print(f"  {label}: loading {cache_path.name} ...")
            start = time.perf_counter()
            self._prog = mxr.load(str(cache_path))
            print(f"  {label}: ready in {time.perf_counter() - start:.1f}s")
        elif onnx_path.exists():
            print(f"  {label}: compiling {onnx_path.name} ...")
            start = time.perf_counter()
            program = mxr.parse_onnx(str(onnx_path))
            if fp16:
                mxr.quantize_fp16(program)
            program.compile(mxr.get_target("gpu"), offload_copy=True)
            print(f"  {label}: compiled in {time.perf_counter() - start:.1f}s")
            mxr.save(program, str(cache_path))
            self._prog = program
        else:
            raise FileNotFoundError(
                f"Neither {cache_path} nor {onnx_path} found"
            )

        self._in_names = self._prog.get_parameter_names()

    def run(self, _output_names, inputs: dict) -> list:
        """Run contiguous NumPy inputs and return NumPy outputs."""
        args = {
            name: self._mxr.argument(np.ascontiguousarray(value))
            for name, value in inputs.items()
        }
        return [np.array(output) for output in self._prog.run(args)]

    def get_providers(self) -> list:
        """Return an ORT-compatible provider list."""
        return ["MIGraphXExecutionProvider"]


class MIGraphXBackbone:
    """Run the precompiled SAM3 vision encoder with MIGraphX.

    A ``tuned_gpuio.mxr`` cache is selected only when explicitly provided and
    present. Otherwise the existing host-I/O ``tuned.mxr`` path is unchanged.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        cache_path: str | Path,
        gpu_io_cache_path: str | Path | None = None,
    ) -> None:
        mxr = _load_migraphx_module()
        self._mxr = mxr

        cache_path = Path(cache_path)
        onnx_path = Path(onnx_path)
        gpu_io_cache_path = (
            Path(gpu_io_cache_path) if gpu_io_cache_path is not None else None
        )
        load_path = (
            gpu_io_cache_path
            if gpu_io_cache_path is not None and gpu_io_cache_path.exists()
            else cache_path
        )
        self.gpu_io = gpu_io_cache_path is not None and load_path == gpu_io_cache_path

        if load_path.exists():
            mode = " GPU-I/O" if self.gpu_io else ""
            print(f"  MIGraphX backbone{mode}: loading {load_path.name} ...")
            start = time.perf_counter()
            self._prog = mxr.load(str(load_path))
            print(
                f"  MIGraphX backbone{mode}: ready in "
                f"{time.perf_counter() - start:.1f}s"
            )
        elif onnx_path.exists():
            print(
                f"  MIGraphX backbone: compiling {onnx_path.name} "
                "with autotuning (~3 min) ..."
            )
            start = time.perf_counter()
            old_skip = os.environ.pop("MIGRAPHX_SKIP_BENCHMARKING", None)
            try:
                program = mxr.parse_onnx(str(onnx_path))
                mxr.quantize_fp16(program)
                program.compile(mxr.get_target("gpu"), offload_copy=True)
            finally:
                if old_skip is not None:
                    os.environ["MIGRAPHX_SKIP_BENCHMARKING"] = old_skip
            print(
                f"  MIGraphX backbone: compiled in "
                f"{time.perf_counter() - start:.1f}s"
            )
            mxr.save(program, str(cache_path))
            print(f"  MIGraphX backbone: cache saved -> {cache_path}")
            self._prog = program
        else:
            raise FileNotFoundError(
                f"MIGraphX backbone needs {cache_path} (pre-compiled) "
                f"or {onnx_path} (to compile from scratch); neither found."
            )

        self._gpu_output_names = sorted(
            (
                name
                for name in self._prog.get_parameter_names()
                if "#output_" in name
            ),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        if self.gpu_io and not self._gpu_output_names:
            raise RuntimeError(
                f"{load_path} has no GPU output parameters; rebuild it with "
                "compile_backbone_mxr.py --gpu-io"
            )
        self._gpu_io_poisoned = False
        self._failed_run_keepalives = []

    def _drain_failed_run(self, keepalive, device) -> None:
        """Retain raw-pointer buffers unless the device can be drained safely."""
        import torch

        self._failed_run_keepalives.append(keepalive)
        try:
            torch.cuda.synchronize(device=device)
        except BaseException:
            self._gpu_io_poisoned = True
        else:
            self._failed_run_keepalives.remove(keepalive)

    def warmup(self, n: int = 3) -> None:
        """Warm up the selected host-I/O or GPU-I/O program."""
        if self.gpu_io:
            import torch

            shape = list(
                self._prog.get_parameter_shapes()["pixel_values"].lens()
            )
            data = torch.randn(*shape, device="cuda", dtype=torch.float32)
            for _ in range(n):
                self.run_torch(data)
            return

        shape = list(self._prog.get_parameter_shapes()["pixel_values"].lens())
        data = np.random.randn(*shape).astype(np.float32)
        argument = self._mxr.argument(data)
        for _ in range(n):
            self._prog.run({"pixel_values": argument})

    def __call__(self, img_np: np.ndarray):
        """Run a host-I/O program and return at least four NumPy outputs."""
        if self.gpu_io:
            raise RuntimeError(
                "This backbone uses GPU-resident I/O; call run_torch() with a "
                "Torch CUDA/HIP tensor."
            )

        img_contiguous = np.ascontiguousarray(img_np)
        argument = self._mxr.argument(img_contiguous)
        outputs = self._prog.run({"pixel_values": argument})
        arrays = [np.array(output) for output in outputs]
        if arrays and not arrays[0].flags.c_contiguous:
            arrays = [np.ascontiguousarray(array) for array in arrays]
        while len(arrays) < 4:
            arrays.append(None)
        return tuple(arrays)

    def run_torch(self, pixel_values):
        """Run a GPU-I/O program synchronously on Torch CUDA/HIP allocations."""
        if not self.gpu_io:
            raise RuntimeError("run_torch() requires a tuned_gpuio.mxr program")
        if self._gpu_io_poisoned:
            raise RuntimeError(
                "GPU-I/O backbone is unusable after a failed device drain"
            )

        import torch

        if pixel_values.device.type != "cuda":
            raise ValueError(
                "GPU-resident backbone requires a CUDA/HIP tensor, got "
                f"{pixel_values.device}"
            )
        input_tensor = pixel_values.detach().to(
            dtype=torch.float32
        ).contiguous()

        mgx_to_torch = {
            "bool_type": torch.bool,
            "uint8_type": torch.uint8,
            "int8_type": torch.int8,
            "int16_type": torch.int16,
            "int32_type": torch.int32,
            "int64_type": torch.int64,
            "half_type": torch.float16,
            "float_type": torch.float32,
            "double_type": torch.float64,
        }
        torch_to_mgx = {value: key for key, value in mgx_to_torch.items()}

        def to_argument(tensor):
            shape = self._mxr.shape(
                type=torch_to_mgx[tensor.dtype],
                lens=list(tensor.shape),
                strides=list(tensor.stride()),
            )
            return self._mxr.argument_from_pointer(shape, tensor.data_ptr())

        args = {"pixel_values": to_argument(input_tensor)}
        outputs = []
        parameter_shapes = self._prog.get_parameter_shapes()
        for name in self._gpu_output_names:
            shape = parameter_shapes[name]
            tensor = torch.empty_strided(
                list(shape.lens()),
                list(shape.strides()),
                dtype=mgx_to_torch[shape.type_string()],
                device=pixel_values.device,
            )
            outputs.append(tensor)
            args[name] = to_argument(tensor)

        keepalive = (input_tensor, args, outputs)
        stream = torch.cuda.current_stream(device=pixel_values.device)
        try:
            self._prog.run_async(args, stream.cuda_stream, "ihipStream_t")
            torch.cuda.synchronize(device=pixel_values.device)
        except BaseException:
            self._drain_failed_run(keepalive, pixel_values.device)
            raise
        del keepalive
        return tuple(outputs)


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """Convert BGR uint8 input to normalized float32 NCHW."""
    image = cv2.resize(img_bgr[:, :, ::-1], (imgsz, imgsz))
    normalized = (image.astype(np.float32) / 255.0 - _MEAN) / _STD
    return normalized.transpose(2, 0, 1)[None]


def retarget_resolution(model, new_imgsz: int) -> None:
    """Re-initialize all RoPE buffers for a different input resolution."""
    import torch
    from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoVisionRotaryEmbedding,
    )

    new_height = new_imgsz // 14
    model.config.image_size = new_imgsz
    model.config.memory_attention_rope_feat_sizes = [new_height, new_height]
    model.image_size = new_imgsz
    prompt_encoder = getattr(model, "prompt_encoder", None) or getattr(
        getattr(model, "tracker_model", None), "prompt_encoder", None
    )
    if prompt_encoder is not None:
        prompt_encoder.image_embedding_size = (new_height, new_height)
        prompt_encoder.mask_input_size = (4 * new_height, 4 * new_height)
        prompt_encoder.input_image_size = new_imgsz

    for _, module in model.named_modules():
        device = getattr(module, "rope_embeddings_cos", torch.tensor(0)).device
        dtype = getattr(
            module, "rope_embeddings_cos", torch.tensor(0.0)
        ).dtype
        if isinstance(module, Sam3ViTRotaryEmbedding) and module.end_x > new_height:
            module.end_x = module.end_y = new_height
            frequencies = 1.0 / (
                module.rope_theta
                ** (
                    torch.arange(0, module.dim, 4)[: module.dim // 4].float()
                    / module.dim
                )
            )
            flat = torch.arange(new_height * new_height, dtype=torch.long)
            x_positions = (flat % new_height).float() * module.scale
            y_positions = torch.div(
                flat, new_height, rounding_mode="floor"
            ).float() * module.scale
            inverse = torch.cat(
                [
                    torch.outer(x_positions, frequencies),
                    torch.outer(y_positions, frequencies),
                ],
                dim=-1,
            ).repeat_interleave(2, dim=-1)
            module.register_buffer(
                "rope_embeddings_cos",
                inverse.cos().to(device, dtype),
                persistent=False,
            )
            module.register_buffer(
                "rope_embeddings_sin",
                inverse.sin().to(device, dtype),
                persistent=False,
            )
        elif isinstance(module, Sam3TrackerVideoVisionRotaryEmbedding):
            module.end_x = module.end_y = new_height
            inverse = module.create_inv_freq()
            module.register_buffer(
                "rope_embeddings_cos",
                inverse.cos().to(device, dtype),
                persistent=False,
            )
            module.register_buffer(
                "rope_embeddings_sin",
                inverse.sin().to(device, dtype),
                persistent=False,
            )


class MemoryBank:
    """FIFO of spatial memory entries with temporal positional encodings."""

    def __init__(self, temporal_pe: np.ndarray, max_slots: int = 7):
        self.temporal_pe = temporal_pe
        self.max_slots = max_slots
        self._entries: deque = deque(maxlen=max_slots)

    def push(
        self,
        maskmem_features: np.ndarray,
        maskmem_pos_enc: np.ndarray,
    ) -> None:
        """Insert the newest memory entry at the front of the bank."""
        self._entries.appendleft((maskmem_features, maskmem_pos_enc))

    def build_attention_inputs(self, fixed_slots: int = 0):
        """Build ordered memory and position tensors for attention."""
        if not self._entries:
            return None, None
        memory_list = []
        position_list = []
        for index, (features, positions) in enumerate(self._entries):
            position_list.append(positions + self.temporal_pe[index])
            memory_list.append(features)
        if fixed_slots > 0 and len(memory_list) < fixed_slots:
            last_features = memory_list[0]
            last_positions = position_list[0]
            while len(memory_list) < fixed_slots:
                memory_list.append(last_features)
                position_list.append(last_positions)
        return (
            np.concatenate(memory_list, axis=0),
            np.concatenate(position_list, axis=0),
        )

    def reset(self) -> None:
        """Clear all stored entries."""
        self._entries = deque(maxlen=self.max_slots)

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "MIGraphXBackbone",
    "MIGraphXSession",
    "MemoryBank",
    "preprocess_image",
    "retarget_resolution",
]
