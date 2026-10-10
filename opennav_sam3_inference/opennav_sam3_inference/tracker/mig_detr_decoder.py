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

"""Fixed-shape 504px DETR decoder executed by direct MIGraphX GPU I/O."""
from __future__ import annotations

import hashlib
from pathlib import Path

import migraphx
import torch
import torch.nn as nn

from transformers.models.sam3.modeling_sam3 import Sam3DETRDecoderOutput


FIXED_DETR_DECODER_SHA256 = (
    '141bab3fb6940327dff8548c907ad205267ff6546d24aaae3f413fc6eb30ded3'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class MIGFixedDetrDecoder(nn.Module):
    """
    Drop-in direct-MXR replacement for the fixed 504px DETR decoder.

    The graph boundary is FP32 while the surrounding SAM3 model remains FP16.
    Persistent input/output buffers are exposed to MIGraphX by pointer, and the
    program runs on the current Torch HIP stream.  The original decoder is kept
    alive because SAM3 applies its external ``box_head`` to returned states.
    """

    _INPUT_NAMES = (
        'vision_features',
        'text_features',
        'vision_pos_encoding',
        'text_cross_attn_mask',
    )
    _INPUT_LENS = {
        'vision_features': (1, 1296, 256),
        'text_features': (1, 32, 256),
        'vision_pos_encoding': (1, 1296, 256),
        'text_cross_attn_mask': (1, 1, 1, 32),
    }
    _OUTPUT_LENS = (
        (6, 1, 200, 256),
        (6, 1, 200, 4),
        (6, 1, 1),
    )

    def __init__(
        self,
        mxr_path: str | Path,
        original_decoder: nn.Module,
        *,
        verify_sha256: bool = True,
    ) -> None:
        super().__init__()
        self.mxr_path = Path(mxr_path)
        if verify_sha256:
            actual = _sha256(self.mxr_path)
            checksum = self.mxr_path.with_suffix(self.mxr_path.suffix + '.sha256')
            expected = FIXED_DETR_DECODER_SHA256
            if checksum.is_file():
                fields = checksum.read_text(encoding='ascii').split()
                if len(fields) != 2 or fields[1] != self.mxr_path.name:
                    raise RuntimeError(f'invalid fixed decoder checksum file: {checksum}')
                expected = fields[0]
            if actual != expected:
                raise RuntimeError(
                    'fixed DETR decoder MXR hash mismatch: '
                    f'{actual} != {expected}'
                )

        self.config = original_decoder.config
        # Avoid registering the original module tree twice while retaining the
        # box-head owner and its parameter lifetime.
        object.__setattr__(self, '_original_decoder_keepalive', original_decoder)
        object.__setattr__(self, 'box_head', original_decoder.box_head)

        self.program = migraphx.load(str(self.mxr_path))
        self.parameter_shapes = self.program.get_parameter_shapes()
        missing = [
            name for name in self._INPUT_NAMES if name not in self.parameter_shapes
        ]
        if missing:
            raise RuntimeError(f'fixed decoder MXR is missing inputs: {missing}')
        for name, expected in self._INPUT_LENS.items():
            shape = self.parameter_shapes[name]
            if tuple(shape.lens()) != expected or shape.type_string() != 'float_type':
                raise RuntimeError(
                    f'unexpected fixed decoder parameter {name}: '
                    f'{shape.type_string()} {tuple(shape.lens())}'
                )

        self.all_output_names = tuple(
            sorted(
                (name for name in self.parameter_shapes if '#output_' in name),
                key=lambda name: int(name.rsplit('_', 1)[1]),
            )
        )
        selected = []
        for expected in self._OUTPUT_LENS:
            matches = [
                name for name in self.all_output_names
                if tuple(self.parameter_shapes[name].lens()) == expected
                and self.parameter_shapes[name].type_string() == 'float_type'
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f'fixed decoder requires one output shaped {expected}, got {matches}'
                )
            selected.append(matches[0])
        self.output_names = tuple(selected)
        self._output_indices = tuple(
            self.all_output_names.index(name) for name in self.output_names
        )

        self._buffers_device: torch.device | None = None
        self._fp32_inputs: tuple[torch.Tensor, ...] | None = None
        self._fp32_outputs: tuple[torch.Tensor, ...] | None = None
        self._arguments = None

    @staticmethod
    def _argument(tensor: torch.Tensor):
        type_names = {
            torch.float32: 'float_type',
            torch.float16: 'half_type',
            torch.int64: 'int64_type',
            torch.bool: 'bool_type',
        }
        shape = migraphx.shape(
            type=type_names[tensor.dtype],
            lens=list(tensor.shape),
            strides=list(tensor.stride()),
        )
        return migraphx.argument_from_pointer(shape, tensor.data_ptr())

    def _initialize_buffers(self, inputs: tuple[torch.Tensor, ...]) -> None:
        self._buffers_device = inputs[0].device
        self._fp32_inputs = tuple(
            torch.empty(tuple(tensor.shape), device=tensor.device, dtype=torch.float32)
            for tensor in inputs
        )
        self._fp32_outputs = tuple(
            torch.empty_strided(
                list(self.parameter_shapes[name].lens()),
                list(self.parameter_shapes[name].strides()),
                dtype=torch.float32,
                device=inputs[0].device,
            )
            for name in self.all_output_names
        )
        self._arguments = {
            name: self._argument(tensor)
            for name, tensor in zip(self._INPUT_NAMES, self._fp32_inputs)
        }
        self._arguments.update(
            {
                name: self._argument(tensor)
                for name, tensor in zip(self.all_output_names, self._fp32_outputs)
            }
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        **kwargs,
    ) -> Sam3DETRDecoderOutput:
        del kwargs
        inputs_by_name = {
            'vision_features': vision_features,
            'text_features': text_features,
            'vision_pos_encoding': vision_pos_encoding,
        }
        for name, tensor in inputs_by_name.items():
            if tensor.device.type != 'cuda':
                raise ValueError('fixed MIG DETR decoder requires GPU tensors')
            if tuple(tensor.shape) != self._INPUT_LENS[name]:
                raise ValueError(
                    f'expected {name} {self._INPUT_LENS[name]}, got '
                    f'{tuple(tensor.shape)}'
                )
        if spatial_shapes is not None and tuple(spatial_shapes.shape) != (1, 2):
            raise ValueError(
                f'expected spatial_shapes [1,2], got {tuple(spatial_shapes.shape)}'
            )

        dtype = vision_features.dtype
        if text_mask is None:
            cross_mask = torch.zeros(
                self._INPUT_LENS['text_cross_attn_mask'],
                device=vision_features.device,
                dtype=dtype,
            )
        else:
            if tuple(text_mask.shape) != (1, 32):
                raise ValueError(
                    f'expected text_mask [1,32], got {tuple(text_mask.shape)}'
                )
            cross_mask = torch.where(
                text_mask,
                torch.tensor(0.0, device=text_mask.device, dtype=dtype),
                torch.tensor(
                    torch.finfo(dtype).min,
                    device=text_mask.device,
                    dtype=dtype,
                ),
            )[:, None, None, :]

        inputs = (
            vision_features,
            text_features,
            vision_pos_encoding,
            cross_mask,
        )
        if self._arguments is None:
            self._initialize_buffers(inputs)
        elif vision_features.device != self._buffers_device:
            raise RuntimeError(
                f'fixed decoder buffers are on {self._buffers_device}, got '
                f'{vision_features.device}'
            )

        assert self._fp32_inputs is not None
        assert self._fp32_outputs is not None
        for destination, source in zip(self._fp32_inputs, inputs):
            destination.copy_(source)
        stream = torch.cuda.current_stream(device=vision_features.device)
        try:
            self.program.run_async(
                self._arguments,
                stream.cuda_stream,
                'ihipStream_t',
            )
        except BaseException:
            torch.cuda.synchronize(device=vision_features.device)
            raise

        outputs = tuple(
            self._fp32_outputs[index].to(dtype=dtype)
            for index in self._output_indices
        )
        return Sam3DETRDecoderOutput(
            intermediate_hidden_states=outputs[0],
            reference_boxes=outputs[1],
            presence_logits=outputs[2],
        )


__all__ = ['FIXED_DETR_DECODER_SHA256', 'MIGFixedDetrDecoder']
