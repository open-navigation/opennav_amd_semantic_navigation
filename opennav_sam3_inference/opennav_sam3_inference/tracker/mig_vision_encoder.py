"""MIGraphX-backed vision_encoder shim for Sam3VideoModel.

Drop-in replacement for `Sam3VideoModel.detector_model.vision_encoder`. Routes
the forward pass through a precompiled MIGraphX `.mxr` (the detector backbone
exported with last_hidden_state — see `export/backbone/export_backbone_single.py
--backbone-source detector`). Returns a `Sam3VisionEncoderOutput` with all
fields the downstream Sam3VideoModel pipeline expects:

  - `fpn_hidden_states`: 4 FPN levels (consumed by detector path)
  - `fpn_position_encoding`: matching sine PE (re-computed cheaply on host)
  - `last_hidden_state`: raw ViT tokens (consumed by `tracker_neck` to
                         compute tracker FPN — different weights from detector)

The position encoding module is reused from the original PyTorch
vision_encoder.neck (it has no learnable parameters; it's a sinusoidal
embedding parameterized by spatial size + dtype).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from .migraphx_runtime import MIGraphXBackbone


def _to_torch_output(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Move a host output without a GPU-side FP32-to-FP16 cast."""
    if dtype == torch.float16:
        host = np.ascontiguousarray(array, dtype=np.float16)
        return torch.from_numpy(host).to(device=device)
    host = np.ascontiguousarray(array)
    return torch.from_numpy(host).to(device=device, dtype=dtype)


class MIGVisionEncoder(nn.Module):
    """Shim that mimics Sam3VisionModel.forward via a MIGraphX backbone.

    Args:
        mxr_backbone:    a `MIGraphXBackbone` whose underlying `.mxr` was
                         exported with `--backbone-source detector` and includes
                         the `last_hidden_state` output (5 outputs total).
        position_encoding: the sine PE module from the original PT
                         vision_encoder.neck (`detector.vision_encoder.neck.position_encoding`).
                         Has no learnable params; reused for cheap FPN PE.
    """

    def __init__(
        self,
        mxr_backbone: MIGraphXBackbone,
        position_encoding: nn.Module,
    ):
        super().__init__()
        self.mxr = mxr_backbone
        self.position_encoding = position_encoding

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> Sam3VisionEncoderOutput:
        device = pixel_values.device
        dtype = pixel_values.dtype

        if self.mxr.gpu_io:
            outs = self.mxr.run_torch(pixel_values)
        else:
            # Host-I/O fallback for existing tuned.mxr artifacts.
            np_in = (
                pixel_values.detach().float().cpu().numpy().astype(
                    np.float32, copy=False
                )
            )
            outs = self.mxr(np_in)
        if len(outs) < 5 or outs[4] is None:
            raise RuntimeError(
                "MIGVisionEncoder requires a 5-output backbone (4 FPN + last_hidden_state). "
                "Re-export with: python export/backbone/export_backbone_single.py "
                "--backbone-source detector --output-name backbone_detector_lhs_fp32.onnx"
            )
        f0, f1, f2, f3, last_hidden_state = outs

        if self.mxr.gpu_io:
            fpn = [tensor.to(dtype=dtype) for tensor in (f0, f1, f2, f3)]
            last_hidden_state = last_hidden_state.to(dtype=dtype)
        else:
            fpn = [
                _to_torch_output(tensor, device, dtype)
                for tensor in (f0, f1, f2, f3)
            ]
            last_hidden_state = _to_torch_output(
                last_hidden_state, device, dtype
            )
        pe = [self.position_encoding(t.shape, t.device, t.dtype) for t in fpn]

        return Sam3VisionEncoderOutput(
            last_hidden_state=last_hidden_state,
            fpn_hidden_states=tuple(fpn),
            fpn_position_encoding=tuple(pe),
            hidden_states=None,
            attentions=None,
        )


def patch_sam3_video_model_with_mig(model, mxr_backbone: MIGraphXBackbone) -> None:
    """In-place replace `model.detector_model.vision_encoder` with a MIG shim.

    `model` must be a `Sam3VideoModel` already loaded and moved to the desired
    device/dtype. After this, every call to `model(...)` /
    `_det_track_one_frame` runs the MIGraphX backbone for the vision encoder
    and the original PyTorch detector head, tracker_neck, and tracker_model.
    """
    pe = model.detector_model.vision_encoder.neck.position_encoding
    shim = MIGVisionEncoder(mxr_backbone, pe)
    # Move to model's device + dtype so children behave consistently
    target_dtype = next(model.detector_model.parameters()).dtype
    target_device = next(model.detector_model.parameters()).device
    shim = shim.to(device=target_device, dtype=target_dtype)
    model.detector_model.vision_encoder = shim
