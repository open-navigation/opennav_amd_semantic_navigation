"""MIGraphX-backed Sam3DetrEncoder shim for Sam3VideoModel.

Drop-in replacement for `Sam3VideoModel.detector_model.detr_encoder`. Routes
the encoder forward through ONNX Runtime's MIGraphX execution provider —
NOT direct migraphx.parse_onnx, because that path has the same FP16
attention numerical bug as memory_attention (Finding #8 in project_summary).
ORT MIG EP gives correct results and is still ~5× faster than the pure
PyTorch detr_encoder.

The cross-attention mask (text padding) is precomputed on the host before
the call — the original `create_bidirectional_mask` path uses dynamic
control flow that does not trace cleanly to ONNX.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from transformers.models.sam3.modeling_sam3 import Sam3DETREncoderOutput


def precompute_cross_attn_mask(text_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, T) bool padding mask → (B, 1, 1, T) additive bias, broadcast over heads + queries."""
    neg_inf = torch.finfo(dtype).min
    mask = torch.where(
        text_mask,
        torch.tensor(0.0, dtype=dtype, device=text_mask.device),
        torch.tensor(neg_inf, dtype=dtype, device=text_mask.device),
    )
    return mask[:, None, None, :]


class MIGDetrEncoder(nn.Module):
    """Replaces Sam3DetrEncoder with a single MIGraphX session call.

    Original forward signature:
        detr_encoder(vision_features=[t], text_features=t, vision_pos_embeds=[t], text_mask=t)
            → Sam3DETREncoderOutput(last_hidden_state, pos_embeds_flattened, text_features, spatial_shapes)
    """

    def __init__(self, onnx_path: Path, original_encoder: nn.Module,
                 ort_cache_dir: Path | None = None):
        super().__init__()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cache_dir = Path(ort_cache_dir) if ort_cache_dir else (
            Path(onnx_path).parent / "ort_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers = [
            ("MIGraphXExecutionProvider", {
                "migraphx_fp16_enable": "1",
                "migraphx_model_cache_dir": str(cache_dir),
            }),
            "CPUExecutionProvider",
        ]
        print(f"  detr_encoder (ORT MIG EP, fp16): compiling from {Path(onnx_path).name} ...")
        import time
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
        elapsed = time.perf_counter() - t0
        prov = self.session.get_providers()[0]
        print(f"  detr_encoder (ORT MIG EP): ready in {elapsed:.1f}s on {prov}")
        # Keep the original encoder around as a fallback / so patched module
        # still passes `isinstance(...) ` checks downstream if any exist.
        self._original = original_encoder

    def forward(
        self,
        vision_features,
        text_features,
        vision_pos_embeds=None,
        text_mask=None,
        spatial_sizes=None,
        **kwargs,
    ):
        # Unwrap singleton lists to tensors
        if isinstance(vision_features, (list, tuple)):
            vision_feature = vision_features[0]
        else:
            vision_feature = vision_features
        if isinstance(vision_pos_embeds, (list, tuple)):
            vision_pos = vision_pos_embeds[0]
        else:
            vision_pos = vision_pos_embeds

        device = vision_feature.device
        dtype = vision_feature.dtype

        # If features came in pre-flattened with spatial_sizes, reshape back.
        if spatial_sizes is not None and vision_feature.ndim == 3:
            h, w = spatial_sizes[0]
            bsz = vision_feature.shape[1]
            vision_feature = vision_feature.reshape(h, w, bsz, -1).permute(2, 3, 0, 1)
            vision_pos = vision_pos.reshape(h, w, bsz, -1).permute(2, 3, 0, 1)

        # Precompute cross-attention mask
        if text_mask is None:
            # Treat all text tokens as valid
            text_mask = torch.ones(
                text_features.shape[0], text_features.shape[1],
                dtype=torch.bool, device=device,
            )
        cross_mask = precompute_cross_attn_mask(text_mask, dtype)

        # Run via ORT MIG EP (numpy fp32 in/out)
        out_np = self.session.run(None, {
            "vision_feature":  vision_feature.detach().float().cpu().numpy(),
            "vision_pos":      vision_pos.detach().float().cpu().numpy(),
            "text_features":   text_features.detach().float().cpu().numpy(),
            "cross_attn_mask": cross_mask.detach().float().cpu().numpy(),
        })
        last_hidden_state = torch.from_numpy(out_np[0]).to(device=device, dtype=dtype)

        # Compute pos_embeds_flattened on host (cheap reshape — needed downstream)
        pos_flat = vision_pos.flatten(2).transpose(1, 2)
        h, w = vision_feature.shape[-2:]
        spatial_shapes = torch.tensor([(h, w)], dtype=torch.long, device=device)

        return Sam3DETREncoderOutput(
            last_hidden_state=last_hidden_state,
            pos_embeds_flattened=pos_flat,
            text_features=text_features,
            spatial_shapes=spatial_shapes,
        )


def patch_sam3_video_model_detr_encoder(model, onnx_path: Path) -> None:
    """In-place replace `model.detector_model.detr_encoder` with the ORT MIG EP shim.

    `onnx_path` should be the simplified .onnx (e.g.
    `onnx_files_1008/detector_modules/detr_encoder_simplified.onnx`); the ORT
    MIG EP compiles it on first use and caches the resulting .mxr next door.

    Call after `patch_sam3_video_model_with_mig` (or before — independent).
    """
    original = model.detector_model.detr_encoder
    shim = MIGDetrEncoder(Path(onnx_path), original)
    target_dtype = next(model.detector_model.parameters()).dtype
    target_device = next(model.detector_model.parameters()).device
    shim = shim.to(device=target_device, dtype=target_dtype)
    model.detector_model.detr_encoder = shim
