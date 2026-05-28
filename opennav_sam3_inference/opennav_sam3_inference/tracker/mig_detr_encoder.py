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
        # Detect the ONNX-compiled prompt seq_len so we can pad runtime
        # inputs to match. ONNX text_features shape: [1, seq_len, 256].
        ts_input = next(i for i in self.session.get_inputs() if i.name == "text_features")
        self.expected_seq_len = int(ts_input.shape[1])
        print(f"  detr_encoder (ORT MIG EP): ready in {elapsed:.1f}s on {prov} "
              f"(prompt_seq_len={self.expected_seq_len})")
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

        # Pad text_features + text_mask LOCALLY to the ONNX-compiled fixed
        # seq_len so variable-length prompts (text only, text+box, text+more
        # box) all run on the same precompiled MIG model. The original
        # text_features (un-padded) is preserved for the downstream decoder,
        # which still uses the ORIGINAL text_mask shape — passing padded
        # tensors downstream would break the PyTorch detr_decoder.
        orig_text_features = text_features
        cur_seq_len = text_features.shape[1]
        target_seq_len = self.expected_seq_len
        mig_text_features = text_features
        mig_text_mask = text_mask
        if cur_seq_len > target_seq_len:
            if not getattr(self, "_warned_truncate", False):
                print(f"  [MIGDetrEncoder] WARN: prompt seq {cur_seq_len} > "
                      f"compiled {target_seq_len}, truncating", flush=True)
                self._warned_truncate = True
            mig_text_features = text_features[:, :target_seq_len, :]
            mig_text_mask = text_mask[:, :target_seq_len]
        elif cur_seq_len < target_seq_len:
            pad_len = target_seq_len - cur_seq_len
            pad_features = torch.zeros(
                text_features.shape[0], pad_len, text_features.shape[2],
                dtype=text_features.dtype, device=text_features.device,
            )
            mig_text_features = torch.cat([text_features, pad_features], dim=1)
            pad_mask = torch.zeros(
                text_mask.shape[0], pad_len,
                dtype=text_mask.dtype, device=text_mask.device,
            )
            mig_text_mask = torch.cat([text_mask, pad_mask], dim=1)

        cross_mask = precompute_cross_attn_mask(mig_text_mask, dtype)

        # Run via ORT MIG EP (numpy fp32 in/out)
        out_np = self.session.run(None, {
            "vision_feature":  vision_feature.detach().float().cpu().numpy(),
            "vision_pos":      vision_pos.detach().float().cpu().numpy(),
            "text_features":   mig_text_features.detach().float().cpu().numpy(),
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
            text_features=orig_text_features,  # un-padded — downstream uses orig mask
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
