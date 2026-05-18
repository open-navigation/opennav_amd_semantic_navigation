#!/usr/bin/env python3
"""Export Sam3DetrEncoder to ONNX → simplify → MIGraphX .mxr.

Self-contained one-shot script (vs backbone's 3-stage pipeline) because
detr_encoder is small enough that simplify + compile run in seconds.

Inputs (text-prompt path, fixed shapes after Sam3 preprocessor):
    vision_feature: (1, 256, P, P)   — fpn_2 (image embedding level)
    vision_pos:     (1, 256, P, P)
    text_features:  (1, 32, 256)     — CLIP text encoder pooler_output, padded to 32
    text_mask:      (1, 32) bool     — True where token is real (post-pad)
where P = imgsz // 14 (P=72 at 1008px, P=36 at 504px).

Output:
    last_hidden_state: (1, P*P, 256)

Outputs go to <onnx-dir>/detector_modules/{detr_encoder_fp32.onnx,
detr_encoder_simplified.onnx, detr_encoder.mxr}.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Script lives at <repo>/export/detector/<this>.py — go up THREE levels.
WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, default=Path("/home/amd/project/sam3/model/sam3"))
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="Resolution root (e.g. onnx_files_1008). Outputs go to "
                        "<onnx-dir>/detector_modules/. Defaults from --imgsz.")
    p.add_argument("--imgsz", type=int, default=1008)
    p.add_argument("--text-seq-len", type=int, default=32)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-fp16", action="store_true",
                   help="Skip migraphx.quantize_fp16 (default: enabled)")
    p.add_argument("--build-mxr", action="store_true",
                   help="Also compile a direct-MIG .mxr (default: skip; the runtime shim uses ORT MIG EP because direct-MIG FP16 attention has a numerical bug)")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip ONNX-CPU vs PyTorch verification")
    args = p.parse_args()
    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE / f"onnx_files_{args.imgsz}"
    args.out_dir = args.onnx_dir / "detector_modules"
    return args


def precompute_cross_attn_mask(text_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert (B, T) bool padding mask → (B, 1, 1, T) additive bias.

    Matches what `create_bidirectional_mask` produces for cross-attention:
    valid token = 0.0, padding = a very negative number (so post-softmax weight is 0).
    Broadcastable across (heads, query_len) dims of attention scores.
    """
    neg_inf = torch.finfo(dtype).min
    mask = torch.where(text_mask, torch.tensor(0.0, dtype=dtype),
                       torch.tensor(neg_inf, dtype=dtype))
    return mask[:, None, None, :]


class DetrEncoderWrapper(nn.Module):
    """Pure tensor-in/tensor-out shim around Sam3DetrEncoder.

    The cross-attention mask is precomputed externally (pass `cross_attn_mask`
    of shape (B, 1, 1, T) as input) so the dynamic
    `create_bidirectional_mask` call is kept out of the ONNX graph.
    """

    def __init__(self, detr_encoder):
        super().__init__()
        self.encoder = detr_encoder

    def forward(self, vision_feature, vision_pos, text_features, cross_attn_mask):
        # Flatten the single FPN level: (1, 256, H, W) -> (1, H*W, 256)
        feat = vision_feature.flatten(2).transpose(1, 2)
        pos  = vision_pos.flatten(2).transpose(1, 2)

        hidden = feat
        for layer in self.encoder.layers:
            hidden = layer(
                hidden,
                prompt_feats=text_features,
                vision_pos_encoding=pos,
                prompt_cross_attn_mask=cross_attn_mask,
            )
        return hidden  # (1, H*W, 256)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.out_dir / "detr_encoder_fp32.onnx"
    simp_path = args.out_dir / "detr_encoder_simplified.onnx"
    mxr_path  = args.out_dir / "detr_encoder.mxr"

    P = args.imgsz // 14

    # ----- 1. Build model + wrapper -----
    print(f"Loading Sam3VideoModel.detector_model from {args.checkpoint} ...")
    from transformers import Sam3VideoModel
    model = Sam3VideoModel.from_pretrained(
        str(args.checkpoint), attn_implementation="eager"
    ).cpu().eval()
    detector = model.detector_model
    wrapper = DetrEncoderWrapper(detector.detr_encoder).cpu().eval()

    # ----- 2. Dummy inputs -----
    vision_feature = torch.randn(1, 256, P, P)
    vision_pos     = torch.randn(1, 256, P, P)
    text_features  = torch.randn(1, args.text_seq_len, 256)
    text_mask      = torch.ones(1, args.text_seq_len, dtype=torch.bool)
    cross_mask     = precompute_cross_attn_mask(text_mask, dtype=text_features.dtype)

    # ----- 3. ONNX export (FP32) -----
    print(f"\n[1/3] Exporting FP32 ONNX → {fp32_path.name} ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (vision_feature, vision_pos, text_features, cross_mask),
            str(fp32_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["vision_feature", "vision_pos", "text_features", "cross_attn_mask"],
            output_names=["last_hidden_state"],
        )
    sz = fp32_path.stat().st_size / 1e6
    print(f"      saved: {fp32_path.name} ({sz:.1f} MB) in {time.perf_counter()-t0:.1f}s")

    # ----- 4. Simplify -----
    print(f"\n[2/3] Simplifying with onnxsim → {simp_path.name} ...")
    import onnx
    import onnxsim
    t0 = time.perf_counter()
    m = onnx.load(str(fp32_path))
    n_before = len(m.graph.node)
    simplified, check = onnxsim.simplify(
        m,
        overwrite_input_shapes={
            "vision_feature":  [1, 256, P, P],
            "vision_pos":      [1, 256, P, P],
            "text_features":   [1, args.text_seq_len, 256],
            "cross_attn_mask": [1, 1, 1, args.text_seq_len],
        },
    )
    n_after = len(simplified.graph.node)
    onnx.save(simplified, str(simp_path))
    sz = simp_path.stat().st_size / 1e6
    print(f"      simplified: {n_before} → {n_after} nodes, check={check}")
    print(f"      saved: {simp_path.name} ({sz:.1f} MB) in {time.perf_counter()-t0:.1f}s")
    if not check:
        raise SystemExit("onnxsim verification failed")

    # ----- 5. ONNX-CPU vs PyTorch correctness check -----
    if not args.skip_verify:
        print(f"\n[verify] ONNX-CPU vs PyTorch ...")
        import onnxruntime as ort
        sess = ort.InferenceSession(str(simp_path), providers=["CPUExecutionProvider"])
        np_in = {
            "vision_feature":  vision_feature.numpy(),
            "vision_pos":      vision_pos.numpy(),
            "text_features":   text_features.numpy(),
            "cross_attn_mask": cross_mask.numpy(),
        }
        onnx_out = sess.run(None, np_in)[0]
        with torch.inference_mode():
            ref = wrapper(vision_feature, vision_pos, text_features, cross_mask)
        diff = (ref.numpy() - onnx_out)
        print(f"      ONNX out: shape={onnx_out.shape} std={onnx_out.std():.4f}")
        print(f"      |diff|: max={np.abs(diff).max():.5f}  mean={np.abs(diff).mean():.6f}")
        if np.abs(diff).max() > 1e-3:
            print(f"      ⚠ ONNX-CPU differs from PT (>{1e-3} max diff)")
        else:
            print(f"      ✓ ONNX-CPU matches PT within numeric noise")

    if not args.build_mxr:
        print("\nDone — runtime uses ORT MIG EP. Pass --build-mxr to also build a direct-MIG .mxr (currently broken: FP16 NaN, FP32 0.05 drift breaks detector).")
        return

    # ----- 6. MIGraphX compile -----
    print(f"\n[3/3] MIGraphX compile + autotune → {mxr_path.name} ...")
    os.environ.pop("MIGRAPHX_SKIP_BENCHMARKING", None)
    import migraphx
    print(f"      migraphx from: {migraphx.__file__}")
    t0 = time.perf_counter()
    prog = migraphx.parse_onnx(str(simp_path))
    if not args.no_fp16:
        migraphx.quantize_fp16(prog)
    prog.compile(migraphx.get_target("gpu"), offload_copy=True)
    elapsed = time.perf_counter() - t0
    migraphx.save(prog, str(mxr_path))
    sz = mxr_path.stat().st_size / 1e6
    print(f"      compiled in {elapsed:.0f}s, saved {mxr_path.name} ({sz:.1f} MB)")
    print(f"\nDone. Use this .mxr in tracker/mig_detr_encoder.py shim.")


if __name__ == "__main__":
    main()
