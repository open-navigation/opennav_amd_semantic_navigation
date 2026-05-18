#!/usr/bin/env python3
"""Export Sam3TrackerVideoModel vision_encoder as a single-session ONNX backbone.

Produces one fp32 ONNX (`backbone_<source>/single_fp32.onnx` by default) that takes
`pixel_values` (1, 3, H, H) and returns the four FPN levels
(`fpn_0`, `fpn_1`, `fpn_2`, `fpn_3`) — `fpn_3` (smallest, scale=0.5) is needed
by the SAM3 detector path (text-prompt). The tracker (box-prompt) only consumes
the first three; tracker.py indexes `outputs[0..2]` so the extra output is
backward compatible.

Pipeline:
    [export_backbone_single.py]  ->  backbone_<source>/single_fp32.onnx
    [simplify_backbone.py]       ->  backbone_<source>/single_simplified.onnx
    [compile_backbone_mxr.py]    ->  backbone_<source>/tuned.mxr  (the runtime cache)

Usage:
    python export/backbone/export_backbone_single.py --imgsz 504 --onnx-dir onnx_files_504
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Script lives at <repo>/export/backbone/<this>.py — go up THREE levels.
WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, default=WORKSPACE / "model" / "sam3")
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="Resolution root, e.g. onnx_files_504 or onnx_files_1008. "
                        "Defaults based on --imgsz.")
    p.add_argument("--imgsz", type=int, default=504)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--backbone-source",
        choices=["tracker", "detector"],
        default="tracker",
        help=("Which vision_encoder weights to export. Output goes to "
              "<onnx-dir>/backbone_<source>/single_fp32.onnx. "
              "'tracker' uses Sam3TrackerVideoModel.vision_encoder — produces "
              "FPN features compatible with the box-prompt tracker (existing "
              "8.21 FPS path). 'detector' uses Sam3VideoModel.detector_model"
              ".vision_encoder — needed for the SAM3 text-prompt detector path. "
              "The two share ViT weights but have DIFFERENT FPN proj weights "
              "for fpn[0..2]; fpn[3] weights match. Run both to support both."),
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Run CPU ONNX vs PyTorch comparison after export",
    )
    return p.parse_args()


class SingleSessionBackbone(nn.Module):
    """Whole vision_encoder (ViT backbone + FPN neck) as one graph.

    Returns five tensors:
        fpn_0: (1, 256, 4*P, 4*P)   high-res FPN     (scale=4.0)
        fpn_1: (1, 256, 2*P, 2*P)   mid-res FPN      (scale=2.0)
        fpn_2: (1, 256,   P,   P)   image emb FPN    (scale=1.0)
        fpn_3: (1, 256, P/2, P/2)   small FPN (det)  (scale=0.5)
        last_hidden_state: (1, P*P, 1024)   raw ViT pre-FPN tokens
    where P = imgsz // 14. fpn_3 is consumed by the detector path; the
    box-prompt tracker uses fpn_0..fpn_2. last_hidden_state is needed by
    Sam3VideoModel's tracker_neck (separate FPN with tracker weights) which
    runs on the raw ViT output rather than detector FPN features.
    """

    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, pixel_values):
        out = self.vision_encoder(pixel_values, return_dict=True)
        f0, f1, f2, f3 = out.fpn_hidden_states[:4]
        return f0, f1, f2, f3, out.last_hidden_state


def main():
    args = parse_args()
    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE / f"onnx_files_{args.imgsz}"
    sub_dir = args.onnx_dir / f"backbone_{args.backbone_source}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_path = sub_dir / "single_fp32.onnx"

    from tracker.tracker import retarget_resolution

    if args.backbone_source == "tracker":
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoModel,
        )
        print(f"Loading Sam3TrackerVideoModel (tracker FPN weights) from {args.checkpoint} ...")
        model = (
            Sam3TrackerVideoModel.from_pretrained(
                str(args.checkpoint), attn_implementation="eager"
            )
            .cpu()
            .eval()
        )
        if args.imgsz != 1008:
            retarget_resolution(model, args.imgsz)
        vision_encoder = model.vision_encoder
    else:  # detector
        from transformers import Sam3VideoModel
        print(f"Loading Sam3VideoModel.detector_model (detector FPN weights) from {args.checkpoint} ...")
        model = (
            Sam3VideoModel.from_pretrained(
                str(args.checkpoint), attn_implementation="eager"
            )
            .cpu()
            .eval()
        )
        if args.imgsz != 1008:
            retarget_resolution(model, args.imgsz)
        vision_encoder = model.detector_model.vision_encoder

    P = args.imgsz // 14
    print(f"  imgsz={args.imgsz}px  feature map={P}x{P}  source={args.backbone_source}")

    wrapper = SingleSessionBackbone(vision_encoder).eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)

    print("\nExporting ...")
    with torch.no_grad():
        f0, f1, f2, f3, lhs = wrapper(dummy)
    print(
        f"  Output shapes: fpn_0={tuple(f0.shape)} "
        f"fpn_1={tuple(f1.shape)} fpn_2={tuple(f2.shape)} fpn_3={tuple(f3.shape)} "
        f"last_hidden_state={tuple(lhs.shape)}"
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["pixel_values"],
            output_names=["fpn_0", "fpn_1", "fpn_2", "fpn_3", "last_hidden_state"],
        )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path}  ({size_mb:.0f} MB)")

    # ONNX > 2GB triggers external-data files (.data sidecar). Note this for the user.
    data_path = out_path.with_suffix(out_path.suffix + ".data")
    if data_path.exists():
        data_mb = data_path.stat().st_size / 1e6
        print(f"  External weights: {data_path.name}  ({data_mb:.0f} MB)")

    if args.verify:
        print("\n[verify] Comparing ONNX vs PyTorch outputs ...")
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
        o0, o1, o2, o3 = sess.run(None, {"pixel_values": x})
        print(f"  ONNX: fpn_0={o0.shape}  fpn_1={o1.shape}  fpn_2={o2.shape}  fpn_3={o3.shape}")

        pv = torch.from_numpy(x)
        with torch.inference_mode():
            vis = vision_encoder(pv, return_dict=True)
        for i, (onnx_out, pt_out) in enumerate(
            zip([o0, o1, o2, o3], vis.fpn_hidden_states[:4])
        ):
            diff = np.abs(onnx_out - pt_out.numpy()).max()
            print(f"  fpn_{i} max_diff: {diff:.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
