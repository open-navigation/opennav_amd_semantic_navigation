#!/usr/bin/env python3
"""Simplify backbone ONNX with onnx-simplifier (constant folding, dead-code).

Reads the fp32 single-session backbone produced by `export_backbone_single.py`
and writes a smaller, leaner ONNX with all dynamic Shape/If branches resolved
for the fixed input shape.

This is required before MIGraphX compile: stock backbone has ~9000 nodes with
hundreds of constant-folded Shape paths and If branches that MIGraphX can't
fuse. After onnxsim it drops to ~2200 nodes, enabling far better kernel fusion.

Usage:
    python export/backbone/simplify_backbone.py --imgsz 504 --onnx-dir onnx_files_504
"""

from __future__ import annotations
import argparse
from pathlib import Path

import onnx
import onnxsim


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Resolution root (e.g. onnx_files_504). Reads "
                        "<onnx-dir>/backbone_<source>/single_fp32.onnx, writes "
                        "<onnx-dir>/backbone_<source>/single_simplified.onnx.")
    p.add_argument("--imgsz", type=int, default=504)
    p.add_argument("--backbone-source", choices=["tracker", "detector"],
                   default="tracker",
                   help="Which backbone subdir to operate on")
    return p.parse_args()


def main():
    args = parse_args()
    sub_dir = args.onnx_dir / f"backbone_{args.backbone_source}"
    src = sub_dir / "single_fp32.onnx"
    dst = sub_dir / "single_simplified.onnx"

    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run export/backbone/export_backbone_single.py first."
        )

    print(f"Loading {src} ({src.stat().st_size / 1e6:.0f} MB)...")
    model = onnx.load(str(src))
    n_before = len(model.graph.node)
    n_if = sum(1 for n in model.graph.node if n.op_type == "If")
    n_shape = sum(1 for n in model.graph.node if n.op_type == "Shape")
    print(f"Before: {n_before} nodes  (If={n_if}, Shape={n_shape})")

    shape = [1, 3, args.imgsz, args.imgsz]
    print(f"Running onnxsim with overwrite_input_shapes={{pixel_values: {shape}}} ...")
    simplified, check = onnxsim.simplify(
        model,
        overwrite_input_shapes={"pixel_values": shape},
    )

    n_after = len(simplified.graph.node)
    n_if2 = sum(1 for n in simplified.graph.node if n.op_type == "If")
    n_shape2 = sum(1 for n in simplified.graph.node if n.op_type == "Shape")
    pct = (n_before - n_after) / n_before * 100
    print(f"After:  {n_after} nodes  (If={n_if2}, Shape={n_shape2})  check={check}")
    print(f"Removed: {n_before - n_after} nodes ({pct:.1f}%)")

    onnx.save(simplified, str(dst))
    print(f"Saved: {dst}  ({dst.stat().st_size / 1e6:.0f} MB)")
    if not check:
        raise SystemExit("onnxsim verification check failed — output may be incorrect")


if __name__ == "__main__":
    main()
