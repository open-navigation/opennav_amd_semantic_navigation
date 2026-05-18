#!/usr/bin/env python3
"""Compile simplified backbone ONNX to a MIGraphX .mxr cache with autotuning.

Reads backbone_<source>/single_simplified.onnx and produces backbone_<source>/tuned.mxr —
the runtime cache that tracker.py loads in ~3s instead of recompiling each
session.

Autotuning runs ~3 minutes for 504px and ~9 minutes for 1008px the first
time; the resulting .mxr is hardware-specific (gfx1151) and locked to the
MIGraphX build that produced it (currently 2.15+patches.20260511).

Requires PYTHONPATH=/opt/rocm-7.2.x/lib so the patched migraphx Python
binding loads. The wrapping setup.sh sets this; if running standalone:

    PYTHONPATH=/opt/rocm-7.2.x/lib${PYTHONPATH:+:$PYTHONPATH} \\
        python export/backbone/compile_backbone_mxr.py --imgsz 504 --onnx-dir onnx_files_504
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--onnx-dir", type=Path, required=True,
                   help="Resolution root (e.g. onnx_files_504). Reads "
                        "<onnx-dir>/backbone_<source>/single_simplified.onnx, writes "
                        "<onnx-dir>/backbone_<source>/tuned.mxr.")
    p.add_argument("--imgsz", type=int, default=504)
    p.add_argument("--backbone-source", choices=["tracker", "detector"],
                   default="tracker",
                   help="Which backbone subdir to operate on")
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Skip migraphx.quantize_fp16 (default: enabled).",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-compile output sanity check.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sub_dir = args.onnx_dir / f"backbone_{args.backbone_source}"
    src = sub_dir / "single_simplified.onnx"
    dst = sub_dir / "tuned.mxr"

    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run export/backbone/simplify_backbone.py first."
        )

    # Make sure autotuning is enabled (env can disable it for fast iteration).
    os.environ.pop("MIGRAPHX_SKIP_BENCHMARKING", None)

    # Route attention subgraphs through rocMLIR-compiled kernels.
    # Validated on gfx1151: 18% faster (169ms vs 200ms/frame) with identical
    # mask quality (IoU=0.9995 vs baseline). Not the default on RDNA/gfx11;
    # must be set explicitly. Does not affect non-attention ops.
    os.environ.setdefault("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", "attention")

    # Defer the import: it touches the dynamic linker and prints whatever
    # warning the patched library emits.
    # MIGraphX Python binding lives in /opt/rocm-7.2.x/lib — add it if not
    # already on sys.path (e.g. when invoked as a subprocess without PYTHONPATH).
    import glob as _g
    _mxr_lib = (
        (os.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
        if os.path.isdir(os.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
        else next(
            (p for p in sorted(_g.glob("/opt/rocm-7.2.*/lib"), reverse=True)
             if os.path.isdir(p)), "/opt/rocm-7.2.0/lib"
        )
    )
    if _mxr_lib not in sys.path:
        sys.path.insert(0, _mxr_lib)
    import migraphx

    print(f"migraphx from: {migraphx.__file__}")
    print(f"Compiling {src} ...")
    print(f"  (autotuning enabled — first run takes ~3 min @504px / ~9 min @1008px)")

    t0 = time.perf_counter()
    prog = migraphx.parse_onnx(str(src))
    if not args.no_fp16:
        migraphx.quantize_fp16(prog)
    prog.compile(migraphx.get_target("gpu"), offload_copy=True)
    elapsed = time.perf_counter() - t0
    print(f"  Compiled in {elapsed:.0f}s")

    migraphx.save(prog, str(dst))
    size_mb = dst.stat().st_size / 1e6
    print(f"  Saved: {dst}  ({size_mb:.0f} MB)")

    if args.skip_verify:
        return

    print("\n[verify] Running compiled backbone, checking outputs are C-contiguous ...")
    inp = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    arg = migraphx.argument(inp)
    # Warmup so first-call kernel JIT doesn't pollute the check.
    for _ in range(3):
        prog.run({"pixel_values": arg})
    outs = prog.run({"pixel_values": arg})
    all_ok = True
    for i, o in enumerate(outs):
        a = np.array(o)
        c = a.flags.c_contiguous
        all_ok = all_ok and c
        print(f"  fpn_{i}: shape={a.shape} C_contiguous={c}")
    if not all_ok:
        raise SystemExit(
            "Outputs are NOT C-contiguous — patched MIGraphX (NHWC fix) is "
            "not in effect. Reinstall via tools/install_migraphx_patched.sh."
        )
    print("  OK — all outputs C-contiguous")


if __name__ == "__main__":
    main()
