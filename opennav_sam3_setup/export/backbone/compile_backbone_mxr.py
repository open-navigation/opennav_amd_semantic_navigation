#!/usr/bin/env python3
"""Compile simplified backbone ONNX to a MIGraphX .mxr cache with autotuning.

Reads backbone_<source>/single_simplified.onnx and produces either
backbone_<source>/tuned.mxr (host I/O) or tuned_gpuio.mxr (Torch GPU I/O).
Both are runtime caches that load in seconds instead of recompiling each
session; the GPU-I/O artifact is additive and never overwrites tuned.mxr.

Autotuning runs ~3 minutes for 504px and ~9 minutes for 1008px the first
time; the resulting .mxr is hardware-specific (gfx1151) and locked to the
MIGraphX 2.17 build that produced it.

The Conda activation hook installed by setup.sh selects the published
MIGraphX binary prefix. Activate that environment before running standalone.

    conda activate opennav-sam3-inference
    python export/backbone/compile_backbone_mxr.py --imgsz 504 \\
        --onnx-dir onnx_files_504
"""

from __future__ import annotations
import argparse
import os
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
    p.add_argument(
        "--gpu-io",
        action="store_true",
        help="Compile with GPU-resident inputs/outputs and write tuned_gpuio.mxr.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sub_dir = args.onnx_dir / f"backbone_{args.backbone_source}"
    src = sub_dir / "single_simplified.onnx"
    dst = sub_dir / ("tuned_gpuio.mxr" if args.gpu_io else "tuned.mxr")

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

    # Defer the import until after compile-specific environment flags are set.
    # setup.sh's Conda activation hook supplies the binary prefix.
    import migraphx

    print(f"migraphx from: {migraphx.__file__}")
    print(f"Compiling {src} ...")
    print(f"  (autotuning enabled — first run takes ~3 min @504px / ~9 min @1008px)")

    t0 = time.perf_counter()
    prog = migraphx.parse_onnx(str(src))
    if not args.no_fp16:
        migraphx.quantize_fp16(prog)
    prog.compile(migraphx.get_target("gpu"), offload_copy=not args.gpu_io)
    elapsed = time.perf_counter() - t0
    print(f"  Compiled in {elapsed:.0f}s")

    migraphx.save(prog, str(dst))
    size_mb = dst.stat().st_size / 1e6
    print(f"  Saved: {dst}  ({size_mb:.0f} MB)")

    if args.skip_verify:
        return

    if args.gpu_io:
        print("\n[verify] Running GPU-resident backbone ...")
        params = {
            name: migraphx.to_gpu(migraphx.generate_argument(shape))
            for name, shape in prog.get_parameter_shapes().items()
        }
        outs = prog.run(params)
        migraphx.gpu_sync()
        for i, output in enumerate(outs):
            array = np.array(migraphx.from_gpu(output))
            if not np.isfinite(array).all():
                raise SystemExit(f"GPU output {i} contains non-finite values")
            print(f"  output_{i}: shape={array.shape} finite=True")
        print("  OK — GPU-resident inputs and outputs are valid")
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
            "Outputs are NOT C-contiguous; verify the published MIGraphX 2.17 "
            "runtime prefix is active."
        )
    print("  OK — all outputs C-contiguous")


if __name__ == "__main__":
    main()
