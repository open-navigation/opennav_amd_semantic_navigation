#!/usr/bin/env python3
"""
Pre-compile and cache all ORT MIGraphX sessions for a given onnx_dir.

Must be run BEFORE the tracker to populate the MIGraphX compile cache.
Run WITHOUT the backbone loaded (no GPU memory pressure) so FP16 compilations
succeed at 1008px without OOM.

Usage:
    python export/tracker_modules/prewarm_ort_cache.py --onnx-dir onnx_files_504        # 504px
    python export/tracker_modules/prewarm_ort_cache.py --onnx-dir onnx_files_1008   # 1008px
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
os.environ.setdefault("MIGRAPHX_GPU_HIP_FLAGS",
                      "-Wno-error -Wno-lifetime-safety-intra-tu-suggestions")
# Do NOT set MIGRAPHX_SKIP_BENCHMARKING here — kernel autotuning is critical:
# memory_attention: 758ms without autotuning vs 53ms with autotuning (14× difference).
# ORT sessions use their own provider options for SKIP_BENCHMARKING.
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"

import onnxruntime as ort

# Direct MIGraphX Python API for the dec_/mem_ sessions further down.
# Imported with alias `_mxr` so the name doesn't clash with the cache path
# variable (`mxr_cache_path`) or the cache directory.  The bindings live at
# /opt/rocm-7.2.x/lib (added to PYTHONPATH by the wrapping setup.sh).
import sys
import glob as _g, os as _o
_rocm_lib = (
    (_o.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
    if _o.path.isdir(_o.environ.get("ROCM_PATH", "").rstrip("/") + "/lib")
    else next(
        (p for p in sorted(_g.glob("/opt/rocm-7.2.*/lib"), reverse=True)
         if _o.path.isdir(p)), "/opt/rocm-7.2.0/lib"
    )
)
sys.path.insert(0, _rocm_lib)
import migraphx as _mxr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_504"),
                   help="Resolution root, e.g. onnx_files_504 or onnx_files_1008. "
                        "Operates on <onnx-dir>/tracker_modules/.")
    p.add_argument("--warmup", type=int, default=1,
                   help="Number of warmup inference runs per session")
    return p.parse_args()


def make_providers(fp16: bool, cache_dir: str):
    opts = {"migraphx_model_cache_dir": cache_dir}
    if fp16:
        opts["migraphx_fp16_enable"] = "1"
    return [("MIGraphXExecutionProvider", opts), ("CPUExecutionProvider", {})]


def warmup_session(sess: ort.InferenceSession, inputs: dict, n: int = 1):
    for i in range(n):
        sess.run(None, inputs)


def main():
    args = parse_args()
    modules_dir = args.onnx_dir.resolve() / "tracker_modules"
    cache_dir = str(modules_dir / "mxr_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Infer resolution from directory name / feature map size
    # memory_attention_fixed_N7.onnx input shapes tell us HW
    ma_path = modules_dir / "memory_attention_fixed_N7.onnx"
    if not ma_path.exists():
        raise FileNotFoundError(
            f"{ma_path} not found. Run export/tracker_modules/export_tracker_modules.py first.")

    # Peek at first input shape to determine HW
    _tmp = ort.InferenceSession(str(ma_path), providers=["CPUExecutionProvider"])
    HW = _tmp.get_inputs()[0].shape[0]   # current_vision_features: (HW, 1, 256)
    H = W = int(HW ** 0.5)
    del _tmp

    # Determine whether to use FP16 for memory_attention
    # At 1008px (HW=5184) we want FP16 — the whole point of this script is to
    # compile it here WITHOUT the backbone so there's no OOM.
    use_fp16_attn = True   # always compile FP16 (no backbone loaded here)

    print(f"Pre-warming ORT MIGraphX cache for {modules_dir}")
    print(f"  Resolution: ~{H*14}px  HW={HW}  cache: {cache_dir}")
    print(f"  FP16 memory_attention: {'yes' if use_fp16_attn else 'no'}")
    print()

    # Dummy inputs
    CF = np.zeros((HW, 1, 256), dtype=np.float32)
    CP = np.zeros((HW, 1, 256), dtype=np.float32)
    MF = np.zeros((7 * HW, 1, 64), dtype=np.float32)
    MP = np.zeros((7 * HW, 1, 64), dtype=np.float32)
    MK = np.zeros((1, 1, H * 16, W * 16), dtype=np.float32)
    fpn0 = np.zeros((1, 256, H * 4, W * 4), dtype=np.float32)
    fpn1 = np.zeros((1, 256, H * 2, W * 2), dtype=np.float32)
    fpn2 = np.zeros((1, 256, H, W), dtype=np.float32)
    box  = np.array([[[[0, 0, H * 14 * 0.8, W * 14 * 0.8]]]], dtype=np.float32)
    pts  = np.zeros((1, 1, 1, 2), dtype=np.float32)
    lbl  = np.full((1, 1, 1), -1, dtype=np.int32)

    # ── 1. memory_attention: skipped — now uses ORT MIGraphX EP ──────────────
    # NOTE: memory_attention was reverted from direct MIGraphX Python API back to
    # ORT MIGraphX EP because the direct API has a numerical correctness bug for
    # attention-type modules (similar to ROCm/AMDMIGraphX#3596 — incorrect results
    # when attention is compiled via direct API, regardless of FP16/FP32).
    # ORT EP handles the attention numerics correctly. ORT manages its own
    # compilation cache internally; no explicit prewarm needed here.
    print(f"  memory_attention: skipped (uses ORT MIGraphX EP, not direct API)")
    t0 = time.perf_counter()

    # ── 2. Direct migrachx sessions (dec_prop, mem_enc, dec_init — all FP32) ──
    # Direct MIG API eliminates ORT's ReorderInput/Output CPU overhead:
    #   dec_prop: ORT 99ms → direct MIG 4.7ms | dec_init: ORT CPU 118ms → 4.9ms
    direct_sessions = [
        ("dec_propagate FP32 (direct migraphx)",
         str(modules_dir / "mask_decoder_propagate.onnx"),
         "dec_prop_fp32.mxr",
         {"fpn_2_cond": fpn2, "fpn_0": fpn0, "fpn_1": fpn1}),
        ("memory_encoder FP32 (direct migraphx)",
         str(modules_dir / "memory_encoder.onnx"),
         "mem_enc_fp32.mxr",
         {"vision_features": fpn2, "masks": MK}),
        ("dec_init FP32 (direct migraphx)",
         str(modules_dir / "mask_decoder_init.onnx"),
         "dec_init_fp32.mxr",
         {"fpn_2": fpn2, "fpn_0": fpn0, "fpn_1": fpn1,
          "input_points": pts, "input_labels": lbl, "input_boxes": box}),
    ]

    total_t = time.perf_counter() - t0
    for label, onnx_path, cache_name, inputs in direct_sessions:
        mxr_cache_path = Path(cache_dir) / cache_name
        print(f"  {label} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            if mxr_cache_path.exists():
                prog = _mxr.load(str(mxr_cache_path))
                print(f"loaded existing  ({time.perf_counter()-t0:.1f}s)")
            else:
                prog = _mxr.parse_onnx(onnx_path)
                prog.compile(_mxr.get_target("gpu"), offload_copy=True)
                _mxr.save(prog, str(mxr_cache_path))
                print(f"compiled+saved  ({time.perf_counter()-t0:.1f}s)")
            _args = {k: _mxr.argument(np.ascontiguousarray(v)) for k, v in inputs.items()}
            for _ in range(args.warmup): prog.run(_args)
            total_t += time.perf_counter() - t0
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nTotal compile time: {total_t:.0f}s")
    print(f"Cache saved to: {cache_dir}")
    print("\nTracker will now load from cache on startup (~2s per session).")


if __name__ == "__main__":
    main()
