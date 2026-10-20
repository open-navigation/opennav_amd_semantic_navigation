#!/usr/bin/env python3
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

"""Populate the runtime ORT caches of a 504px artifact root at build time.

Constructs SAM3Live exactly as the ROS node does (MIG, fixed DETR decoder) and
runs a few synthetic frames. This compiles the DETR encoder ORT cache with
full autotuning, so the node does not compile it on its first frame, and
checks that the runtime reuses the build-time memory-attention caches: a new
file there means the runtime cache key does not match the build and the node
would recompile S1..S10 on every fresh artifact root.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "opennav_sam3_inference"

# Caches that compile_memory_attention.py / compile_fixed_detr_decoder.py own.
BUILD_OWNED_CACHES = (
    Path("tracker_modules") / "ort_cache_mem_attn",
    Path("detr_decoder_fixed") / "ort_cache",
)


def cache_files(onnx_dir: Path) -> set[Path]:
    return {path.relative_to(onnx_dir) for path in onnx_dir.rglob("*.mxr")}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Runtime caches must be autotuned like every other build artifact.
    os.environ.pop("MIGRAPHX_SKIP_BENCHMARKING", None)
    os.environ.pop("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", None)
    sys.path.insert(0, str(PACKAGE_ROOT))

    import numpy as np
    import torch

    from opennav_sam3_inference.tracker.live_inference import SAM3Live

    onnx_dir = args.onnx_dir.resolve()
    before = cache_files(onnx_dir)
    started = time.perf_counter()
    live = SAM3Live(
        checkpoint=args.checkpoint,
        prompts=["object"],
        onnx_dir=onnx_dir,
        imgsz=504,
        dtype=torch.float16,
        mig=True,
        fixed_detr_decoder=True,
        redetect_every=1,
    )
    generator = np.random.default_rng(20261019)
    try:
        for _ in range(args.frames):
            frame = generator.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
            live.infer(frame)
    finally:
        live.close()
    elapsed = time.perf_counter() - started

    created = sorted(cache_files(onnx_dir) - before)
    print(f"prewarm finished in {elapsed:.1f}s; new runtime caches: "
          f"{[str(path) for path in created] or 'none'}")
    rebuilt = [
        path for path in created
        if any(path.is_relative_to(owned) for owned in BUILD_OWNED_CACHES)
    ]
    if rebuilt:
        raise RuntimeError(
            "the runtime compiled caches that the build already produced: "
            f"{[str(path) for path in rebuilt]}; the build and runtime cache "
            "keys differ"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
