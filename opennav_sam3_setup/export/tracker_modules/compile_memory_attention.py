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

"""Compile fixed-shape memory-attention ONNX graphs with full autotuning.

Each shape is compiled in a fresh child process so compiler state and GPU
allocations cannot leak between S1..SN.  The validated 504px policy routes
S1-S7 and S9-S10 through the attention-specific MLIR path, while S8 uses the
generic autotuner because the attention-specific S8 program failed the mask
correctness gate.

The runtime loads these caches from <onnx-dir>/tracker_modules/
ort_cache_mem_attn; without them the node would compile S1..S10 on start.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


GENERIC_AUTOTUNE_SLOTS = frozenset({8})
POLICY_FILENAME = "compile_policy.json"
PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "opennav_sam3_inference"


def specific_ops_for_slot(slot: int) -> str:
    """Return the checksum-validated MLIR policy for one spatial shape."""
    return "" if slot in GENERIC_AUTOTUNE_SLOTS else "attention"


def configure_worker_environment(slot: int, environ=None) -> str:
    """Force full benchmarking and the slot-specific MLIR policy."""
    env = os.environ if environ is None else environ
    env.pop("MIGRAPHX_SKIP_BENCHMARKING", None)
    specific_ops = specific_ops_for_slot(slot)
    if specific_ops:
        env["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] = specific_ops
    else:
        env.pop("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", None)
    return specific_ops


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True,
                        help="504px artifact root, e.g. /models/onnx_files_504")
    parser.add_argument("--ptr-tokens", type=int, required=True)
    parser.add_argument("--max-spatial-slots", type=int, default=10)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the dedicated memory-attention cache before compiling.",
    )
    parser.add_argument("--worker-slot", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def graph_path(onnx_dir: Path, slot: int, ptr_tokens: int) -> Path:
    return (
        onnx_dir / "tracker_modules"
        / f"memory_attention_fixed_S{slot}_P{ptr_tokens}.onnx"
    )


def cache_path(onnx_dir: Path) -> Path:
    return onnx_dir / "tracker_modules" / "ort_cache_mem_attn"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_policy(onnx_dir: Path, ptr_tokens: int,
                    max_spatial_slots: int) -> dict:
    return {
        "schema": 1,
        "ptr_tokens": ptr_tokens,
        "max_spatial_slots": max_spatial_slots,
        "migraphx_skip_benchmarking": False,
        "specific_ops_by_slot": {
            f"S{slot}": specific_ops_for_slot(slot)
            for slot in range(1, max_spatial_slots + 1)
        },
        "onnx_sha256_by_slot": {
            f"S{slot}": sha256(graph_path(onnx_dir, slot, ptr_tokens))
            for slot in range(1, max_spatial_slots + 1)
        },
    }


def verified_existing_cache(cache: Path, expected: dict) -> bool:
    policy_path = cache / POLICY_FILENAME
    try:
        recorded = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key, value in expected.items():
        if recorded.get(key) != value:
            return False
    cache_files = recorded.get("cache_files")
    if not isinstance(cache_files, list):
        return False
    actual = {
        path.name: path.stat().st_size for path in cache.glob("*.mxr")
    }
    wanted = {
        row.get("name"): row.get("size")
        for row in cache_files if isinstance(row, dict)
    }
    return actual == wanted and len(actual) == expected["max_spatial_slots"]


def compile_worker(onnx_dir: Path, slot: int, ptr_tokens: int) -> None:
    specific_ops = configure_worker_environment(slot)

    # Import the provider only after the environment is final. MIGraphX reads
    # these variables during provider/library initialization.
    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(PACKAGE_ROOT))
    from opennav_sam3_inference.tracker.ort_plugin_ep import create_mig_session

    graph = graph_path(onnx_dir, slot, ptr_tokens)
    cache = cache_path(onnx_dir)
    if not graph.is_file():
        raise FileNotFoundError(graph)
    cache.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in cache.glob("*.mxr")}

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    started = time.perf_counter()
    # Raises if any node is not assigned to the MIGraphX EP.
    session = create_mig_session(graph, options, cache)

    dtype_for = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
    }
    inputs = {}
    for value in session.get_inputs():
        if value.type not in dtype_for:
            raise TypeError(f"unsupported input type {value.name}: {value.type}")
        try:
            shape = tuple(int(dim) for dim in value.shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"S{slot} input {value.name} is not fixed-shape: {value.shape}"
            ) from exc
        inputs[value.name] = np.zeros(shape, dtype=dtype_for[value.type])

    session.run(None, inputs)
    elapsed = time.perf_counter() - started
    del session
    gc.collect()

    after = {path.name for path in cache.glob("*.mxr")}
    created = sorted(after - before)
    if not created:
        raise RuntimeError(
            f"S{slot} did not create a new MXR in the clean build cache"
        )
    print(
        f"compiled S{slot} with full autotuning, "
        f"specific_ops={specific_ops!r}, in {elapsed:.1f}s; created={created}",
        flush=True,
    )


def compile_all(
    onnx_dir: Path,
    *,
    ptr_tokens: int,
    max_spatial_slots: int,
    force: bool,
) -> None:
    if ptr_tokens < 1:
        raise ValueError("ptr_tokens must be positive")
    if max_spatial_slots < 1:
        raise ValueError("max_spatial_slots must be positive")

    missing = [
        graph_path(onnx_dir, slot, ptr_tokens)
        for slot in range(1, max_spatial_slots + 1)
        if not graph_path(onnx_dir, slot, ptr_tokens).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing memory-attention ONNX graphs: "
            + ", ".join(str(path) for path in missing)
        )

    expected = expected_policy(onnx_dir, ptr_tokens, max_spatial_slots)
    cache = cache_path(onnx_dir)
    if cache.exists():
        if not force and verified_existing_cache(cache, expected):
            print(
                "memory-attention cache already matches the recorded "
                f"full-autotune policy: {cache}"
            )
            return
        if not force and any(cache.iterdir()):
            raise RuntimeError(
                "refusing to delete an unverified memory-attention cache "
                f"without --force: {cache}. Select a new artifact root or "
                "rerun explicitly with --force."
            )
        if force:
            print(f"resetting memory-attention cache (--force): {cache}")
            shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve()
    for slot in range(1, max_spatial_slots + 1):
        env = dict(os.environ)
        configure_worker_environment(slot, env)
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--onnx-dir", str(onnx_dir),
                "--ptr-tokens", str(ptr_tokens),
                "--max-spatial-slots", str(max_spatial_slots),
                "--worker-slot", str(slot),
            ],
            check=True,
            env=env,
        )

    caches = sorted(cache.glob("*.mxr"))
    if len(caches) != max_spatial_slots:
        raise RuntimeError(
            "memory-attention cache coverage mismatch: "
            f"expected {max_spatial_slots} MXRs, found {len(caches)} in {cache}"
        )
    policy = dict(expected)
    policy["cache_files"] = [
        {"name": path.name, "size": path.stat().st_size} for path in caches
    ]
    (cache / POLICY_FILENAME).write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"memory-attention cache complete: {len(caches)} independently "
        f"autotuned MXRs in {cache}"
    )


def main() -> int:
    args = parse_args()
    if args.worker_slot is not None:
        if not 1 <= args.worker_slot <= args.max_spatial_slots:
            raise ValueError(
                f"worker slot must be in 1..{args.max_spatial_slots}"
            )
        compile_worker(args.onnx_dir, args.worker_slot, args.ptr_tokens)
    else:
        compile_all(
            args.onnx_dir,
            ptr_tokens=args.ptr_tokens,
            max_spatial_slots=args.max_spatial_slots,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
