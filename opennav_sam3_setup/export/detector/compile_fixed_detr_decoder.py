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

"""Compile the fixed 504px DETR decoder ONNX into a direct-I/O MXR."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"skip: {args.output} already exists")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / "ort_cache"
    if cache.exists() and args.force:
        shutil.rmtree(cache)
    cache.mkdir(exist_ok=True)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=options,
        providers=[(
            "MIGraphXExecutionProvider",
            {"migraphx_fp16_enable": "1", "migraphx_model_cache_dir": str(cache)},
        )],
    )
    if session.get_providers()[0] != "MIGraphXExecutionProvider":
        raise RuntimeError(f"MIGraphX is not primary: {session.get_providers()}")

    generator = np.random.default_rng(20260831)
    feed = {}
    for item in session.get_inputs():
        value = generator.standard_normal(tuple(int(x) for x in item.shape)).astype(np.float32)
        if item.name == "text_cross_attn_mask":
            value.fill(-65504.0)
            value[..., :5] = 0.0
        feed[item.name] = value
    outputs = session.run(None, feed)
    if not all(np.isfinite(value).all() for value in outputs):
        raise RuntimeError("fixed decoder produced non-finite output")

    caches = sorted(cache.glob("*.mxr"))
    if len(caches) != 1:
        raise RuntimeError(f"expected one decoder MXR cache, found: {caches}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(caches[0], temporary)
    temporary.replace(args.output)
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )
    print(f"fixed decoder: {args.output} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
