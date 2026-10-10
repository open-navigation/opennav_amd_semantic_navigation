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

"""Export the fixed-shape SAM3 DETR decoder without loading the full model.

This is an experiment-only exporter.  It keeps the complete six decoder
layers, per-layer box refinement, per-layer BoxRPB generation, and the three
native Sam3DETRDecoder outputs.  Dynamic mask creation and ModelOutput
decorators are intentionally kept outside the ONNX graph.

The exported boundary is all FP32 so it can use the existing ORT/MIGraphX
Torch GPU-I/O convention:

  vision_features       [1, H*W, 256]
  text_features         [1, T, 256]
  vision_pos_encoding   [1, H*W, 256]
  text_cross_attn_mask  [1, 1, 1, T]

Outputs:

  intermediate_hidden_states  [6, 1, 200, 256]
  reference_boxes             [6, 1, 200, 4]
  presence_logits             [6, 1, 1]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import Sam3VideoConfig
from transformers.models.sam3.modeling_sam3 import (
    Sam3DetrDecoder,
    box_cxcywh_to_xyxy,
    create_bidirectional_mask,
    inverse_sigmoid,
)


class FixedDetrDecoder(nn.Module):
    """Tensor-only, static-spatial wrapper matching Sam3DetrDecoder.forward."""

    def __init__(self, decoder: Sam3DetrDecoder, height: int, width: int):
        super().__init__()
        self.decoder = decoder
        self.height = int(height)
        self.width = int(width)

        # Fixed grid is an ONNX initializer.  The RPB itself remains dynamic:
        # it is recomputed from each layer's refined reference boxes.
        self.register_buffer(
            "coords_h", torch.arange(self.height, dtype=torch.float32) / self.height
        )
        self.register_buffer(
            "coords_w", torch.arange(self.width, dtype=torch.float32) / self.width
        )

    def _get_rpb_matrix(self, reference_boxes: torch.Tensor) -> torch.Tensor:
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape

        deltas_y = self.coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(batch_size, num_queries, self.height, 2)
        deltas_x = self.coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(batch_size, num_queries, self.width, 2)

        deltas_x_log = deltas_x * 8
        deltas_x_log = (
            torch.sign(deltas_x_log)
            * torch.log2(torch.abs(deltas_x_log) + 1.0)
            / math.log2(8)
        )
        deltas_y_log = deltas_y * 8
        deltas_y_log = (
            torch.sign(deltas_y_log)
            * torch.log2(torch.abs(deltas_y_log) + 1.0)
            / math.log2(8)
        )

        deltas_x = self.decoder.box_rpb_embed_x(deltas_x_log)
        deltas_y = self.decoder.box_rpb_embed_y(deltas_y_log)
        rpb_matrix = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
        return rpb_matrix.flatten(2, 3).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_cross_attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = vision_features.shape[0]
        query_embeds = self.decoder.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = self.decoder.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = reference_boxes.sigmoid()
        presence_token = self.decoder.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = torch.cat([presence_token, query_embeds], dim=1)

        intermediate_outputs = []
        intermediate_boxes = []
        intermediate_presence_logits = []

        for layer in self.decoder.layers:
            query_sine_embed = self.decoder.position_encoding.encode_boxes(reference_boxes)
            query_pos = self.decoder.ref_point_head(query_sine_embed)

            rpb_matrix = self._get_rpb_matrix(reference_boxes)
            # Presence token attends uniformly to vision tokens.
            presence_rpb = torch.zeros_like(rpb_matrix[:, :, :1, :])
            vision_cross_attn_mask = torch.cat([presence_rpb, rpb_matrix], dim=2)

            hidden_states = layer(
                hidden_states,
                query_pos=query_pos,
                text_features=text_features,
                vision_features=vision_features,
                vision_pos_encoding=vision_pos_encoding,
                text_cross_attn_mask=text_cross_attn_mask,
                vision_cross_attn_mask=vision_cross_attn_mask,
            )

            query_hidden_states = hidden_states[:, 1:]
            normalized_queries = self.decoder.output_layer_norm(query_hidden_states)
            delta_boxes = self.decoder.box_head(normalized_queries)
            new_reference_boxes = (delta_boxes + inverse_sigmoid(reference_boxes)).sigmoid()

            intermediate_outputs.append(normalized_queries)
            # Native decoder returns the references used by each layer, not
            # the post-layer update.  The outer detector applies box_head once
            # more to these outputs to produce that layer's prediction.
            intermediate_boxes.append(reference_boxes)

            presence_hidden = hidden_states[:, :1]
            presence_logits = self.decoder.presence_head(
                self.decoder.presence_layer_norm(presence_hidden)
            ).squeeze(-1)
            intermediate_presence_logits.append(
                presence_logits.clamp(
                    min=-self.decoder.clamp_presence_logit_max_val,
                    max=self.decoder.clamp_presence_logit_max_val,
                )
            )
            reference_boxes = new_reference_boxes.detach()

        return (
            torch.stack(intermediate_outputs),
            torch.stack(intermediate_boxes),
            torch.stack(intermediate_presence_logits),
        )


def load_decoder(checkpoint: Path) -> Sam3DetrDecoder:
    config = Sam3VideoConfig.from_pretrained(str(checkpoint)).detector_config.detr_decoder_config
    config._attn_implementation = "eager"
    decoder = Sam3DetrDecoder(config).cpu().eval()

    weights_path = checkpoint / "model.safetensors"
    prefix = "detector_model.detr_decoder."
    with safe_open(str(weights_path), framework="pt", device="cpu") as weights:
        state = {
            name[len(prefix) :]: weights.get_tensor(name)
            for name in weights.keys()
            if name.startswith(prefix)
        }
    decoder.load_state_dict(state, strict=True)
    return decoder


def make_inputs(height: int, width: int, text_len: int):
    generator = torch.Generator(device="cpu").manual_seed(20260831)
    vision_features = torch.randn(1, height * width, 256, generator=generator)
    text_features = torch.randn(1, text_len, 256, generator=generator)
    vision_pos_encoding = torch.randn(1, height * width, 256, generator=generator)
    text_mask = torch.zeros(1, text_len, dtype=torch.bool)
    text_mask[:, :5] = True
    mask_value = torch.finfo(text_features.dtype).min
    cross_mask = torch.where(
        text_mask,
        torch.zeros_like(text_mask, dtype=text_features.dtype),
        torch.full_like(text_mask, mask_value, dtype=text_features.dtype),
    )[:, None, None, :]
    return (vision_features, text_features, vision_pos_encoding, cross_mask), text_mask


def compare_outputs(label: str, reference, candidate, atol: float) -> list[dict]:
    rows = []
    names = ["intermediate_hidden_states", "reference_boxes", "presence_logits"]
    for name, expected, actual in zip(names, reference, candidate):
        expected_np = expected.detach().cpu().numpy() if torch.is_tensor(expected) else expected
        actual_np = actual.detach().cpu().numpy() if torch.is_tensor(actual) else actual
        diff = np.abs(expected_np - actual_np)
        row = {
            "name": name,
            "shape": list(actual_np.shape),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "allclose": bool(np.allclose(expected_np, actual_np, rtol=1e-4, atol=atol)),
        }
        rows.append(row)
        print(
            f"  {label} {name}: shape={tuple(actual_np.shape)} "
            f"max_abs={row['max_abs']:.8g} mean_abs={row['mean_abs']:.8g} "
            f"allclose={row['allclose']}"
        )
    return rows


def graph_stats(model_path: Path) -> dict:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)
    initializer_bytes = sum(
        tensor.ByteSize() for tensor in model.graph.initializer
    )
    return {
        "path": str(model_path),
        "file_bytes": model_path.stat().st_size,
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_proto_bytes": initializer_bytes,
        "operator_histogram": dict(sorted(ops.items())),
        "inputs": [
            {
                "name": value.name,
                "shape": [dim.dim_value for dim in value.type.tensor_type.shape.dim],
                "elem_type": value.type.tensor_type.elem_type,
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "shape": [dim.dim_value for dim in value.type.tensor_type.shape.dim],
                "elem_type": value.type.tensor_type.elem_type,
            }
            for value in model.graph.output
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--width", type=int, default=36)
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-simplify", action="store_true")
    parser.add_argument("--skip-ort-run", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "detr_decoder_fixed_fp32.onnx"
    simplified_path = args.output_dir / "detr_decoder_fixed_simplified.onnx"
    report_path = args.output_dir / "export_report.json"

    print("Loading only Sam3DetrDecoder weights ...", flush=True)
    decoder = load_decoder(args.checkpoint)
    wrapper = FixedDetrDecoder(decoder, args.height, args.width).cpu().eval()
    inputs, text_mask = make_inputs(args.height, args.width, args.text_len)

    # Prove that the tensor-only wrapper preserves the native decoder exactly.
    print("Checking wrapper against native PyTorch decoder ...", flush=True)
    with torch.inference_mode():
        native = decoder(
            vision_features=inputs[0],
            text_features=inputs[1],
            vision_pos_encoding=inputs[2],
            text_mask=text_mask,
            spatial_shapes=torch.tensor([[args.height, args.width]], dtype=torch.long),
            return_dict=True,
        )
        wrapped = wrapper(*inputs)
    native_tuple = (
        native.intermediate_hidden_states,
        native.reference_boxes,
        native.presence_logits,
    )
    wrapper_check = compare_outputs("wrapper-vs-native", native_tuple, wrapped, atol=0.0)
    if not all(row["max_abs"] == 0.0 for row in wrapper_check):
        raise RuntimeError("Fixed wrapper is not bit-exact with native decoder")

    print(f"Exporting fixed FP32 ONNX to {raw_path} ...", flush=True)
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapper,
        inputs,
        str(raw_path),
        opset_version=args.opset,
        dynamo=False,
        input_names=[
            "vision_features",
            "text_features",
            "vision_pos_encoding",
            "text_cross_attn_mask",
        ],
        output_names=[
            "intermediate_hidden_states",
            "reference_boxes",
            "presence_logits",
        ],
        do_constant_folding=True,
    )
    export_seconds = time.perf_counter() - t0

    import onnx

    raw_model = onnx.load(str(raw_path))
    onnx.checker.check_model(raw_model, full_check=True)
    raw_stats = graph_stats(raw_path)
    print(
        f"Raw graph: {raw_stats['nodes']} nodes, "
        f"{raw_stats['file_bytes'] / 1e6:.1f} MB, export {export_seconds:.1f}s"
    )

    simplify_seconds = None
    simplify_check = None
    selected_path = raw_path
    simplified_stats = None
    if not args.skip_simplify:
        import onnxsim

        print(f"Simplifying to {simplified_path} ...", flush=True)
        t0 = time.perf_counter()
        simplified, simplify_check = onnxsim.simplify(
            raw_model,
            overwrite_input_shapes={
                "vision_features": [1, args.height * args.width, 256],
                "text_features": [1, args.text_len, 256],
                "vision_pos_encoding": [1, args.height * args.width, 256],
                "text_cross_attn_mask": [1, 1, 1, args.text_len],
            },
        )
        simplify_seconds = time.perf_counter() - t0
        if not simplify_check:
            raise RuntimeError("onnxsim numerical check failed")
        onnx.checker.check_model(simplified, full_check=True)
        onnx.save(simplified, str(simplified_path))
        selected_path = simplified_path
        simplified_stats = graph_stats(simplified_path)
        print(
            f"Simplified graph: {simplified_stats['nodes']} nodes, "
            f"{simplified_stats['file_bytes'] / 1e6:.1f} MB, "
            f"simplify {simplify_seconds:.1f}s"
        )

    ort_parse_seconds = None
    ort_run_seconds = None
    ort_check = None
    print(f"Creating CPU ORT session from {selected_path.name} ...", flush=True)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        str(selected_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    ort_parse_seconds = time.perf_counter() - t0
    print(f"ORT parsed/optimized graph in {ort_parse_seconds:.1f}s", flush=True)
    if not args.skip_ort_run:
        feed = {
            session_input.name: tensor.detach().cpu().numpy()
            for session_input, tensor in zip(session.get_inputs(), inputs)
        }
        t0 = time.perf_counter()
        ort_outputs = session.run(None, feed)
        ort_run_seconds = time.perf_counter() - t0
        ort_check = compare_outputs("ort-vs-wrapper", wrapped, ort_outputs, atol=1e-4)
        if not all(row["allclose"] for row in ort_check):
            raise RuntimeError("ORT CPU output is outside tolerance")
        print(f"ORT CPU correctness run completed in {ort_run_seconds:.1f}s", flush=True)

    report = {
        "fixed_contract": {
            "batch": 1,
            "object_queries": decoder.config.num_queries,
            "queries_including_presence": decoder.config.num_queries + 1,
            "vision_tokens": args.height * args.width,
            "vision_height": args.height,
            "vision_width": args.width,
            "text_tokens": args.text_len,
            "hidden_size": decoder.config.hidden_size,
            "layers": decoder.config.num_layers,
            "heads": decoder.config.num_attention_heads,
            "dtype": "float32",
        },
        "wrapper_vs_native": wrapper_check,
        "raw": raw_stats,
        "simplified": simplified_stats,
        "timings_seconds": {
            "export": export_seconds,
            "simplify": simplify_seconds,
            "ort_parse": ort_parse_seconds,
            "ort_cpu_run": ort_run_seconds,
        },
        "onnxsim_check": simplify_check,
        "ort_vs_wrapper": ort_check,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
