#!/usr/bin/env python3
"""Export memory_attention with FIXED spatial(7) + FIXED object-pointer slots(64).

Sam3VideoModel adds variable-length object pointers to the memory bank that
the existing box-prompt `memory_attention_fixed_N7.onnx` (which bakes
`num_object_pointer_tokens=0`) cannot consume. This script exports a
padded variant:

    memory shape: (7*HW + K, 1, 64)   K=64  →  steady state at frame 16+
    num_object_pointer_tokens=K       (excludes pointer slots from RoPE)

Output: <onnx-dir>/tracker_modules/memory_attention_fixed_S7_P64.onnx
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Script lives at <repo>/export/tracker_modules/<this>.py — go up THREE levels.
WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Re-use the wrapper class from the main tracker_modules export to keep
# the wrapper interface in one place.
from export.tracker_modules.export_tracker_modules import MemoryAttentionWrapper


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("/home/amd/project/sam3/model/sam3"))
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="Resolution root (e.g. onnx_files_1008). Defaults from --imgsz.")
    p.add_argument("--imgsz", type=int, default=1008)
    p.add_argument("--spatial-slots", type=int, default=7,
                   help="Number of spatial memory frames (default 7, matches max_obj_ptrs_in_encoder)")
    p.add_argument("--ptr-tokens", type=int, default=64,
                   help="Number of object-pointer tokens (default 64 = "
                        "max_object_pointers_in_encoder * 4). Bakes "
                        "num_object_pointer_tokens=K into the graph.")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE / f"onnx_files_{args.imgsz}"
    args.out_dir = args.onnx_dir / "tracker_modules"
    return args


class PaddedMemoryAttentionWrapper(MemoryAttentionWrapper):
    """MemoryAttentionWrapper with non-zero num_object_pointer_tokens baked in."""

    def __init__(self, memory_attention, H: int, W: int, num_ptr_tokens: int):
        super().__init__(memory_attention, H, W)
        self.num_ptr_tokens = num_ptr_tokens

    def forward(self, current_vision_features, memory, current_vis_pos_embed, memory_pos_embed):
        out = self.memory_attention(
            current_vision_features=current_vision_features,
            current_vision_position_embeddings=current_vis_pos_embed,
            memory=memory,
            memory_posision_embeddings=memory_pos_embed,
            num_object_pointer_tokens=self.num_ptr_tokens,
        )
        out = out.squeeze(1).permute(0, 2, 1).view(1, 256, self.H, self.W)
        return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    P = args.imgsz // 14
    HW = P * P
    total_mem = args.spatial_slots * HW + args.ptr_tokens
    out_name = f"memory_attention_fixed_S{args.spatial_slots}_P{args.ptr_tokens}.onnx"
    out_path = args.out_dir / out_name

    print(f"Loading Sam3VideoModel.tracker_model from {args.checkpoint} ...")
    from transformers import Sam3VideoModel, Sam3VideoConfig
    config = Sam3VideoConfig.from_pretrained(str(args.checkpoint))
    if args.imgsz != 1008:
        # Rewrite config BEFORE from_pretrained so RoPE buffers are sized correctly.
        # Same pattern as demo_text.py config-based init.
        config.image_size = args.imgsz
        config.low_res_mask_size = 4 * args.imgsz // 14
    model = Sam3VideoModel.from_pretrained(
        str(args.checkpoint), config=config, attn_implementation="eager"
    ).cpu().eval()
    trk = model.tracker_model

    wrapper = PaddedMemoryAttentionWrapper(trk.memory_attention, P, P, args.ptr_tokens).cpu().eval()

    # Dummy inputs
    cur_feat = torch.randn(HW, 1, 256)
    memory   = torch.randn(total_mem, 1, 64)
    cur_pos  = torch.randn(HW, 1, 256)
    mem_pos  = torch.randn(total_mem, 1, 64)

    print(f"\nSpatial slots: {args.spatial_slots}  HW={HW}  ptr-tokens: {args.ptr_tokens}")
    print(f"Total memory tokens: {total_mem}")

    print(f"\nExporting → {out_name} ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        # Sanity check first
        out = wrapper(cur_feat, memory, cur_pos, mem_pos)
        print(f"  test output: shape={tuple(out.shape)}")
        torch.onnx.export(
            wrapper,
            (cur_feat, memory, cur_pos, mem_pos),
            str(out_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["current_vision_features", "memory",
                         "current_vis_pos_embed", "memory_pos_embed"],
            output_names=["conditioned_features"],
        )
    sz = out_path.stat().st_size / 1e6
    print(f"  saved: {out_path.name} ({sz:.0f} MB) in {time.perf_counter()-t0:.1f}s")

    print("\nDone. Next:")
    print(f"  - Wire MIGMemoryAttention shim that uses ORT MIG EP on {out_name}")
    print(f"  - Pad pointer tokens to {args.ptr_tokens} at runtime; fall back to PT for early frames")


if __name__ == "__main__":
    main()
