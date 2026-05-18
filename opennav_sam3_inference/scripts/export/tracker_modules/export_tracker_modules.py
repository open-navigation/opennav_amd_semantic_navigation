#!/usr/bin/env python3
"""Export SAM3 video tracking modules to ONNX.

Exports 4 ONNX models enabling mask-level video tracking on AMD MIGraphX:
  mask_decoder_init.onnx       - First frame: prompt_encoder + mask_decoder (box or point prompt)
  mask_decoder_propagate.onnx  - Subsequent frames: mask_decoder with empty prompts
  memory_encoder.onnx          - Encode mask + vision features into memory entry
  memory_attention.onnx        - Fuse current frame features with memory bank

Key design decisions:
  - RoPE in memory_attention uses cos/sin (real-valued), no view_as_complex → ONNX-compatible
  - memory bank length is dynamic (up to 7 spatial frames × 5184 tokens each)
  - num_object_pointer_tokens=0 for this version (object pointers appended to memory but RoPE applied)


Usage:
    PYTHONPATH not required (transformers>=5.8.0 includes sam3_tracker_video)\
        python export/tracker_modules/export_tracker_modules.py \\
        [--onnx-dir onnx_files_1008] [--imgsz 1008] [--opset 17] [--verify]
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Script lives at <repo>/export/tracker_modules/<this>.py — go up THREE levels.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_HF_MODEL  = WORKSPACE_ROOT / "model" / "sam3"

os.environ["TRANSFORMERS_OFFLINE"] = "1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="Resolution root (e.g. onnx_files_504). Outputs go to "
                        "<onnx-dir>/tracker_modules/. Defaults based on --imgsz.")
    p.add_argument("--imgsz", type=int, default=1008)
    p.add_argument("--checkpoint", type=Path, default=LOCAL_HF_MODEL)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify", action="store_true", help="Run CPU accuracy test after export")
    p.add_argument("--fixed-slots", type=int, default=7, metavar="N",
                   help="Also export memory_attention_fixed_N<N>.onnx with static shapes for MIGraphX (0=skip)")
    args = p.parse_args()
    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE_ROOT / f"onnx_files_{args.imgsz}"
    args.output_dir = args.onnx_dir / "tracker_modules"
    return args


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class MaskDecoderInitWrapper(nn.Module):
    """First-frame decoder: prompt_encoder + mask_decoder.

    Bakes in:
      - image_wide_positional_embeddings (constant, from model)
      - no_memory_embedding added to fpn_2 (no prior memory on first frame)

    Supports box and/or point prompts:
      - For box: pass input_boxes (1,1,4) = [x1,y1,x2,y2], leave input_points=None
      - For points: pass input_points (1,1,N,2) + input_labels (1,1,N)
      - For both: pass all three

    Inputs:
        fpn_2            (1, 256, H, W)  backbone coarse feature map
        fpn_0            (1, 256, 4H, 4W) backbone high-res level 0 (feat_s0)
        fpn_1            (1, 256, 2H, 2W) backbone high-res level 1 (feat_s1)
        input_points     (1, 1, N, 2)   point coordinates in pixel space, or zeros if unused
        input_labels     (1, 1, N)      point labels (1=fg, 0=bg), or empty
        input_boxes      (1, 1, 4)      box [x1,y1,x2,y2] in pixel space, or zeros if unused
        has_points       scalar bool tensor — whether input_points/labels are valid
        has_boxes        scalar bool tensor — whether input_boxes is valid

    Outputs:
        pred_masks       (1, 1, imgsz, imgsz)  logits (threshold at 0 for binary)
        object_pointer   (1, 1, 256)
        object_score_logits (1, 1, 1)
    """

    def __init__(self, model, image_pe: torch.Tensor, H: int, W: int, image_size: int):
        super().__init__()
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder   = model.mask_decoder
        self.image_size = image_size
        self.H, self.W  = H, W
        # bake in constant tensors
        self.register_buffer("image_pe", image_pe)
        # no_memory_embedding is (1,1,256) seq-first; reshape to (1,256,1,1) for BCHW addition
        no_mem = model.no_memory_embedding.detach().view(1, -1, 1, 1)
        self.register_buffer("no_memory_embedding", no_mem)
        self.register_buffer("no_object_pointer", model.no_object_pointer.detach())
        self.object_pointer_proj = model.object_pointer_proj
        # conv_s0/s1: project backbone high-res features to decoder channel counts
        # conv_s0: 256 → hidden_size//8 = 32  (feat_s0, 288×288)
        # conv_s1: 256 → hidden_size//4 = 64  (feat_s1, 144×144)
        self.conv_s0 = model.mask_decoder.conv_s0
        self.conv_s1 = model.mask_decoder.conv_s1

    def forward(
        self,
        fpn_2: torch.Tensor,       # (1, 256, H, W)
        fpn_0: torch.Tensor,       # (1, 256, 4H, 4W)
        fpn_1: torch.Tensor,       # (1, 256, 2H, 2W)
        input_points: torch.Tensor,  # (1, 1, N, 2) — coords in pixel space
        input_labels: torch.Tensor,  # (1, 1, N)
        input_boxes: torch.Tensor,   # (1, 1, 4) — [x1,y1,x2,y2]
    ):
        B = fpn_2.shape[0]

        # Add no_memory_embedding: signals "no prior memory" on first frame
        image_embed = fpn_2 + self.no_memory_embedding

        # Project high-res backbone features to decoder channel counts (done by model before mask_decoder call)
        feat_s0 = self.conv_s0(fpn_0)  # (1, 256, 4H, 4W) → (1, 32, 4H, 4W)
        feat_s1 = self.conv_s1(fpn_1)  # (1, 256, 2H, 2W) → (1, 64, 2H, 2W)

        # Prompt encoding
        sparse_emb_pts,  dense_emb = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=None,
        )

        image_pe = self.image_pe.expand(B, -1, -1, -1)

        # Mask decoder
        low_res, iou_scores, sam_output_tokens, obj_score_logits = self.mask_decoder(
            image_embeddings=image_embed,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_emb_pts,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            high_resolution_features=[feat_s0, feat_s1],
        )

        # Upsample to full resolution: (B, n_masks=1, H, W)
        pred_masks = F.interpolate(
            low_res.squeeze(1).float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, imgsz, imgsz)

        # Object pointer
        is_obj = (obj_score_logits > 0).float()
        sam_token = sam_output_tokens[:, :, 0]
        obj_ptr = self.object_pointer_proj(sam_token)
        obj_ptr = is_obj * obj_ptr + (1 - is_obj) * self.no_object_pointer

        return pred_masks, obj_ptr, obj_score_logits


class MaskDecoderPropWrapper(nn.Module):
    """Propagation decoder: mask_decoder with empty prompts.

    Input fpn_2_cond should be the output of memory_attention (conditioned features).

    Inputs:
        fpn_2_cond   (1, 256, H, W)   conditioned image features (from memory_attention)
        fpn_0        (1, 256, 4H, 4W)
        fpn_1        (1, 256, 2H, 2W)

    Outputs:
        pred_masks       (1, 1, imgsz, imgsz)
        object_pointer   (1, 1, 256)
        object_score_logits (1, 1, 1)
    """

    def __init__(self, model, image_pe: torch.Tensor, H: int, W: int, image_size: int):
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.prompt_encoder = model.prompt_encoder
        self.image_size = image_size
        self.H, self.W  = H, W
        self.register_buffer("image_pe", image_pe)
        self.register_buffer("no_object_pointer", model.no_object_pointer.detach())
        self.object_pointer_proj = model.object_pointer_proj
        self.conv_s0 = model.mask_decoder.conv_s0
        self.conv_s1 = model.mask_decoder.conv_s1

    def forward(
        self,
        fpn_2_cond: torch.Tensor,  # (1, 256, H, W) — output of memory_attention
        fpn_0: torch.Tensor,       # (1, 256, 4H, 4W) raw backbone feature
        fpn_1: torch.Tensor,       # (1, 256, 2H, 2W) raw backbone feature
    ):
        B = fpn_2_cond.shape[0]
        image_pe = self.image_pe.expand(B, -1, -1, -1)

        feat_s0 = self.conv_s0(fpn_0)  # → (1, 32, 4H, 4W)
        feat_s1 = self.conv_s1(fpn_1)  # → (1, 64, 2H, 2W)

        # Empty prompts: no points, no boxes, no masks
        sparse_emb, dense_emb = self.prompt_encoder(
            input_points=None,
            input_labels=None,
            input_boxes=None,
            input_masks=None,
        )
        # sparse_emb is None when no prompts — create empty tensor
        if sparse_emb is None:
            sparse_emb = torch.zeros(B, 1, 0, self.mask_decoder.hidden_size,
                                     dtype=fpn_2_cond.dtype, device=fpn_2_cond.device)

        low_res, iou_scores, sam_output_tokens, obj_score_logits = self.mask_decoder(
            image_embeddings=fpn_2_cond,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            high_resolution_features=[feat_s0, feat_s1],
        )

        # (B, n_masks=1, imgsz, imgsz) — no unsqueeze, low_res.squeeze(1) already removes point_batch dim
        pred_masks = F.interpolate(
            low_res.squeeze(1).float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        is_obj = (obj_score_logits > 0).float()
        sam_token = sam_output_tokens[:, :, 0]
        obj_ptr = self.object_pointer_proj(sam_token)
        obj_ptr = is_obj * obj_ptr + (1 - is_obj) * self.no_object_pointer

        return pred_masks, obj_ptr, obj_score_logits


class MemoryEncoderWrapper(nn.Module):
    """Memory encoder: encodes a mask + vision features into a memory entry.

    Caller must resize mask to (1, 1, H*16, W*16) before passing.
    For 1008px (H=W=72): mask size = 1152×1152.

    Inputs:
        vision_features   (1, 256, H, W)   — backbone_fpn_2 (raw, no no_memory_embedding)
        masks             (1, 1, H*16, W*16) — pred_mask resized to memory size

    Outputs:
        maskmem_features  (H*W, 1, 64)   — spatial memory features, flattened seq-first
        maskmem_pos_enc   (H*W, 1, 64)   — positional encoding, flattened seq-first
    """

    def __init__(self, memory_encoder: nn.Module):
        super().__init__()
        self.memory_encoder = memory_encoder

    def forward(self, vision_features: torch.Tensor, masks: torch.Tensor):
        # Apply sigmoid + scale/bias (as in the original _encode_new_memory)
        mask_for_mem = torch.sigmoid(masks)
        # Default scale=20.0, bias=-10.0 from SAM3 config
        mask_for_mem = mask_for_mem * 20.0 - 10.0

        mf, mp = self.memory_encoder(vision_features, mask_for_mem)

        # Flatten from (B, C, H, W) to (H*W, B, C) — seq-first for memory_attention
        mf = mf.flatten(2).permute(2, 0, 1)
        mp = mp.flatten(2).permute(2, 0, 1)

        return mf, mp


class MemoryAttentionWrapper(nn.Module):
    """Memory attention: conditions current frame features on memory bank.

    Inputs:
        current_vision_features  (H*W, 1, 256)  — fpn_2 flattened (seq-first)
        memory                   (N,   1, 64)   — concatenated memory bank (dynamic N)
        current_vis_pos_embed    (H*W, 1, 256)  — vision_pos_enc_2 flattened (seq-first)
        memory_pos_embed         (N,   1, 64)   — spatial+temporal PE (assembled by host)

    Output:
        conditioned_features     (1, 256, H, W) — reshaped output for mask_decoder
    """

    def __init__(self, memory_attention: nn.Module, H: int, W: int):
        super().__init__()
        self.memory_attention = memory_attention
        self.H, self.W = H, W

    def forward(
        self,
        current_vision_features: torch.Tensor,  # (HW, 1, 256)
        memory: torch.Tensor,                   # (N, 1, 64) — dynamic
        current_vis_pos_embed: torch.Tensor,    # (HW, 1, 256)
        memory_pos_embed: torch.Tensor,         # (N, 1, 64)
    ):
        out = self.memory_attention(
            current_vision_features=current_vision_features,
            current_vision_position_embeddings=current_vis_pos_embed,
            memory=memory,
            memory_posision_embeddings=memory_pos_embed,
            num_object_pointer_tokens=0,
        )
        # out shape: (1, 1, HW, 256) — squeeze batch dims and reshape to BCHW
        B = 1
        out = out.squeeze(1).permute(0, 2, 1).view(B, 256, self.H, self.W)
        return out


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def get_image_pe(model, H: int, W: int) -> torch.Tensor:
    """Precompute the image-wide positional embedding (constant per image size)."""
    with torch.no_grad():
        pe = model.get_image_wide_positional_embeddings()
    return pe.detach()  # (1, 256, H, W)


def export_mask_decoder_init(model, args, image_pe, H, W, output_path: Path) -> None:
    wrapper = MaskDecoderInitWrapper(model, image_pe, H, W, args.imgsz).cpu().eval()

    # Dummy inputs — box prompt (truck box from truck.jpg, pixel coords at 1008px)
    dummy_fpn2  = torch.randn(1, 256, H, W)
    dummy_fpn0  = torch.randn(1, 256, H*4, W*4)
    dummy_fpn1  = torch.randn(1, 256, H*2, W*2)
    # Box: x1,y1,x2,y2 = 85,281,1710,850  → shape (1,1,4)
    dummy_box   = torch.tensor([[[[85., 281., 1710., 850.]]]])   # (1,1,4) not used in export shape
    # Points: one foreground click (N=1) + one box gives N=3 total (2 box corners + 1 point)
    dummy_pts   = torch.randn(1, 1, 1, 2)  # N=1 point
    dummy_lbls  = torch.ones(1, 1, 1, dtype=torch.int32)

    with torch.no_grad():
        out = wrapper(dummy_fpn2, dummy_fpn0, dummy_fpn1, dummy_pts, dummy_lbls, dummy_box)
    print(f"  MaskDecoderInit forward: masks={out[0].shape} ptr={out[1].shape} score={out[2].shape}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_fpn2, dummy_fpn0, dummy_fpn1, dummy_pts, dummy_lbls, dummy_box),
            str(output_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["fpn_2", "fpn_0", "fpn_1", "input_points", "input_labels", "input_boxes"],
            output_names=["pred_masks", "object_pointer", "object_score_logits"],
            dynamic_axes={
                "input_points": {2: "num_points"},
                "input_labels": {2: "num_points"},
            },
        )
    print(f"  Saved: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_mask_decoder_propagate(model, args, image_pe, H, W, output_path: Path) -> None:
    wrapper = MaskDecoderPropWrapper(model, image_pe, H, W, args.imgsz).cpu().eval()

    dummy_fpn2_cond = torch.randn(1, 256, H, W)
    dummy_fpn0      = torch.randn(1, 256, H*4, W*4)
    dummy_fpn1      = torch.randn(1, 256, H*2, W*2)

    with torch.no_grad():
        out = wrapper(dummy_fpn2_cond, dummy_fpn0, dummy_fpn1)
    print(f"  MaskDecoderProp forward: masks={out[0].shape} ptr={out[1].shape}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_fpn2_cond, dummy_fpn0, dummy_fpn1),
            str(output_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["fpn_2_cond", "fpn_0", "fpn_1"],
            output_names=["pred_masks", "object_pointer", "object_score_logits"],
        )
    print(f"  Saved: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_memory_encoder(model, args, H, W, output_path: Path) -> None:
    wrapper = MemoryEncoderWrapper(model.memory_encoder).cpu().eval()

    dummy_vis  = torch.randn(1, 256, H, W)
    dummy_mask = torch.randn(1, 1, H*16, W*16)

    with torch.no_grad():
        mf, mp = wrapper(dummy_vis, dummy_mask)
    print(f"  MemoryEncoder forward: mf={mf.shape} mp={mp.shape}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_vis, dummy_mask),
            str(output_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["vision_features", "masks"],
            output_names=["maskmem_features", "maskmem_pos_enc"],
        )
    print(f"  Saved: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_memory_attention(model, args, H, W, output_path: Path) -> None:
    wrapper = MemoryAttentionWrapper(model.memory_attention, H, W).cpu().eval()

    HW   = H * W
    N    = 3 * HW  # dummy: 3 memory frames × HW tokens each
    dummy_cur_feat  = torch.randn(HW, 1, 256)
    dummy_memory    = torch.randn(N, 1, 64)
    dummy_cur_pos   = torch.randn(HW, 1, 256)
    dummy_mem_pos   = torch.randn(N, 1, 64)

    with torch.no_grad():
        cond = wrapper(dummy_cur_feat, dummy_memory, dummy_cur_pos, dummy_mem_pos)
    print(f"  MemoryAttention forward: conditioned={cond.shape}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_cur_feat, dummy_memory, dummy_cur_pos, dummy_mem_pos),
            str(output_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["current_vision_features", "memory",
                         "current_vis_pos_embed", "memory_pos_embed"],
            output_names=["conditioned_features"],
            dynamic_axes={
                "memory":          {0: "num_memory_tokens"},
                "memory_pos_embed":{0: "num_memory_tokens"},
            },
        )
    print(f"  Saved: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Memory attention — fixed-size ONNX (for MIGraphX)
# ---------------------------------------------------------------------------

def export_memory_attention_fixed(model, args, H: int, W: int, num_slots: int, output_path: Path) -> None:
    wrapper = MemoryAttentionWrapper(model.memory_attention, H, W).cpu().eval()

    HW  = H * W
    N   = num_slots * HW  # fixed memory bank: num_maskmem frames x HW tokens
    dummy_cur_feat = torch.randn(HW, 1, 256)
    dummy_memory   = torch.randn(N, 1, 64)
    dummy_cur_pos  = torch.randn(HW, 1, 256)
    dummy_mem_pos  = torch.randn(N, 1, 64)

    with torch.no_grad():
        cond = wrapper(dummy_cur_feat, dummy_memory, dummy_cur_pos, dummy_mem_pos)
    print(f"  MemoryAttention forward (fixed N={num_slots}): conditioned={cond.shape}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_cur_feat, dummy_memory, dummy_cur_pos, dummy_mem_pos),
            str(output_path),
            opset_version=args.opset,
            dynamo=False,
            input_names=["current_vision_features", "memory",
                         "current_vis_pos_embed", "memory_pos_embed"],
            output_names=["conditioned_features"],
        )
    print(f"  Saved: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(args, output_dir: Path) -> None:
    import onnxruntime as ort
    from PIL import Image
    import torchvision.transforms.v2 as T

    os.environ.setdefault("MIGRAPHX_SKIP_BENCHMARKING", "1")

    CPU = ["CPUExecutionProvider"]

    def load(name):
        return ort.InferenceSession(str(output_dir / name), providers=CPU)

    # --- memory_encoder ---
    enc_sess = load("memory_encoder.onnx")
    inp_names = [i.name for i in enc_sess.get_inputs()]
    H = W = args.imgsz // 14
    rng = np.random.default_rng(0)
    vis_np  = rng.standard_normal((1, 256, H, W)).astype(np.float32)
    mask_np = rng.standard_normal((1, 1, H*16, W*16)).astype(np.float32)
    mf, mp = enc_sess.run(None, {"vision_features": vis_np, "masks": mask_np})
    print(f"  [verify] memory_encoder: mf={mf.shape} mp={mp.shape}")

    # --- memory_attention ---
    attn_sess = load("memory_attention.onnx")
    HW = H * W
    N  = 2 * HW
    cur_feat = rng.standard_normal((HW, 1, 256)).astype(np.float32)
    memory   = rng.standard_normal((N, 1, 64)).astype(np.float32)
    cur_pos  = rng.standard_normal((HW, 1, 256)).astype(np.float32)
    mem_pos  = rng.standard_normal((N, 1, 64)).astype(np.float32)
    cond = attn_sess.run(None, {
        "current_vision_features": cur_feat, "memory": memory,
        "current_vis_pos_embed": cur_pos, "memory_pos_embed": mem_pos,
    })[0]
    print(f"  [verify] memory_attention: conditioned={cond.shape}")

    # --- mask_decoder_propagate ---
    prop_sess = load("mask_decoder_propagate.onnx")
    fpn2 = rng.standard_normal((1, 256, H, W)).astype(np.float32)
    fpn0 = rng.standard_normal((1, 256, H*4, W*4)).astype(np.float32)
    fpn1 = rng.standard_normal((1, 256, H*2, W*2)).astype(np.float32)
    masks, ptr, score = prop_sess.run(None, {"fpn_2_cond": fpn2, "fpn_0": fpn0, "fpn_1": fpn1})
    print(f"  [verify] mask_decoder_prop: masks={masks.shape} ptr={ptr.shape}")

    print("  ✅ All ONNX models loaded and ran successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def retarget_resolution(model, new_imgsz: int) -> None:
    """Re-initialize all RoPE buffers for a different input resolution."""
    from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoVisionRotaryEmbedding

    new_H = new_imgsz // 14
    model.config.image_size = new_imgsz
    model.config.memory_attention_rope_feat_sizes = [new_H, new_H]
    model.image_size = new_imgsz

    # Update prompt encoder sizes (affects dense_prompt_embeddings and image_pe shape)
    pe = model.prompt_encoder
    pe.image_embedding_size = (new_H, new_H)
    pe.mask_input_size = (4 * new_H, 4 * new_H)
    pe.input_image_size = new_imgsz

    # Shared image embedding (for get_image_wide_positional_embeddings) - update size reference
    # Sam3TrackerVideoPositionalEmbedding uses prompt_encoder.image_embedding_size via config
    model.config.vision_config.backbone_feature_sizes = [(new_H * 4, new_H * 4),
                                                          (new_H * 2, new_H * 2),
                                                          (new_H, new_H)]

    n = 0
    for _, mod in model.named_modules():
        dev = getattr(mod, 'rope_embeddings_cos', torch.tensor(0)).device
        dtype = getattr(mod, 'rope_embeddings_cos', torch.tensor(0.0)).dtype
        if isinstance(mod, Sam3ViTRotaryEmbedding) and mod.end_x > new_H:
            mod.end_x = mod.end_y = new_H
            freqs = 1.0 / (mod.rope_theta ** (torch.arange(0, mod.dim, 4)[:mod.dim // 4].float() / mod.dim))
            flat = torch.arange(new_H * new_H, dtype=torch.long)
            xp = (flat % new_H).float() * mod.scale
            yp = torch.div(flat, new_H, rounding_mode="floor").float() * mod.scale
            inv = torch.cat([torch.outer(xp, freqs), torch.outer(yp, freqs)], dim=-1).repeat_interleave(2, dim=-1)
            mod.register_buffer("rope_embeddings_cos", inv.cos().to(dev, dtype), persistent=False)
            mod.register_buffer("rope_embeddings_sin", inv.sin().to(dev, dtype), persistent=False)
            n += 1
        elif isinstance(mod, Sam3TrackerVideoVisionRotaryEmbedding):
            mod.end_x = mod.end_y = new_H
            inv = mod.create_inv_freq()
            mod.register_buffer("rope_embeddings_cos", inv.cos().to(dev, dtype), persistent=False)
            mod.register_buffer("rope_embeddings_sin", inv.sin().to(dev, dtype), persistent=False)
            n += 1
    print(f"  Retargeted {n} RoPE modules → {new_imgsz}px ({new_H}×{new_H}={new_H**2} tokens)")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoModel

    print(f"Loading Sam3TrackerVideoModel from {args.checkpoint} ...")
    model = Sam3TrackerVideoModel.from_pretrained(
        str(args.checkpoint), attn_implementation="eager"
    ).cpu().eval()

    H = W = args.imgsz // 14   # patch_size=14
    print(f"  Image size: {args.imgsz}px  Feature map: {H}×{W}={H*W} tokens")

    if args.imgsz != 1008:
        retarget_resolution(model, args.imgsz)

    image_pe = get_image_pe(model, H, W)
    print(f"  image_pe: {image_pe.shape}")

    temporal_pe_path = args.output_dir / "temporal_pe.npy"
    temporal_pe = model.memory_temporal_positional_encoding.detach().cpu().numpy()
    np.save(temporal_pe_path, temporal_pe)
    print(f"  temporal_pe: {temporal_pe.shape} → {temporal_pe_path.name}")

    paths = {
        "mask_decoder_init":       args.output_dir / "mask_decoder_init.onnx",
        "mask_decoder_propagate":  args.output_dir / "mask_decoder_propagate.onnx",
        "memory_encoder":          args.output_dir / "memory_encoder.onnx",
        "memory_attention":        args.output_dir / "memory_attention.onnx",
        "temporal_pe":             temporal_pe_path,
    }

    print("\n[1/4] Exporting mask_decoder_init ...")
    export_mask_decoder_init(model, args, image_pe, H, W, paths["mask_decoder_init"])

    print("\n[2/4] Exporting mask_decoder_propagate ...")
    export_mask_decoder_propagate(model, args, image_pe, H, W, paths["mask_decoder_propagate"])

    print("\n[3/4] Exporting memory_encoder ...")
    export_memory_encoder(model, args, H, W, paths["memory_encoder"])

    print("\n[4/4] Exporting memory_attention ...")
    export_memory_attention(model, args, H, W, paths["memory_attention"])

    if args.fixed_slots > 0:
        fixed_name = f"memory_attention_fixed_N{args.fixed_slots}.onnx"
        paths["memory_attention_fixed"] = args.output_dir / fixed_name
        print(f"\n[4b] Exporting {fixed_name} (fixed shapes for MIGraphX) ...")
        export_memory_attention_fixed(model, args, H, W, args.fixed_slots, paths["memory_attention_fixed"])

    print(f"\nExported to {args.output_dir}:")
    for name, p in paths.items():
        exists = "✅" if p.exists() else "❌ MISSING"
        size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "-"
        print(f"  {exists}  {p.name}  ({size})")

    if args.verify:
        print("\n[verify] Running quick shape-only test ...")
        verify(args, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
