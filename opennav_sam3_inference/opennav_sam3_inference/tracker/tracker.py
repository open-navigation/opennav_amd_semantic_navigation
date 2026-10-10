"""
SAM3OnnxTracker — mask-level video tracker using SAM3 + ONNX Runtime.

Pipeline:
  Frame 0 (init):  box/point prompt → mask_decoder_init.onnx → mask + object pointer
  Frame t (prop):  memory_attention.onnx (MIGraphX) → conditioned features
                   mask_decoder_propagate.onnx → mask + object pointer
  Every frame:     memory_encoder.onnx → memory entry → FIFO bank (max 7 frames)

Backbone: MIGraphX compiled (backbone_<source>/tuned.mxr, ~88ms at 504px) when available,
          falling back to PyTorch ROCm FP16 (139ms). Pass backbone="pytorch" to force
          PyTorch. All other modules are CPU ONNX, except memory_attention which runs
          on MIGraphX when the fixed-N7 file is present.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .migraphx_runtime import (  # noqa: F401
    MemoryBank,
    MIGraphXBackbone,
    MIGraphXSession,
    preprocess_image,
    retarget_resolution,
)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SharedTrackerResources:
    """Heavy-weight tracker resources loaded ONCE, shared across many
    SAM3OnnxTracker instances.

    Use case: hybrid pipeline runs N per-object trackers per frame; each one
    needs its own MemoryBank but they can share backbone + ORT modules.

    Pass an instance to ``SAM3OnnxTracker(shared=resources)`` to skip the
    heavy module loading and just allocate per-object state.
    """

    def __init__(
        self,
        onnx_dir: str | Path,
        imgsz: int = 504,
        num_maskmem: int = 7,
        backbone: str = "auto",
    ):
        # Build a temporary SAM3OnnxTracker (with checkpoint=None — backbone="auto"
        # will go through MIGraphX since onnx_files_504/backbone_tracker/tuned.mxr
        # exists). Then steal its module references.
        owner = SAM3OnnxTracker(
            checkpoint=None,
            onnx_dir=onnx_dir,
            imgsz=imgsz,
            num_maskmem=num_maskmem,
            backbone=backbone,
        )
        # Shared (read-only from per-tracker POV) refs
        self.imgsz            = owner.imgsz
        self.H                = owner.H
        self.W                = owner.W
        self.HW               = owner.HW
        self.mask_mem_size    = owner.mask_mem_size
        self.num_maskmem      = num_maskmem
        self._mxr_backbone    = owner._mxr_backbone
        self.vision_encoder   = owner.vision_encoder
        self.device           = owner.device
        self.dec_init         = owner.dec_init
        self.dec_prop         = owner.dec_prop
        self.mem_enc          = owner.mem_enc
        self.mem_attn         = owner.mem_attn
        self._mem_attn_slots  = owner._mem_attn_slots
        # temporal_pe came from the owner's memory_bank
        self._temporal_pe     = owner.memory_bank.temporal_pe

    def run_backbone(self, img_np):
        """Shared backbone call: (1, 3, H, W) float32 → (fpn_0, fpn_1, fpn_2, pos_2)."""
        if self._mxr_backbone is not None:
            return self._mxr_backbone(img_np)
        import torch
        pv = torch.from_numpy(img_np).to(self.device).half()
        with torch.inference_mode():
            vis = self.vision_encoder(pv, return_dict=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return (
            vis.fpn_hidden_states[0].float().cpu().numpy(),
            vis.fpn_hidden_states[1].float().cpu().numpy(),
            vis.fpn_hidden_states[2].float().cpu().numpy(),
            (vis.fpn_position_encoding[2].float().cpu().numpy()
             if vis.fpn_position_encoding is not None else None),
        )


class SAM3OnnxTracker:
    """
    SAM3 mask-level video tracker using ONNX tracking modules.

    Args:
        checkpoint:   Path to facebook/sam3 model directory.
        onnx_dir:     Directory containing the exported ONNX files.
        imgsz:        Input resolution (default 504; use 1008 for higher quality).
        num_maskmem:  Memory bank size (default 7).
        backbone:     "auto" (default) — use MIGraphX if cache/onnx present, else PyTorch.
                      "migraphx"       — require MIGraphX backbone (88ms at 504px).
                      "pytorch"        — require PyTorch ROCm backbone (139ms at 504px).
    """

    def __init__(
        self,
        checkpoint: str | Path,
        onnx_dir: str | Path,
        imgsz: int = 504,
        num_maskmem: int = 7,
        backbone: str = "auto",
        shared: "SharedTrackerResources | None" = None,
    ):
        # Shared-resources fast path: skip all heavy module loads, just copy
        # refs + allocate own MemoryBank. Used by the hybrid pipeline.
        if shared is not None:
            self.imgsz            = shared.imgsz
            self.H                = shared.H
            self.W                = shared.W
            self.HW               = shared.HW
            self.num_maskmem      = num_maskmem
            self.mask_mem_size    = shared.mask_mem_size
            self._mxr_backbone    = shared._mxr_backbone
            self.vision_encoder   = shared.vision_encoder
            self.device           = shared.device
            self.dec_init         = shared.dec_init
            self.dec_prop         = shared.dec_prop
            self.mem_enc          = shared.mem_enc
            self.mem_attn         = shared.mem_attn
            self._mem_attn_slots  = shared._mem_attn_slots
            self.memory_bank      = MemoryBank(shared._temporal_pe, max_slots=num_maskmem)
            self._frame_idx       = 0
            self._timings: dict[str, list] = {
                k: [] for k in ["backbone", "mem_attn", "dec_init", "dec_prop", "mem_enc", "total"]
            }
            return

        import onnxruntime as ort

        self.imgsz         = imgsz
        self.H = self.W    = imgsz // 14
        self.HW            = self.H * self.W
        self.num_maskmem   = num_maskmem
        self.mask_mem_size = self.H * 16
        # onnx_dir is the resolution root (onnx_files_504 / onnx_files_1008).
        # Files live in fixed subdirectories:
        #   <onnx_dir>/backbone_tracker/  — backbone .onnx + .mxr (tracker FPN weights)
        #   <onnx_dir>/tracker_modules/   — mask_decoder_*, memory_*, temporal_pe, caches
        onnx_dir = Path(onnx_dir)
        backbone_dir = onnx_dir / "backbone_tracker"
        modules_dir  = onnx_dir / "tracker_modules"

        _mxr_cache  = backbone_dir / "tuned.mxr"
        _mxr_onnx   = backbone_dir / "single_simplified.onnx"
        _temporal_pe = modules_dir / "temporal_pe.npy"
        _use_mxr    = (backbone == "migraphx") or (
            backbone == "auto" and (_mxr_cache.exists() or _mxr_onnx.exists())
        )

        if _use_mxr:
            # MIGraphX backbone path: no torch dependency whatsoever.
            # temporal_pe is loaded from a pre-saved .npy file (generated once during export).
            if not _temporal_pe.exists():
                raise FileNotFoundError(
                    f"{_temporal_pe} not found. Run export/tracker_modules/export_tracker_modules.py "
                    f"once to generate it, or copy it from another onnx_files_<RES>/tracker_modules/ "
                    f"directory."
                )
            temporal_pe = np.load(str(_temporal_pe))

            self._mxr_backbone  = MIGraphXBackbone(_mxr_onnx, _mxr_cache)
            self.vision_encoder = None
            self.device         = None

        else:
            # ---- Backbone: PyTorch ROCm GPU FP16 ----
            import torch
            from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
                Sam3TrackerVideoModel,
            )

            self._mxr_backbone = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading SAM3 from {checkpoint} (device={self.device}) ...")
            model = Sam3TrackerVideoModel.from_pretrained(
                str(checkpoint), attn_implementation="eager"
            ).to(self.device).half().eval()

            if imgsz != 1008:
                retarget_resolution(model, imgsz)

            self.vision_encoder = model.vision_encoder
            temporal_pe = model.memory_temporal_positional_encoding.detach().cpu().numpy()

            # ---- AMD TunableOp: GEMM kernel autotuner ----
            _tunableop = False
            if self.device.type == "cuda":
                try:
                    torch.cuda.tunable.enable(val=True)
                    torch.cuda.tunable.tuning_enable(val=True)
                    _tunableop = True
                except Exception:
                    pass

            # ---- Warmup (also triggers TunableOp tuning) ----
            print("  Warming up GPU kernels ...")
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=self.device, dtype=torch.float16)
            n_warmup = 8 if _tunableop else 1
            with torch.inference_mode():
                for _ in range(n_warmup):
                    self.vision_encoder(dummy, return_dict=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                if _tunableop:
                    torch.cuda.tunable.tuning_enable(val=False)
                    print("  TunableOp tuning complete.")
            print("  Warmup complete.")

        # ---- ORT session options ----
        cpu_opts = ort.SessionOptions()
        cpu_opts.intra_op_num_threads = 8
        cpu_opts.inter_op_num_threads = 1
        cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        CPU = ["CPUExecutionProvider"]

        # MIGraphX provider helpers with persistent compile cache.
        # Cache key includes model hash + fp16 flag, so FP16 and FP32 variants are stored
        # separately. Run export/tracker_modules/prewarm_ort_cache.py once per onnx_dir to populate.
        _cache = str(modules_dir / "mxr_cache")
        def _mig(fp16: bool = False) -> list:
            opts: dict = {"migraphx_model_cache_dir": _cache}
            if fp16:
                opts["migraphx_fp16_enable"] = "1"
            return [("MIGraphXExecutionProvider", opts), ("CPUExecutionProvider", {})]

        MIG     = _mig(fp16=False)
        MIG_FP16 = _mig(fp16=True)

        # FP16 memory_attention is 2.76× faster and numerically safe (max_diff=0.012).
        # With a pre-compiled cache, FP16 is used at ALL resolutions (no OOM risk:
        # cache loading does not trigger GPU compilation workspace allocation).
        # Without a cache, FP16 at 1008px OOMs with backbone in memory → fall back to FP32.
        _cache_exists = any(Path(_cache).glob("*")) if Path(_cache).exists() else False
        _attn_prov = MIG_FP16 if (_cache_exists or self.HW <= 2000) else MIG

        # ---- Load ONNX modules via direct migraphx Python API ----
        # Direct API eliminates ORT EP's ReorderInput/Output CPU overhead (NCHW↔NCHWc),
        # which dominated latency for all these modules:
        #   dec_prop:  ORT EP FP32 = 99ms  → direct MIG FP32 = 4.7ms  (21×)
        #   mem_enc:   ORT EP FP16 = 7ms   → direct MIG FP32 = 3.3ms
        #   dec_init:  ORT CPU    = 118ms  → direct MIG FP32 = 4.9ms  (24×)
        # FP16 corrupts ConvTranspose upsampling for all three — keep FP32.
        def _mig_direct(onnx_path, cache_name, label):
            cache_path = Path(_cache) / cache_name
            return MIGraphXSession(
                onnx_path=onnx_path,
                cache_path=cache_path,
                fp16=False,
                label=label,
            )

        print("  Loading ONNX tracking modules ...")
        self.dec_init = _mig_direct(
            modules_dir / "mask_decoder_init.onnx", "dec_init_fp32.mxr", "dec_init")
        self.dec_prop = _mig_direct(
            modules_dir / "mask_decoder_propagate.onnx", "dec_prop_fp32.mxr", "dec_propagate")
        self.mem_enc  = _mig_direct(
            modules_dir / "memory_encoder.onnx", "mem_enc_fp32.mxr", "memory_encoder")

        # memory_attention: ORT MIGraphX EP.
        # NOTE: direct MIGraphX Python API produces numerically wrong cond output
        # (max_diff~6.7 vs CPU ORT reference) for this model — root cause is a
        # MIGraphX compiler difference for this specific attention architecture.
        # ORT MIGraphX EP gives correct results. At 1008px with backbone in GPU
        # memory, ORT EP may fall back to CPU (silent); acceptable since this
        # module contributes <10% of total latency. FP16 via migraphx_fp16_enable.
        mem_attn_fixed  = modules_dir / "memory_attention_fixed_N7.onnx"
        if mem_attn_fixed.exists():
            ort_opts = ort.SessionOptions()
            ort_opts.intra_op_num_threads = 1
            # Persist ORT-compiled .mxr to disk so subsequent runs skip the 5-min recompile.
            # First run: ~5 min compilation → cache saved. Subsequent: ~1.3s load from cache.
            _ort_cache_dir = modules_dir / "ort_mig_cache"
            _ort_cache_dir.mkdir(parents=True, exist_ok=True)
            _attn_prov_cached = []
            for (pname, popts) in _attn_prov:
                if pname == "MIGraphXExecutionProvider":
                    popts = dict(popts, migraphx_model_cache_dir=str(_ort_cache_dir))
                _attn_prov_cached.append((pname, popts))
            try:
                self.mem_attn = ort.InferenceSession(
                    str(mem_attn_fixed), providers=_attn_prov_cached, sess_options=ort_opts)
                # Check which provider actually ran
                prov = self.mem_attn.get_providers()[0]
                fp16_on = "FP16" if _attn_prov == MIG_FP16 else ""
                print(f"  memory_attention (ORT {prov[:3]} {fp16_on}): loaded")
            except Exception as e:
                self.mem_attn = ort.InferenceSession(str(mem_attn_fixed), providers=CPU)
                print(f"  memory_attention: CPU fallback ({str(e)[:50]})")
            self._mem_attn_slots = num_maskmem
        else:
            self.mem_attn = ort.InferenceSession(
                str(modules_dir / "memory_attention.onnx"), providers=CPU)
            self._mem_attn_slots = 0
            print("  memory_attention: CPU (dynamic) — export fixed_N7 for MIGraphX speedup")

        self.memory_bank = MemoryBank(temporal_pe, max_slots=num_maskmem)
        self._frame_idx  = 0
        self._timings: dict[str, list] = {
            k: [] for k in ["backbone", "mem_attn", "dec_init", "dec_prop", "mem_enc", "total"]
        }

        # Re-warmup MIGraphX backbone after all ORT sessions are created.
        # ORT's MIGraphX EP initialization can reset GPU state, invalidating earlier warmup.
        if self._mxr_backbone is not None:
            self._mxr_backbone.warmup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backbone(self, img_np: np.ndarray):
        t0 = time.perf_counter()
        if self._mxr_backbone is not None:
            result = self._mxr_backbone(img_np)
        else:
            import torch
            pv = torch.from_numpy(img_np).to(self.device).half()
            with torch.inference_mode():
                vis = self.vision_encoder(pv, return_dict=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            result = (
                vis.fpn_hidden_states[0].float().cpu().numpy(),
                vis.fpn_hidden_states[1].float().cpu().numpy(),
                vis.fpn_hidden_states[2].float().cpu().numpy(),
                (vis.fpn_position_encoding[2].float().cpu().numpy()
                 if vis.fpn_position_encoding is not None else None),
            )
        self._timings["backbone"].append(time.perf_counter() - t0)
        return result

    def _encode_memory(self, fpn_2: np.ndarray, pred_masks: np.ndarray) -> None:
        t0 = time.perf_counter()
        mask_2d = pred_masks.squeeze().astype(np.float32)
        mask_r  = cv2.resize(mask_2d, (self.mask_mem_size, self.mask_mem_size),
                             interpolation=cv2.INTER_LINEAR)[None, None]
        mf, mp = self.mem_enc.run(None, {"vision_features": fpn_2, "masks": mask_r})
        self._timings["mem_enc"].append(time.perf_counter() - t0)
        self.memory_bank.push(mf, mp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_frame(
        self,
        img_np: np.ndarray,
        box: list[float],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """
        Initialize tracking on the first frame.

        Args:
            img_np:        Preprocessed image (1, 3, H, W) float32.
            box:           [x1, y1, x2, y2] in imgsz pixel coordinates.
            point_coords:  Optional extra clicks (1, 1, N, 2).
            point_labels:  Optional click labels  (1, 1, N).

        Returns:
            (binary_mask, score)  —  mask shape (imgsz, imgsz) bool.
        """
        t_total = time.perf_counter()
        fpn_0, fpn_1, fpn_2, _ = self._backbone(img_np)

        box_np = np.array([[[[box[0], box[1], box[2], box[3]]]]], dtype=np.float32)
        if point_coords is None:
            pts  = np.zeros((1, 1, 1, 2), dtype=np.float32)
            lbls = np.full((1, 1, 1), -1, dtype=np.int32)
        else:
            pts  = point_coords.astype(np.float32)
            lbls = point_labels.astype(np.int32)

        t0 = time.perf_counter()
        masks, _, score = self.dec_init.run(None, {
            "fpn_2": fpn_2, "fpn_0": fpn_0, "fpn_1": fpn_1,
            "input_points": pts, "input_labels": lbls, "input_boxes": box_np,
        })
        self._timings["dec_init"].append(time.perf_counter() - t0)

        binary_mask = masks.squeeze() > 0
        self._encode_memory(fpn_2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary_mask, float(score.flat[0])

    def propagate_frame(self, img_np: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Propagate tracking to the next frame using memory bank.

        Args:
            img_np:  Preprocessed image (1, 3, H, W) float32.

        Returns:
            (binary_mask, score)  —  mask shape (imgsz, imgsz) bool.
        """
        t_total = time.perf_counter()
        fpn_0, fpn_1, fpn_2, pos_2 = self._backbone(img_np)

        cur_feat = fpn_2.reshape(1, 256, self.HW).transpose(2, 0, 1)
        # pos_2 is the 4th backbone output; only usable as positional encoding
        # if its spatial size matches fpn_2 (self.HW). At 1008px the tracker
        # backbone ONNX has 4 FPN levels where fpn_3 (36x36) != fpn_2 (72x72).
        pos_2_valid = (pos_2 is not None and pos_2.size == 256 * self.HW)
        cur_pos  = (pos_2.reshape(1, 256, self.HW).transpose(2, 0, 1)
                    if pos_2_valid else np.zeros_like(cur_feat))

        memory, memory_pos = self.memory_bank.build_attention_inputs(
            fixed_slots=self._mem_attn_slots)
        if memory is None:
            cond = fpn_2
        else:
            t0 = time.perf_counter()
            cond = self.mem_attn.run(None, {
                "current_vision_features": cur_feat,
                "memory":                  memory,
                "current_vis_pos_embed":   cur_pos,
                "memory_pos_embed":        memory_pos,
            })[0]
            self._timings["mem_attn"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        masks, _, score = self.dec_prop.run(None, {
            "fpn_2_cond": cond, "fpn_0": fpn_0, "fpn_1": fpn_1,
        })
        self._timings["dec_prop"].append(time.perf_counter() - t0)

        binary_mask = masks.squeeze() > 0
        self._encode_memory(fpn_2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary_mask, float(score.flat[0])

    # ------------------------------------------------------------------
    # Shared-backbone API (hybrid pipeline)
    # ------------------------------------------------------------------

    def run_backbone(self, img_np: np.ndarray):
        """Public alias of the internal backbone call.

        Returns: (fpn_0, fpn_1, fpn_2, pos_2). pos_2 may be None when the
        loaded backbone .mxr has only 3 outputs (older box-prompt build).
        """
        return self._backbone(img_np)

    def init_with_features(
        self,
        fpn_0: np.ndarray,
        fpn_1: np.ndarray,
        fpn_2: np.ndarray,
        pos_2,  # may be None / unused
        box: list[float],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """Same as init_frame but caller supplies pre-computed backbone features.

        Lets several trackers share one backbone call per frame.
        """
        t_total = time.perf_counter()
        box_np = np.array([[[[box[0], box[1], box[2], box[3]]]]], dtype=np.float32)
        if point_coords is None:
            pts  = np.zeros((1, 1, 1, 2), dtype=np.float32)
            lbls = np.full((1, 1, 1), -1, dtype=np.int32)
        else:
            pts  = point_coords.astype(np.float32)
            lbls = point_labels.astype(np.int32)

        t0 = time.perf_counter()
        masks, _, score = self.dec_init.run(None, {
            "fpn_2": fpn_2, "fpn_0": fpn_0, "fpn_1": fpn_1,
            "input_points": pts, "input_labels": lbls, "input_boxes": box_np,
        })
        self._timings["dec_init"].append(time.perf_counter() - t0)

        binary_mask = masks.squeeze() > 0
        self._encode_memory(fpn_2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary_mask, float(score.flat[0])

    def propagate_with_features(
        self,
        fpn_0: np.ndarray,
        fpn_1: np.ndarray,
        fpn_2: np.ndarray,
        pos_2,
    ) -> tuple[np.ndarray, float]:
        """Same as propagate_frame but caller supplies backbone features."""
        t_total = time.perf_counter()
        cur_feat = fpn_2.reshape(1, 256, self.HW).transpose(2, 0, 1)
        pos_2_valid = (pos_2 is not None and pos_2.size == 256 * self.HW)
        cur_pos  = (pos_2.reshape(1, 256, self.HW).transpose(2, 0, 1)
                    if pos_2_valid else np.zeros_like(cur_feat))

        memory, memory_pos = self.memory_bank.build_attention_inputs(
            fixed_slots=self._mem_attn_slots)
        if memory is None:
            cond = fpn_2
        else:
            t0 = time.perf_counter()
            cond = self.mem_attn.run(None, {
                "current_vision_features": cur_feat,
                "memory":                  memory,
                "current_vis_pos_embed":   cur_pos,
                "memory_pos_embed":        memory_pos,
            })[0]
            self._timings["mem_attn"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        masks, _, score = self.dec_prop.run(None, {
            "fpn_2_cond": cond, "fpn_0": fpn_0, "fpn_1": fpn_1,
        })
        self._timings["dec_prop"].append(time.perf_counter() - t0)

        binary_mask = masks.squeeze() > 0
        self._encode_memory(fpn_2, masks[0])
        self._timings["total"].append(time.perf_counter() - t_total)
        self._frame_idx += 1
        return binary_mask, float(score.flat[0])

    def reset(self) -> None:
        """Reset tracker state for a new video sequence."""
        self.memory_bank.reset()
        self._frame_idx = 0

    def print_timings(self) -> None:
        """Print per-module average latency and FPS."""
        print("\nPer-frame timing (ms):")
        for name, vals in self._timings.items():
            if vals:
                avg = np.mean(vals[1:] if len(vals) > 1 else vals) * 1000
                print(f"  {name:30s}: {avg:6.1f} ms")
        n = len(self._timings["total"])
        if n > 1:
            fps = 1.0 / np.mean(list(self._timings["total"])[1:])
            print(f"  {'FPS':30s}: {fps:.2f}")
