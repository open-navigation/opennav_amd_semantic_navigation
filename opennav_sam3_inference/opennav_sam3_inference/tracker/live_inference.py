"""Streaming single-frame inference for SAM3.

Designed for live sensor input (robotics, webcam, RTSP) where frames arrive
one-at-a-time and the ~O(N_frames) preprocessing inside
``Sam3VideoProcessor.init_video_session`` is not acceptable.

Uses the HF streaming entry point:

    session = processor.init_video_session(video=None, ...)   # empty session
    model(inference_session=session, frame=pixel_values)      # per-frame call

which is explicitly designed for "frames provided one at a time" mode
(``streaming=True`` inside the model). All existing MIG / ROCm patches from
``tracker/`` are reused so steady-state per-frame latency matches
``demo_text.py --mig``.

Quick start
-----------

    from tracker.live_inference import SAM3Live

    live = SAM3Live(
        checkpoint="model/sam3",
        onnx_dir="onnx_files_504",
        prompts=["car", "sidewalk", "grass"],
        imgsz=504,
        mig=True,
    )

    for frame_bgr in video_stream():
        out = live.infer(frame_bgr)
        for prompt, obj_ids in out["prompt_to_obj_ids"].items():
            for oid in obj_ids:
                mask = out["masks"][oid]           # HxW bool
                score = out["scores"][oid]         # float
                box = out["boxes"][oid]            # (x1, y1, x2, y2)
                # ... use mask ...

    # Operating-context switch (e.g. user changes ROI class list):
    live.reset_prompts(["pedestrian", "vehicle"])
"""
from __future__ import annotations

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")

import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# Memory-attention pointer-token cap per resolution (must match
# memory_attention_fixed_S7_P{K}.onnx artifact that was compiled by build.py).
_K_PER_IMGSZ = {504: 64, 1008: 48}


class SAM3Live:
    """One-frame-at-a-time SAM3 inference wrapper.

    Construct once at startup (model load + optional MIG warmup are slow but
    one-shot). Then call ``infer(frame_bgr)`` per incoming sensor frame.

    Threading model: single-threaded, single resident session. Do not share
    one instance across threads.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        prompts: Sequence[str],
        *,
        onnx_dir: str | Path | None = None,
        imgsz: int = 1008,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        mig: bool = False,
        max_vision_features_cache_size: int = 1,
        keep_recent_frames: int = 0,
        redetect_every: int = 1,
        max_objects_per_prompt: int | dict[str, int] | None = 5,
        bootstrap_frames: int = 0,
        bootstrap_min_score: float = 0.3,
    ):
        """
        Args:
            checkpoint: HF model dir (contains model.safetensors).
            prompts: initial text prompts (e.g. ["car", "sidewalk"]).
            onnx_dir: required if ``mig=True`` (e.g. ``onnx_files_504``).
            imgsz: 504 or 1008. Must match the MIG artifacts under ``onnx_dir``.
            dtype: fp16 (default) or fp32.
            device: torch device, defaults to cuda if available.
            mig: enable MIGraphX accelerated paths (vision encoder, DETR encoder,
                memory attention, batched mask decoder). Highly recommended.
            max_vision_features_cache_size: HF vision-feature LRU size. Default 1
                — only keeps the most recent frame's features.
            keep_recent_frames: bound on number of past raw frame tensors kept
                in the session. **Default 0 = no pruning** because the SAM3
                tracker maintains per-frame state in output_dict_per_obj that
                is not safe to drop without also pruning processed_frames
                consistently (pruning only processed_frames leaves the
                tracker walking stale per-frame entries → O(N^2) growth).
                Raw pixel growth is ~1.5 MB/frame @504 — for long-running
                streams call ``reset_tracking()`` periodically (e.g. every
                few minutes) to fully reclaim state instead.
            redetect_every: full SAM3 (detector + tracker) runs on every Nth
                frame; intermediate frames run tracker propagation only
                (faster, but no new objects are discovered). Default 1 =
                full detection every frame. Frame 0 and the first frame
                after any reset_*() call always run full detection
                regardless of this schedule. Set to e.g. 3-5 to recover
                FPS on multi-prompt workloads.
            max_objects_per_prompt: cap on simultaneously tracked objects
                per prompt. Excess (lowest detection score) are evicted via
                ``session.remove_object``, freeing tracker memory and
                bounding compute. **Default 5** — without a cap the session
                silently accumulates ghost objects (low-score detections
                that get hidden from output but still cost full tracker
                propagation every frame). On a 2-prompt scene we measured
                450 ms/frame uncapped vs 212 ms/frame at cap=1; on a
                3-prompt dense scene 7 s/frame uncapped vs 300 ms at cap=1.
                - ``int`` (default 5): same cap for all prompts
                - ``dict``: per-prompt cap, e.g. ``{"tree": 3, "human": 5}``;
                  prompts not listed are uncapped
                - ``None``: unlimited — only safe when you know the scene's
                  true object count is small (e.g. single tracked target).
            bootstrap_frames: PoC switch — if > 0, run text prompts on the
                first N frames AS USUAL to anchor the concept, then auto-extract
                visual exemplar embeddings from the high-confidence decoder
                queries on those frames and swap them into the cached
                ``session.prompt_embeddings[prompt_id]`` so frame N+ runs
                detection against site-specific visual signatures instead of
                CLIP text. The customer API (text prompts in yaml) is unchanged.
                Default 0 = disabled (pure text mode, original behavior).
            bootstrap_min_score: per-query confidence floor (sigmoid of
                pred_logits * presence) for a query to be collected into the
                bootstrap pool. Default 0.3 — somewhat permissive to collect
                more samples in the few frames available; tighten if exemplar
                quality is bad.
        """
        # Lazy import so that consumers of this module don't pay model import cost
        # if they only want the class definition.
        from transformers import Sam3VideoModel, AutoProcessor, Sam3VideoConfig

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device
        self.dtype = dtype
        self.imgsz = imgsz
        self.keep_recent_frames = keep_recent_frames
        self.redetect_every = max(1, int(redetect_every))
        self.max_objects_per_prompt = max_objects_per_prompt
        # Force full detection on next infer() (frame 0, or first after reset).
        self._force_detect_next = True
        # Monotonic count of calls to infer() — drives the redetect schedule.
        self._infer_calls = 0
        # Bootstrap-to-exemplar PoC state (see bootstrap_frames docstring).
        # Approach (Path C - box prompts): during bootstrap collect high-conf
        # pred_boxes per prompt; after bootstrap, inject them as input_boxes
        # on every subsequent detector call. Boxes are stored in
        # CxCyWH normalized [0,1] format (matches Sam3GeometryEncoder input).
        self.bootstrap_frames = max(0, int(bootstrap_frames))
        self.bootstrap_min_score = float(bootstrap_min_score)
        self._bootstrap_remaining: dict[int, int] = {}
        # During bootstrap: per-prompt accumulator of high-conf pred_boxes.
        # After bootstrap done: per-prompt stored exemplar boxes (tensor).
        self._exemplar_box_pool: dict[int, list[torch.Tensor]] = {}
        self._exemplar_boxes: dict[int, torch.Tensor] = {}  # [num_boxes, 4] cxcywh [0,1]
        # Forward-hook capture for the current infer() call. List of
        # (pred_logits, pred_boxes, presence_logits) per detector call.
        self._bootstrap_capture: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        # Per-infer counter of detector_model calls — used by the box-inject
        # pre-hook to figure out which prompt is being processed.
        self._detector_call_counter = 0
        self._bootstrap_hook_handle = None
        self._box_inject_hook_handle = None
        # Drift detection + auto re-bootstrap. After bootstrap completes for
        # a prompt, we capture a baseline avg-score. Per-frame avg score is
        # appended to a rolling window; every check_interval we compare the
        # window mean vs baseline. A drop greater than drop_threshold
        # triggers re-bootstrap (clears exemplar boxes, restarts the 5-frame
        # text capture so the system adapts to the new scene). Env-var
        # tunable; defaults are conservative.
        import os as _os
        self._drift_enabled = (
            _os.environ.get("SAM3_DRIFT_REBOOTSTRAP", "1") == "1"
            and self.bootstrap_frames > 0
        )
        self._drift_window = int(_os.environ.get("SAM3_DRIFT_WINDOW_FRAMES", "30"))
        self._drift_check_interval = int(_os.environ.get("SAM3_DRIFT_CHECK_INTERVAL", "30"))
        self._drift_drop_threshold = float(_os.environ.get("SAM3_DRIFT_DROP_THRESHOLD", "0.4"))
        self._drift_grace_frames = int(_os.environ.get("SAM3_DRIFT_GRACE_FRAMES", "30"))
        self._drift_baseline_score: dict[int, float] = {}
        from collections import deque as _deque
        self._drift_recent_scores: dict[int, "_deque[float]"] = {}
        self._drift_frames_since_bootstrap: dict[int, int] = {}
        self._drift_pending_rebootstrap: bool = False
        # Optional periodic re-bootstrap. Independent of the condition-based
        # drift detection above (both can be enabled simultaneously — OR
        # logic). Useful as a safety net in environments where you want to
        # force fresh exemplars on a fixed schedule regardless of score
        # signal (e.g., long-running deployments, slow gradual drift that
        # never trips the score threshold). Default 0 = off.
        self._periodic_rebootstrap_seconds = float(
            _os.environ.get("SAM3_PERIODIC_REBOOTSTRAP_SECONDS", "0")
        )
        self._last_bootstrap_complete_time: float = 0.0
        # Monotonic frame_idx we hand to model.forward. We MUST assign this
        # ourselves rather than letting HF auto-assign, because HF computes
        # the next idx as ``len(processed_frames)`` — which collides with
        # old indices after _prune_old_frames() shrinks the dict, corrupting
        # the tracker's per-frame state. See infer() for details.
        self._next_frame_idx = 0

        t = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))

        config = Sam3VideoConfig.from_pretrained(str(checkpoint))
        if imgsz != config.image_size:
            config.image_size = imgsz
            config.low_res_mask_size = 4 * imgsz // 14
            new_size = {"height": imgsz, "width": imgsz}
            new_mask = {"height": 4 * imgsz // 14, "width": 4 * imgsz // 14}
            for sub in (getattr(self.processor, "image_processor", None),
                        getattr(self.processor, "video_processor", None)):
                if sub is not None:
                    if hasattr(sub, "size"):
                        sub.size = new_size
                    if hasattr(sub, "mask_size"):
                        sub.mask_size = new_mask
            if hasattr(self.processor, "target_size"):
                self.processor.target_size = imgsz

        self.model = (
            Sam3VideoModel.from_pretrained(str(checkpoint), config=config)
            .to(device).to(dtype).eval()
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[SAM3Live] model loaded in {time.perf_counter() - t:.1f}s "
              f"(imgsz={imgsz}, dtype={dtype})")

        if mig:
            if onnx_dir is None:
                raise ValueError("mig=True requires onnx_dir")
            self._apply_mig_patches(Path(onnx_dir), imgsz)

        # Detector-skip patch — always applied; controlled per-frame via
        # model._skip_detection. Idempotent.
        from .redetect_schedule import patch_redetect_schedule
        patch_redetect_schedule(self.model)

        # Empty (streaming) session — no preloaded frames, no O(N) preprocess.
        self.session = self.processor.init_video_session(
            video=None,
            inference_device=device,
            dtype=dtype,
            max_vision_features_cache_size=max_vision_features_cache_size,
        )

        # Bootstrap hook on detector_model — captures decoder_hidden_states
        # + pred_logits per detector call. No-op if bootstrap_frames=0.
        if self.bootstrap_frames > 0:
            self._install_bootstrap_hook()

        # Add initial prompts (deduped + encoded by the processor).
        self.set_prompts(prompts)
        print(f"[SAM3Live] ready. prompts={list(self.session.prompts.values())} "
              f"bootstrap_frames={self.bootstrap_frames}")

    # ------------------------------------------------------------------
    # MIG patch wiring (mirrors demo_text.py)
    # ------------------------------------------------------------------
    def _apply_mig_patches(self, onnx_dir: Path, imgsz: int) -> None:
        from .tracker import MIGraphXBackbone
        from .mig_vision_encoder import patch_sam3_video_model_with_mig

        det_dir = onnx_dir / "backbone_detector"
        print(f"[SAM3Live] patching vision_encoder with MIGraphX backbone ...")
        t = time.perf_counter()
        mxr = MIGraphXBackbone(
            onnx_path=det_dir / "single_simplified.onnx",
            cache_path=det_dir / "tuned.mxr",
        )
        mxr.warmup(n=2)
        patch_sam3_video_model_with_mig(self.model, mxr)
        print(f"  vision_encoder MIG ready in {time.perf_counter() - t:.1f}s")

        detr_onnx = onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx"
        if detr_onnx.exists():
            from .mig_detr_encoder import patch_sam3_video_model_detr_encoder
            patch_sam3_video_model_detr_encoder(self.model, detr_onnx)
            print(f"  detr_encoder MIG ready")
        else:
            print(f"  (skip detr_encoder MIG: {detr_onnx} not found)")

        # K is resolution-dependent (MLIR attention perf cliff).
        k = _K_PER_IMGSZ.get(imgsz, 32)
        mem_attn_onnx = onnx_dir / "tracker_modules" / f"memory_attention_fixed_S7_P{k}.onnx"
        if not mem_attn_onnx.exists():
            for alt in (32, 48, 64, 16, 4):
                alt_path = onnx_dir / "tracker_modules" / f"memory_attention_fixed_S7_P{alt}.onnx"
                if alt_path.exists():
                    mem_attn_onnx = alt_path
                    break
        if mem_attn_onnx.exists():
            from .mig_memory_attention import patch_sam3_video_model_memory_attention
            patch_sam3_video_model_memory_attention(self.model, mem_attn_onnx)
            print(f"  memory_attention MIG ready ({mem_attn_onnx.name})")

        # batched_mask_decoder patch — previously had an obj-allocation cascade
        # bug when combined with bootstrap box-prompt (produced 4/5 empty masks
        # per frame) because the slow_path always treated propagation frames
        # as is_init_cond_frame=True. Fixed in batched_mask_decoder.py: the
        # check is now gated on has_new_inputs to match HF native behavior.
        # Patch is safe to apply unconditionally.
        import os as _os
        force_skip = _os.environ.get("SAM3_DISABLE_BATCHED_DECODER", "0") == "1"
        if force_skip:
            print(f"  batched_mask_decoder patch SKIPPED (env override)")
        else:
            from .batched_mask_decoder import patch_batched_mask_decoder
            patch_batched_mask_decoder(self.model)
            print(f"  batched_mask_decoder patch applied (active for N>1 obj)")

    # ------------------------------------------------------------------
    # Bootstrap-to-exemplar PoC plumbing
    # ------------------------------------------------------------------
    def _install_bootstrap_hook(self) -> None:
        """Install two hooks on ``detector_model``:

        1. **Forward hook** (capture) — collects ``pred_logits``, ``pred_boxes``,
           ``presence_logits`` from each detector call. Used during bootstrap
           to extract high-confidence boxes per prompt.
        2. **Forward pre-hook** (inject) — once bootstrap is complete for a
           prompt, injects that prompt's stored exemplar boxes as
           ``input_boxes`` into the detector forward kwargs. Per-call
           prompt attribution uses ``self._detector_call_counter`` which is
           reset to 0 at the start of each infer().
        """
        detector = self.model.detector_model

        def detector_hook(module, args, kwargs, output):
            try:
                pred_logits = getattr(output, "pred_logits", None)
                pred_boxes = getattr(output, "pred_boxes", None)
                presence_logits = getattr(output, "presence_logits", None)
                if pred_logits is None or pred_boxes is None:
                    if not getattr(self, "_bootstrap_hook_warned", False):
                        attrs = [a for a in dir(output) if not a.startswith("_")]
                        print(f"[SAM3Live] bootstrap hook: pred_logits={pred_logits is not None} "
                              f"pred_boxes={pred_boxes is not None}. Attrs: {attrs[:20]}",
                              flush=True)
                        self._bootstrap_hook_warned = True
                    return
                self._bootstrap_capture.append(
                    (pred_logits.detach(), pred_boxes.detach(),
                     presence_logits.detach() if presence_logits is not None else None)
                )
            except Exception as e:
                print(f"[SAM3Live] bootstrap hook exception: {e}", flush=True)

        def box_inject_pre_hook(module, args, kwargs):
            # Look up which prompt is currently being processed via call counter.
            prompt_ids = list(self.session.prompts.keys())
            idx = self._detector_call_counter
            self._detector_call_counter += 1
            if idx >= len(prompt_ids):
                return  # call out of expected range — skip injection
            pid = prompt_ids[idx]
            boxes = self._exemplar_boxes.get(pid)
            if boxes is None:
                return  # bootstrap not yet done for this prompt
            if kwargs.get("input_boxes") is not None:
                return  # caller already supplied boxes — don't override
            num_boxes = boxes.shape[0]
            # Inject:
            #   input_boxes:        [1, num_boxes, 4]  (cxcywh [0,1])
            #   input_boxes_labels: [1, num_boxes]     (1 = positive)
            kwargs["input_boxes"] = boxes.to(
                dtype=self.dtype, device=self.device,
            ).unsqueeze(0)
            kwargs["input_boxes_labels"] = torch.ones(
                1, num_boxes, dtype=torch.long, device=self.device,
            )
            return args, kwargs

        self._bootstrap_hook_handle = detector.register_forward_hook(detector_hook, with_kwargs=True)
        self._box_inject_hook_handle = detector.register_forward_pre_hook(
            box_inject_pre_hook, with_kwargs=True,
        )
        print(f"[SAM3Live] bootstrap hooks installed (capture + box-inject)")

    def _process_bootstrap_capture(self) -> None:
        """Consume ``self._bootstrap_capture`` from the just-completed infer
        call. Collects high-confidence pred_boxes per prompt; when a prompt
        has used its full bootstrap budget, stores the representative box
        set in ``self._exemplar_boxes[prompt_id]``. From the next infer onward,
        the box-inject pre-hook automatically passes those boxes as
        ``input_boxes`` on the per-prompt detector call so SAM3's geometry
        encoder spatially grounds detection.
        """
        if self.bootstrap_frames <= 0 or not self._bootstrap_capture:
            return
        prompt_ids = list(self.session.prompts.keys())
        if len(self._bootstrap_capture) != len(prompt_ids):
            return  # call mismatch, skip
        for prompt_id, (pred_logits, pred_boxes, presence_logits) in zip(prompt_ids, self._bootstrap_capture):
            remaining = self._bootstrap_remaining.get(prompt_id, 0)
            if remaining <= 0:
                continue
            # pred_logits: [1, num_queries] | pred_boxes: [1, num_queries, 4] xyxy [0,1]
            scores = pred_logits[0].sigmoid()
            if presence_logits is not None:
                scores = scores * presence_logits[0].sigmoid()
            keep = scores > self.bootstrap_min_score
            n_kept = int(keep.sum().item()) if keep.numel() else 0
            if n_kept > 0:
                boxes_xyxy = pred_boxes[0, keep]  # [n_kept, 4]
                # Convert xyxy → cxcywh (normalized [0,1]).
                x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                w = (x2 - x1).clamp(min=1e-6)
                h = (y2 - y1).clamp(min=1e-6)
                boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
                self._exemplar_box_pool.setdefault(prompt_id, []).append(boxes_cxcywh)
            self._bootstrap_remaining[prompt_id] = remaining - 1
            prompt_text = self.session.prompts.get(prompt_id, "?")
            print(
                f"[SAM3Live] bootstrap prompt={prompt_text!r} (id={prompt_id}): "
                f"frame_consumed → remaining={remaining - 1} kept_boxes={n_kept}",
                flush=True,
            )
            # If this prompt just finished bootstrap → freeze exemplar boxes.
            if self._bootstrap_remaining[prompt_id] == 0:
                pool = self._exemplar_box_pool.get(prompt_id) or []
                total = sum(t.shape[0] for t in pool)
                if total == 0:
                    print(
                        f"[SAM3Live] bootstrap FAIL for prompt={prompt_text!r}: "
                        f"no high-conf boxes during bootstrap window. "
                        f"Keeping pure text-prompt detection.",
                        flush=True,
                    )
                    continue
                all_boxes = torch.cat(pool, dim=0)  # [total, 4] cxcywh
                # Cap to a manageable number — empirically cap=3 is the
                # sweet spot. cap=1 too restrictive (range tighter on area
                # but misses spatial diversity); cap>=5 starts bleeding the
                # class concept into adjacent-class edge regions (e.g. floor
                # bleeding into baseboards/踢脚线 when cap=10 on Steve's
                # hallway bag). Top-K-by-area keeps the largest exemplar
                # patches — most representative of the central class region.
                # Env-var overridable for tuning.
                import os as _os
                cap = int(_os.environ.get("SAM3_BOOTSTRAP_BOX_CAP", "3"))
                if all_boxes.shape[0] > cap:
                    areas = all_boxes[:, 2] * all_boxes[:, 3]
                    topk = torch.topk(areas, cap).indices
                    all_boxes = all_boxes[topk]
                self._exemplar_boxes[prompt_id] = all_boxes
                # Capture drift-detection baseline: mean of the box-area-pooled
                # bootstrap scores. We use a proxy of "what's a healthy score
                # for this prompt in this scene"; rolling per-frame mean is
                # compared against it later.
                baseline = (pred_logits[0].sigmoid() *
                            (presence_logits[0].sigmoid() if presence_logits is not None else 1.0))
                # Only use baseline from boxes that passed the threshold this frame
                if keep.any():
                    baseline_val = float(baseline[keep].mean().item())
                else:
                    baseline_val = float(self.bootstrap_min_score)
                self._drift_baseline_score[prompt_id] = baseline_val
                from collections import deque as _deque
                self._drift_recent_scores[prompt_id] = _deque(maxlen=self._drift_window)
                self._drift_frames_since_bootstrap[prompt_id] = 0
                # Record wall-clock time for the periodic re-bootstrap timer.
                # (Last completion wins if multiple prompts finish at same frame.)
                import time as _time
                self._last_bootstrap_complete_time = _time.perf_counter()
                print(
                    f"[SAM3Live] bootstrap DONE for prompt={prompt_text!r}: "
                    f"stored {all_boxes.shape[0]} exemplar boxes (cxcywh) "
                    f"from {total} pooled candidates — box-inject active from next frame "
                    f"(drift baseline score={baseline_val:.3f})",
                    flush=True,
                )
                self._exemplar_box_pool[prompt_id] = []

    # ------------------------------------------------------------------
    # Drift detection + auto re-bootstrap
    # ------------------------------------------------------------------

    def _drift_record_and_check(self, result: dict) -> None:
        """Append per-prompt avg score from this frame, then (every
        check_interval frames) compare rolling mean vs baseline. If any
        prompt's mean drops below baseline * (1 - drop_threshold), schedule
        a re-bootstrap on the next infer().
        """
        if not self._drift_enabled or not self._drift_baseline_score:
            return
        prompt_to_obj_ids = result.get("prompt_to_obj_ids", {})
        scores = result.get("scores", {})
        # Find prompt_id for each prompt text (need to look up via session)
        text_to_id = {v: k for k, v in self.session.prompts.items()}
        for prompt_text, oids in prompt_to_obj_ids.items():
            pid = text_to_id.get(prompt_text)
            if pid is None or pid not in self._drift_recent_scores:
                continue
            self._drift_frames_since_bootstrap[pid] = (
                self._drift_frames_since_bootstrap.get(pid, 0) + 1
            )
            if oids:
                avg = sum(float(scores.get(o, 0.0)) for o in oids) / len(oids)
            else:
                avg = 0.0
            self._drift_recent_scores[pid].append(avg)

        # Throttled drift check
        if (self._infer_calls % self._drift_check_interval) != 0:
            return
        for pid, baseline in self._drift_baseline_score.items():
            grace = self._drift_frames_since_bootstrap.get(pid, 0)
            if grace < self._drift_grace_frames:
                continue  # not enough post-bootstrap data yet
            recent = self._drift_recent_scores.get(pid)
            if recent is None or len(recent) < max(5, self._drift_window // 4):
                continue
            current_mean = sum(recent) / len(recent)
            min_acceptable = baseline * (1.0 - self._drift_drop_threshold)
            if current_mean < min_acceptable:
                prompt_text = self.session.prompts.get(pid, "?")
                pct = (1.0 - current_mean / max(baseline, 1e-6)) * 100
                print(
                    f"[SAM3Live] f={self._infer_calls} DRIFT detected on prompt={prompt_text!r} "
                    f"(baseline={baseline:.3f} → rolling mean={current_mean:.3f}, "
                    f"-{pct:.0f}%). Scheduling re-bootstrap on next infer.",
                    flush=True,
                )
                self._drift_pending_rebootstrap = True
                return  # one prompt is enough to trigger full re-bootstrap

    def _check_periodic_rebootstrap(self) -> None:
        """If SAM3_PERIODIC_REBOOTSTRAP_SECONDS > 0 and elapsed wall-clock
        time since last bootstrap completion exceeds it, schedule a
        re-bootstrap. Independent of (additive to) drift detection.
        """
        if self._periodic_rebootstrap_seconds <= 0:
            return
        if self._last_bootstrap_complete_time <= 0:
            return  # never bootstrapped yet
        if self._drift_pending_rebootstrap:
            return  # already scheduled
        import time as _time
        elapsed = _time.perf_counter() - self._last_bootstrap_complete_time
        if elapsed >= self._periodic_rebootstrap_seconds:
            print(
                f"[SAM3Live] f={self._infer_calls} PERIODIC re-bootstrap "
                f"({elapsed:.1f}s since last bootstrap, "
                f"threshold {self._periodic_rebootstrap_seconds:.0f}s). "
                f"Scheduling re-bootstrap on next infer.",
                flush=True,
            )
            self._drift_pending_rebootstrap = True

    def _drift_perform_rebootstrap(self) -> None:
        """Clear exemplar boxes and reset bootstrap counters so the next
        bootstrap_frames frames re-capture against the new scene. Tracker
        state is NOT reset (we want continuity through the re-bootstrap
        window if possible).
        """
        self._exemplar_boxes.clear()
        self._exemplar_box_pool.clear()
        self._drift_baseline_score.clear()
        self._drift_recent_scores.clear()
        self._drift_frames_since_bootstrap.clear()
        # Reset per-prompt bootstrap counter to original value
        for pid in self.session.prompts.keys():
            self._bootstrap_remaining[pid] = self.bootstrap_frames
            self._exemplar_box_pool.setdefault(pid, [])
        self._drift_pending_rebootstrap = False
        # Restart the periodic timer (will be set anew when next bootstrap
        # completes via _process_bootstrap_capture).
        self._last_bootstrap_complete_time = 0.0
        print(f"[SAM3Live] f={self._infer_calls} RE-BOOTSTRAP started — next "
              f"{self.bootstrap_frames} frames use text prompts to capture fresh "
              f"exemplar boxes", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Add prompts to the current session. Duplicates are deduped by
        the processor. Incremental — does NOT clear existing prompts.
        """
        prompts = [p for p in prompts if p]
        if not prompts:
            return
        self.processor.add_text_prompt(self.session, list(prompts))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # Init bootstrap counter for any newly-added prompts.
        if self.bootstrap_frames > 0:
            for pid in self.session.prompts.keys():
                self._bootstrap_remaining.setdefault(pid, self.bootstrap_frames)
                self._exemplar_box_pool.setdefault(pid, [])

    def reset_prompts(self, prompts: Sequence[str]) -> None:
        """Drop ALL existing prompts + tracked objects + cache, install new
        prompt set. Use when the operating context changes (e.g. user
        switches from "indoor objects" to "outdoor objects").
        """
        # reset_state() clears prompts, tracking, and vision cache.
        # processed_frames is preserved (raw pixel tensors stay), but we
        # don't reuse old frame_idx so this is OK.
        self.session.reset_state()
        # reset_state() does NOT clear processed_frames — drop the stale
        # raw pixel buffers ourselves so memory doesn't leak and our reset
        # counter doesn't collide with old indices on the next infer().
        if self.session.processed_frames is not None:
            self.session.processed_frames.clear()
        self._next_frame_idx = 0
        # Reset bootstrap state — new prompts get a fresh bootstrap cycle.
        self._bootstrap_remaining.clear()
        self._exemplar_box_pool.clear()
        self._exemplar_boxes.clear()
        # Reset drift state too — baseline must be re-learned after re-prompts.
        self._drift_baseline_score.clear()
        self._drift_recent_scores.clear()
        self._drift_frames_since_bootstrap.clear()
        self._drift_pending_rebootstrap = False
        self._last_bootstrap_complete_time = 0.0
        self.set_prompts(prompts)
        # Force detection on next frame — no tracked objects to propagate.
        self._force_detect_next = True

    def reset_tracking(self) -> None:
        """Drop tracked-object history but keep prompts. Use when the scene
        has changed enough that re-detection from scratch is desired.

        NOTE: does NOT re-run bootstrap. The exemplar embeddings (if already
        installed) stay in session.prompt_embeddings. To re-bootstrap, use
        reset_prompts() with the same prompt list.
        """
        self.session.reset_inference_session()
        if self.session.processed_frames is not None:
            self.session.processed_frames.clear()
        self._next_frame_idx = 0
        self._force_detect_next = True

    def infer(
        self,
        frame_bgr: np.ndarray,
        *,
        full_detection: bool | None = None,
    ) -> dict:
        """Run one streaming inference step on a single sensor frame.

        Args:
            frame_bgr: HxWx3 uint8 BGR image (OpenCV convention).
            full_detection: per-call override of the redetect schedule.
                - ``None`` (default): use the ``redetect_every`` schedule
                  set at construction time.
                - ``True``: force full SAM3 (detector + tracker) this frame.
                - ``False``: force tracker propagation only — no detector,
                  no new objects discovered. Useful in ROS callbacks where
                  the node decides per-message based on its own policy
                  (e.g. queue depth, time since last detect, UI flag).

            Note: when the wrapper requires detection (frame 0, first frame
            after any reset, or no prompts encoded yet), ``full_detection=False``
            is silently overridden to True. This avoids a no-op frame.

        Returns:
            dict with:
                object_ids:         list[int]            tracked object IDs this frame
                scores:             dict[int, float]     detection score per obj_id
                masks:              dict[int, ndarray]   HxW bool mask at original res
                boxes:              dict[int, tuple]     (x1,y1,x2,y2) per obj_id
                prompt_to_obj_ids:  dict[str, list[int]] grouping by prompt text
                frame_idx:          int                  session-internal frame counter
                detected:           bool                 True if detector ran this frame
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR, got shape {frame_bgr.shape}")
        H, W = frame_bgr.shape[:2]

        # Preprocess via the same video_processor that init_video_session uses,
        # to guarantee numeric identity with the preloaded-video path.
        # videos=[pil] → treated as one video of one frame.
        pil = _bgr_to_pil(frame_bgr)
        processed = self.processor.video_processor(
            videos=[pil],
            device=self.device,
            return_tensors="pt",
        )
        # pixel_values_videos: [1, 1, C, H, W]  →  pick (C, H, W)
        pixel_values = processed.pixel_values_videos[0][0].to(self.dtype)

        # Decide whether this frame runs full detection or tracker-only.
        # Precedence: forced-detect (frame 0 / post-reset) > caller override
        # > schedule.
        if self._force_detect_next:
            skip_detection = False
            self._force_detect_next = False
        elif full_detection is not None:
            skip_detection = not full_detection
        else:
            skip_detection = (self._infer_calls % self.redetect_every) != 0
        self.model._skip_detection = skip_detection
        self._infer_calls += 1

        # Streaming forward. We pass frame_idx explicitly (instead of letting
        # HF auto-assign it as ``len(processed_frames)``) because we prune old
        # entries from processed_frames for memory bounds — auto-assignment
        # would then collide with frame indices that the tracker still has
        # per-frame state for, silently corrupting output_dict_per_obj and
        # producing masks that lag the actual frame by N frames.
        # Setting streaming=True is implicit when ``frame`` is provided.
        frame_idx_for_this_call = self._next_frame_idx
        self._next_frame_idx += 1
        # Reset bootstrap state for this frame.
        if self.bootstrap_frames > 0:
            self._bootstrap_capture.clear()
            self._detector_call_counter = 0
            # If drift detection scheduled a re-bootstrap, perform it BEFORE
            # this frame's detector call so the pre-hook stops injecting old
            # exemplar boxes and the bootstrap counter restarts capture.
            if self._drift_pending_rebootstrap:
                self._drift_perform_rebootstrap()
        with torch.inference_mode():
            raw_out = self.model(
                inference_session=self.session,
                frame=pixel_values,
                frame_idx=frame_idx_for_this_call,
            )
        # Consume any captured queries — possibly install exemplar embeddings.
        if self.bootstrap_frames > 0 and not skip_detection:
            self._process_bootstrap_capture()

        frame_idx = raw_out.frame_idx

        # Enforce per-prompt cap BEFORE postprocess. Evicted objects:
        #   (a) are removed from session state via session.remove_object, so
        #       the tracker stops propagating them next frame (bounds compute)
        #   (b) are stripped from this frame's raw output so they don't appear
        #       once before disappearing
        evicted_ids = self._enforce_per_prompt_cap(raw_out.obj_id_to_tracker_score)

        def _strip(d):
            if not evicted_ids or not isinstance(d, dict):
                return d
            return {k: v for k, v in d.items() if k not in evicted_ids}

        # Postprocess: low-res masks → original resolution + multi-prompt grouping.
        model_outputs = {
            "obj_id_to_mask": _strip(raw_out.obj_id_to_mask),
            "obj_id_to_score": _strip(raw_out.obj_id_to_score),
            "obj_id_to_tracker_score": _strip(raw_out.obj_id_to_tracker_score),
            "suppressed_obj_ids": raw_out.suppressed_obj_ids,
        }
        pp = self.processor.postprocess_outputs(
            inference_session=self.session,
            model_outputs=model_outputs,
            original_sizes=[[H, W]],
        )

        obj_ids = pp["object_ids"].tolist()
        scores_list = pp["scores"].tolist()
        if len(obj_ids):
            masks_np = pp["masks"].cpu().numpy()
            boxes_np = pp["boxes"].cpu().numpy()
        else:
            masks_np = np.zeros((0, H, W), dtype=bool)
            boxes_np = np.zeros((0, 4), dtype=np.float32)

        # Bound memory growth from accumulating raw frame tensors.
        if self.keep_recent_frames > 0:
            self._prune_old_frames(keep_last=self.keep_recent_frames)

        result = {
            "object_ids": obj_ids,
            "scores": {oid: float(s) for oid, s in zip(obj_ids, scores_list)},
            "masks": {oid: masks_np[i] for i, oid in enumerate(obj_ids)},
            "boxes": {oid: tuple(float(v) for v in boxes_np[i]) for i, oid in enumerate(obj_ids)},
            "prompt_to_obj_ids": pp["prompt_to_obj_ids"],
            "frame_idx": frame_idx,
            "detected": not skip_detection,
        }
        # Drift detection: record this frame's per-prompt avg score, and
        # throttled-check whether any prompt has dropped enough from its
        # bootstrap baseline to trigger a re-bootstrap on the next infer.
        if self._drift_enabled:
            self._drift_record_and_check(result)
        # Periodic re-bootstrap timer (independent of drift; user-opt-in
        # safety net via SAM3_PERIODIC_REBOOTSTRAP_SECONDS).
        if self.bootstrap_frames > 0:
            self._check_periodic_rebootstrap()
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cap_for(self, prompt_text: str) -> int | None:
        cap = self.max_objects_per_prompt
        if cap is None:
            return None
        if isinstance(cap, int):
            return cap
        if isinstance(cap, dict):
            return cap.get(prompt_text)
        return None

    def _enforce_per_prompt_cap(self, tracker_scores: dict) -> set[int]:
        """Evict excess objects per prompt, keeping the most-persistent ones.

        Sort key: live ``tracker_score`` from this frame's raw output.
        The detection score in ``session.obj_id_to_score`` is frozen at
        first-detect time and never updates, so when fresh detections
        arrive each frame at ~0.94 they tie-break-evict the older tracked
        objects — but those fresh objects haven't built propagation
        history yet and get filtered out by the postprocess tracker_score
        gate, producing empty output frames (the "stops detecting"
        symptom on stuff-class prompts like ``grass`` / ``sidewalk``).
        Sorting by live tracker_score keeps whichever objects are
        actually propagating well right now.

        Uses ``session.remove_object`` which physically drops the object
        from obj_ids + all per-object dicts, so tracker propagation cost
        is freed starting next frame. Returns the set of evicted obj_ids
        so the caller can also strip them from this frame's raw outputs.
        """
        if self.max_objects_per_prompt is None:
            return set()

        by_prompt: dict[str, list[tuple[int, float]]] = {}
        for oid in list(self.session.obj_ids):
            pid = self.session.obj_id_to_prompt_id.get(oid)
            if pid is None:
                continue
            prompt_text = self.session.prompts.get(pid, "?")
            score = float(tracker_scores.get(oid, 0.0))
            by_prompt.setdefault(prompt_text, []).append((oid, score))

        evicted: set[int] = set()
        for prompt_text, items in by_prompt.items():
            cap = self._cap_for(prompt_text)
            if cap is None or len(items) <= cap:
                continue
            items.sort(key=lambda x: x[1], reverse=True)
            for oid, _ in items[cap:]:
                self.session.remove_object(oid, strict=False)
                evicted.add(oid)
        return evicted

    def _prune_old_frames(self, keep_last: int) -> None:
        """Drop processed_frames entries older than the last ``keep_last``.

        Vision features for these frames are already cached (or evicted by the
        LRU). The mask history lives in output_dict_per_obj, not here, so
        dropping raw pixels has no effect on tracker propagation as long as
        memory_attention window <= keep_last.
        """
        pf = self.session.processed_frames
        if pf is None or len(pf) <= keep_last:
            return
        # Keep highest-index `keep_last` entries.
        sorted_idx = sorted(pf.keys())
        to_drop = sorted_idx[:-keep_last]
        for idx in to_drop:
            del pf[idx]


__all__ = ["SAM3Live"]
