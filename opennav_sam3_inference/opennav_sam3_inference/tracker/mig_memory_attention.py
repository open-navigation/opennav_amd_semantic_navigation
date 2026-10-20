"""MIGraphX-backed memory_attention shim for Sam3VideoModel.tracker_model.

Drop-in replacement for `tracker_model.memory_attention.forward`. Routes the
forward through ONNX Runtime's MIGraphX execution provider with a padded
FIXED-shape ONNX (`memory_attention_fixed_S7_P64.onnx` produced by
`export/tracker_modules/export_memory_attention_padded.py`):

    spatial slots: 7  → 7 * 1296 = 9072 spatial tokens at 504 px (steady state)
    pointer tokens: 64 (= max_object_pointers_in_encoder * 4, cap)
    total memory: 9136 tokens

Shape-specialization rule:
    The shim loads each available S1..S10 sibling graph and selects the exact
    spatial-memory shape at runtime. Callers that pass required_spatial_slots
    and allow_pytorch_fallback=False (the live runtime) get an error instead of
    a PyTorch forward for missing shapes. Pointer tokens are padded or
    truncated to the compiled K.

Why fixed-shape pad rather than dynamic shape:
- Dynamic ONNX through ORT MIG EP recompiles per shape. The build therefore
  emits ten bounded spatial shapes and a fixed pointer-token capacity.
- Direct migraphx.parse_onnx + quantize_fp16 has the FP16 attention
  numerical bug (Finding #8 / detr_encoder analog).
"""
from __future__ import annotations

from collections.abc import Iterable
import logging
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from .ort_gpu_io import GpuIoExecutionError, run_float32_gpu
from .ort_plugin_ep import create_mig_session


# ----- shape constants -----
# K is the pointer-token cap. At 504 px the MIGraphX attention kernels show no
# latency cliff for K=4..64, so the build uses K=64 (16 objects). The runtime
# adapts to whatever K is baked into the loaded ONNX.
DEFAULT_PTR_TOKENS = 64  # only the export-script default; runtime reads from ONNX

_LOGGER = logging.getLogger(__name__)


class MIGMemoryAttention(nn.Module):
    """ORT MIG EP shim for tracker_model.memory_attention.

    Shape parameters (HW per spatial frame, K = ptr-token cap) are inferred
    from the ONNX session's input shapes. All sibling files with the same
    pointer capacity are loaded as exact-shape specializations. This covers
    both the seven non-conditioning memory slots and additional conditioning
    frames without falling back to PyTorch.
    """

    def __init__(
        self,
        onnx_path: Path,
        original_forward,
        ort_cache_dir: Path | None = None,
        *,
        required_spatial_slots: Iterable[int] | None = None,
        allow_pytorch_fallback: bool = True,
    ):
        super().__init__()
        # `original_forward` is the BOUND .forward method captured BEFORE we
        # monkey-patch the module's forward. Storing the module + going through
        # __call__ recurses (because the patched forward calls us again).
        self._original_forward = original_forward
        onnx_path = Path(onnx_path)
        cache_dir = Path(ort_cache_dir) if ort_cache_dir else (
            onnx_path.parent / 'ort_cache_mem_attn'
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir
        match = re.search(r'_S(\d+)_P(\d+)\.onnx$', onnx_path.name)
        if match is None:
            raise ValueError(
                f'Cannot infer memory-attention shape from {onnx_path.name}; '
                'expected memory_attention_fixed_S<slots>_P<pointers>.onnx'
            )
        base_slots, self.ptr_tokens = map(int, match.groups())

        self._sessions = {}
        pattern = f'memory_attention_fixed_S*_P{self.ptr_tokens}.onnx'
        for path in sorted(onnx_path.parent.glob(pattern)):
            candidate = re.search(r'_S(\d+)_P(\d+)\.onnx$', path.name)
            if candidate is None:
                continue
            slots, pointer_tokens = map(int, candidate.groups())
            if pointer_tokens == self.ptr_tokens:
                self._sessions[slots] = self._load_session(path)

        if base_slots not in self._sessions:
            raise FileNotFoundError(onnx_path)
        required_slots = (
            tuple(sorted({int(slot) for slot in required_spatial_slots}))
            if required_spatial_slots is not None
            else None
        )
        if required_slots is not None:
            if not required_slots or required_slots[0] < 1:
                raise ValueError(
                    f'invalid required memory-attention slots: {required_slots}'
                )
            missing_slots = sorted(set(required_slots) - self._sessions.keys())
            if missing_slots:
                raise RuntimeError(
                    'incomplete MIG memory-attention artifact coverage: '
                    f'required S={list(required_slots)}, '
                    f'available S={sorted(self._sessions)}, '
                    f'missing S={missing_slots}; refusing silent PyTorch fallback'
                )
            non_mig_slots = [
                slot for slot in required_slots if not self._sessions[slot][3]
            ]
            if non_mig_slots:
                raise RuntimeError(
                    'MIG memory-attention sessions did not select '
                    f'MIGraphXExecutionProvider for S={non_mig_slots}'
                )
        self.session = self._sessions[base_slots][0]
        self.required_spatial_slots = required_slots
        self.allow_pytorch_fallback = bool(allow_pytorch_fallback)

        # Infer HW from the canonical largest-slot session.
        in_shapes = {x.name: x.shape for x in self.session.get_inputs()}
        self.HW = int(in_shapes["current_vision_features"][0])
        total_mem = int(in_shapes["memory"][0])
        expected_total = base_slots * self.HW + self.ptr_tokens
        if total_mem != expected_total:
            raise RuntimeError(
                f'ONNX memory length {total_mem} != expected {expected_total} '
                f'(HW={self.HW}, S={base_slots}, K={self.ptr_tokens})'
            )
        print(
            '  memory_attention shape sessions ready: '
            f'S={sorted(self._sessions)} (HW={self.HW}, K={self.ptr_tokens})'
        )

        # Stats
        self._mig_calls = 0
        self._pt_fallback_calls = 0
        self._gpu_io_disabled_slots = set()
        self._warned_fallback_reasons = set()

    def _load_session(self, path: Path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print(
            f"  memory_attention S{path.stem.split('_S')[-1].split('_')[0]}: "
            f'loading {path.name} ...'
        )
        t0 = time.perf_counter()
        session = create_mig_session(path, opts, self._cache_dir)
        elapsed = time.perf_counter() - t0
        provider = session.get_providers()[0]
        output = session.get_outputs()[0]
        output_shape = tuple(int(dim) for dim in output.shape)
        print(f'    ready in {elapsed:.1f}s on {provider}')
        return session, output.name, output_shape, provider == 'MIGraphXExecutionProvider'

    def forward(
        self,
        current_vision_features,
        memory,
        current_vision_position_embeddings=None,
        memory_posision_embeddings=None,
        num_object_pointer_tokens: int = 0,
    ):
        spatial_part = memory.shape[0] - num_object_pointer_tokens
        spatial_slots, remainder = divmod(spatial_part, self.HW)
        session_info = self._sessions.get(spatial_slots)
        # Each exact spatial shape has its own compiled session. Pointer count
        # may exceed self.ptr_tokens, in which case the oldest pointers are
        # truncated as before.
        ok = (remainder == 0
              and session_info is not None
              and num_object_pointer_tokens >= 0
              and current_vision_features.shape[0] == self.HW)

        if not ok:
            self._pt_fallback_calls += 1
            reasons = []
            if remainder != 0:
                reasons.append(
                    f'spatial token count {spatial_part} is not divisible by HW={self.HW}'
                )
            if session_info is None:
                reasons.append(
                    f'missing S{spatial_slots} specialization; '
                    f'available S={sorted(self._sessions)}'
                )
            if num_object_pointer_tokens < 0:
                reasons.append(
                    f'negative pointer-token count {num_object_pointer_tokens}'
                )
            if current_vision_features.shape[0] != self.HW:
                reasons.append(
                    'current vision token count '
                    f'{current_vision_features.shape[0]} != HW={self.HW}'
                )
            detail = '; '.join(reasons) or 'unknown shape-contract mismatch'
            message = (
                'MIG memory_attention cannot serve this call: '
                f'{detail}. PyTorch fallback count={self._pt_fallback_calls}'
            )
            if not self.allow_pytorch_fallback:
                raise RuntimeError(message + '; fallback is disabled')
            warning_key = (spatial_slots, remainder, num_object_pointer_tokens,
                           int(current_vision_features.shape[0]))
            if warning_key not in self._warned_fallback_reasons:
                _LOGGER.warning(message)
                self._warned_fallback_reasons.add(warning_key)
            return self._original_forward(
                current_vision_features=current_vision_features,
                memory=memory,
                current_vision_position_embeddings=current_vision_position_embeddings,
                memory_posision_embeddings=memory_posision_embeddings,
                num_object_pointer_tokens=num_object_pointer_tokens,
            )

        self._mig_calls += 1
        device = current_vision_features.device
        dtype = current_vision_features.dtype
        session, output_name, output_shape, gpu_io_enabled = session_info

        # Object pointers are at the END of the memory tensor:
        #   [spatial(spatial_part) | actual_ptrs(N)]
        # We need exactly self.ptr_tokens slots:
        #   N <= ptr_tokens:  pad with zeros (no info loss)
        #   N >  ptr_tokens:  keep conditioning and newest temporal pointers
        #                      (HF orders them from newest to oldest)
        spatial = memory[:spatial_part]
        spatial_pos = memory_posision_embeddings[:spatial_part]
        ptrs = memory[spatial_part:]
        ptrs_pos = memory_posision_embeddings[spatial_part:]

        if num_object_pointer_tokens <= self.ptr_tokens:
            pad_n = self.ptr_tokens - num_object_pointer_tokens
            if pad_n > 0:
                zero_pad = torch.zeros(pad_n, 1, 64, dtype=memory.dtype, device=memory.device)
                ptrs = torch.cat([ptrs, zero_pad], dim=0)
                ptrs_pos = torch.cat([ptrs_pos, zero_pad], dim=0)
        else:
            # HF orders conditioning pointers first, followed by temporal
            # pointers from newest to oldest, so retain the first K tokens.
            ptrs = ptrs[:self.ptr_tokens]
            ptrs_pos = ptrs_pos[:self.ptr_tokens]

        memory_padded = torch.cat([spatial, ptrs], dim=0)
        mem_pos_padded = torch.cat([spatial_pos, ptrs_pos], dim=0)

        ort_inputs = {
            'current_vision_features': current_vision_features,
            'memory': memory_padded,
            'current_vis_pos_embed': current_vision_position_embeddings,
            'memory_pos_embed': mem_pos_padded,
        }
        out_4d = None
        if (gpu_io_enabled
                and spatial_slots not in self._gpu_io_disabled_slots
                and device.type == 'cuda'):
            try:
                out_4d = run_float32_gpu(
                    session,
                    ort_inputs,
                    output_name,
                    output_shape,
                )
            except GpuIoExecutionError:
                raise
            except Exception as exc:
                print(
                    f'  [MIGMemoryAttention] S{spatial_slots} GPU I/O '
                    f'binding disabled: {exc}'
                )
                self._gpu_io_disabled_slots.add(spatial_slots)
        if out_4d is None:
            out_np = session.run(None, {
                name: tensor.detach().float().cpu().numpy()
                for name, tensor in ort_inputs.items()
            })
            out_4d = torch.from_numpy(out_np[0]).to(device=device)

        # ONNX exports `conditioned_features` as (1, 256, H, W); PT returns
        # (1, 1, HW, 256). Caller (_prepare_memory_conditioned_features) does:
        #   .squeeze(1).permute(0,2,1).view(B, C, H, W)
        # so we must return the (1, 1, HW, 256) shape PT does.
        out_4d = out_4d.flatten(2).permute(0, 2, 1).unsqueeze(0)
        return out_4d.to(device=device, dtype=dtype)


def patch_sam3_video_model_memory_attention(
    model,
    onnx_path: Path,
    *,
    required_spatial_slots: Iterable[int] | None = None,
    allow_pytorch_fallback: bool = True,
) -> None:
    """Hot-patch `model.tracker_model.memory_attention.forward` in place.

    We do NOT swap the whole module — that would break parameter ownership
    and any nn.Module child checks elsewhere. Instead we capture the original
    bound forward, build a shim that knows how to call it, and rebind
    `.forward` to the shim.
    """
    trk = model.tracker_model
    original_forward = trk.memory_attention.forward  # captured BEFORE monkey-patch
    shim = MIGMemoryAttention(
        Path(onnx_path),
        original_forward,
        required_spatial_slots=required_spatial_slots,
        allow_pytorch_fallback=allow_pytorch_fallback,
    )
    trk.memory_attention.forward = shim.forward
    trk.memory_attention._mig_shim = shim  # keep ref alive
