#!/usr/bin/env python3
"""Build all MIG artefacts needed for `demo_text.py --mig` in one command.

Runs the text-prompt MIG export pipeline:
  1. export_backbone_single   — ViT backbone ONNX (FP32, 4 FPN + last_hidden_state)
  2. simplify_backbone        — onnxsim graph simplification
  3. compile_backbone_mxr     — MIGraphX kernel autotune → tuned.mxr  (~12 min @1008)
  4. export_detr_encoder      — DETR encoder ONNX + onnxsim
  5. export_memory_attention_padded — padded memory_attention ONNX (ORT MIG EP)
  6. export_fixed_detr_decoder — fixed 504px decoder ONNX
  7. compile_fixed_detr_decoder — direct-I/O decoder MXR + checksum

Each step skips if its output file already exists (use --force to rebuild).

Usage:
  # Build 504px artefacts (~15 min)
  python export/build_text_prompt_mig.py --imgsz 504

  # Build both resolutions
  python export/build_text_prompt_mig.py --imgsz 504 1008

  # Rebuild backbone only (force)
  python export/build_text_prompt_mig.py --imgsz 504 --force --steps backbone
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--imgsz", type=int, nargs="+", default=[504],
                   choices=[504, 1008],
                   help="Resolution(s) to build (default: 504). Pass both to build all.")
    p.add_argument("--checkpoint", type=Path,
                   default=WORKSPACE / "/sam3",
                   help="Path to /sam3 (default: /sam3)")
    p.add_argument("--onnx-root", type=Path, default=None,
                   help="Root for onnx_files_<imgsz>/ dirs (default: project root)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if output files already exist")
    p.add_argument("--steps", nargs="+",
                   choices=["backbone", "detr_encoder", "memory_attention",
                            "fixed_decoder", "all"],
                   default=["all"],
                   help="Which steps to run (default: all)")
    p.add_argument("--ptr-tokens", type=int, default=None,
                   help="Pointer token slots for memory_attention. "
                        "Default: 64 at 504px, 48 at 1008px (highest safe K per kernel cliff). "
                        "Set explicitly to override.")
    return p.parse_args()


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess; return True on success."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=WORKSPACE)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED after {elapsed:.0f}s (exit {result.returncode})")
        return False
    print(f"\n  ✓ done in {elapsed:.0f}s")
    return True


def _display(path: Path) -> Path | str:
    """Path relative to WORKSPACE when possible, else the absolute path.

    Output dirs can live outside the repo (e.g. --onnx-root pointing at an
    external models directory), so relative_to() may not apply.
    """
    try:
        return path.relative_to(WORKSPACE)
    except ValueError:
        return path


def exists(path: Path, label: str, force: bool) -> bool:
    """Return True (skip) if path exists and not forcing."""
    if not force and path.exists():
        print(f"  skip: {label} already exists at {_display(path)}")
        return True
    return False


def build_for_imgsz(imgsz: int, args) -> bool:
    onnx_root = args.onnx_root or WORKSPACE
    onnx_dir = onnx_root / f"onnx_files_{imgsz}"
    det_dir = onnx_dir / "backbone_detector"
    mod_dir = onnx_dir / "detector_modules"
    trk_dir = onnx_dir / "tracker_modules"
    fixed_dir = onnx_dir / "detr_decoder_fixed"

    steps = set(args.steps)
    run_all = "all" in steps
    ok = True

    # ── Step 1: export backbone ONNX ─────────────────────────────────────
    if run_all or "backbone" in steps:
        out = det_dir / "single_fp32.onnx"
        if not exists(out, "backbone ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/export_backbone_single.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "detector",
                "--checkpoint", str(args.checkpoint),
            ], f"[1/5] Export backbone ONNX @{imgsz}px")

        # ── Step 2: simplify backbone ─────────────────────────────────────
        out = det_dir / "single_simplified.onnx"
        if not exists(out, "simplified backbone ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/simplify_backbone.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "detector",
            ], f"[2/5] Simplify backbone @{imgsz}px")

        # ── Step 3: compile .mxr ─────────────────────────────────────────
        out = det_dir / "tuned.mxr"
        if not exists(out, "tuned.mxr", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/compile_backbone_mxr.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "detector",
                "--skip-verify",
            ], f"[3/5] Compile backbone .mxr @{imgsz}px  (~12 min at 1008px)")

    # ── Step 4: export DETR encoder ───────────────────────────────────────
    if run_all or "detr_encoder" in steps:
        out = mod_dir / "detr_encoder_simplified.onnx"
        if not exists(out, "detr_encoder ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/detector/export_detr_encoder.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--checkpoint", str(args.checkpoint),
            ], f"[4/5] Export DETR encoder @{imgsz}px")

    # ── Step 5: export memory_attention ──────────────────────────────────
    if run_all or "memory_attention" in steps:
        # Resolve None default per imgsz (504→64, 1008→48 — kernel cliff aware)
        ptr_tokens = args.ptr_tokens if args.ptr_tokens is not None else {504: 64, 1008: 48}.get(imgsz, 32)
        name = f"memory_attention_fixed_S7_P{ptr_tokens}.onnx"
        out = trk_dir / name
        if not exists(out, name, args.force):
            ok = ok and run([
                sys.executable,
                "export/tracker_modules/export_memory_attention_padded.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--ptr-tokens", str(ptr_tokens),
                "--checkpoint", str(args.checkpoint),
            ], f"[5/5] Export memory_attention (S7_P{ptr_tokens}) @{imgsz}px")

    if run_all or "fixed_decoder" in steps:
        if imgsz != 504:
            if "fixed_decoder" in steps:
                print("  fixed decoder currently supports only 504px")
                return False
            print("  skip: fixed decoder currently supports only 504px")
        else:
            fixed_onnx = fixed_dir / "detr_decoder_fixed_simplified.onnx"
            if not exists(fixed_onnx, "fixed decoder ONNX", args.force):
                ok = ok and run([
                    sys.executable,
                    "export/detector/export_fixed_detr_decoder.py",
                    "--checkpoint", str(args.checkpoint),
                    "--output-dir", str(fixed_dir),
                ], "[6/7] Export fixed DETR decoder")
            fixed_mxr = fixed_dir / "direct_gpuio.mxr"
            if not exists(fixed_mxr, "fixed decoder MXR", args.force):
                command = [
                    sys.executable,
                    "export/detector/compile_fixed_detr_decoder.py",
                    "--onnx", str(fixed_onnx),
                    "--output", str(fixed_mxr),
                ]
                if args.force:
                    command.append("--force")
                ok = ok and run(command, "[7/7] Compile fixed DETR decoder MXR")

    return ok


def main():
    args = parse_args()
    t_start = time.perf_counter()
    all_ok = True

    for imgsz in args.imgsz:
        print(f"\n{'='*60}")
        print(f"  Building text-prompt MIG artefacts @ {imgsz}px")
        print(f"  checkpoint : {args.checkpoint}")
        print(f"  onnx_files : {args.onnx_root}/onnx_files_{imgsz}")
        print(f"{'='*60}")
        ok = build_for_imgsz(imgsz, args)
        all_ok = all_ok and ok
        status = "✓ complete" if ok else "✗ FAILED"
        print(f"\n  {imgsz}px: {status}")

    total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Total: {total/60:.1f} min  |  {'ALL OK' if all_ok else 'SOME STEPS FAILED'}")
    print(f"{'='*60}")

    if all_ok:
        print("Build complete!")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
