#!/usr/bin/env python3
"""Build MIG/ONNX artefacts for SAM3 demo pipelines.

Two pipelines, pick one or both:

  box   — SAM3OnnxTracker (demo.py): tracker modules + MIGraphX backbone
            ~10 min @504px / ~20 min @1008px
  text  — Sam3VideoModel (demo_text.py --mig): detector backbone + DETR encoder
            + padded memory_attention
            ~18 min @504px / ~30 min @1008px

Usage:
  # Box-prompt only (fastest, good first step)
  python export/build.py --pipeline box --imgsz 504

  # Text-prompt MIG only
  python export/build.py --pipeline text --imgsz 504

  # Everything at 504px
  python export/build.py --pipeline all --imgsz 504

  # Both pipelines, both resolutions (~90 min total)
  python export/build.py --pipeline all --imgsz 504 1008

  # Rebuild one stage only
  python export/build.py --pipeline text --imgsz 1008 --steps backbone
  python export/build.py --pipeline box  --imgsz 504  --force

Each step skips if its output already exists. Safe to re-run after interruption.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent

G  = "\033[32m"
B  = "\033[1m"
NC = "\033[0m"
Y  = "\033[33m"


def banner(msg: str) -> None:
    print(f"\n{B}{'═'*62}{NC}")
    print(f"{B}  {msg}{NC}")
    print(f"{B}{'═'*62}{NC}")


def step_header(msg: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  {msg}")
    print(f"{'─'*62}")


def run(cmd: list, label: str) -> bool:
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=WORKSPACE)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"\n  {Y}✗ FAILED{NC} after {elapsed:.0f}s (exit {r.returncode})")
        return False
    print(f"  {G}✓{NC} done in {elapsed:.0f}s")
    return True


def skip_if_exists(path: Path, label: str, force: bool) -> bool:
    if not force and path.exists():
        print(f"  skip  {label}  ({path.relative_to(WORKSPACE)})")
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Box-prompt pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_box(imgsz: int, args) -> bool:
    onnx_dir  = WORKSPACE / f"onnx_files_{imgsz}"
    mod_dir   = onnx_dir / "tracker_modules"
    bb_dir    = onnx_dir / "backbone_tracker"
    steps     = set(args.steps)
    run_all   = "all" in steps
    ok        = True

    step_header(f"Box-prompt @ {imgsz}px  →  {onnx_dir.name}/")

    # Step 1: tracker modules (memory_attention, mask_decoder_*, memory_encoder)
    if run_all or "tracker_modules" in steps:
        sentinel = mod_dir / "temporal_pe.npy"
        if not skip_if_exists(sentinel, "tracker modules", args.force):
            ok = ok and run([
                sys.executable,
                "export/tracker_modules/export_tracker_modules.py",
                "--checkpoint", str(args.checkpoint),
                "--imgsz", str(imgsz),
                "--onnx-dir", str(onnx_dir),
            ], "Export tracker ONNX modules")

    # Step 2: backbone export
    if run_all or "backbone" in steps:
        out = bb_dir / "single_fp32.onnx"
        if not skip_if_exists(out, "backbone ONNX", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/export_backbone_single.py",
                "--imgsz", str(imgsz),
                "--backbone-source", "tracker",
                "--checkpoint", str(args.checkpoint),
            ], "Export backbone ONNX (tracker FPN)")

        # Step 3: simplify
        out = bb_dir / "single_simplified.onnx"
        if not skip_if_exists(out, "simplified backbone", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/simplify_backbone.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "tracker",
            ], "Simplify backbone (onnxsim)")

        # Step 4: compile .mxr
        out = bb_dir / "tuned.mxr"
        if not skip_if_exists(out, "tuned.mxr", args.force):
            ok = ok and run([
                sys.executable,
                "export/backbone/compile_backbone_mxr.py",
                "--onnx-dir", str(onnx_dir),
                "--imgsz", str(imgsz),
                "--backbone-source", "tracker",
                "--skip-verify",
            ], f"Compile backbone .mxr  (~3 min @504 / ~9 min @1008)")

    # Step 5: prewarm direct-MIG caches
    if run_all or "prewarm" in steps:
        sentinel = onnx_dir / "tracker_modules" / "mxr_cache" / "dec_prop_fp32.mxr"
        if not skip_if_exists(sentinel, "prewarm cache", args.force):
            ok = ok and run([
                sys.executable,
                "export/tracker_modules/prewarm_ort_cache.py",
                "--onnx-dir", str(onnx_dir),
            ], "Pre-warm direct-MIG caches")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Text-prompt pipeline  (delegates to build_text_prompt_mig.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_text(imgsz: int, args) -> bool:
    step_header(f"Text-prompt MIG @ {imgsz}px  →  onnx_files_{imgsz}/")

    cmd = [
        sys.executable,
        "export/build_text_prompt_mig.py",
        "--imgsz", str(imgsz),
        "--checkpoint", str(args.checkpoint),
    ]
    # Only forward --ptr-tokens if user explicitly set it; otherwise let
    # build_text_prompt_mig.py pick the per-imgsz default (504→64, 1008→48).
    if args.ptr_tokens is not None:
        cmd += ["--ptr-tokens", str(args.ptr_tokens)]
    if args.force:
        cmd.append("--force")
    if "all" not in args.steps:
        cmd += ["--steps"] + args.steps

    return run(cmd, f"Build text-prompt MIG artefacts @{imgsz}px")


# ─────────────────────────────────────────────────────────────────────────────
# Final demo hints
# ─────────────────────────────────────────────────────────────────────────────

LD = (f"LD_PRELOAD=" + _rocm_base + "/lib/libmigraphx_c.so.3:"
      f"{_rocm_base}/lib/migraphx/lib/libmigraphx.so.2016000.0")

def print_hints(pipeline: str, imgsz_list: list[int], checkpoint: Path) -> None:
    banner("Done — next steps")
    first = imgsz_list[0]

    if pipeline in ("box", "all"):
        print(f"""
{B}Box-prompt demo:{NC}
  python demo.py --checkpoint {checkpoint} --onnx-dir onnx_files_{first} \\
      --video YOUR_VIDEO.mp4 --box x1,y1,x2,y2
""")

    if pipeline in ("text", "all"):
        print(f"""\
{B}Text-prompt demo (MIG-accelerated):{NC}
  {LD} \\
      python demo_text.py --checkpoint {checkpoint} \\
          --video YOUR_VIDEO.mp4 --text "object" \\
          --imgsz {first} --mig --onnx-dir onnx_files_{first}
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pipeline", choices=["box", "text", "all"], default="box",
                   help="Which pipeline to build (default: box)")
    p.add_argument("--imgsz", type=int, nargs="+", default=[504],
                   choices=[504, 1008],
                   help="Resolution(s) to build (default: 504)")
    p.add_argument("--checkpoint", type=Path, default=WORKSPACE / "model/sam3")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if output files already exist")
    p.add_argument("--steps", nargs="+", default=["all"],
                   help=("Limit to specific stages. Box: tracker_modules, backbone, prewarm. "
                         "Text: backbone, detr_encoder, memory_attention."))
    p.add_argument("--ptr-tokens", type=int, default=None,  # 504→64, 1008→48
                   help="Pointer token slots for memory_attention. Default: 64 at 504px, 48 at 1008px (highest safe K per kernel cliff). Set explicitly to override.")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.perf_counter()

    banner(f"SAM3 artifact build  |  pipeline={args.pipeline}  "
           f"imgsz={args.imgsz}  checkpoint={args.checkpoint.name}")

    all_ok = True
    for imgsz in args.imgsz:
        if args.pipeline in ("box", "all"):
            all_ok = build_box(imgsz, args) and all_ok
        if args.pipeline in ("text", "all"):
            all_ok = build_text(imgsz, args) and all_ok

    elapsed = time.perf_counter() - t_start
    status = f"{G}ALL OK{NC}" if all_ok else f"{Y}SOME STEPS FAILED{NC}"
    banner(f"Total: {elapsed/60:.1f} min  |  {status}")

    if all_ok:
        print_hints(args.pipeline, args.imgsz, args.checkpoint)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
