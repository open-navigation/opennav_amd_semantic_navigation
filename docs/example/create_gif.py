#!/usr/bin/env python3
# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Create a GIF from all images in a directory.
#
# Usage:
#   python example/create_gif.py                          # from processed_data/
#   python example/create_gif.py --input-dir raw_data     # from raw_data/

import argparse
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def create_gif(input_dir: Path, output_path: Path, duration_ms: int = 1000):
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"No images found in {input_dir}")
        return

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((640, 480), Image.LANCZOS)
        frames.append(img)
        print(f"  Added {p.name}")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"\nSaved {len(frames)}-frame GIF to {output_path}")


def main():
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create a GIF from a directory of images")
    parser.add_argument("--input-dir", default="processed_data",
                        help="Input directory relative to example/ (default: processed_data)")
    parser.add_argument("--output", default=str(example_dir / "data.gif"),
                        help="Output GIF path (default: example/data.gif)")
    parser.add_argument("--duration", type=int, default=1000,
                        help="Duration per frame in ms (default: 1000)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = example_dir / input_dir

    create_gif(input_dir, Path(args.output), args.duration)


if __name__ == "__main__":
    main()
