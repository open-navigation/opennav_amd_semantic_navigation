#!/usr/bin/env python3
# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone SAM3 labeling script for example images.
# Requires opennav_sam3_inference to be importable (pip install -e opennav_sam3_inference/).
#
# Usage:
#   python example/label_images.py --checkpoint models/sam3 --onnx-dir onnx_files_504

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

# Filename -> list of text prompts to segment with.
IMAGE_PROMPTS = {
    "cables.jpeg":                 ["Cables and chargers along the floor"], # TODO try small objects on the ground
    "cables2.jpg":                 ["cables on the floor"], # TODO try small objects on the ground
    "forklift_ground_forks.jpg":   ["forklift forks", "ground or floor"],
    "glass_wall.jpg":              ["glass wall or door", "carpet", "wood, concrete, or tile floor"],
    "lego.jpg":                    ["lego or legos", "carpet"], # TODO try small objects on the ground
    "overhang_sign.jpg":           ["hanging or overhanging sign", "shelves", "floor or walkway"],
    "pothole.jpg":                 ["pothole", "road"],
    "puddle.png":                  ["puddle or water", "road"],
    "tall_grass_push_through.jpg": ["tall grass a robot can drive over"],
    "shovel.jpg":                  ["tools on the ground", "hole", "pile of dirt", "navigable flat ground"], #TODO last one
}


def _render_overlay(rgb: np.ndarray, instances, alpha: float = 0.5) -> np.ndarray:
    """Draw colored semi-transparent masks on an RGB image."""
    out = rgb.copy()
    for mask, class_id in instances:
        color = np.random.default_rng(class_id).integers(64, 256, size=3, dtype=np.uint8)
        out[mask] = ((1.0 - alpha) * out[mask] + alpha * color).astype(np.uint8)
    return out


def process_images(args):
    from opennav_sam3_inference.tracker.live_inference import SAM3Live

    example_dir = Path(__file__).resolve().parent
    input_dir = example_dir / "raw_data"
    output_dir = example_dir / "processed_data"
    os.makedirs(output_dir, exist_ok=True)

    sorted_items = sorted(IMAGE_PROMPTS.items())
    first_prompts = sorted_items[0][1]

    print(f"Loading SAM3 from {args.checkpoint} on {args.device} (imgsz={args.imgsz}) ...")
    live = SAM3Live(
        checkpoint=args.checkpoint,
        prompts=first_prompts,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        dtype=torch.float16,
        device=args.device,
        mig=not args.no_mig,
        max_objects_per_prompt=args.max_objects_per_prompt,
        redetect_every=1,
    )

    processed = 0
    for filename, prompts in sorted_items:
        img_path = input_dir / filename
        if not img_path.exists():
            print(f"  SKIP {filename}: file not found")
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  SKIP {filename}: failed to load")
            continue

        live.reset_prompts(prompts)
        result = live.infer(bgr, full_detection=True)

        prompt_to_class_id = {p: i + 1 for i, p in enumerate(prompts)}
        H, W = bgr.shape[:2]
        label_map = np.zeros((H, W), dtype=np.uint8)
        score_map = np.zeros((H, W), dtype=np.float32)
        instances = []

        for prompt_text, obj_ids in result["prompt_to_obj_ids"].items():
            class_id = prompt_to_class_id.get(prompt_text, 0)
            for oid in obj_ids:
                score = result["scores"][oid]
                if score < args.score_threshold:
                    continue
                mask = result["masks"][oid]
                update = mask & (score > score_map)
                label_map[update] = class_id
                score_map[update] = score
                instances.append((mask, class_id))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay = _render_overlay(rgb, instances)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / filename), overlay_bgr)
        processed += 1

        print(f"  {filename}: {len(instances)} instances from prompts {prompts}")

    print(f"\nDone. {processed} images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Label example images with SAM3")
    parser.add_argument("--checkpoint", default="models/sam3",
                        help="HuggingFace model directory (default: models/sam3)")
    parser.add_argument("--onnx-dir", default="onnx_files_504",
                        help="ONNX artifact directory (default: onnx_files_504)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda)")
    parser.add_argument("--imgsz", type=int, default=504,
                        help="Image resolution, 504 or 1008 (default: 504)")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Minimum detection score (default: 0.5)")
    parser.add_argument("--no-mig", action="store_true",
                        help="Disable MIGraphX acceleration")
    parser.add_argument("--max-objects-per-prompt", type=int, default=5,
                        help="Max tracked objects per prompt (default: 5)")
    args = parser.parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
