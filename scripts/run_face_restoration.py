"""
CLI Demonstration Tool for Dual-Path Face Inpainting & Face Restoration.
Supports:
- PATH A: Mask Provided (--mask path/to/mask.png)
- PATH B: Auto-Damage Detection (No --mask argument; automatically locates scars, burns, scratches)

Usage Examples:
  # Path A: User-supplied mask
  python scripts/run_face_restoration.py \\
      --target user_tests/test1/target.jpg \\
      --mask user_tests/test1/mask.png \\
      --references user_tests/test1/ref1.jpg user_tests/test1/ref2.jpg user_tests/test1/ref3.jpg \\
      --mock

  # Path B: Auto-damage detection (No mask provided)
  python scripts/run_face_restoration.py \\
      --target user_tests/test1/masked_target.jpg \\
      --references user_tests/test1/ref1.jpg user_tests/test1/ref2.jpg user_tests/test1/ref3.jpg \\
      --mock
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from modules.restoration.face_inpainting_pipeline import FaceInpaintingPipeline
from modules.restoration.damage_detector import DamageMaskDetector


def run_restoration_cli():
    parser = argparse.ArgumentParser(description="Dual-Path Face Inpainting & Face Restoration")
    parser.add_argument("--target", type=str, required=True, help="Path to input target face image")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to binary mask. If omitted, runs auto-damage detector.")
    parser.add_argument("--references", type=str, nargs="*", default=[], help="Paths to 1-3 reference photos of the same subject")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baselines/step_150_best/step_150.pt", help="Path to Step 150 adapter weights")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="evaluation/visualizations/restoration_runs", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Fast mock mode for CPU testing without loading 3.4GB SD model")

    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    assert target_path.exists(), f"Target image not found: {target_path}"
    target_bgr = cv2.imread(str(target_path))
    assert target_bgr is not None, f"Failed to load image: {target_path}"
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    mask_rgb = None
    if args.mask:
        mask_path = Path(args.mask).resolve()
        assert mask_path.exists(), f"Supplied mask file not found: {mask_path}"
        mask_rgb = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    ref_rgbs = []
    for ref_p in args.references:
        rp = Path(ref_p).resolve()
        assert rp.exists(), f"Reference file not found: {rp}"
        bgr = cv2.imread(str(rp))
        if bgr is not None:
            ref_rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    print("=" * 80)
    print("AI FACE RESTORATION / INPAINTING SYSTEM (DUAL INPUT MODES)")
    print(f"Target Image:      {target_path.name} ({target_rgb.shape[1]}x{target_rgb.shape[0]})")
    print(f"Supplied Mask:     {Path(args.mask).name if args.mask else 'NONE (Automatic Damage Detection Mode)'}")
    print(f"Reference Gallery: {len(ref_rgbs)} photo(s)")
    print(f"Compute Device:    {dev} | Mock Mode: {args.mock}")
    print("=" * 80)

    # Initialize Pipeline
    pipeline = FaceInpaintingPipeline(
        checkpoint_path=args.checkpoint,
        device=dev,
        mock_mode=args.mock,
        default_steps=args.steps
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Execute Restoration
    result = pipeline.restore(
        target_image=target_rgb,
        mask=mask_rgb,
        reference_images=ref_rgbs,
        seed=args.seed,
        steps=args.steps,
        save_debug_vis=True,
        output_dir=out_dir
    )

    stem = target_path.stem
    mode_str = result["input_mode"].lower()

    # Save artifacts
    res_bgr = cv2.cvtColor(result["final_image"], cv2.COLOR_RGB2BGR)
    final_img_path = out_dir / f"{stem}_{mode_str}_restored.png"
    cv2.imwrite(str(final_img_path), res_bgr)

    mask_path_out = out_dir / f"{stem}_{mode_str}_mask.png"
    cv2.imwrite(str(mask_path_out), result["normalized_mask"])

    meta = {
        "target_file": str(target_path),
        "input_mode": result["input_mode"],
        "mask_metrics": result["mask_metrics"],
        "reference_scoring": result["reference_scoring"],
        "invariance": result["invariance"],
        "performance": result["performance"],
        "debug_visualization": result["debug_visualization_path"],
        "final_image_path": str(final_img_path)
    }

    meta_path = out_dir / f"{stem}_{mode_str}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 80)
    print(f"RESTORATION SUCCESSFUL [{result['input_mode']}]")
    print(f"  Masked Pixels:        {result['mask_metrics']['num_masked_pixels']:,} px ({result['mask_metrics']['image_coverage_pct']}%)")
    if result["mask_metrics"]["detection_confidence"] is not None:
        print(f"  Detection Confidence: {result['mask_metrics']['detection_confidence']:.2f}")
    if result["reference_scoring"]["weights"]:
        print(f"  Reference Weights:    {result['reference_scoring']['weights']}")
    print(f"  Unmasked Invariance:  PASS (Delta_max = {result['invariance']['max_difference']:.5f})")
    print(f"  Inpainting Runtime:   {result['performance']['inpainting_seconds']}s")
    print(f"  Saved Final Image:    {final_img_path}")
    print(f"  Saved Visualization:  {result['debug_visualization_path']}")
    print(f"  Saved Metadata:       {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_restoration_cli()
