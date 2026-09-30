"""
CLI Execution Script for Mode 3: Deblurring.
Applies edge-preserving deblurring, local contrast restoration, and noise suppression
to any input image, saving the restored image and a side-by-side comparison figure.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.deblurring.deblurrer import ImageDeblurrer


def main():
    parser = argparse.ArgumentParser(description="Mode 3: Deblurring CLI")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="user_tests/natural_damage/case01/target.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="evaluation/visualizations/deblurring",
        help="Directory to save output artifacts"
    )
    parser.add_argument(
        "--strength", "-s",
        type=float,
        default=1.2,
        help="Deblurring intensity multiplier (default 1.2)"
    )
    parser.add_argument(
        "--simulate_blur",
        action="store_true",
        help="If set, applies synthetic motion/defocus blur to input before restoring (for benchmark demo)"
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    assert input_path.exists(), f"Input file not found: {input_path}"

    out_dir = PROJECT_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODE 3: DEBLURRING (INDEPENDENT PIPELINE)")
    print("=" * 80)

    orig_pil = Image.open(input_path).convert("RGB")
    source_img = np.array(orig_pil)

    # Optional blur simulation if requested or if input is pristine
    if args.simulate_blur:
        print("Simulating optical defocus blur (Gaussian kernel 9x9, sigma=2.5)...")
        blurred = cv2.GaussianBlur(source_img, (9, 9), 2.5)
        eval_input = blurred
        prefix = "simulated_blur_"
    else:
        eval_input = source_img
        prefix = ""

    deblurrer = ImageDeblurrer(strength=args.strength, enable_clahe=True)
    res = deblurrer.deblur(eval_input, return_dict=True)

    deblurred = res["deblurred_image"]
    initial_m = res["initial_metrics"]
    final_m = res["final_metrics"]
    h, w = deblurred.shape[:2]

    print(f"\nExecution successful in {res['runtime_ms']:.1f}ms on CPU!")
    print(f"Image dimensions: {w}x{h}")
    print(f"Initial Laplacian Variance: {initial_m['laplacian_variance']:.1f}")
    print(f"Final Laplacian Variance:   {final_m['laplacian_variance']:.1f} ({res['sharpness_gain']}x gain)")

    # 1. Save restored image
    restored_path = out_dir / f"{prefix}{input_path.stem}_deblurred.jpg"
    deblurrer.save_result(deblurred, restored_path)
    print(f"Saved deblurred image: {restored_path}")

    # 2. Save side-by-side comparison
    divider = np.ones((h, 4, 3), dtype=np.uint8) * 128
    comparison_strip = np.hstack([eval_input, divider, deblurred])
    comp_path = out_dir / f"{prefix}{input_path.stem}_comparison.jpg"
    Image.fromarray(comparison_strip).save(comp_path, quality=95)
    print(f"Saved side-by-side comparison: {comp_path}")

    print("=" * 80)
    print("MODE 3 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
