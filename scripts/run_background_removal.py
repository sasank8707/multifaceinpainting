"""
CLI Execution Script for Mode 2: Background Removal.
Takes any input image, extracts foreground person/object, and saves a transparent RGBA PNG
with a visual side-by-side comparison on a checkerboard background.
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

from modules.background_removal.background_remover import BackgroundRemover


def create_checkerboard(width: int, height: int, check_size: int = 16) -> np.ndarray:
    """Generates a standard checkerboard background for visualizing transparency."""
    y, x = np.indices((height, width))
    checker = ((x // check_size) + (y // check_size)) % 2
    board = np.where(checker[:, :, np.newaxis] == 0, 204, 255).astype(np.uint8)
    return np.repeat(board, 3, axis=2) if board.ndim == 2 else board


def composite_on_checkerboard(rgba: np.ndarray, check_size: int = 16) -> np.ndarray:
    """Composites an RGBA image onto a checkerboard background."""
    h, w = rgba.shape[:2]
    checker = create_checkerboard(w, h, check_size=check_size)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
    blended = rgb * alpha + checker.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Mode 2: Background Removal CLI")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="user_tests/natural_damage/case01/target.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="evaluation/visualizations/background_removal",
        help="Directory to save output artifacts"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Probability threshold for foreground segmentation"
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    assert input_path.exists(), f"Input file not found: {input_path}"

    out_dir = PROJECT_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODE 2: BACKGROUND REMOVAL (INDEPENDENT PIPELINE)")
    print("=" * 80)
    print(f"Input image: {input_path}")
    print(f"Output directory: {out_dir}")

    remover = BackgroundRemover(threshold=args.threshold, edge_refinement=True)
    res = remover.remove_background(input_path, return_dict=True)

    rgba = res["rgba_image"]
    alpha = res["alpha_mask"]
    orig_rgb = res["original_rgb"]
    h, w = rgba.shape[:2]

    print(f"\nExecution successful in {res['runtime_ms']:.1f}ms on {res['device']}!")
    print(f"Image dimensions: {w}x{h}")
    print(f"Foreground detected: {res['foreground_pixels']:,} px ({res['foreground_percentage']}%)")

    # 1. Save transparent PNG
    transparent_path = out_dir / f"{input_path.stem}_transparent.png"
    remover.save_result(rgba, transparent_path)
    print(f"Saved transparent PNG: {transparent_path}")

    # 2. Save alpha mask
    alpha_path = out_dir / f"{input_path.stem}_alpha_mask.png"
    Image.fromarray(alpha).save(alpha_path)
    print(f"Saved alpha mask: {alpha_path}")

    # 3. Save side-by-side visualization on checkerboard
    checker_composite = composite_on_checkerboard(rgba, check_size=16)
    divider = np.ones((h, 4, 3), dtype=np.uint8) * 128
    comparison_strip = np.hstack([orig_rgb, divider, checker_composite])
    
    comp_path = out_dir / f"{input_path.stem}_side_by_side.png"
    Image.fromarray(comparison_strip).save(comp_path)
    print(f"Saved comparison on checkerboard: {comp_path}")

    print("=" * 80)
    print("MODE 2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
