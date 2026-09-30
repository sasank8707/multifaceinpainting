"""
Verification Script for Authentic Full Diffusion Restoration (Path A).
Tests real 50-step diffusion on CPU using Step 150 checkpoint with:
- Input image containing a visible solid-gray painted masked region
- Matching binary mask
- 3 same-person reference photos
- O1 reference scoring + O2 regional identity fusion
- Inward feather r=7 boundary harmonization
- Anti-copy audit (inside-mask regenerated vs outside-mask invariance)
"""

import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import torch

from modules.restoration.face_inpainting_pipeline import FaceInpaintingPipeline
from modules.restoration.boundary_harmonizer import BoundaryHarmonizer, verify_unmasked_invariance


def main():
    print("=" * 80)
    print("AUTHENTIC FULL DIFFUSION RESTORATION TEST (PATH A - 50 DDIM STEPS)")
    print("=" * 80)

    # 1. Load inputs
    clean_target = np.array(Image.open("user_tests/test1/target.jpg").convert("RGB"))
    mask = np.array(Image.open("user_tests/test1/mask.png").convert("L"))
    ref1 = np.array(Image.open("user_tests/test1/ref1.jpg").convert("RGB"))
    ref2 = np.array(Image.open("user_tests/test1/ref2.jpg").convert("RGB"))
    ref3 = np.array(Image.open("user_tests/test1/ref3.jpg").convert("RGB"))

    h, w = clean_target.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask > 127).astype(np.uint8) * 255
    total_masked_pixels = int(np.sum(binary_mask > 0))

    # 2. Create partially masked/damaged face with solid gray region (RGB 128, 128, 128)
    painted_input = clean_target.copy()
    painted_input[binary_mask > 0] = (128, 128, 128)

    out_dir = Path("evaluation/visualizations/full_diffusion_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = out_dir / "01_input_with_gray_mask.png"
    mask_path = out_dir / "02_binary_mask.png"
    Image.fromarray(painted_input).save(input_path)
    Image.fromarray(binary_mask).save(mask_path)
    print(f"Created visible gray-masked input: {input_path} ({w}x{h}, {total_masked_pixels:,} masked pixels)")

    # 3. Initialize authentic pipeline with Step 150 checkpoint
    checkpoint_path = "checkpoints/baselines/step_150_best/step_150.pt"
    assert Path(checkpoint_path).exists(), f"Checkpoint missing: {checkpoint_path}"
    print(f"Loading Step 150 checkpoint: {checkpoint_path}")

    pipeline = FaceInpaintingPipeline(
        checkpoint_path=checkpoint_path,
        mock_mode=False,
        identity_scale=0.10,
        guidance_scale=7.5,
        default_steps=50,
        device="cpu"
    )

    # 4. Run authentic restoration
    print("Executing 50-step diffusion denoising on CPU with 3 references...")
    t0 = time.time()
    res = pipeline.restore(
        target_image=painted_input,
        mask=binary_mask,
        reference_images=[ref1, ref2, ref3],
        seed=42,
        steps=50,
        guidance_scale=7.5,
        identity_scale=0.10,
        save_debug_vis=False
    )
    t1 = time.time()
    runtime = t1 - t0
    print(f"Diffusion execution completed in {runtime:.1f} seconds ({runtime/60:.2f} min)")

    raw_diff = res["final_image"]
    final_mask = res["normalized_mask"]

    # 5. Apply Boundary Harmonization r=7
    harmonizer = BoundaryHarmonizer(feather_radius=7.0, blend_mode="feathered")
    composited, _ = harmonizer.composite(painted_input, raw_diff, final_mask)

    # 6. Audit Outside-Mask Invariance
    passed_inv, max_diff, violated_outside = verify_unmasked_invariance(painted_input, composited, final_mask)

    # 7. Audit Inside-Mask Regeneration (Anti-Copy check against gray input)
    inside_mask = (final_mask > 0)
    inside_diff = np.abs(composited[inside_mask].astype(int) - painted_input[inside_mask].astype(int))
    # Count pixels that differ from the gray input by > 5 in any channel
    inside_changed_pixels = int(np.sum(np.max(inside_diff, axis=-1) > 5))
    regen_ratio = float(inside_changed_pixels) / max(1, total_masked_pixels)

    # Check whether solid gray remains in the center of the patch
    patch_pixels = composited[inside_mask]
    gray_residual = np.sum(np.all(np.abs(patch_pixels - 128) < 3, axis=-1))

    print("\n" + "=" * 80)
    print("AUDIT & VERIFICATION RESULTS:")
    print("=" * 80)
    print(f"Total Masked Pixels:               {total_masked_pixels:,}")
    print(f"Inside-Mask Regenerated Pixels:    {inside_changed_pixels:,} / {total_masked_pixels:,} ({regen_ratio*100:.2f}%)")
    print(f"Gray Residual Pixels (unchanged):  {gray_residual} (0 expected)")
    print(f"Outside-Mask Invariance Passed:    {passed_inv} (Violated: {violated_outside} px, Max Delta: {max_diff})")
    print(f"Reference Weights (O1 Scoring):    {res.get('reference_scoring', {}).get('weights', [])}")
    print(f"Total Runtime:                     {runtime:.2f}s")

    assert regen_ratio >= 0.95, f"Expected >95% masked pixels regenerated, got {regen_ratio*100:.1f}%"
    assert gray_residual < 10, f"Gray mask remained in output! {gray_residual} pixels"
    assert passed_inv is True, f"Outside-mask invariance violated! Max diff: {max_diff}"

    # 8. Save output and side-by-side comparison
    output_path = out_dir / "03_full_diffusion_restored.png"
    Image.fromarray(composited).save(output_path)

    # Create Side-by-Side: [Input Gray Masked | Mask Overlay | Restored Output]
    overlay_input = painted_input.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_input, contours, -1, (0, 230, 255), 2)

    side_by_side = np.hstack([painted_input, overlay_input, composited])
    side_by_side_path = out_dir / "04_side_by_side_comparison.png"
    Image.fromarray(side_by_side).save(side_by_side_path)

    # Save metrics JSON
    metrics = {
        "status": "SUCCESS",
        "checkpoint": checkpoint_path,
        "diffusion_steps": 50,
        "runtime_seconds": round(runtime, 2),
        "total_masked_pixels": total_masked_pixels,
        "inside_mask_changed_pixels": inside_changed_pixels,
        "inside_mask_regenerated_ratio": round(regen_ratio, 4),
        "gray_residual_pixels": int(gray_residual),
        "outside_mask_invariance_passed": passed_inv,
        "outside_mask_max_diff": max_diff,
        "outside_mask_violated_pixels": violated_outside,
        "reference_weights": res.get("reference_scoring", {}).get("weights", []),
        "output_path": str(output_path),
        "side_by_side_path": str(side_by_side_path)
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOutputs saved to: {out_dir}")
    print(f"- Input:        {input_path}")
    print(f"- Restored:     {output_path}")
    print(f"- Side-by-side: {side_by_side_path}")
    print(f"- Metrics:      {out_dir / 'metrics.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
