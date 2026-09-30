"""
Phase 7 Ablation 1: Vanilla Diffusion Inpainting Baseline.
Evaluates the anatomical reconstruction capability of pretrained runwayml/stable-diffusion-inpainting
on the exact same target (CelebA-HQ 7367) and exact same mask, without Phase 4 identity tokens,
without Phase 5 coarse guidance, using a natural text prompt through the standard CLIP text encoder.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import numpy as np
import cv2
import torch
import skimage.metrics
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance


def compute_laplacian_error(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Computes 5x5 Laplacian error inside the mask (Wei et al. 2019)."""
    laplace_5x5 = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=np.float32)

    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    
    lap_gt = np.zeros_like(gt_f)
    lap_pred = np.zeros_like(pred_f)
    for c in range(3):
        lap_gt[:, :, c] = cv2.filter2D(gt_f[:, :, c], -1, laplace_5x5)
        lap_pred[:, :, c] = cv2.filter2D(pred_f[:, :, c], -1, laplace_5x5)
        
    mask_bool = (mask > 0.5)
    diff = np.abs(lap_gt - lap_pred)
    return float(np.mean(diff[mask_bool]))


def compute_sobel_error(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Computes Sobel gradient magnitude error inside the mask."""
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    gx_gt = cv2.Sobel(gt_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_gt = cv2.Sobel(gt_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_gt = np.sqrt(gx_gt**2 + gy_gt**2)

    gx_pred = cv2.Sobel(pred_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_pred = cv2.Sobel(pred_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_pred = np.sqrt(gx_pred**2 + gy_pred**2)

    mask_bool = (mask > 0.5)
    return float(np.mean(np.abs(mag_gt - mag_pred)[mask_bool]))


def run_ablation1():
    print("=" * 70)
    print("STARTING ABLATION 1: VANILLA DIFFUSION INPAINTING BASELINE")
    print("=" * 70)
    start_total_time = time.time()

    # Paths
    phase7_dir = Path("experiments/exp_phase7_first_real_reconstruction")
    ablation_dir = Path("experiments/exp_phase7_ablation1_vanilla_baseline")
    artifacts_dir = ablation_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SAME Target and SAME Mask
    print("\n[Step 1/5] Loading verified Phase 7 Target and Mask...")
    target_path = phase7_dir / "artifacts" / "01_original_target.png"
    mask_path = phase7_dir / "artifacts" / "02_mask.png"
    phase7_recon_path = phase7_dir / "artifacts" / "05_final_diffusion_reconstruction.png"

    if not target_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Target or mask not found in {phase7_dir / 'artifacts'}")

    target_bgr = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    target_img = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)
    phase7_recon_bgr = cv2.imread(str(phase7_recon_path), cv2.IMREAD_COLOR)
    phase7_recon = cv2.cvtColor(phase7_recon_bgr, cv2.COLOR_BGR2RGB)

    h, w = target_img.shape[:2]
    print(f"Target Image: {w}x{h}, Masked Pixels: {int(mask.sum())} ({mask.sum() / (h*w)*100:.2f}%)")

    # Mask BBox
    ys, xs = np.where(mask > 0.5)
    mask_bbox = (int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs)))

    # 2. Letterbox to 512x512 Working Resolution (Same Preprocessing)
    print("\n[Step 2/5] Letterboxing to 512x512 working resolution...")
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
        image=target_img,
        mask=mask,
        target_size=(512, 512)
    )
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    # 3. Load Real Pipeline from Local Cache
    print("\n[Step 3/5] Loading real runwayml/stable-diffusion-inpainting pipeline...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0] if snapshots else "runwayml/stable-diffusion-inpainting"
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    )
    pipe = pipe.to("cpu")

    # 4. Execute Vanilla Inpainting (No identity tokens, no coarse guidance, standard text prompt)
    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    print(f"\n[Step 4/5] Running Diffusion Inference (5 steps, CPU, seed 42)...")
    print(f"Prompt: \"{prompt}\"")
    generator = torch.Generator(device="cpu").manual_seed(42)

    t_inpaint_start = time.time()
    result_pil = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=5,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    t_inpaint = time.time() - t_inpaint_start
    print(f"Diffusion generation completed in {t_inpaint:.2f}s.")

    # 5. Unletterbox and Strict Unmasked Compositing
    padded_candidate = np.array(result_pil, dtype=np.uint8)
    final_baseline = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_candidate,
        original_image=target_img,
        original_mask=mask,
        transform_info=transform_info
    )

    # 6. Verify Unmasked Invariance
    passed, max_diff, violated_pixels = verify_unmasked_invariance(target_img, final_baseline, mask)
    print(f"\n[Step 5/5] Verifying Bit-Exact Unmasked Invariance...")
    print(f"Unmasked Invariance Passed: {passed}, Max Diff Outside Mask: {max_diff}, Violated Pixels: {violated_pixels}")
    assert passed, f"Unmasked invariance violated: {max_diff}"

    # 7. Compute Quantitative Metrics
    mask_bool = (mask > 0.5)
    mae_diff = np.abs(final_baseline[mask_bool].astype(float) - target_img[mask_bool].astype(float))
    base_mae = float(np.mean(mae_diff))
    base_rmse = float(np.sqrt(np.mean(mae_diff**2)))
    base_psnr = float(20 * np.log10(255.0 / (base_rmse + 1e-9)))

    # SSIM on masked crop
    min_y, max_y, min_x, max_x = mask_bbox
    crop_gt = target_img[min_y:max_y, min_x:max_x]
    crop_baseline = final_baseline[min_y:max_y, min_x:max_x]
    crop_phase7 = phase7_recon[min_y:max_y, min_x:max_x]
    base_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_baseline, channel_axis=2))

    # Edge and structural metrics
    base_sobel_err = compute_sobel_error(target_img, final_baseline, mask)
    base_lap_err = compute_laplacian_error(target_img, final_baseline, mask)

    # Load Phase 7 metrics for direct comparison
    with open(phase7_dir / "metrics.json") as f:
        p7_metrics = json.load(f)["quantitative_metrics"]["final_diffusion"]

    # 8. Save Visual Artifacts
    print("\nGenerating and saving visual artifacts...")
    # 01 Baseline Final Reconstruction
    cv2.imwrite(str(artifacts_dir / "01_baseline_final_reconstruction.png"), cv2.cvtColor(final_baseline, cv2.COLOR_RGB2BGR))

    # 02 Masked Region Crop Comparison
    pad_y = int((max_y - min_y) * 0.1)
    pad_x = int((max_x - min_x) * 0.1)
    c_min_y = max(0, min_y - pad_y)
    c_max_y = min(h, max_y + pad_y)
    c_min_x = max(0, min_x - pad_x)
    c_max_x = min(w, max_x + pad_x)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=150)
    axes[0].imshow(target_img[c_min_y:c_max_y, c_min_x:c_max_x])
    axes[0].set_title("1. Ground Truth Target (Crop)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(phase7_recon[c_min_y:c_max_y, c_min_x:c_max_x])
    axes[1].set_title("2. Phase 7 Run (With P4/P5 Tokens)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(final_baseline[c_min_y:c_max_y, c_min_x:c_max_x])
    axes[2].set_title("3. Ablation 1 (Vanilla Baseline)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(artifacts_dir / "02_masked_region_crop_comparison.png"), bbox_inches="tight")
    plt.close()

    # 03 Difference Map
    diff_mag = np.mean(np.abs(target_img.astype(float) - final_baseline.astype(float)), axis=2)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(diff_mag, cmap="hot", vmin=0, vmax=100)
    ax.set_title("Ablation 1 Difference Map (|Baseline - Target|)", fontsize=12, pad=8)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Pixel Error [0-255]", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(artifacts_dir / "03_difference_map.png"), bbox_inches="tight")
    plt.close()

    # 04 Side-by-side Full Images
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=150)
    axes[0].imshow(target_img)
    axes[0].set_title("Ground Truth Target", fontsize=11)
    axes[0].axis("off")

    masked_in = target_img.copy()
    masked_in[mask > 0.5] = 0
    axes[1].imshow(masked_in)
    axes[1].set_title("Masked Input Canvas", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(phase7_recon)
    axes[2].set_title("Phase 7 Run (Failed Acceptance)", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(final_baseline)
    axes[3].set_title("Ablation 1 (Vanilla Baseline)", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(artifacts_dir / "04_side_by_side_comparison.png"), bbox_inches="tight")
    plt.close()

    total_time = time.time() - start_total_time

    # 9. Save Metrics JSON
    metrics_data = {
        "experiment": {
            "name": "Phase 7 Ablation 1: Vanilla Diffusion Inpainting Baseline",
            "model": "runwayml/stable-diffusion-inpainting",
            "prompt": prompt,
            "negative_prompt": None,
            "target_image": "7367.jpg",
            "seed": 42,
            "num_inference_steps": 5,
            "guidance_scale": 7.5,
            "working_resolution": [512, 512],
            "original_resolution": [1024, 1024],
            "device": "cpu",
            "pipeline_runtime_seconds": t_inpaint,
            "total_runtime_seconds": total_time
        },
        "unmasked_invariance": {
            "passed": passed,
            "max_unmasked_diff": float(max_diff),
            "violated_pixels": int(violated_pixels)
        },
        "quantitative_metrics_comparison": {
            "ablation1_vanilla_baseline": {
                "masked_mae": base_mae,
                "masked_rmse": base_rmse,
                "masked_psnr_db": base_psnr,
                "masked_crop_ssim": base_ssim,
                "sobel_gradient_error": base_sobel_err,
                "laplacian_5x5_error": base_lap_err
            },
            "phase7_run": {
                "masked_mae": p7_metrics["masked_mae"],
                "masked_rmse": p7_metrics["masked_rmse"],
                "masked_psnr_db": p7_metrics["masked_psnr_db"],
                "masked_crop_ssim": p7_metrics["masked_crop_ssim"],
                "sobel_gradient_error": p7_metrics["sobel_gradient_error"],
                "laplacian_5x5_error": p7_metrics["laplacian_5x5_error"]
            },
            "delta_ablation_minus_phase7": {
                "mae_delta": base_mae - p7_metrics["masked_mae"],
                "rmse_delta": base_rmse - p7_metrics["masked_rmse"],
                "psnr_delta_db": base_psnr - p7_metrics["masked_psnr_db"],
                "ssim_delta": base_ssim - p7_metrics["masked_crop_ssim"],
                "sobel_delta": base_sobel_err - p7_metrics["sobel_gradient_error"],
                "laplacian_delta": base_lap_err - p7_metrics["laplacian_5x5_error"]
            }
        }
    }

    with open(ablation_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("\n" + "=" * 70)
    print("ABLATION 1 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Metrics saved to: {ablation_dir / 'metrics.json'}")
    print(f"Artifacts saved to: {artifacts_dir}")
    print(f"Baseline PSNR: {base_psnr:.2f} dB (Phase 7: {p7_metrics['masked_psnr_db']:.2f} dB)")
    print(f"Baseline SSIM: {base_ssim:.4f} (Phase 7: {p7_metrics['masked_crop_ssim']:.4f})")
    print(f"Baseline MAE:  {base_mae:.2f} (Phase 7: {p7_metrics['masked_mae']:.2f})")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation1()
