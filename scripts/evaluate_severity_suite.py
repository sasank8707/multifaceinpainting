"""
Multi-Severity Occlusion Evaluation Benchmark Suite (25%, 50%, 75%, 100% Full-Face).
Evaluates both Step 150 (Protected Baseline) and Step 500 across 4 canonical severity tiers:
- Tier 1: 25% face occlusion (canonical_eval_mask_25pct.png, 84,250 pixels, 19.85% face area)
- Tier 2: 50% face occlusion (canonical_eval_mask_50pct.png, 209,887 pixels, 49.44% face area)
- Tier 3: 75% face occlusion (canonical_eval_mask_75pct.png, 317,951 pixels, 74.90% face area)
- Tier 4: 100% full-face occlusion (canonical_eval_mask_100pct.png, 424,496 pixels, 100.00% face area)

Strict Invariance & Metrics:
- Biometrics: SimRef, SimGT, SimDiff, Identity Gap
- Quality: PSNR, SSIM, MAE, RMSE, Sobel, Laplacian
- Invariance: Bit-Exact Unmasked Invariance (Delta_max == 0.0)
- Generation/Copy Audit: Cross-correlation against reference patches
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage.metrics
from PIL import Image

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from diffusers import StableDiffusionInpaintPipeline


def compute_laplacian_error(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
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
    return float(np.mean(np.abs(lap_gt - lap_pred)[mask_bool]))


def compute_sobel_error(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
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


def compute_generation_copy_audit(recon_crop: np.ndarray, ref_crops: List[np.ndarray]) -> Dict[str, float]:
    """
    Computes generation vs copy audit metrics:
    Checks whether reconstructed crop is a direct pixel copy of any reference.
    """
    recon_f = recon_crop.astype(np.float32) / 255.0
    max_ssim = 0.0
    max_corr = 0.0

    for ref_c in ref_crops:
        ref_resized = cv2.resize(ref_c, (recon_crop.shape[1], recon_crop.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        # SSIM against reference
        s = skimage.metrics.structural_similarity(recon_f, ref_resized, channel_axis=2, data_range=1.0)
        max_ssim = max(max_ssim, float(s))
        # Normalized cross-correlation
        r_norm = recon_f - recon_f.mean()
        ref_norm = ref_resized - ref_resized.mean()
        std_prod = (r_norm.std() * ref_norm.std()) + 1e-8
        corr = float(np.mean(r_norm * ref_norm) / std_prod)
        max_corr = max(max_corr, corr)

    is_copy = bool(max_corr > 0.95 or max_ssim > 0.90)
    return {
        "max_ref_pixel_ssim": round(max_ssim, 4),
        "max_ref_cross_correlation": round(max_corr, 4),
        "is_verbatim_copy": is_copy,
        "is_novel_generation": not is_copy
    }


def find_sd_model():
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
    if cache_base.exists():
        dirs = [d for d in cache_base.iterdir() if d.is_dir()]
        if dirs:
            return str(dirs[0])
    return "runwayml/stable-diffusion-inpainting"


def run_severity_suite(
    checkpoints: List[str] = [
        "checkpoints/baselines/step_150_best/step_150.pt",
        "checkpoints/cloud_training/step_500.pt"
    ],
    device_str: str = "auto",
    output_dir: str = "evaluation/reports",
    steps: int = 50,
    seed: int = 42,
    smoke_test: bool = False
):
    print("=" * 95)
    print("OCCLUSION-SEVERITY BENCHMARK SUITE: 25%, 50%, 75%, 100% FULL-FACE")
    print("=" * 95)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    print(f"Compute Device: {device}")

    # Canonical evaluation assets
    target_p = PROJECT_ROOT / "data/processed/dataset_cache/7367.jpg"
    ref_a_p = PROJECT_ROOT / "data/processed/dataset_cache/13401.jpg"
    ref_b_p = PROJECT_ROOT / "data/processed/dataset_cache/23058.jpg"
    ref_c_p = PROJECT_ROOT / "data/processed/dataset_cache/23414.jpg"
    diff_p = PROJECT_ROOT / "data/processed/dataset_cache/3616.jpg"

    target_rgb = cv2.cvtColor(cv2.imread(str(target_p)), cv2.COLOR_BGR2RGB)
    ref_a_rgb = cv2.cvtColor(cv2.imread(str(ref_a_p)), cv2.COLOR_BGR2RGB)
    ref_b_rgb = cv2.cvtColor(cv2.imread(str(ref_b_p)), cv2.COLOR_BGR2RGB)
    ref_c_rgb = cv2.cvtColor(cv2.imread(str(ref_c_p)), cv2.COLOR_BGR2RGB)
    diff_rgb = cv2.cvtColor(cv2.imread(str(diff_p)), cv2.COLOR_BGR2RGB)

    tiers = [
        ("25pct", "data/masks/canonical_eval_mask_25pct.png", "Tier 1 (25% Occlusion)"),
        ("50pct", "data/masks/canonical_eval_mask_50pct.png", "Tier 2 (50% Occlusion)"),
        ("75pct", "data/masks/canonical_eval_mask_75pct.png", "Tier 3 (75% Occlusion)"),
        ("100pct", "data/masks/canonical_eval_mask_100pct.png", "Tier 4 (100% Full-Face)")
    ]

    if smoke_test:
        print("[SMOKE TEST MODE: 1 tier, 1 step]")
        tiers = tiers[:1]
        steps = 1

    def prep_256(img):
        r = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    def prep_112(img):
        r = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    # Initialize models
    print("\n[1/3] Initializing ArcFace and Pipeline Backbones...")
    diff_arcface_path = str(PROJECT_ROOT / "models/pretrained/w600k_r50.onnx")
    diff_arcface = DifferentiableArcFaceBackbone(diff_arcface_path).to(device)
    diff_arcface.eval()
    arcface = RegionalIdentityEncoder(token_dim=512)
    arcface.eval()
    phase4 = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(device)
    phase4.eval()

    with torch.no_grad():
        tok_A = arcface(prep_256(ref_a_rgb))["spatial_tokens"].to(device)
        tok_B = arcface(prep_256(ref_b_rgb))["spatial_tokens"].to(device)
        tok_C = arcface(prep_256(ref_c_rgb))["spatial_tokens"].to(device)
        w_multi = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32, device=device)

        e_ref_A = diff_arcface.extract_embedding(prep_112(ref_a_rgb))
        e_ref_B = diff_arcface.extract_embedding(prep_112(ref_b_rgb))
        e_ref_C = diff_arcface.extract_embedding(prep_112(ref_c_rgb))
        e_ref_mean = F.normalize((e_ref_A + e_ref_B + e_ref_C) / 3.0, p=2, dim=-1)
        e_diff = diff_arcface.extract_embedding(prep_112(diff_rgb))
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))

    sd_model = find_sd_model()
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float32,
            safety_checker=None,
            local_files_only=True
        ).to(device)
    except Exception:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(device)

    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=0.10)
    manager.install(zero_init_v=False)
    manager.to(device=device, dtype=pipe.unet.dtype)

    all_results = []
    print("\n[2/3] Executing Evaluation Matrix across Severity Tiers...")

    for tier_id, mask_rel_p, tier_name in tiers:
        mask_file = PROJECT_ROOT / mask_rel_p
        assert mask_file.exists(), f"Missing mask: {mask_file}"
        mask_gray = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        mask = (mask_gray > 127).astype(np.float32)
        masked_pixels = int(np.sum(mask))
        mask_pct = float(masked_pixels / (mask.shape[0] * mask.shape[1]) * 100)

        mask_bool = (mask > 0.5)
        ys, xs = np.where(mask_bool)
        crop_coords = (int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs)))
        min_y, max_y, min_x, max_x = crop_coords
        crop_gt = target_rgb[min_y:max_y, min_x:max_x]
        ref_crops = [ref_a_rgb[min_y:max_y, min_x:max_x], ref_b_rgb[min_y:max_y, min_x:max_x], ref_c_rgb[min_y:max_y, min_x:max_x]]

        # Fuse tokens conditioned on this tier's mask
        mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            t_fused = phase4(
                target_tokens=None,
                target_mask=mask_tensor_256,
                reference_tokens_list=[tok_A, tok_B, tok_C],
                reference_weights=w_multi
            )["fused_tokens"]

        padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(target_rgb, mask, target_size=(512, 512))
        pil_img = Image.fromarray(padded_img)
        pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

        prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"

        for ckpt_p_str in checkpoints:
            ckpt_p = (PROJECT_ROOT / ckpt_p_str).resolve()
            if not ckpt_p.exists():
                print(f"Skipping missing checkpoint: {ckpt_p}")
                continue

            ckpt_name = ckpt_p.stem
            print(f"\n--- Tier: {tier_name} ({tier_id}) | Checkpoint: {ckpt_name} ---")

            ckpt_data = manager.load_checkpoint(str(ckpt_p), strict=True)
            manager.to(device=device, dtype=pipe.unet.dtype)
            manager.set_scale(0.10)
            manager.set_identity_tokens(t_fused.to(device=device, dtype=pipe.unet.dtype))

            gen = torch.Generator(device="cpu").manual_seed(seed) if device.type == "cpu" else torch.Generator(device=device).manual_seed(seed)

            t0 = time.time()
            diff_out = pipe(
                prompt=prompt,
                image=pil_img,
                mask_image=pil_mask,
                num_inference_steps=steps,
                guidance_scale=7.5,
                generator=gen
            ).images[0]
            gen_dur = time.time() - t0

            padded_cand = np.array(diff_out, dtype=np.uint8)
            final_recon = AspectRatioPreserver.unletterbox(
                padded_candidate=padded_cand,
                original_image=target_rgb,
                original_mask=mask,
                transform_info=transform_info
            )

            passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, final_recon, mask)
            crop_recon = final_recon[min_y:max_y, min_x:max_x]

            # Quality metrics
            mae_diff = np.abs(final_recon[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
            recon_mae = float(np.mean(mae_diff))
            recon_rmse = float(np.sqrt(np.mean(mae_diff**2)))
            recon_psnr = float(20 * np.log10(255.0 / (recon_rmse + 1e-9)))
            recon_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_recon, channel_axis=2))
            recon_sobel = compute_sobel_error(target_rgb, final_recon, mask)
            recon_lap = compute_laplacian_error(target_rgb, final_recon, mask)

            # Biometrics
            with torch.no_grad():
                e_recon = diff_arcface.extract_embedding(prep_112(final_recon))
                cos_ref = float(F.cosine_similarity(e_recon, e_ref_mean, dim=-1).item())
                cos_tgt = float(F.cosine_similarity(e_recon, e_target, dim=-1).item())
                cos_diff = float(F.cosine_similarity(e_recon, e_diff, dim=-1).item())
                bio_gap = cos_ref - cos_diff

            # Copy vs generation audit
            copy_audit = compute_generation_copy_audit(crop_recon, ref_crops)

            result_entry = {
                "tier_id": tier_id,
                "tier_name": tier_name,
                "masked_pixels": masked_pixels,
                "pct_image_area": mask_pct,
                "checkpoint": ckpt_name,
                "checkpoint_path": str(ckpt_p),
                "step": ckpt_data.get("step", "unknown"),
                "runtime_seconds": gen_dur,
                "invariance": {
                    "passed": passed_inv,
                    "max_diff": max_diff,
                    "violated_pixels": violated_pixels
                },
                "biometrics": {
                    "sim_ref": round(cos_ref, 4),
                    "sim_gt": round(cos_tgt, 4),
                    "sim_diff": round(cos_diff, 4),
                    "bio_gap": round(bio_gap, 4)
                },
                "reconstruction": {
                    "psnr": round(recon_psnr, 2),
                    "ssim": round(recon_ssim, 4),
                    "mae": round(recon_mae, 2),
                    "rmse": round(recon_rmse, 2),
                    "sobel_error": round(recon_sobel, 4),
                    "laplacian_error": round(recon_lap, 4)
                },
                "copy_audit": copy_audit
            }
            all_results.append(result_entry)
            print(f"  [RESULT] SimRef: {cos_ref:.4f} | SimGT: {cos_tgt:.4f} | SimDiff: {cos_diff:.4f} | Gap: {bio_gap:+.4f} | PSNR: {recon_psnr:.2f} dB | SSIM: {recon_ssim:.4f}")
            print(f"  [AUDIT] Invariance: {passed_inv} (violated={violated_pixels}) | Max Ref SSIM: {copy_audit['max_ref_pixel_ssim']:.4f} | Copy: {copy_audit['is_verbatim_copy']}")

    # Print Formatted Comparison Table
    print("\n" + "=" * 110)
    print("MULTI-SEVERITY BENCHMARK COMPARATIVE RESULTS TABLE")
    print("=" * 110)
    print(f"| {'Tier':<8} | {'Masked Px':<10} | {'Checkpoint':<10} | {'SimRef':<8} | {'SimGT':<8} | {'SimDiff':<8} | {'Bio Gap':<8} | {'PSNR (dB)':<10} | {'SSIM':<7} | {'Inv':<5} |")
    print("-" * 110)
    for r in all_results:
        b = r["biometrics"]
        q = r["reconstruction"]
        inv_str = "PASS" if r["invariance"]["passed"] else "FAIL"
        print(f"| {r['tier_id']:<8} | {r['masked_pixels']:<10,d} | {r['checkpoint']:<10} | {b['sim_ref']:<8.4f} | {b['sim_gt']:<8.4f} | {b['sim_diff']:<8.4f} | {b['bio_gap']:<+8.4f} | {q['psnr']:<10.2f} | {q['ssim']:<7.4f} | {inv_str:<5} |")
    print("=" * 110)

    # Save JSON & CSV reports
    out_dir_p = Path(output_dir)
    if not out_dir_p.is_absolute():
        out_dir_p = (PROJECT_ROOT / out_dir_p).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    json_p = out_dir_p / "severity_suite_results.json"
    with open(json_p, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_p = out_dir_p / "severity_suite_results.csv"
    with open(csv_p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tier_id", "tier_name", "masked_pixels", "pct_image_area", "checkpoint", "step",
            "sim_ref", "sim_gt", "sim_diff", "bio_gap", "psnr", "ssim", "mae", "rmse",
            "sobel_error", "laplacian_error", "invariance_passed", "max_ref_pixel_ssim", "is_copy", "runtime_seconds"
        ])
        for r in all_results:
            b = r["biometrics"]
            q = r["reconstruction"]
            c = r["copy_audit"]
            writer.writerow([
                r["tier_id"], r["tier_name"], r["masked_pixels"], r["pct_image_area"], r["checkpoint"], r["step"],
                b["sim_ref"], b["sim_gt"], b["sim_diff"], b["bio_gap"], q["psnr"], q["ssim"], q["mae"], q["rmse"],
                q["sobel_error"], q["laplacian_error"], r["invariance"]["passed"], c["max_ref_pixel_ssim"], c["is_verbatim_copy"], round(r["runtime_seconds"], 2)
            ])

    print(f"\n[3/3] Reports successfully saved:")
    print(f"  - JSON Report: {json_p}")
    print(f"  - CSV Report:  {csv_p}")
    print("=" * 95)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Severity Occlusion Evaluation Benchmark Suite")
    parser.add_argument("--checkpoints", nargs="+", default=[
        "checkpoints/baselines/step_150_best/step_150.pt",
        "checkpoints/cloud_training/step_500.pt"
    ])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="evaluation/reports")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 tier with 1 step to verify execution flow")
    args = parser.parse_args()

    run_severity_suite(
        checkpoints=args.checkpoints,
        device_str=args.device,
        output_dir=args.output_dir,
        steps=args.steps,
        seed=args.seed,
        smoke_test=args.smoke_test
    )


if __name__ == "__main__":
    main()
