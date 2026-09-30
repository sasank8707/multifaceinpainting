"""
Controlled External-Image Inference on user_tests/test1/
Using Trained Decoupled Identity Cross-Attention (step_150.pt).

Configuration:
- Target: user_tests/test1/target.jpg (484x369)
- Mask: user_tests/test1/mask.png (484x369, 11,036 masked pixels, 21.5% face coverage)
- References: ref1.jpg, ref2.jpg, ref3.jpg
- Checkpoint: checkpoints/decoupled_identity_composite_loss/step_150.pt
- CFG Calibration: id_tokens repeated for batch_size=2, lambda=0.10 preserved
- Dynamic Weights: Computed via MaskAwareReferenceScorer (pose, quality, visibility, overlap)
- Diffusion: 5 steps, CFG 7.5, seed 42, CPU FP32
- Zero Training: purely inference evaluation.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage.metrics
import matplotlib.pyplot as plt
from PIL import Image

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.reference_scoring.scorer import MaskAwareReferenceScorer
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


def run_external_inference(
    test_dir: str = "user_tests/test1",
    checkpoint_path: str = "checkpoints/decoupled_identity_composite_loss/step_150.pt",
    steps: int = 5,
    guidance_scale: float = 7.5,
    lambda_scale: float = 0.1,
    seed: int = 42
):
    print("=" * 80)
    print("EXTERNAL IMAGE INFERENCE: USER_TESTS/TEST1")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Diffusion Steps: {steps}, CFG: {guidance_scale}, Lambda: {lambda_scale}, Seed: {seed}")
    print("=" * 80)
    start_time = time.time()

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    base_path = PROJECT_ROOT / test_dir
    output_dir = base_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load External Images
    print("\n[Step 1/6] Loading Target, Binary Mask, and 3 References...")
    target_rgb = cv2.cvtColor(cv2.imread(str(base_path / "target.jpg")), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(base_path / "mask.png"), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)

    ref1_rgb = cv2.cvtColor(cv2.imread(str(base_path / "ref1.jpg")), cv2.COLOR_BGR2RGB)
    ref2_rgb = cv2.cvtColor(cv2.imread(str(base_path / "ref2.jpg")), cv2.COLOR_BGR2RGB)
    ref3_rgb = cv2.cvtColor(cv2.imread(str(base_path / "ref3.jpg")), cv2.COLOR_BGR2RGB)

    h, w = target_rgb.shape[:2]
    num_masked_pixels = int(mask.sum())
    print(f"Target Dimensions: {w}x{h}, Masked Pixels: {num_masked_pixels} ({num_masked_pixels/(h*w)*100:.2f}%)")

    # 2. Phase 2 & 3: Run Face Analysis & MaskAwareReferenceScorer
    print("\n[Step 2/6] Running Face Analysis and MaskAwareReferenceScorer...")
    face_pipeline = FaceAnalysisPipeline()
    target_analysis = face_pipeline.analyze(target_rgb, mask)
    ref_analyses = [
        face_pipeline.analyze(ref1_rgb),
        face_pipeline.analyze(ref2_rgb),
        face_pipeline.analyze(ref3_rgb)
    ]
    reference_names = ["ref1", "ref2", "ref3"]

    scorer = MaskAwareReferenceScorer(
        pose_weight=0.30,
        quality_weight=0.25,
        visibility_weight=0.30,
        overlap_weight=0.15,
        top_k=3,
        strict_identity=False
    )
    scoring_report = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=ref_analyses,
        target_identity_id="user_test1",
        reference_identity_ids=["user_test1", "user_test1", "user_test1"],
        reference_names=reference_names
    )

    comp_0 = scoring_report["components"][0]
    norm_weights = comp_0["normalized_weights"]
    ref_weights = [norm_weights[name] for name in reference_names]
    sum_w = sum(ref_weights)
    ref_weights = [w_i / sum_w for w_i in ref_weights] if sum_w > 0 else [1.0/3, 1.0/3, 1.0/3]

    print("\n" + "=" * 70)
    print("COMPUTED DYNAMIC REFERENCE WEIGHTS:")
    for i, name in enumerate(reference_names):
        bd = comp_0["score_breakdown"].get(name, {})
        print(f"  {name}: weight={ref_weights[i]:.4f} (Composite={bd.get('composite_score',0):.4f}, Pose={bd.get('pose_score',0):.4f}, Quality={bd.get('quality_score',0):.4f}, Visibility={bd.get('visibility_score',0):.4f})")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    # 3. Extract Reference Identity Tokens & Phase-4 Regional Fusion
    print("[Step 3/6] Extracting Biometric Representations & Regional Fusion...")
    arcface = RegionalIdentityEncoder(token_dim=512).to(device)
    arcface.eval()
    for p in arcface.parameters(): p.requires_grad = False

    phase4 = RegionalIdentityTransformer(
        token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0
    ).to(device)
    phase4.eval()
    for p in phase4.parameters(): p.requires_grad = False

    diff_arcface = DifferentiableArcFaceBackbone("models/pretrained/w600k_r50.onnx").to(device)
    diff_arcface.eval()
    for p in diff_arcface.parameters(): p.requires_grad = False

    def prep_img(rgb_img, size=(256, 256)):
        resized = cv2.resize(rgb_img, size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    def prep_112(rgb_img):
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        tok_1 = arcface(prep_img(ref1_rgb))["spatial_tokens"]
        tok_2 = arcface(prep_img(ref2_rgb))["spatial_tokens"]
        tok_3 = arcface(prep_img(ref3_rgb))["spatial_tokens"]

        w_multi = torch.tensor([ref_weights], dtype=torch.float32, device=device)

        # STRICT TARGET ANTI-LEAKAGE: target_tokens is None
        t_fused = phase4(
            target_tokens=None,
            target_mask=mask_tensor_256,
            reference_tokens_list=[tok_1, tok_2, tok_3],
            reference_weights=w_multi
        )["fused_tokens"]

        e_ref1 = diff_arcface.extract_embedding(prep_112(ref1_rgb))
        e_ref2 = diff_arcface.extract_embedding(prep_112(ref2_rgb))
        e_ref3 = diff_arcface.extract_embedding(prep_112(ref3_rgb))
        e_ref_mean = F.normalize((e_ref1 + e_ref2 + e_ref3) / 3.0, p=2, dim=-1)
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))

    print(f"Fused identity tokens shape: {t_fused.shape}, Mean Norm: {t_fused.norm(dim=-1).mean():.4f}")

    # 4. Load Pretrained Pipeline & Decoupled Cross-Attention
    print("\n[Step 4/6] Loading Stable Diffusion Inpainting & step_150 Checkpoint...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0] if snapshots else "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    ).to(device)

    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=lambda_scale)
    manager.install(zero_init_v=False)
    ckpt_file = PROJECT_ROOT / checkpoint_path
    assert ckpt_file.exists(), f"Checkpoint {ckpt_file} not found!"
    manager.load_checkpoint(str(ckpt_file))
    manager.set_scale(lambda_scale)
    manager.set_identity_tokens(t_fused)
    print(f"Successfully loaded trained weights from: {ckpt_file}")

    # 5. Execute End-to-End Diffusion Generation
    print("\n[Step 5/6] Letterboxing and Executing Diffusion Generation...")
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
        image=target_rgb, mask=mask, target_size=(512, 512)
    )
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))
    prompt = "high quality photorealistic portrait of a young man, natural facial anatomy, natural skin texture"

    gen = torch.Generator(device="cpu").manual_seed(seed)
    t_diff_start = time.time()
    diff_output = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=gen
    ).images[0]
    t_diff = time.time() - t_diff_start
    print(f"Diffusion generation completed in {t_diff:.2f} seconds.")

    # 6. Unletterbox, Strict Invariance Verification, and Metrics
    print("\n[Step 6/6] Unletterboxing, Compositing, and Evaluating...")
    padded_cand = np.array(diff_output, dtype=np.uint8)
    final_reconstruction = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_cand,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )

    passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, final_reconstruction, mask)
    print(f"Bit-Exact Unmasked Invariance: Passed={passed_inv}, Max Delta={max_diff}, Violated Pixels={violated_pixels}")
    assert passed_inv, f"Unmasked invariance violated: {max_diff}"

    mask_bool = (mask > 0.5)
    ys, xs = np.where(mask_bool)
    min_y, max_y, min_x, max_x = int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs))
    crop_gt = target_rgb[min_y:max_y, min_x:max_x]
    crop_recon = final_reconstruction[min_y:max_y, min_x:max_x]

    # Reconstruction Metrics
    mae_diff = np.abs(final_reconstruction[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
    recon_mae = float(np.mean(mae_diff))
    recon_rmse = float(np.sqrt(np.mean(mae_diff**2)))
    recon_psnr = float(20 * np.log10(255.0 / (recon_rmse + 1e-9)))
    recon_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_recon, channel_axis=2))
    recon_sobel = compute_sobel_error(target_rgb, final_reconstruction, mask)
    recon_lap = compute_laplacian_error(target_rgb, final_reconstruction, mask)

    # Biometric ArcFace Evaluation
    with torch.no_grad():
        e_recon = diff_arcface.extract_embedding(prep_112(final_reconstruction))
        cos_ref_mean = float(F.cosine_similarity(e_recon, e_ref_mean, dim=-1).item())
        cos_ref1 = float(F.cosine_similarity(e_recon, e_ref1, dim=-1).item())
        cos_ref2 = float(F.cosine_similarity(e_recon, e_ref2, dim=-1).item())
        cos_ref3 = float(F.cosine_similarity(e_recon, e_ref3, dim=-1).item())
        cos_target = float(F.cosine_similarity(e_recon, e_target, dim=-1).item())

    print("\nBiometric Identity Results:")
    print(f"  Final Recon vs Reference Mean: {cos_ref_mean:.4f}")
    print(f"  Final Recon vs Ref 1 (3/4):    {cos_ref1:.4f}")
    print(f"  Final Recon vs Ref 2 (Up):     {cos_ref2:.4f}")
    print(f"  Final Recon vs Ref 3 (Front):  {cos_ref3:.4f}")
    print(f"  Final Recon vs Target GT:      {cos_target:.4f}")

    print("\nMasked Region Reconstruction Metrics:")
    print(f"  PSNR:       {recon_psnr:.2f} dB")
    print(f"  SSIM:       {recon_ssim:.4f}")
    print(f"  MAE:        {recon_mae:.2f}")
    print(f"  RMSE:       {recon_rmse:.2f}")
    print(f"  Sobel Err:  {recon_sobel:.4f}")
    print(f"  Laplace Err:{recon_lap:.4f}")

    # Save Output Images
    Image.fromarray(final_reconstruction).save(output_dir / "final_reconstruction.png")
    Image.fromarray(crop_recon).save(output_dir / "masked_crop.png")

    # Side-by-Side Comparison Panel
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(target_rgb)
    axes[0].set_title(f"Target Ground Truth\n({w}x{h})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    masked_vis = target_rgb.copy()
    masked_vis[mask_bool] = [255, 60, 60]
    axes[1].imshow(masked_vis)
    axes[1].set_title(f"Masked Input\n({num_masked_pixels} px, 21.5%)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(final_reconstruction)
    axes[2].set_title(f"Reconstruction (Step 150)\ncos(Ref Mean): {cos_ref_mean:.4f}\nPSNR: {recon_psnr:.2f} dB | SSIM: {recon_ssim:.4f}", fontsize=11, fontweight="bold", color="darkgreen")
    axes[2].axis("off")

    ref_tile = np.hstack([cv2.resize(ref1_rgb, (160, 200)), cv2.resize(ref2_rgb, (160, 200)), cv2.resize(ref3_rgb, (160, 200))])
    axes[3].imshow(ref_tile)
    axes[3].set_title(f"References (w={ref_weights[0]:.2f}, {ref_weights[1]:.2f}, {ref_weights[2]:.2f})", fontsize=11)
    axes[3].axis("off")

    crop_comp = np.hstack([cv2.resize(crop_gt, (140, 160)), cv2.resize(crop_recon, (140, 160))])
    axes[4].imshow(crop_comp)
    axes[4].set_title("GT vs Reconstructed Organ", fontsize=11)
    axes[4].axis("off")

    plt.tight_layout()
    comparison_path = output_dir / "side_by_side_comparison.png"
    plt.savefig(comparison_path, dpi=160)
    plt.close()

    # Save to Brain Artifacts
    brain_dir = Path("C:/Users/sasan/.gemini/antigravity/brain/fbe3e04a-609f-4ac9-8552-a9b57e8d7a2a/user_test1_inference")
    brain_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(output_dir / "final_reconstruction.png"), str(brain_dir / "final_reconstruction.png"))
    shutil.copy(str(output_dir / "masked_crop.png"), str(brain_dir / "masked_crop.png"))
    shutil.copy(str(comparison_path), str(brain_dir / "side_by_side_comparison.png"))

    # Save Metrics JSON
    results = {
        "test_case": "user_tests/test1",
        "checkpoint": checkpoint_path,
        "diffusion_runtime_seconds": t_diff,
        "total_runtime_seconds": time.time() - start_time,
        "reference_scoring": {
            "ref1_weight": ref_weights[0],
            "ref2_weight": ref_weights[1],
            "ref3_weight": ref_weights[2],
            "breakdown": comp_0.get("score_breakdown", {})
        },
        "unmasked_invariance": {
            "passed": passed_inv,
            "max_difference": max_diff,
            "violated_pixels": violated_pixels
        },
        "biometric_identity": {
            "cos_recon_vs_ref_mean": cos_ref_mean,
            "cos_recon_vs_ref1": cos_ref1,
            "cos_recon_vs_ref2": cos_ref2,
            "cos_recon_vs_ref3": cos_ref3,
            "cos_recon_vs_target_gt": cos_target
        },
        "reconstruction_quality": {
            "psnr": recon_psnr,
            "ssim": recon_ssim,
            "mae": recon_mae,
            "rmse": recon_rmse,
            "sobel_error": recon_sobel,
            "laplacian_error": recon_lap
        }
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    shutil.copy(str(output_dir / "metrics.json"), str(brain_dir / "metrics.json"))

    print(f"\nResults and figures saved to: {output_dir}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_external_inference()
