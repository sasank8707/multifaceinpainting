"""
Controlled End-to-End Inference using Trained Decoupled Identity Cross-Attention.

Configuration:
- Checkpoint: checkpoints/decoupled_identity_composite_loss/step_150.pt
- Target: CelebA-HQ 7367 (Identity 3111)
- References: CelebA-HQ 13401, 23058, 23414 (Verified Same-Identity 3111)
- Mask: Exact deterministic synthetic mask from Phase 7 (84,250 masked pixels, ~20% face coverage)
- Diffusion Backbone: runwayml/stable-diffusion-inpainting (CPU, FP32, 5 steps, CFG 7.5, seed 42)
- Coupling Scale: lambda = 0.1
- Zero Training: purely inference evaluation.

Outputs:
- Generated inpainting composite with strict unmasked bit-exact invariance.
- Biometric ArcFace identity evaluation (Same-ID References, Target GT, Different ID 3616, Vanilla Baseline).
- Masked region metrics: PSNR, SSIM, MAE, RMSE, Sobel, Laplacian errors.
- Comprehensive side-by-side comparison figure and metrics JSON.
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
import torch
import torch.nn.functional as F
import skimage.metrics
import matplotlib.pyplot as plt
from PIL import Image

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import (
    DecoupledIdentityCrossAttentionManager,
    DecoupledIdentityAttnProcessor
)
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from diffusers import StableDiffusionInpaintPipeline


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


def run_inference(
    checkpoint_path: str = "checkpoints/decoupled_identity_composite_loss/step_150.pt",
    steps: int = 5,
    guidance_scale: float = 7.5,
    lambda_scale: float = 0.1,
    seed: int = 42
):
    print("=" * 80)
    print("CONTROLLED END-TO-END INFERENCE: DECOUPLED IDENTITY CROSS-ATTENTION")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Diffusion Steps: {steps}, CFG: {guidance_scale}, Lambda: {lambda_scale}, Seed: {seed}")
    print("=" * 80)
    total_start_time = time.time()

    # Optimal CPU threading
    torch.set_num_threads(4)
    device = torch.device("cpu")

    out_dir = PROJECT_ROOT / "experiments/exp_decoupled_identity_inference"
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1/6] Loading Target, Mask, References, and Vanilla Baseline...")
    phase7_artifacts = PROJECT_ROOT / "experiments/exp_phase7_first_real_reconstruction/artifacts"
    phase7_data = PROJECT_ROOT / "experiments/exp_phase7_first_real_reconstruction/data"
    ablation1_artifacts = PROJECT_ROOT / "experiments/exp_phase7_ablation1_vanilla_baseline/artifacts"

    target_path = phase7_artifacts / "01_original_target.png"
    mask_path = phase7_artifacts / "02_mask.png"
    ref_a_path = phase7_data / "ref1_13401.jpg"
    ref_b_path = phase7_data / "ref2_23058.jpg"
    ref_c_path = phase7_data / "ref3_23414.jpg"
    diff_id_path = phase7_data / "diff_id_3616.jpg"
    vanilla_path = ablation1_artifacts / "01_baseline_final_reconstruction.png"

    target_rgb = cv2.cvtColor(cv2.imread(str(target_path)), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)

    ref_a_rgb = cv2.cvtColor(cv2.imread(str(ref_a_path)), cv2.COLOR_BGR2RGB)
    ref_b_rgb = cv2.cvtColor(cv2.imread(str(ref_b_path)), cv2.COLOR_BGR2RGB)
    ref_c_rgb = cv2.cvtColor(cv2.imread(str(ref_c_path)), cv2.COLOR_BGR2RGB)
    diff_id_rgb = cv2.cvtColor(cv2.imread(str(diff_id_path)), cv2.COLOR_BGR2RGB)

    vanilla_rgb = None
    if vanilla_path.exists():
        vanilla_rgb = cv2.cvtColor(cv2.imread(str(vanilla_path)), cv2.COLOR_BGR2RGB)

    h, w = target_rgb.shape[:2]
    num_masked_pixels = int(mask.sum())
    print(f"Target size: {w}x{h}, Masked Pixels: {num_masked_pixels} ({num_masked_pixels/(h*w)*100:.2f}%)")

    # 2. Phase 2 & 3: Run Face Analysis & MaskAwareReferenceScorer
    print("\n[Step 2/6] Running Phase 2 Face Analysis & Phase 3 MaskAwareReferenceScorer...")
    from modules.face_analysis.pipeline import FaceAnalysisPipeline
    from modules.reference_scoring.scorer import MaskAwareReferenceScorer

    face_pipeline = FaceAnalysisPipeline()
    target_analysis = face_pipeline.analyze(target_rgb, mask)
    ref_analyses = [
        face_pipeline.analyze(ref_a_rgb),
        face_pipeline.analyze(ref_b_rgb),
        face_pipeline.analyze(ref_c_rgb)
    ]

    reference_names = ["13401", "23058", "23414"]
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
        target_identity_id="3111",
        reference_identity_ids=["3111", "3111", "3111"],
        reference_names=reference_names
    )

    comp_0 = scoring_report["components"][0]
    norm_weights = comp_0["normalized_weights"]
    ref_weights = [norm_weights[name] for name in reference_names]
    # Normalize to exact sum 1.0
    sum_w = sum(ref_weights)
    ref_weights = [w_i / sum_w for w_i in ref_weights] if sum_w > 0 else [1.0/3, 1.0/3, 1.0/3]

    print("\n" + "=" * 70)
    print("ACTUAL REFERENCE WEIGHTS (from MaskAwareReferenceScorer):")
    print(f"  Ref A (13401): weight = {ref_weights[0]:.4f}")
    print(f"  Ref B (23058): weight = {ref_weights[1]:.4f}")
    print(f"  Ref C (23414): weight = {ref_weights[2]:.4f}")
    print("\n  Per-Reference Scoring Breakdown (Pose, Quality, Visibility, Overlap):")
    score_breakdown = comp_0.get("score_breakdown", {})
    for name in reference_names:
        bd = score_breakdown.get(name, {})
        print(f"    Ref {name}: Composite={bd.get('composite_score', 0.0):.4f} | Pose={bd.get('pose_score', 0.0):.4f} | Quality={bd.get('quality_score', 0.0):.4f} | Visibility={bd.get('visibility_score', 0.0):.4f} | Overlap={bd.get('overlap_score', 0.0):.4f}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    # 3. Extract Reference Identity Tokens & Phase-4 Regional Fusion
    print("[Step 3/6] Extracting Biometric Representations & Phase-4 Regional Fusion...")
    arcface = RegionalIdentityEncoder(token_dim=512).to(device)
    arcface.eval()
    for p in arcface.parameters():
        p.requires_grad = False

    phase4 = RegionalIdentityTransformer(
        token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0
    ).to(device)
    phase4.eval()
    for p in phase4.parameters():
        p.requires_grad = False

    diff_arcface = DifferentiableArcFaceBackbone("models/pretrained/w600k_r50.onnx").to(device)
    diff_arcface.eval()
    for p in diff_arcface.parameters():
        p.requires_grad = False

    def prep_img(rgb_img, size=(256, 256)):
        resized = cv2.resize(rgb_img, size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    def prep_112(rgb_img):
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        tok_A = arcface(prep_img(ref_a_rgb))["spatial_tokens"]
        tok_B = arcface(prep_img(ref_b_rgb))["spatial_tokens"]
        tok_C = arcface(prep_img(ref_c_rgb))["spatial_tokens"]

        w_multi = torch.tensor([ref_weights], dtype=torch.float32, device=device)

        # STRICT TARGET ANTI-LEAKAGE: target_tokens is None
        t_fused = phase4(
            target_tokens=None,
            target_mask=mask_tensor_256,
            reference_tokens_list=[tok_A, tok_B, tok_C],
            reference_weights=w_multi
        )["fused_tokens"] # [1, 256, 512]

        # ArcFace reference embeddings
        e_ref_A = diff_arcface.extract_embedding(prep_112(ref_a_rgb))
        e_ref_B = diff_arcface.extract_embedding(prep_112(ref_b_rgb))
        e_ref_C = diff_arcface.extract_embedding(prep_112(ref_c_rgb))
        e_ref_mean = F.normalize((e_ref_A + e_ref_B + e_ref_C) / 3.0, p=2, dim=-1)
        e_diff = diff_arcface.extract_embedding(prep_112(diff_id_rgb))
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))

    print(f"Fused identity tokens shape: {t_fused.shape}, Mean Token Norm: {t_fused.norm(dim=-1).mean():.4f}")
    sys.stdout.flush()

    # 3. Load Pretrained Pipeline & Decoupled Cross-Attention
    print("\n[Step 3/6] Loading Pretrained Stable Diffusion Inpainting & step_150 Checkpoint...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0] if snapshots else "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    ).to(device)

    # Attach Decoupled Identity Manager
    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=lambda_scale)
    manager.install(zero_init_v=False)

    ckpt_file = PROJECT_ROOT / checkpoint_path
    assert ckpt_file.exists(), f"Checkpoint {ckpt_file} not found!"
    manager.load_checkpoint(str(ckpt_file))
    print(f"Successfully loaded trained weights from: {ckpt_file}")

    # Configure identity tokens and coupling scale
    manager.set_scale(lambda_scale)
    manager.set_identity_tokens(t_fused)

    # 4. Prepare Letterboxed Target and Mask (512x512)
    print("\n[Step 4/6] Letterboxing to 512x512 working resolution...")
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
        image=target_rgb, mask=mask, target_size=(512, 512)
    )
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    # 5. Execute End-to-End Diffusion Generation
    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    print(f"\n[Step 5/6] Running End-to-End Diffusion Inference ({steps} steps, CFG {guidance_scale}, seed {seed})...")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    t_diff_start = time.time()
    diff_output = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    t_diff = time.time() - t_diff_start
    print(f"Diffusion generation completed in {t_diff:.2f} seconds.")

    # 6. Unletterbox and Strict Unmasked Compositing
    print("\n[Step 6/6] Compositing, Invariance Verification, and Biometric Evaluation...")
    padded_candidate = np.array(diff_output, dtype=np.uint8)
    final_reconstruction = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_candidate,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )

    # Verify bit-exact unmasked invariance
    passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, final_reconstruction, mask)
    print(f"Bit-Exact Unmasked Invariance: Passed={passed_inv}, Max Delta={max_diff}, Violated Pixels={violated_pixels}")
    assert passed_inv, f"Unmasked invariance violated: {max_diff}"

    # Masked region extraction
    mask_bool = (mask > 0.5)
    ys, xs = np.where(mask_bool)
    min_y, max_y, min_x, max_x = int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs))

    crop_gt = target_rgb[min_y:max_y, min_x:max_x]
    crop_recon = final_reconstruction[min_y:max_y, min_x:max_x]
    crop_vanilla = vanilla_rgb[min_y:max_y, min_x:max_x] if vanilla_rgb is not None else None

    # Quantitative reconstruction metrics inside mask
    mae_diff = np.abs(final_reconstruction[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
    recon_mae = float(np.mean(mae_diff))
    recon_rmse = float(np.sqrt(np.mean(mae_diff**2)))
    recon_psnr = float(20 * np.log10(255.0 / (recon_rmse + 1e-9)))
    recon_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_recon, channel_axis=2))
    recon_sobel = compute_sobel_error(target_rgb, final_reconstruction, mask)
    recon_lap = compute_laplacian_error(target_rgb, final_reconstruction, mask)

    # Biometric ArcFace Evaluation on Final Composite
    with torch.no_grad():
        e_recon = diff_arcface.extract_embedding(prep_112(final_reconstruction))
        cos_recon_ref = float(F.cosine_similarity(e_recon, e_ref_mean, dim=-1).item())
        cos_recon_target = float(F.cosine_similarity(e_recon, e_target, dim=-1).item())
        cos_recon_diff = float(F.cosine_similarity(e_recon, e_diff, dim=-1).item())
        biom_gap = cos_recon_ref - cos_recon_diff

        if vanilla_rgb is not None:
            e_vanilla = diff_arcface.extract_embedding(prep_112(vanilla_rgb))
            cos_vanilla_ref = float(F.cosine_similarity(e_vanilla, e_ref_mean, dim=-1).item())
            cos_vanilla_target = float(F.cosine_similarity(e_vanilla, e_target, dim=-1).item())
            cos_vanilla_diff = float(F.cosine_similarity(e_vanilla, e_diff, dim=-1).item())
            vanilla_biom_gap = cos_vanilla_ref - cos_vanilla_diff
        else:
            cos_vanilla_ref, cos_vanilla_target, cos_vanilla_diff, vanilla_biom_gap = 0.0, 0.0, 0.0, 0.0

    print("\nBiometric Identity Results:")
    print(f"  Final Recon vs Same-ID Ref Mean:  {cos_recon_ref:.4f}  (Vanilla: {cos_vanilla_ref:.4f})")
    print(f"  Final Recon vs Target GT:         {cos_recon_target:.4f}  (Vanilla: {cos_vanilla_target:.4f})")
    print(f"  Final Recon vs Different ID 3616: {cos_recon_diff:.4f}  (Vanilla: {cos_vanilla_diff:.4f})")
    print(f"  Biometric Gap (Ref - Diff):       {biom_gap:+.4f} (Vanilla: {vanilla_biom_gap:+.4f})")

    print("\nMasked Region Quality Metrics:")
    print(f"  PSNR:       {recon_psnr:.2f} dB")
    print(f"  SSIM:       {recon_ssim:.4f}")
    print(f"  MAE:        {recon_mae:.2f}")
    print(f"  RMSE:       {recon_rmse:.2f}")
    print(f"  Sobel Err:  {recon_sobel:.4f}")
    print(f"  Laplace Err:{recon_lap:.4f}")

    # Save Output Images
    Image.fromarray(final_reconstruction).save(artifacts_dir / "01_identity_reconstruction_step150.png")
    Image.fromarray(crop_recon).save(artifacts_dir / "02_masked_crop_step150.png")

    # Generate Comparison Figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(target_rgb)
    axes[0].set_title("Ground Truth Target (7367)")
    axes[0].axis("off")

    masked_view = target_rgb.copy()
    masked_view[mask_bool] = [255, 100, 100]
    axes[1].imshow(masked_view)
    axes[1].set_title(f"Masked Input ({num_masked_pixels} px)")
    axes[1].axis("off")

    if vanilla_rgb is not None:
        axes[2].imshow(vanilla_rgb)
        axes[2].set_title(f"Vanilla SD Inpainting\ncos(Ref): {cos_vanilla_ref:.4f} | Gap: {vanilla_biom_gap:+.4f}")
        axes[2].axis("off")
    else:
        axes[2].axis("off")

    axes[3].imshow(final_reconstruction)
    axes[3].set_title(f"Decoupled Identity (Step 150)\ncos(Ref): {cos_recon_ref:.4f} | Gap: {biom_gap:+.4f}")
    axes[3].axis("off")

    # 3 references tiled
    ref_tile = np.hstack([cv2.resize(ref_a_rgb, (170, 170)), cv2.resize(ref_b_rgb, (170, 170)), cv2.resize(ref_c_rgb, (170, 170))])
    axes[4].imshow(ref_tile)
    axes[4].set_title("3 Same-ID References\n(13401, 23058, 23414)")
    axes[4].axis("off")

    plt.tight_layout()
    comparison_path = artifacts_dir / "03_full_side_by_side_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    print(f"Comparison figure saved to: {comparison_path}")

    # Copy to Brain Artifacts
    brain_dir = Path("C:/Users/sasan/.gemini/antigravity/brain/fbe3e04a-609f-4ac9-8552-a9b57e8d7a2a/inference_step150")
    brain_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(comparison_path), str(brain_dir / "03_full_side_by_side_comparison.png"))
    shutil.copy(str(artifacts_dir / "01_identity_reconstruction_step150.png"), str(brain_dir / "01_identity_reconstruction_step150.png"))
    shutil.copy(str(artifacts_dir / "02_masked_crop_step150.png"), str(brain_dir / "02_masked_crop_step150.png"))

    # Save Metrics JSON
    results = {
        "checkpoint": checkpoint_path,
        "diffusion_config": {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "lambda_scale": lambda_scale,
            "seed": seed,
            "runtime_sec": t_diff
        },
        "unmasked_invariance": {
            "passed": passed_inv,
            "max_difference": max_diff,
            "violated_pixels": violated_pixels
        },
        "biometric_identity": {
            "cos_recon_vs_ref_mean": cos_recon_ref,
            "cos_recon_vs_target_gt": cos_recon_target,
            "cos_recon_vs_diff_id_3616": cos_recon_diff,
            "biometric_identity_gap": biom_gap,
            "vanilla_cos_vs_ref_mean": cos_vanilla_ref,
            "vanilla_cos_vs_diff_id": cos_vanilla_diff,
            "vanilla_biometric_gap": vanilla_biom_gap
        },
        "reconstruction_quality": {
            "psnr": recon_psnr,
            "ssim": recon_ssim,
            "mae": recon_mae,
            "rmse": recon_rmse,
            "sobel_error": recon_sobel,
            "laplacian_error": recon_lap
        },
        "reference_scoring": {
            "reference_names": reference_names,
            "normalized_weights": {reference_names[i]: float(ref_weights[i]) for i in range(len(reference_names))},
            "score_breakdown": comp_0.get("score_breakdown", {})
        }
    }

    with open(out_dir / "inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - total_start_time
    print(f"\nTotal Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} min).")
    print(f"Results saved to: {out_dir}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_inference()
