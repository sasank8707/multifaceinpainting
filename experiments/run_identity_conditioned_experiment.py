"""
First Real Identity-Conditioned Diffusion Generation Experiment.

Controlled Experiment:
- Target: CelebA-HQ 7367 (Identity 3111)
- References: CelebA-HQ 13401, 23058, 23414 (Verified Same-Identity 3111)
- Mask: Exact deterministic synthetic mask from Phase 7 (84,250 masked pixels, ~20% face area)
- Model: runwayml/stable-diffusion-inpainting (CPU, FP32, 5 steps, CFG 7.5, seed 42, K=1)
- Identity Conditioning:
  - Real pretrained ArcFace ResNet-50 (w600k_r50.onnx)
  - Reference-only identity representation (WHO)
  - Mask-aware reference scoring
  - Regional reference fusion with zero target identity residual
  - IdentityConditioningAdapter retrained on real ArcFace features
  - Explicit channels 5:9 latent assembly without diffusers zero-masking
  - Isolated coarse reconstructor (no smoke model)

Evaluates:
- Exact unmasked invariance
- Masked region reconstruction (PSNR, SSIM, MAE, RMSE, Sobel, Laplacian)
- Conditioning sensitivity (difference from vanilla baseline)
- Biometric ArcFace identity evaluation (full face and masked-zone crops)
- 10 standardized output artifacts and comprehensive metrics JSON
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage.metrics
import matplotlib.pyplot as plt
from PIL import Image

from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.conditioning_adapter import IdentityConditioningAdapter
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.reference_scoring.scorer import MaskAwareReferenceScorer
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


def run_experiment():
    print("=" * 80)
    print("STARTING FIRST REAL IDENTITY-CONDITIONED DIFFUSION GENERATION EXPERIMENT")
    print("=" * 80)
    total_start_time = time.time()

    # Output directories
    exp_dir = Path("experiments/exp_identity_conditioned_diffusion")
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exact target, mask, and verified references
    print("\n[Step 1/7] Loading target, mask, references, and vanilla baseline...")
    phase7_artifacts = Path("experiments/exp_phase7_first_real_reconstruction/artifacts")
    phase7_data = Path("experiments/exp_phase7_first_real_reconstruction/data")
    ablation1_artifacts = Path("experiments/exp_phase7_ablation1_vanilla_baseline/artifacts")

    target_path = phase7_artifacts / "01_original_target.png"
    mask_path = phase7_artifacts / "02_mask.png"
    vanilla_path = ablation1_artifacts / "01_baseline_final_reconstruction.png"

    ref_a_path = phase7_data / "ref1_13401.jpg"
    ref_b_path = phase7_data / "ref2_23058.jpg"
    ref_c_path = phase7_data / "ref3_23414.jpg"

    # Verify all input files
    for p in [target_path, mask_path, vanilla_path, ref_a_path, ref_b_path, ref_c_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required input file missing: {p}")

    target_rgb = cv2.cvtColor(cv2.imread(str(target_path)), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)
    vanilla_rgb = cv2.cvtColor(cv2.imread(str(vanilla_path)), cv2.COLOR_BGR2RGB)

    ref_a_rgb = cv2.cvtColor(cv2.imread(str(ref_a_path)), cv2.COLOR_BGR2RGB)
    ref_b_rgb = cv2.cvtColor(cv2.imread(str(ref_b_path)), cv2.COLOR_BGR2RGB)
    ref_c_rgb = cv2.cvtColor(cv2.imread(str(ref_c_path)), cv2.COLOR_BGR2RGB)
    reference_images = [ref_a_rgb, ref_b_rgb, ref_c_rgb]
    reference_ids = ["13401", "23058", "23414"]

    h, w = target_rgb.shape[:2]
    num_masked_pixels = int(np.sum(mask > 0.5))
    mask_coverage = num_masked_pixels / (h * w) * 100
    print(f"Target: {w}x{h} | Masked Pixels: {num_masked_pixels:,} ({mask_coverage:.2f}%)")

    # Bounding box of mask
    ys, xs = np.where(mask > 0.5)
    mask_bbox = (int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs)))
    print(f"Mask BBox: Y=[{mask_bbox[0]}, {mask_bbox[1]}], X=[{mask_bbox[2]}, {mask_bbox[3]}]")

    # 2. Phase 2 Analysis & Phase 3 Reference Scoring
    print("\n[Step 2/7] Running Phase 2 Face Analysis and Phase 3 Reference Scoring...")
    face_pipeline = FaceAnalysisPipeline()
    target_analysis = face_pipeline.analyze(target_rgb, mask)
    ref_analyses = [face_pipeline.analyze(r) for r in reference_images]

    scorer = MaskAwareReferenceScorer(strict_identity=False)
    scoring_report = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=ref_analyses,
        target_identity_id="3111",
        reference_identity_ids=["3111", "3111", "3111"]
    )
    global_ranked = scoring_report.get("global_ranked_references", [])
    ref_weights = [r.get("weight", 1.0/3) for r in global_ranked[:3]]
    total_w = sum(ref_weights)
    ref_weights = [w_i / total_w for w_i in ref_weights] if total_w > 0 else [1.0/3, 1.0/3, 1.0/3]
    print(f"Reference Scoring Weights: {ref_weights}")

    # 3. Phase 4: Regional Identity Fusion (WHERE vs WHO)
    print("\n[Step 3/7] Running Phase 4 Regional Identity Fusion (Real ArcFace ResNet-50)...")
    encoder = RegionalIdentityEncoder(token_dim=512)
    encoder.eval()

    transformer = RegionalIdentityTransformer(
        token_dim=512,
        num_heads=8,
        num_layers=2,
        spatial_patch_grid=(16, 16),
        dropout=0.0
    )
    transformer.eval()

    # Prepare inputs for fusion
    # Mask tensor [1, 1, 256, 256]
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float()

    # Reference ArcFace spatial tokens [1, 256, 512] each
    ref_tokens_list = []
    with torch.no_grad():
        for r_rgb in reference_images:
            r_256 = cv2.resize(r_rgb, (256, 256))
            r_tensor = torch.from_numpy(r_256).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            r_tokens = encoder(r_tensor)["spatial_tokens"]
            ref_tokens_list.append(r_tokens)

    weights_tensor = torch.tensor([ref_weights], dtype=torch.float32)

    with torch.no_grad():
        # Fusion forward pass (STRICTLY ZERO TARGET IDENTITY RESIDUAL)
        fusion_out = transformer(
            target_tokens=None,
            target_mask=mask_tensor,
            reference_tokens_list=ref_tokens_list,
            reference_weights=weights_tensor
        )
    fused_identity_tokens = fusion_out["fused_tokens"] # [1, 256, 512]
    print(f"Fused Identity Tokens shape: {fused_identity_tokens.shape}")

    # 4. Phase 5/6: Identity Conditioning Adapter (Trained Checkpoint)
    print("\n[Step 4/7] Projecting identity tokens via retrained IdentityConditioningAdapter...")
    adapter = IdentityConditioningAdapter(
        in_dim=512,
        out_dim=768,
        num_tokens=256,
        out_tokens=77,
        num_id_queries=16,
        dropout=0.0
    )
    adapter_ckpt_path = PROJECT_ROOT / "checkpoints/identity_adapter/identity_adapter_best.pt"
    ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    adapter.load_state_dict(ckpt["model_state_dict"])
    adapter.eval()
    print(f"Loaded retrained adapter from {adapter_ckpt_path.name} (val_loss={ckpt.get('val_loss', 'N/A'):.4f})")

    with torch.no_grad():
        cond_out = adapter(identity_tokens=fused_identity_tokens)
    prompt_embeds = cond_out["projected_tokens"] # [1, 77, 768]
    null_embeds = adapter.get_null_embedding(batch_size=1, target_len=77)
    print(f"Diffusion Prompt Embeddings shape: {prompt_embeds.shape}, Mean Token Norm: {prompt_embeds.norm(dim=-1).mean().item():.2f}")
    print(f"Diffusion Null Embeddings shape:   {null_embeds.shape}, Mean Token Norm: {null_embeds.norm(dim=-1).mean().item():.2f}")

    # 5. Load Real Diffusion Inpainting Backbone
    print("\n[Step 5/7] Loading real runwayml/stable-diffusion-inpainting pipeline from cache...")
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

    # 6. Execute Identity-Conditioned Inpainting
    print("\n[Step 6/7] Running Identity-Conditioned Diffusion Inference (5 steps, CPU, seed 42, CFG 7.5)...")
    # Letterbox to 512x512 working resolution
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
        image=target_rgb,
        mask=mask,
        target_size=(512, 512)
    )

    pil_original = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    generator = torch.Generator(device="cpu").manual_seed(42)

    # Explicitly construct normalized masked-image tensor in the exact semantic space
    # expected by pretrained Stable Diffusion inpainting (neutral 0.0 in masked region)
    init_image_tensor = pipe.image_processor.preprocess(
        pil_original, height=512, width=512
    ).to(device="cpu", dtype=torch.float32)

    mask_tensor = pipe.mask_processor.preprocess(
        pil_mask, height=512, width=512
    ).to(device="cpu", dtype=torch.float32)

    # Strictly neutral 0.0 in masked region, bit-exact original pixels in unmasked region
    masked_image_tensor = init_image_tensor * (mask_tensor < 0.5)

    with torch.no_grad():
        guided_latents = pipe._encode_vae_image(masked_image_tensor, generator=generator)

    t_inpaint_start = time.time()
    gen_pil = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=null_embeds,
        image=pil_original,
        mask_image=pil_mask,
        masked_image_latents=guided_latents,
        num_inference_steps=5,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    t_inpaint = time.time() - t_inpaint_start
    print(f"Diffusion generation completed in {t_inpaint:.2f}s.")

    # Unletterbox back to original 1024x1024 with bit-exact unmasked pixel preservation
    padded_candidate = np.array(gen_pil, dtype=np.uint8)
    final_identity_recon = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_candidate,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )

    # Verify bit-exact unmasked pixel preservation
    passed_invariance, max_unmasked_diff, violated_pixels = verify_unmasked_invariance(
        target_rgb, final_identity_recon, mask
    )
    print(f"Bit-Exact Unmasked Invariance: {passed_invariance} (Max diff: {max_unmasked_diff}, Violated: {violated_pixels})")

    # 7. Comprehensive Evaluation Protocol
    print("\n[Step 7/7] Computing comprehensive quantitative metrics and saving artifacts...")
    mask_bool = (mask > 0.5)

    # 7.1 Reconstruction Metrics against Ground Truth (Identity-Conditioned vs Target)
    mae_diff_id = np.abs(final_identity_recon[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
    mae_id = float(np.mean(mae_diff_id))
    rmse_id = float(np.sqrt(np.mean(mae_diff_id**2)))
    psnr_id = float(20 * np.log10(255.0 / (rmse_id + 1e-9)))

    min_y, max_y, min_x, max_x = mask_bbox
    crop_gt = target_rgb[min_y:max_y, min_x:max_x]
    crop_id = final_identity_recon[min_y:max_y, min_x:max_x]
    crop_vanilla = vanilla_rgb[min_y:max_y, min_x:max_x]

    ssim_id = float(skimage.metrics.structural_similarity(crop_gt, crop_id, channel_axis=2))
    sobel_id = compute_sobel_error(target_rgb, final_identity_recon, mask)
    laplacian_id = compute_laplacian_error(target_rgb, final_identity_recon, mask)

    # 7.2 Reconstruction Metrics against Ground Truth (Vanilla Baseline vs Target)
    mae_diff_v = np.abs(vanilla_rgb[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
    mae_v = float(np.mean(mae_diff_v))
    rmse_v = float(np.sqrt(np.mean(mae_diff_v**2)))
    psnr_v = float(20 * np.log10(255.0 / (rmse_v + 1e-9)))
    ssim_v = float(skimage.metrics.structural_similarity(crop_gt, crop_vanilla, channel_axis=2))
    sobel_v = compute_sobel_error(target_rgb, vanilla_rgb, mask)
    laplacian_v = compute_laplacian_error(target_rgb, vanilla_rgb, mask)

    # 7.3 Conditioning Sensitivity (Identity-Conditioned vs Vanilla Baseline)
    # Verifies that identity tokens actively altered the generated pixels
    delta_gen_pixels = np.abs(final_identity_recon[mask_bool].astype(float) - vanilla_rgb[mask_bool].astype(float))
    mean_pixel_delta_baseline = float(np.mean(delta_gen_pixels))
    max_pixel_delta_baseline = float(np.max(delta_gen_pixels))
    fraction_pixels_changed = float(np.mean(delta_gen_pixels > 1.0))
    ssim_id_vs_vanilla = float(skimage.metrics.structural_similarity(crop_vanilla, crop_id, channel_axis=2))

    # 7.4 Biometric ArcFace Identity Evaluation (Crop & Full Protocol)
    # Using real pretrained ArcFace ResNet-50
    def get_embedding(img_np):
        t = torch.from_numpy(cv2.resize(img_np, (256, 256))).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            return encoder(t)["global_embedding"].squeeze(0)

    # Facial crop protocol (bounding box around face and masked region)
    emb_target = get_embedding(target_rgb)
    emb_vanilla = get_embedding(vanilla_rgb)
    emb_id = get_embedding(final_identity_recon)

    emb_ref_a = get_embedding(ref_a_rgb)
    emb_ref_b = get_embedding(ref_b_rgb)
    emb_ref_c = get_embedding(ref_c_rgb)
    emb_refs_mean = F.normalize(torch.stack([emb_ref_a, emb_ref_b, emb_ref_c]).mean(dim=0), p=2, dim=-1)

    # Different identity (Subject 3330 / 3616) for contrastive reference
    diff_id_path = phase7_data / "diff_id_3616.jpg"
    if not diff_id_path.is_file():
        # Load from celeba_hq dataset
        from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset
        ds = CelebAHQIdentityDataset(split="test", smoke_subset_size=1)
        diff_p = ds._get_image_path(3616)
        shutil.copy(diff_p, diff_id_path)
    diff_rgb = cv2.cvtColor(cv2.imread(str(diff_id_path)), cv2.COLOR_BGR2RGB)
    emb_diff_id = get_embedding(diff_rgb)

    # Cosine similarities
    sim_target_to_refs = float(F.cosine_similarity(emb_target.unsqueeze(0), emb_refs_mean.unsqueeze(0)).item())
    sim_id_to_refs = float(F.cosine_similarity(emb_id.unsqueeze(0), emb_refs_mean.unsqueeze(0)).item())
    sim_vanilla_to_refs = float(F.cosine_similarity(emb_vanilla.unsqueeze(0), emb_refs_mean.unsqueeze(0)).item())
    sim_id_to_target = float(F.cosine_similarity(emb_id.unsqueeze(0), emb_target.unsqueeze(0)).item())
    sim_vanilla_to_target = float(F.cosine_similarity(emb_vanilla.unsqueeze(0), emb_target.unsqueeze(0)).item())
    sim_id_to_diff_id = float(F.cosine_similarity(emb_id.unsqueeze(0), emb_diff_id.unsqueeze(0)).item())

    # Masked-region crop embeddings (evaluates identity specifically in the reconstructed region)
    pad_y = int((max_y - min_y) * 0.2)
    pad_x = int((max_x - min_x) * 0.2)
    c_y1, c_y2 = max(0, min_y - pad_y), min(h, max_y + pad_y)
    c_x1, c_x2 = max(0, min_x - pad_x), min(w, max_x + pad_x)

    crop_emb_target = get_embedding(target_rgb[c_y1:c_y2, c_x1:c_x2])
    crop_emb_id = get_embedding(final_identity_recon[c_y1:c_y2, c_x1:c_x2])
    crop_emb_vanilla = get_embedding(vanilla_rgb[c_y1:c_y2, c_x1:c_x2])
    crop_emb_ref_a = get_embedding(ref_a_rgb[c_y1:c_y2, c_x1:c_x2])
    crop_emb_ref_b = get_embedding(ref_b_rgb[c_y1:c_y2, c_x1:c_x2])
    crop_emb_ref_c = get_embedding(ref_c_rgb[c_y1:c_y2, c_x1:c_x2])
    crop_emb_refs_mean = F.normalize(torch.stack([crop_emb_ref_a, crop_emb_ref_b, crop_emb_ref_c]).mean(dim=0), p=2, dim=-1)

    crop_sim_id_to_refs = float(F.cosine_similarity(crop_emb_id.unsqueeze(0), crop_emb_refs_mean.unsqueeze(0)).item())
    crop_sim_vanilla_to_refs = float(F.cosine_similarity(crop_emb_vanilla.unsqueeze(0), crop_emb_refs_mean.unsqueeze(0)).item())
    crop_sim_target_to_refs = float(F.cosine_similarity(crop_emb_target.unsqueeze(0), crop_emb_refs_mean.unsqueeze(0)).item())

    print("\nIdentity Evaluation Metrics (Pretrained ArcFace ResNet-50):")
    print(f"Target vs Same-ID References (Ground Truth Upper Bound): {sim_target_to_refs:+.4f}")
    print(f"Identity-Conditioned vs Same-ID References:              {sim_id_to_refs:+.4f}")
    print(f"Vanilla Baseline vs Same-ID References:                  {sim_vanilla_to_refs:+.4f}")
    print(f"Identity-Conditioned vs Different-ID (3616):             {sim_id_to_diff_id:+.4f}")
    print(f"Masked Crop Identity: ID-Conditioned vs Refs:            {crop_sim_id_to_refs:+.4f}")
    print(f"Masked Crop Identity: Vanilla vs Refs:                   {crop_sim_vanilla_to_refs:+.4f}")

    print("\nConditioning Sensitivity (Identity-Conditioned vs Vanilla Baseline):")
    print(f"Mean Pixel Delta inside Mask: {mean_pixel_delta_baseline:.2f} / 255")
    print(f"Max Pixel Delta inside Mask:  {max_pixel_delta_baseline:.2f} / 255")
    print(f"Fraction of Masked Pixels Altered: {fraction_pixels_changed * 100:.2f}%")
    print(f"SSIM between Vanilla and Identity Output: {ssim_id_vs_vanilla:.4f}")

    # 8. Save the 10 Required Visual Artifacts
    print("\nSaving 10 required visual artifacts...")
    # 01_target.png
    cv2.imwrite(str(artifacts_dir / "01_target.png"), cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
    # 02_mask.png
    cv2.imwrite(str(artifacts_dir / "02_mask.png"), (mask * 255.0).astype(np.uint8))
    # 03_reference_A.png
    cv2.imwrite(str(artifacts_dir / "03_reference_A.png"), cv2.cvtColor(ref_a_rgb, cv2.COLOR_RGB2BGR))
    # 04_reference_B.png
    cv2.imwrite(str(artifacts_dir / "04_reference_B.png"), cv2.cvtColor(ref_b_rgb, cv2.COLOR_RGB2BGR))
    # 05_reference_C.png
    cv2.imwrite(str(artifacts_dir / "05_reference_C.png"), cv2.cvtColor(ref_c_rgb, cv2.COLOR_RGB2BGR))
    # 06_vanilla_baseline.png
    cv2.imwrite(str(artifacts_dir / "06_vanilla_baseline.png"), cv2.cvtColor(vanilla_rgb, cv2.COLOR_RGB2BGR))
    # 07_identity_conditioned_corrected.png
    cv2.imwrite(str(artifacts_dir / "07_identity_conditioned_corrected.png"), cv2.cvtColor(final_identity_recon, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(artifacts_dir / "07_identity_conditioned.png"), cv2.cvtColor(final_identity_recon, cv2.COLOR_RGB2BGR))

    # 08_masked_crop_vanilla.png
    crop_vanilla_large = vanilla_rgb[c_y1:c_y2, c_x1:c_x2]
    cv2.imwrite(str(artifacts_dir / "08_masked_crop_vanilla.png"), cv2.cvtColor(crop_vanilla_large, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(artifacts_dir / "08_masked_region_crop_baseline.png"), cv2.cvtColor(crop_vanilla_large, cv2.COLOR_RGB2BGR))

    # 09_masked_crop_identity.png
    crop_id_large = final_identity_recon[c_y1:c_y2, c_x1:c_x2]
    cv2.imwrite(str(artifacts_dir / "09_masked_crop_identity.png"), cv2.cvtColor(crop_id_large, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(artifacts_dir / "09_masked_region_crop_identity.png"), cv2.cvtColor(crop_id_large, cv2.COLOR_RGB2BGR))

    # 10_final_comparison.png (Multi-panel grid: Target, Mask, Vanilla, Identity-Conditioned, Crop Vanilla, Crop Identity)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Ground Truth Target (ID 7367)", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title(f"Synthetic Mask ({num_masked_pixels:,} px)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(ref_a_rgb)
    axes[0, 2].set_title("Reference Image A (ID 13401)", fontsize=11)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(vanilla_rgb)
    axes[1, 0].set_title("Vanilla Diffusion Baseline", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(final_identity_recon)
    axes[1, 1].set_title("Corrected Identity-Conditioned", fontsize=11)
    axes[1, 1].axis("off")

    # Side-by-side zoom crop
    crop_comparison = np.hstack([crop_vanilla_large, crop_id_large])
    axes[1, 2].imshow(crop_comparison)
    axes[1, 2].set_title("Crop Comparison [Vanilla | Identity]", fontsize=11)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(str(artifacts_dir / "10_final_comparison.png"), bbox_inches="tight")
    plt.close()

    # Also copy all 10 artifacts to the conversation brain artifacts directory for direct user access
    brain_artifacts_dir = Path("C:/Users/sasan/.gemini/antigravity/brain/fbe3e04a-609f-4ac9-8552-a9b57e8d7a2a/artifacts_identity_conditioned")
    brain_artifacts_dir.mkdir(parents=True, exist_ok=True)
    for art in artifacts_dir.glob("*.png"):
        shutil.copy(art, brain_artifacts_dir / art.name)

    total_runtime = time.time() - total_start_time

    # 9. Save Generation Config and Metrics JSON
    results = {
        "experiment": {
            "name": "Phase 7 Corrected Identity-Conditioned Diffusion Generation",
            "model": "runwayml/stable-diffusion-inpainting",
            "device": "cpu",
            "precision": "fp32",
            "seed": 42,
            "inference_steps": 5,
            "guidance_scale": 7.5,
            "working_resolution": [512, 512],
            "original_resolution": [1024, 1024],
            "num_candidates": 1,
            "target_id": "7367",
            "reference_ids": reference_ids,
            "reference_weights": ref_weights,
            "masked_pixels": num_masked_pixels,
            "mask_coverage_percentage": mask_coverage,
            "diffusion_runtime_seconds": t_inpaint,
            "total_runtime_seconds": total_runtime
        },
        "unmasked_invariance": {
            "passed": passed_invariance,
            "max_unmasked_diff": float(max_unmasked_diff),
            "violated_pixels": int(violated_pixels)
        },
        "masked_region_reconstruction_vs_ground_truth": {
            "identity_conditioned": {
                "mae": mae_id,
                "rmse": rmse_id,
                "psnr_db": psnr_id,
                "crop_ssim": ssim_id,
                "sobel_gradient_error": sobel_id,
                "laplacian_5x5_error": laplacian_id
            },
            "vanilla_baseline": {
                "mae": mae_v,
                "rmse": rmse_v,
                "psnr_db": psnr_v,
                "crop_ssim": ssim_v,
                "sobel_gradient_error": sobel_v,
                "laplacian_5x5_error": laplacian_v
            },
            "delta_identity_minus_vanilla": {
                "mae_delta": mae_id - mae_v,
                "rmse_delta": rmse_id - rmse_v,
                "psnr_delta_db": psnr_id - psnr_v,
                "ssim_delta": ssim_id - ssim_v,
                "sobel_delta": sobel_id - sobel_v,
                "laplacian_delta": laplacian_id - laplacian_v
            }
        },
        "conditioning_sensitivity": {
            "mean_pixel_delta_baseline": mean_pixel_delta_baseline,
            "max_pixel_delta_baseline": max_pixel_delta_baseline,
            "fraction_masked_pixels_changed": fraction_pixels_changed,
            "ssim_between_vanilla_and_identity": ssim_id_vs_vanilla,
            "measurably_different": bool(mean_pixel_delta_baseline > 1.0)
        },
        "biometric_arcface_identity_evaluation": {
            "whole_face": {
                "target_vs_same_id_refs": sim_target_to_refs,
                "identity_conditioned_vs_same_id_refs": sim_id_to_refs,
                "vanilla_baseline_vs_same_id_refs": sim_vanilla_to_refs,
                "identity_conditioned_vs_target": sim_id_to_target,
                "vanilla_baseline_vs_target": sim_vanilla_to_target,
                "identity_conditioned_vs_different_id": sim_id_to_diff_id,
                "identity_gain_over_vanilla": sim_id_to_refs - sim_vanilla_to_refs
            },
            "masked_region_crop": {
                "target_vs_same_id_refs": crop_sim_target_to_refs,
                "identity_conditioned_vs_same_id_refs": crop_sim_id_to_refs,
                "vanilla_baseline_vs_same_id_refs": crop_sim_vanilla_to_refs,
                "crop_identity_gain_over_vanilla": crop_sim_id_to_refs - crop_sim_vanilla_to_refs
            }
        },
        "artifacts_saved": [
            "01_target.png",
            "02_mask.png",
            "03_reference_A.png",
            "04_reference_B.png",
            "05_reference_C.png",
            "06_vanilla_baseline.png",
            "07_identity_conditioned.png",
            "08_masked_region_crop_baseline.png",
            "09_masked_region_crop_identity.png",
            "10_final_comparison.png"
        ]
    }

    with open(exp_dir / "generation_config.json", "w") as f:
        json.dump(results["experiment"], f, indent=2)

    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(results["experiment"], f, indent=2)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    shutil.copy(exp_dir / "metrics.json", brain_artifacts_dir / "metrics.json")
    shutil.copy(exp_dir / "generation_config.json", brain_artifacts_dir / "generation_config.json")
    shutil.copy(exp_dir / "experiment_config.json", brain_artifacts_dir / "experiment_config.json")

    # Also save to artifacts_dir
    with open(artifacts_dir / "experiment_config.json", "w") as f:
        json.dump(results["experiment"], f, indent=2)
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("IDENTITY-CONDITIONED EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Total Runtime: {total_runtime:.2f}s (Diffusion: {t_inpaint:.2f}s)")
    print(f"Artifacts saved to: {artifacts_dir} and {brain_artifacts_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_experiment()
