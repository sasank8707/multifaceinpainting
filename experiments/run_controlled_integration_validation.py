"""
Comprehensive Controlled Integration and Anti-Leakage Validation Script.

Executes:
- PART 1: Identity Adapter Anti-Leakage & Token Trace Analysis
- PART 2: Coarse Reconstructor Validation on Held-Out Identity
- PART 3: Real Inpainting UNet Single-Forward Integration & Reference Swap Test

STRICT CONSTRAINTS:
- ZERO full diffusion generation loops
- ZERO architectural changes after experiment
- Complete empirical reporting without assuming success from numerical sensitivity alone
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from modules.identity_fusion.regional_fusion import RegionalIdentityFusionModule
from models.diffusion.conditioning_adapter import IdentityConditioningAdapter
from modules.structural_guidance.coarse_reconstructor import CoarseReconstructor
from modules.utils.device import get_device_config


def compute_laplacian_5x5(img_rgb: np.ndarray) -> np.ndarray:
    """Computes 5x5 Laplacian response magnitude matching Wei et al."""
    lap_5x5 = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=np.float32)
    img_f = img_rgb.astype(np.float32) / 255.0
    resp = np.zeros_like(img_f)
    for c in range(3):
        resp[:, :, c] = cv2.filter2D(img_f[:, :, c], -1, lap_5x5)
    return np.mean(np.abs(resp), axis=2)


def run_part1_identity_leakage(ds: CelebAHQIdentityDataset, device: torch.device) -> dict:
    print("\n" + "=" * 75)
    print("PART 1: IDENTITY ADAPTER ANTI-LEAKAGE & TOKEN FLOW VERIFICATION")
    print("=" * 75)

    # 1. Target and Reference Selection
    # Target: CelebA-HQ 7367 (Identity 3111)
    target_hq = 7367
    target_id = "3111"
    ref_a_hqs = [13401, 23058, 23414] # Same identity (3111)
    ref_b_hqs = [5447, 6008, 7695]    # Different identity (3330)
    ref_c_hqs = [5447, 24194, 11500]  # Shuffled / random identities (3330, 4745, 862)

    print(f"Target: CelebA-HQ ID {target_hq} (Subject ID '{target_id}')")
    print(f"Ref Set A (Same-Identity): {ref_a_hqs} (Subject ID '3111')")
    print(f"Ref Set B (Different-Identity): {ref_b_hqs} (Subject ID '3330')")
    print(f"Ref Set C (Shuffled-Identities): {ref_c_hqs} (Subjects: 3330, 4745, 862)")

    # Load images as [3, 256, 256] tensors in [-1, 1]
    t_tensor = ds._load_image_tensor(target_hq).unsqueeze(0).to(device)
    refs_a_tensors = [ds._load_image_tensor(h).unsqueeze(0).to(device) for h in ref_a_hqs]
    refs_b_tensors = [ds._load_image_tensor(h).unsqueeze(0).to(device) for h in ref_b_hqs]
    refs_c_tensors = [ds._load_image_tensor(h).unsqueeze(0).to(device) for h in ref_c_hqs]

    # Generate deterministic 20% elliptical mask on target
    mask_256 = ds._generate_severity_mask(idx=0, severity=0.20)
    mask_t = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    # 2. Extract Regional ArcFace Features
    encoder = RegionalIdentityEncoder(token_dim=512).to(device)
    encoder.eval()

    with torch.no_grad():
        out_t = encoder(t_tensor)
        t_spatial = out_t["spatial_tokens"] # [1, 256, 512]
        t_global = out_t["global_embedding"] # [1, 512]

        refs_a_spatial = [encoder(r)["spatial_tokens"] for r in refs_a_tensors]
        refs_b_spatial = [encoder(r)["spatial_tokens"] for r in refs_b_tensors]
        refs_c_spatial = [encoder(r)["spatial_tokens"] for r in refs_c_tensors]

        refs_a_global = torch.stack([encoder(r)["global_embedding"] for r in refs_a_tensors], dim=1).mean(dim=1)
        refs_b_global = torch.stack([encoder(r)["global_embedding"] for r in refs_b_tensors], dim=1).mean(dim=1)
        refs_c_global = torch.stack([encoder(r)["global_embedding"] for r in refs_c_tensors], dim=1).mean(dim=1)

    # 3. Multi-Reference Regional Transformer Fusion
    fusion_module = RegionalIdentityTransformer(
        token_dim=512,
        num_heads=8,
        num_layers=2,
        spatial_patch_grid=(16, 16),
        dropout=0.0
    ).to(device)
    fusion_module.eval()

    # Equal Phase 3 retrieval weights
    ref_weights = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], device=device)

    with torch.no_grad():
        fused_a = fusion_module(
            target_tokens=t_spatial,
            target_mask=mask_t,
            reference_tokens_list=refs_a_spatial,
            reference_weights=ref_weights
        )["fused_tokens"] # [1, 256, 512]

        fused_b = fusion_module(
            target_tokens=t_spatial,
            target_mask=mask_t,
            reference_tokens_list=refs_b_spatial,
            reference_weights=ref_weights
        )["fused_tokens"]

        fused_c = fusion_module(
            target_tokens=t_spatial,
            target_mask=mask_t,
            reference_tokens_list=refs_c_spatial,
            reference_weights=ref_weights
        )["fused_tokens"]

    # 4. Load Trained Identity Conditioning Adapter Checkpoint
    adapter = IdentityConditioningAdapter(
        in_dim=512,
        out_dim=768,
        num_tokens=256,
        out_tokens=77,
        num_id_queries=16,
        dropout=0.0
    ).to(device)

    ckpt_path = PROJECT_ROOT / "checkpoints/identity_adapter/identity_adapter_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        adapter.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained Identity Adapter checkpoint from {ckpt_path.name} (Epoch {ckpt.get('epoch', 'N/A')})")
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found; using initialized weights.")
    adapter.eval()

    with torch.no_grad():
        out_a = adapter(identity_tokens=fused_a)
        out_b = adapter(identity_tokens=fused_b)
        out_c = adapter(identity_tokens=fused_c)

        cond_a = out_a["projected_tokens"] # [1, 77, 768]
        cond_b = out_b["projected_tokens"]
        cond_c = out_c["projected_tokens"]

        id_tokens_a = out_a["id_feature_tokens"] # [1, 16, 768]
        id_tokens_b = out_b["id_feature_tokens"]
        id_tokens_c = out_c["id_feature_tokens"]

        probe_a = out_a["probe_arcface"] # [1, 512]
        probe_b = out_b["probe_arcface"]
        probe_c = out_c["probe_arcface"]

    # 5. Measure Cosine Similarities
    # Fused spatial tokens cosine (flattened)
    sim_fused_ab = F.cosine_similarity(fused_a.flatten(1), fused_b.flatten(1), dim=-1).item()
    sim_fused_ac = F.cosine_similarity(fused_a.flatten(1), fused_c.flatten(1), dim=-1).item()
    sim_fused_bc = F.cosine_similarity(fused_b.flatten(1), fused_c.flatten(1), dim=-1).item()

    # Full conditioning sequence cosine (77 tokens)
    sim_cond_ab = F.cosine_similarity(cond_a.flatten(1), cond_b.flatten(1), dim=-1).item()
    sim_cond_ac = F.cosine_similarity(cond_a.flatten(1), cond_c.flatten(1), dim=-1).item()

    # Queried ID tokens cosine (16 tokens)
    sim_id_ab = F.cosine_similarity(id_tokens_a.flatten(1), id_tokens_b.flatten(1), dim=-1).item()
    sim_id_ac = F.cosine_similarity(id_tokens_a.flatten(1), id_tokens_c.flatten(1), dim=-1).item()

    # Probe vs Target GT ArcFace
    probe_a_vs_target = F.cosine_similarity(probe_a, t_global, dim=-1).item()
    probe_b_vs_target = F.cosine_similarity(probe_b, t_global, dim=-1).item()
    probe_c_vs_target = F.cosine_similarity(probe_c, t_global, dim=-1).item()

    # Probe vs Reference GT ArcFace
    probe_a_vs_refa = F.cosine_similarity(probe_a, refs_a_global, dim=-1).item()
    probe_b_vs_refb = F.cosine_similarity(probe_b, refs_b_global, dim=-1).item()
    probe_c_vs_refc = F.cosine_similarity(probe_c, refs_c_global, dim=-1).item()

    # Ground truth reference-to-target similarity
    gt_target_vs_refa = F.cosine_similarity(t_global, refs_a_global, dim=-1).item()
    gt_target_vs_refb = F.cosine_similarity(t_global, refs_b_global, dim=-1).item()

    # 6. Spatial Token Difference Inside vs Outside Mask
    # Downsample mask to 16x16 grid
    mask_16 = F.interpolate(mask_t, size=(16, 16), mode="nearest").squeeze().bool().cpu().numpy().flatten()
    diff_ab_per_token = torch.norm(fused_a - fused_b, p=2, dim=-1).squeeze().cpu().numpy() # [256]
    diff_ac_per_token = torch.norm(fused_a - fused_c, p=2, dim=-1).squeeze().cpu().numpy()

    mean_diff_ab_inside = float(np.mean(diff_ab_per_token[mask_16]))
    mean_diff_ab_outside = float(np.mean(diff_ab_per_token[~mask_16]))
    mean_diff_ac_inside = float(np.mean(diff_ac_per_token[mask_16]))
    mean_diff_ac_outside = float(np.mean(diff_ac_per_token[~mask_16]))

    results = {
        "target_hq": target_hq,
        "target_id": target_id,
        "ref_a_hqs": ref_a_hqs,
        "ref_b_hqs": ref_b_hqs,
        "ref_c_hqs": ref_c_hqs,
        "sim_fused_ab": sim_fused_ab,
        "sim_fused_ac": sim_fused_ac,
        "sim_fused_bc": sim_fused_bc,
        "sim_cond_ab": sim_cond_ab,
        "sim_cond_ac": sim_cond_ac,
        "sim_id_ab": sim_id_ab,
        "sim_id_ac": sim_id_ac,
        "probe_a_vs_target": probe_a_vs_target,
        "probe_b_vs_target": probe_b_vs_target,
        "probe_c_vs_target": probe_c_vs_target,
        "probe_a_vs_refa": probe_a_vs_refa,
        "probe_b_vs_refb": probe_b_vs_refb,
        "probe_c_vs_refc": probe_c_vs_refc,
        "gt_target_vs_refa": gt_target_vs_refa,
        "gt_target_vs_refb": gt_target_vs_refb,
        "mean_diff_ab_inside_mask": mean_diff_ab_inside,
        "mean_diff_ab_outside_mask": mean_diff_ab_outside,
        "mean_diff_ac_inside_mask": mean_diff_ac_inside,
        "mean_diff_ac_outside_mask": mean_diff_ac_outside,
        "num_masked_patches": int(np.sum(mask_16)),
        "num_unmasked_patches": int(np.sum(~mask_16))
    }

    print("\n--- QUANTITATIVE RESULTS: PART 1 ---")
    print(f"Cosine Similarity T_fused(A, B): {sim_fused_ab:.4f}")
    print(f"Cosine Similarity T_fused(A, C): {sim_fused_ac:.4f}")
    print(f"Cosine Similarity Cond(A, B) [77 tokens]: {sim_cond_ab:.4f}")
    print(f"Cosine Similarity Cond(A, C) [77 tokens]: {sim_cond_ac:.4f}")
    print(f"Cosine Similarity ID-Tokens(A, B) [16 queries]: {sim_id_ab:.4f}")
    print(f"Cosine Similarity ID-Tokens(A, C) [16 queries]: {sim_id_ac:.4f}")
    print(f"GT ArcFace Target vs Ref A (Same ID): {gt_target_vs_refa:.4f}")
    print(f"GT ArcFace Target vs Ref B (Diff ID): {gt_target_vs_refb:.4f}")
    print(f"Probe ArcFace A vs Target GT: {probe_a_vs_target:.4f}")
    print(f"Probe ArcFace B vs Target GT: {probe_b_vs_target:.4f}")
    print(f"Probe ArcFace C vs Target GT: {probe_c_vs_target:.4f}")
    print(f"Probe ArcFace B vs Ref B GT:  {probe_b_vs_refb:.4f}")
    print(f"Token L2 Delta (A vs B) Inside Mask:  {mean_diff_ab_inside:.4f}")
    print(f"Token L2 Delta (A vs B) Outside Mask: {mean_diff_ab_outside:.4f}")

    return results, cond_a, cond_b


def run_part2_coarse_validation(ds: CelebAHQIdentityDataset, device: torch.device) -> dict:
    print("\n" + "=" * 75)
    print("PART 2: COARSE RECONSTRUCTOR VALIDATION ON HELD-OUT IDENTITY")
    print("=" * 75)

    # Held-out validation subject: CelebA-HQ 3616 (Identity '3330' from 'val' split)
    val_hq = 3616
    val_id = "3330"
    print(f"Held-out Validation Image: CelebA-HQ {val_hq} (Subject ID '{val_id}')")

    img_path = ds._get_image_path(val_hq)
    bgr = cv2.imread(str(img_path))
    rgb_gt = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_gt_256 = cv2.resize(rgb_gt, (256, 256), interpolation=cv2.INTER_AREA)

    # 20% elliptical mask
    mask_256 = ds._generate_severity_mask(idx=42, severity=0.20)
    mask_bool = (mask_256 > 0.5)

    # Masked input image
    masked_img_256 = rgb_gt_256.copy()
    masked_img_256[mask_bool] = 0

    t_masked_img = torch.from_numpy(masked_img_256).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t_mask = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float()

    t_masked_img = t_masked_img.to(device)
    t_mask = t_mask.to(device)

    # 1. Untrained Model
    untrained_model = CoarseReconstructor(in_channels=4, out_channels=3, base_channels=32).to(device)
    untrained_model.eval()

    # 2. Trained Model Checkpoint
    trained_model = CoarseReconstructor(in_channels=4, out_channels=3, base_channels=32).to(device)
    ckpt_path = PROJECT_ROOT / "checkpoints/coarse_reconstructor/coarse_reconstructor_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        trained_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained Coarse Reconstructor checkpoint from {ckpt_path.name} (Val Loss: {ckpt.get('val_loss', 'N/A'):.4f})")
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found; evaluation will use uninitialized weights.")
    trained_model.eval()

    with torch.no_grad():
        out_untrained = untrained_model(t_masked_img, t_mask)["composited_output"]
        out_trained = trained_model(t_masked_img, t_mask)["composited_output"]

    # Convert back to uint8 [H, W, 3]
    recon_untrained_np = ((out_untrained.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    recon_trained_np = ((out_trained.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # Metrics inside mask
    # Spatial std per channel and overall
    std_untrained = float(np.std(recon_untrained_np[mask_bool]))
    std_trained = float(np.std(recon_trained_np[mask_bool]))
    std_gt = float(np.std(rgb_gt_256[mask_bool]))

    # Mean RGB color inside mask
    mean_rgb_untrained = [float(np.mean(recon_untrained_np[mask_bool, c])) for c in range(3)]
    mean_rgb_trained = [float(np.mean(recon_trained_np[mask_bool, c])) for c in range(3)]
    mean_rgb_gt = [float(np.mean(rgb_gt_256[mask_bool, c])) for c in range(3)]

    # 5x5 Laplacian edge energy inside mask
    lap_gt = compute_laplacian_5x5(rgb_gt_256)
    lap_untrained = compute_laplacian_5x5(recon_untrained_np)
    lap_trained = compute_laplacian_5x5(recon_trained_np)

    lap_energy_gt = float(np.mean(lap_gt[mask_bool]))
    lap_energy_untrained = float(np.mean(lap_untrained[mask_bool]))
    lap_energy_trained = float(np.mean(lap_trained[mask_bool]))

    # Boundary gradient continuity: difference across dilated mask border (width 5)
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask_256.astype(np.uint8), kernel_5)
    border_zone = (dilated_mask > 0) & (~mask_bool)

    diff_border_untrained = float(np.mean(np.abs(recon_untrained_np[border_zone].astype(float) - rgb_gt_256[border_zone].astype(float))))
    diff_border_trained = float(np.mean(np.abs(recon_trained_np[border_zone].astype(float) - rgb_gt_256[border_zone].astype(float))))

    # Save visual artifact
    artifacts_dir = PROJECT_ROOT / "checkpoints/coarse_reconstructor"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vis_path = artifacts_dir / "held_out_validation_comparison.png"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    axes[0].imshow(rgb_gt_256)
    axes[0].set_title(f"GT (CelebA-HQ {val_hq})", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(masked_img_256)
    axes[1].set_title("Masked Input (20% Face Mask)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(recon_untrained_np)
    axes[2].set_title(f"Untrained Coarse\n(Spatial std={std_untrained:.1f}, Red={mean_rgb_untrained[0]:.0f})", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(recon_trained_np)
    axes[3].set_title(f"Trained Coarse Checkpoint\n(Spatial std={std_trained:.1f}, Lap={lap_energy_trained:.4f})", fontsize=10)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(vis_path), bbox_inches="tight")
    plt.close()
    print(f"Saved validation comparison image to {vis_path}")

    results = {
        "val_hq": val_hq,
        "val_id": val_id,
        "std_gt": std_gt,
        "std_untrained": std_untrained,
        "std_trained": std_trained,
        "mean_rgb_gt": mean_rgb_gt,
        "mean_rgb_untrained": mean_rgb_untrained,
        "mean_rgb_trained": mean_rgb_trained,
        "lap_energy_gt": lap_energy_gt,
        "lap_energy_untrained": lap_energy_untrained,
        "lap_energy_trained": lap_energy_trained,
        "diff_border_untrained": diff_border_untrained,
        "diff_border_trained": diff_border_trained,
        "vis_path": str(vis_path)
    }

    print("\n--- QUANTITATIVE RESULTS: PART 2 ---")
    print(f"Masked Region Spatial Std: GT={std_gt:.2f} | Untrained={std_untrained:.2f} | Trained={std_trained:.2f}")
    print(f"Mean RGB inside mask: GT={mean_rgb_gt} | Untrained={mean_rgb_untrained} | Trained={mean_rgb_trained}")
    print(f"Laplacian 5x5 Energy: GT={lap_energy_gt:.4f} | Untrained={lap_energy_untrained:.4f} | Trained={lap_energy_trained:.4f}")
    print(f"Boundary Continuity Error: Untrained={diff_border_untrained:.2f} | Trained={diff_border_trained:.2f}")

    return results, recon_trained_np


def run_part3_unet_single_forward(
    ds: CelebAHQIdentityDataset,
    part1_res: dict,
    part2_recon: np.ndarray,
    cond_id_a: torch.Tensor,
    cond_id_b: torch.Tensor,
    device: torch.device
) -> dict:
    print("\n" + "=" * 75)
    print("PART 3: REAL UNET SINGLE-FORWARD INTEGRATION & REFERENCE SWAP TEST")
    print("=" * 75)

    from diffusers import StableDiffusionInpaintPipeline

    # 1. Load Pretrained Pipeline Components (from local cache)
    print("Loading pretrained RunwayML SD inpainting UNet, VAE, and Text Encoder...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    local_model_dir = os.path.join(
        cache_dir, "models--runwayml--stable-diffusion-inpainting", "snapshots"
    )
    load_path = "runwayml/stable-diffusion-inpainting"
    if os.path.isdir(local_model_dir):
        snapshots = [os.path.join(local_model_dir, d) for d in os.listdir(local_model_dir)]
        if snapshots:
            load_path = snapshots[0]

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True if os.path.isdir(load_path) else False
    ).to(device)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    unet.eval()
    vae.eval()
    text_encoder.eval()

    # 2. Target and Guidance Setup (Working resolution 512x512, latent size 64x64)
    target_hq = part1_res["target_hq"]
    img_path = ds._get_image_path(target_hq)
    bgr = cv2.imread(str(img_path))
    rgb_gt = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_gt_512 = cv2.resize(rgb_gt, (512, 512), interpolation=cv2.INTER_AREA)

    # 20% elliptical mask resized to 512x512
    mask_256 = ds._generate_severity_mask(idx=0, severity=0.20)
    mask_512 = cv2.resize(mask_256, (512, 512), interpolation=cv2.INTER_NEAREST)
    mask_bool_512 = (mask_512 > 0.5)

    # Prepare VAE image inputs:
    # A: Masked target with mask zeroed out (Standard SD inpainting input)
    masked_target_512 = rgb_gt_512.copy()
    masked_target_512[mask_bool_512] = 0

    # B: Phase-5 coarse structural guidance image (trained coarse prediction composited inside mask)
    coarse_512 = cv2.resize(part2_recon, (512, 512), interpolation=cv2.INTER_CUBIC)
    guided_target_512 = rgb_gt_512.copy()
    guided_target_512[mask_bool_512] = coarse_512[mask_bool_512]

    # Convert to VAE tensors [-1, 1]
    t_masked_target = torch.from_numpy(masked_target_512).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t_guided_target = torch.from_numpy(guided_target_512).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t_mask_512 = torch.from_numpy(mask_512).unsqueeze(0).unsqueeze(0).float()

    t_masked_target = t_masked_target.to(device)
    t_guided_target = t_guided_target.to(device)
    t_mask_512 = t_mask_512.to(device)

    # Encode VAE Latents
    with torch.no_grad():
        # Masked target latents (standard zeroed-out mask)
        latents_masked_target = vae.encode(t_masked_target).latent_dist.sample() * vae.config.scaling_factor # [1, 4, 64, 64]
        # Coarse structural guidance latents (Phase 5 coarse reconstruction inside mask)
        latents_guided_target = vae.encode(t_guided_target).latent_dist.sample() * vae.config.scaling_factor # [1, 4, 64, 64]

        # Downsample mask to latent grid (64x64)
        mask_latent = F.interpolate(t_mask_512, size=(64, 64), mode="nearest") # [1, 1, 64, 64]

    # 3. Construct 9-Channel UNet Input Tensors
    # Channel layout:
    #   0:4 = noisy latents z_t
    #   4:5 = mask_latent
    #   5:9 = masked_image_latents
    torch.manual_seed(42)
    z_t = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float32)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    # 9-channel tensor with zeroed mask latents in 5:9
    unet_input_standard = torch.cat([z_t, mask_latent, latents_masked_target], dim=1) # [1, 9, 64, 64]
    # 9-channel tensor with Phase-5 coarse guidance in 5:9
    unet_input_guided = torch.cat([z_t, mask_latent, latents_guided_target], dim=1)   # [1, 9, 64, 64]

    print(f"Constructed UNet Input Standard Shape: {unet_input_standard.shape} (Channels: 0:4=z_t, 4:5=mask, 5:9=masked_latents)")
    print(f"Constructed UNet Input Guided Shape:   {unet_input_guided.shape} (Channels: 0:4=z_t, 4:5=mask, 5:9=coarse_guided_latents)")

    # 4. Text and Identity Conditionings
    # Condition 1: Vanilla Prompt
    vanilla_prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    text_inputs = tokenizer(
        vanilla_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        vanilla_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0] # [1, 77, 768]

    # Condition 2: Trained Identity Conditioning from Ref Set A (Same ID, passed from Part 1)
    # Condition 4: Trained Identity Conditioning from Ref Set B (Different ID, passed from Part 1)
    # Both are [1, 77, 768] tensors ready for cross-attention

    # 5. Run Single UNet Forward Passes (NO DIFFUSION LOOPS)
    print("\nExecuting Single UNet Forward Passes at timestep t=500...")
    with torch.no_grad():
        # Pass 1: Vanilla Prompt + Standard Zeroed Latents
        eps_vanilla = unet(unet_input_standard, timestep, encoder_hidden_states=vanilla_hidden_states).sample # [1, 4, 64, 64]

        # Pass 2: Trained Identity (Set A) + Standard Zeroed Latents
        eps_id_only = unet(unet_input_standard, timestep, encoder_hidden_states=cond_id_a).sample

        # Pass 3: Trained Identity (Set A) + Coarse Guidance Latents
        eps_id_coarse = unet(unet_input_guided, timestep, encoder_hidden_states=cond_id_a).sample

        # Pass 4 (Reference Swap Test): Trained Identity (Set B - Different Identity) + Standard Zeroed Latents
        eps_id_swap_b = unet(unet_input_standard, timestep, encoder_hidden_states=cond_id_b).sample

    # 6. Quantitative Comparisons of Noise Predictions
    # Comparison 1: Vanilla vs Trained Identity Only
    diff_vanilla_vs_id = torch.norm(eps_vanilla - eps_id_only, p=2).item()
    sim_vanilla_vs_id = F.cosine_similarity(eps_vanilla.flatten(), eps_id_only.flatten(), dim=0).item()

    # Comparison 2: Trained Identity Only vs Trained Identity + Coarse Guidance
    diff_id_vs_coarse = torch.norm(eps_id_only - eps_id_coarse, p=2).item()
    sim_id_vs_coarse = F.cosine_similarity(eps_id_only.flatten(), eps_id_coarse.flatten(), dim=0).item()

    # Comparison 3: Reference Swap Test (Set A vs Set B)
    delta_eps_swap = torch.norm(eps_id_only - eps_id_swap_b, p=2).item()
    sim_swap_ab = F.cosine_similarity(eps_id_only.flatten(), eps_id_swap_b.flatten(), dim=0).item()

    # Spatial breakdown inside mask vs outside mask (mask_latent is [1, 1, 64, 64])
    mask_latent_bool = (mask_latent.squeeze() > 0.5) # [64, 64]
    eps_a_masked = eps_id_only.squeeze()[:, mask_latent_bool] # [4, N_masked]
    eps_b_masked = eps_id_swap_b.squeeze()[:, mask_latent_bool]
    delta_eps_masked = torch.norm(eps_a_masked - eps_b_masked, p=2).item()

    eps_a_unmasked = eps_id_only.squeeze()[:, ~mask_latent_bool]
    eps_b_unmasked = eps_id_swap_b.squeeze()[:, ~mask_latent_bool]
    delta_eps_unmasked = torch.norm(eps_a_unmasked - eps_b_unmasked, p=2).item()

    results = {
        "timestep": 500,
        "seed": 42,
        "input_standard_shape": list(unet_input_standard.shape),
        "input_guided_shape": list(unet_input_guided.shape),
        "diff_vanilla_vs_id": diff_vanilla_vs_id,
        "sim_vanilla_vs_id": sim_vanilla_vs_id,
        "diff_id_vs_coarse": diff_id_vs_coarse,
        "sim_id_vs_coarse": sim_id_vs_coarse,
        "delta_eps_swap_ab": delta_eps_swap,
        "sim_swap_ab": sim_swap_ab,
        "delta_eps_swap_inside_mask": delta_eps_masked,
        "delta_eps_swap_outside_mask": delta_eps_unmasked
    }

    print("\n--- QUANTITATIVE RESULTS: PART 3 ---")
    print(f"Vanilla vs Trained Identity L2: {diff_vanilla_vs_id:.4f} | Cosine: {sim_vanilla_vs_id:.4f}")
    print(f"Trained Identity vs +Coarse Guidance L2: {diff_id_vs_coarse:.4f} | Cosine: {sim_id_vs_coarse:.4f}")
    print(f"Reference Swap Test (Set A vs Set B) Delta Eps L2: {delta_eps_swap:.4f} | Cosine: {sim_swap_ab:.4f}")
    print(f"Reference Swap Delta Eps INSIDE Mask:  {delta_eps_masked:.4f}")
    print(f"Reference Swap Delta Eps OUTSIDE Mask: {delta_eps_unmasked:.4f}")

    return results


def main():
    print("=" * 80)
    print("STARTING CONTROLLED INTEGRATION & ANTI-LEAKAGE VALIDATION")
    print("=" * 80)

    device_cfg = get_device_config()
    device = device_cfg.device
    print(f"Runtime Device: {device_cfg.device_str} (is_cuda={device_cfg.is_cuda})")

    # Load dataset helper to retrieve test/val images
    ds = CelebAHQIdentityDataset(split="test", smoke_subset_size=1)

    # 1. PART 1
    t0 = time.time()
    part1_results, cond_id_a, cond_id_b = run_part1_identity_leakage(ds, device)
    t_part1 = time.time() - t0

    # 2. PART 2
    t0 = time.time()
    part2_results, part2_recon = run_part2_coarse_validation(ds, device)
    t_part2 = time.time() - t0

    # 3. PART 3
    t0 = time.time()
    part3_results = run_part3_unet_single_forward(ds, part1_results, part2_recon, cond_id_a, cond_id_b, device)
    t_part3 = time.time() - t0

    # Save comprehensive report
    all_results = {
        "device": device_cfg.device_str,
        "timing_seconds": {
            "part1": t_part1,
            "part2": t_part2,
            "part3": t_part3
        },
        "part1_identity_anti_leakage": part1_results,
        "part2_coarse_reconstructor_validation": part2_results,
        "part3_unet_single_forward_integration": part3_results
    }

    out_json = PROJECT_ROOT / "experiments/controlled_integration_validation_report.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ALL VALIDATION TASKS COMPLETED SUCCESSFULLY in {t_part1 + t_part2 + t_part3:.2f}s!")
    print(f"Comprehensive report saved to: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
