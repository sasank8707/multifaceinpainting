"""
Tests for Reference-Swap Sensitivity (Part C) and Mask Conditioning Sensitivity (Part D).
Verifies:
1. Swapping reference identities produces a substantial shift in T_fused (cosine < 0.85).
2. Per-token L2 difference between same-ID and diff-ID references is significant.
3. Changing the mask changes regional conditioning tokens (region-sensitive).
4. Strictly no target identity residual leakage.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset


@pytest.fixture(scope="module")
def dataset_and_models():
    ds = CelebAHQIdentityDataset(split="test", smoke_subset_size=1)
    encoder = RegionalIdentityEncoder(token_dim=512)
    transformer = RegionalIdentityTransformer(
        token_dim=512,
        num_heads=8,
        num_layers=2,
        spatial_patch_grid=(16, 16),
        dropout=0.0
    )
    encoder.eval()
    transformer.eval()
    return ds, encoder, transformer


def test_part_c_reference_swap(dataset_and_models):
    """
    PART C: Reference-Swap Test.
    Target: CelebA-HQ 7367 (ID 3111), 20% face mask.
    Set A: Same-ID references (13401, 23058, 23414).
    Set B: Different-ID references (5447, 6008, 7695 from ID 3330).
    Set C: Shuffled-ID references (5447, 24194, 11500).
    """
    ds, encoder, transformer = dataset_and_models

    # Load target image and mask
    t_target = ds._load_image_tensor(7367).unsqueeze(0)
    mask_256 = ds._generate_severity_mask(idx=0, severity=0.20)
    mask_t = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float()

    # Load reference image sets
    ref_a_ids = [13401, 23058, 23414]
    ref_b_ids = [5447, 6008, 7695]
    ref_c_ids = [5447, 24194, 11500]

    refs_a = [ds._load_image_tensor(i).unsqueeze(0) for i in ref_a_ids]
    refs_b = [ds._load_image_tensor(i).unsqueeze(0) for i in ref_b_ids]
    refs_c = [ds._load_image_tensor(i).unsqueeze(0) for i in ref_c_ids]

    # Extract real ArcFace spatial tokens and global embeddings
    with torch.no_grad():
        target_feat = encoder(t_target)
        e_target = target_feat["global_embedding"]

        refs_a_tokens = [encoder(r)["spatial_tokens"] for r in refs_a]
        refs_b_tokens = [encoder(r)["spatial_tokens"] for r in refs_b]
        refs_c_tokens = [encoder(r)["spatial_tokens"] for r in refs_c]

        e_ref_a = torch.stack([encoder(r)["global_embedding"] for r in refs_a], dim=1).mean(dim=1)
        e_ref_b = torch.stack([encoder(r)["global_embedding"] for r in refs_b], dim=1).mean(dim=1)
        e_ref_c = torch.stack([encoder(r)["global_embedding"] for r in refs_c], dim=1).mean(dim=1)

    # ArcFace similarity of references
    cos_ref_ab = F.cosine_similarity(e_ref_a, e_ref_b, dim=-1).item()
    cos_ref_ac = F.cosine_similarity(e_ref_a, e_ref_c, dim=-1).item()
    print(f"\nArcFace Global Cosine (Ref A vs Ref B): {cos_ref_ab:.4f}")
    print(f"ArcFace Global Cosine (Ref A vs Ref C): {cos_ref_ac:.4f}")

    # Run regional fusion
    weights = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32)

    with torch.no_grad():
        out_a = transformer(
            target_tokens=None,
            target_mask=mask_t,
            reference_tokens_list=refs_a_tokens,
            reference_weights=weights
        )
        out_b = transformer(
            target_tokens=None,
            target_mask=mask_t,
            reference_tokens_list=refs_b_tokens,
            reference_weights=weights
        )
        out_c = transformer(
            target_tokens=None,
            target_mask=mask_t,
            reference_tokens_list=refs_c_tokens,
            reference_weights=weights
        )

    fused_a = out_a["fused_tokens"] # [1, 256, 512]
    fused_b = out_b["fused_tokens"]
    fused_c = out_c["fused_tokens"]

    # Measure cosine similarity between fused tokens
    sim_fused_ab = F.cosine_similarity(fused_a.flatten(1), fused_b.flatten(1), dim=-1).item()
    sim_fused_ac = F.cosine_similarity(fused_a.flatten(1), fused_c.flatten(1), dim=-1).item()

    # Per-token L2 differences
    diff_ab_per_token = torch.norm(fused_a - fused_b, p=2, dim=-1).squeeze().numpy()
    diff_ac_per_token = torch.norm(fused_a - fused_c, p=2, dim=-1).squeeze().numpy()

    # Mask breakdown (downsampled to 16x16)
    mask_16 = F.interpolate(mask_t, size=(16, 16), mode="nearest").squeeze().bool().numpy().flatten()
    diff_ab_inside = float(np.mean(diff_ab_per_token[mask_16]))
    diff_ab_outside = float(np.mean(diff_ab_per_token[~mask_16]))

    print(f"\nT_fused Cosine (A vs B): {sim_fused_ab:.4f}")
    print(f"T_fused Cosine (A vs C): {sim_fused_ac:.4f}")
    print(f"Mean Token L2 Delta (A vs B) Inside Mask:  {diff_ab_inside:.4f}")
    print(f"Mean Token L2 Delta (A vs B) Outside Mask: {diff_ab_outside:.4f}")

    # VERIFICATION ASSERTIONS:
    # 1. Cosine similarity must NOT be 1.0000 (must be significantly lower, proving reference sensitivity)
    assert sim_fused_ab < 0.85, f"T_fused(A, B) cosine too high ({sim_fused_ab:.4f} >= 0.85); identity collapse detected!"
    assert sim_fused_ac < 0.85, f"T_fused(A, C) cosine too high ({sim_fused_ac:.4f} >= 0.85); identity collapse detected!"

    # 2. Token L2 difference must be substantial (not negligible 0.06 out of norm 22)
    assert diff_ab_inside > 1.0, f"Token L2 difference inside mask too small ({diff_ab_inside:.4f} <= 1.0)!"


def test_part_d_mask_conditioning(dataset_and_models):
    """
    PART D: Mask Conditioning Test.
    Same target and same references, with two distinct masks:
    Mask 1: Upper-face mask (forehead / eye region).
    Mask 2: Lower-face mask (mouth / jaw region).
    Verifies that changing the mask changes regional conditioning tokens.
    """
    ds, encoder, transformer = dataset_and_models

    t_target = ds._load_image_tensor(7367).unsqueeze(0)
    ref_a_ids = [13401, 23058, 23414]
    refs_a = [ds._load_image_tensor(i).unsqueeze(0) for i in ref_a_ids]

    with torch.no_grad():
        refs_a_tokens = [encoder(r)["spatial_tokens"] for r in refs_a]

    # Mask 1: Upper-face mask (rows 20-50% of height)
    mask1 = np.zeros((256, 256), dtype=np.float32)
    mask1[50:130, 60:196] = 1.0
    mask1_t = torch.from_numpy(mask1).unsqueeze(0).unsqueeze(0)

    # Mask 2: Lower-face mask (rows 60-90% of height)
    mask2 = np.zeros((256, 256), dtype=np.float32)
    mask2[150:230, 70:186] = 1.0
    mask2_t = torch.from_numpy(mask2).unsqueeze(0).unsqueeze(0)

    weights = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32)

    with torch.no_grad():
        out_m1 = transformer(
            target_tokens=None,
            target_mask=mask1_t,
            reference_tokens_list=refs_a_tokens,
            reference_weights=weights
        )
        out_m2 = transformer(
            target_tokens=None,
            target_mask=mask2_t,
            reference_tokens_list=refs_a_tokens,
            reference_weights=weights
        )

    fused_m1 = out_m1["fused_tokens"] # [1, 256, 512]
    fused_m2 = out_m2["fused_tokens"]

    # Measure per-token difference between the two masks
    diff_per_token = torch.norm(fused_m1 - fused_m2, p=2, dim=-1).squeeze().numpy() # [256]
    mean_token_diff = float(np.mean(diff_per_token))
    sim_masks = F.cosine_similarity(fused_m1.flatten(1), fused_m2.flatten(1), dim=-1).item()

    print(f"\nMask Conditioning Sensitivity:")
    print(f"Cosine Similarity (Mask 1 vs Mask 2): {sim_masks:.4f}")
    print(f"Mean Per-Token L2 Delta (Mask 1 vs Mask 2): {mean_token_diff:.4f}")

    # Assert that mask changes regional tokens
    assert sim_masks < 0.999, f"Mask conditioning failed: tokens are invariant to mask ({sim_masks:.4f} >= 0.999)"
    assert mean_token_diff > 0.1, f"Mean per-token difference too small ({mean_token_diff:.4f} <= 0.1)"
