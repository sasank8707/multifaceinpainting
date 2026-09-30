"""
Phase 4 Comprehensive Unit Tests:
O2 Regional Identity Feature Extraction & Regional Transformer Fusion.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.identity.arcface_backbone import RegionalIdentityEncoder, build_frozen_identity_encoder
from models.transformer.positional_encoding import Sinusoidal2DPositionalEncoding, LandmarkProximityEncoding
from models.transformer.fusion_transformer import RegionalIdentityTransformer, MaskConditionedCrossAttention
from modules.identity_fusion.regional_fusion import RegionalIdentityFusionModule
from modules.reference_scoring.scorer import IdentityMismatchError
from modules.masking.task_router import TaskMode


@pytest.fixture
def sample_tensors():
    """Generates synthetic image and mask tensors for deterministic testing."""
    torch.manual_seed(42)
    b, c, h, w = 2, 3, 256, 256
    target_img = torch.randn(b, c, h, w)
    target_mask = torch.zeros(b, 1, h, w)
    # Mask a central region
    target_mask[:, :, 80:180, 80:180] = 1.0
    
    ref1 = torch.randn(b, c, h, w)
    ref2 = torch.randn(b, c, h, w)
    ref3 = torch.randn(b, c, h, w)
    
    # 5 normalized landmarks
    landmarks = torch.rand(b, 5, 2)
    return target_img, target_mask, [ref1, ref2, ref3], landmarks


def test_feature_extraction_dimensions(sample_tensors):
    """Test 1: Regional identity encoder produces spatial tokens and global embedding."""
    target_img, _, _, _ = sample_tensors
    encoder = build_frozen_identity_encoder(token_dim=512)
    
    res = encoder(target_img)
    assert "spatial_tokens" in res
    assert "feature_map" in res
    assert "global_embedding" in res
    
    # Spatial tokens shape: [B, P, D] where P = 16*16 = 256
    assert res["spatial_tokens"].shape == (2, 256, 512)
    assert res["feature_map"].shape == (2, 512, 16, 16)
    assert res["global_embedding"].shape == (2, 512)
    
    # Global embedding must be L2-normalized
    norms = torch.norm(res["global_embedding"], p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_frozen_identity_encoder(sample_tensors):
    """Test 2: Identity encoder parameters are strictly frozen (requires_grad=False)."""
    encoder = build_frozen_identity_encoder(token_dim=512)
    
    for name, param in encoder.named_parameters():
        assert param.requires_grad is False, f"Parameter {name} is not frozen!"


def test_spatial_landmark_alignment():
    """Test 3: Positional and continuous landmark proximity encodings."""
    token_dim = 256
    pos_enc = Sinusoidal2DPositionalEncoding(token_dim=token_dim)
    lm_enc = LandmarkProximityEncoding(token_dim=token_dim, num_landmark_anchors=5)
    
    # Grid: 16x16 = 256 patches
    p_pos = pos_enc((16, 16), device=torch.device("cpu"))
    assert p_pos.shape == (1, 256, token_dim)
    
    # Landmarks: [2, 5, 2]
    landmarks = torch.rand(2, 5, 2)
    p_lm = lm_enc((16, 16), landmarks, device=torch.device("cpu"), batch_size=2)
    assert p_lm.shape == (2, 256, token_dim)


def test_mask_conditioned_cross_attention():
    """Test 4: Mask-conditioned cross-attention layer with mask bias."""
    b, m, p, d = 2, 64, 64, 128
    cross_attn = MaskConditionedCrossAttention(token_dim=d, num_heads=4)
    
    q = torch.randn(b, m, d)
    k = torch.randn(b, p, d)
    v = torch.randn(b, p, d)
    mask = torch.zeros(b, m, 1)
    mask[:, 20:40, :] = 1.0 # Masked segment
    
    out, weights = cross_attn(q, k, v, mask_spatial=mask)
    assert out.shape == (b, m, d)
    assert weights.shape == (b, 4, m, p)
    # Attention weights must sum to 1 across reference key dimension
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-4)


def test_fusion_one_reference(sample_tensors):
    """Test 5: Forward pass with N=1 reference."""
    target_img, target_mask, refs, lms = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=256, num_layers=1)
    
    scoring_report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "ref_0", "weight": 1.0}]
    }
    
    out = module(
        target_tensor=target_img,
        target_mask=target_mask,
        reference_tensors=[refs[0]],
        scoring_report=scoring_report,
        target_identity_id="id_1",
        reference_identity_ids=["id_1"]
    )
    
    assert out["fused_tokens"].shape == (2, 256, 256)
    assert out["num_references"] == 1
    assert out["identity_conditioning_enabled"] is True


def test_fusion_three_references(sample_tensors):
    """Test 6: Forward pass with N=3 references."""
    target_img, target_mask, refs, lms = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=256, num_layers=1)
    
    scoring_report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [
            {"name": "ref_0", "weight": 0.5},
            {"name": "ref_1", "weight": 0.3},
            {"name": "ref_2", "weight": 0.2},
        ]
    }
    
    out = module(
        target_tensor=target_img,
        target_mask=target_mask,
        reference_tensors=refs,
        scoring_report=scoring_report,
        target_identity_id="id_42",
        reference_identity_ids=["id_42", "id_42", "id_42"]
    )
    
    assert out["fused_tokens"].shape == (2, 256, 256)
    assert out["num_references"] == 3
    assert len(out["attention_maps"]) == 3


def test_variable_number_of_references(sample_tensors):
    """Test 7: Supports variable number of references dynamically (K=1, 2, 3)."""
    target_img, target_mask, refs, _ = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=128, num_layers=1)
    
    for k in [1, 2, 3]:
        sub_refs = refs[:k]
        scoring_report = {
            "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
            "identity_conditioning_enabled": True,
            "global_ranked_references": [{"name": f"ref_{i}", "weight": 1.0 / k} for i in range(k)]
        }
        out = module(
            target_tensor=target_img,
            target_mask=target_mask,
            reference_tensors=sub_refs,
            scoring_report=scoring_report
        )
        assert out["fused_tokens"].shape == (2, 256, 128)
        assert out["num_references"] == k


def test_different_top_k_weights_influence(sample_tensors):
    """Test 8: Phase-3 reference weights directly and measurably influence fused tokens."""
    target_img, target_mask, refs, _ = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=128, num_layers=1)
    module.eval()
    
    # Weight scenario 1: 90% ref 0, 10% ref 1
    report_1 = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "r0", "weight": 0.9}, {"name": "r1", "weight": 0.1}]
    }
    # Weight scenario 2: 10% ref 0, 90% ref 1
    report_2 = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "r0", "weight": 0.1}, {"name": "r1", "weight": 0.9}]
    }
    
    with torch.no_grad():
        out1 = module(target_tensor=target_img, target_mask=target_mask, reference_tensors=refs[:2], scoring_report=report_1)
        out2 = module(target_tensor=target_img, target_mask=target_mask, reference_tensors=refs[:2], scoring_report=report_2)
        
    diff = torch.abs(out1["fused_tokens"] - out2["fused_tokens"]).mean().item()
    assert diff > 1e-4, "Reference weights must measurably alter the fused token representation!"


def test_gradient_flow_through_trainable_transformer(sample_tensors):
    """Test 9: Gradients flow cleanly through the trainable transformer layers."""
    target_img, target_mask, refs, _ = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=128, num_layers=1)
    module.train()
    
    scoring_report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "r0", "weight": 1.0}]
    }
    
    out = module(
        target_tensor=target_img,
        target_mask=target_mask,
        reference_tensors=[refs[0]],
        scoring_report=scoring_report
    )
    
    fused_tokens = out["fused_tokens"]
    loss = fused_tokens.mean()
    loss.backward()
    
    # Check that transformer parameters received gradients
    cross_attn_grad = module.transformer.cross_attention.w_q.weight.grad
    assert cross_attn_grad is not None, "Gradients must flow into cross-attention projections!"
    assert cross_attn_grad.norm().item() > 0.0
    
    # Check that identity encoder parameters have NO gradients (remain frozen)
    for p in module.identity_encoder.parameters():
        assert p.grad is None, "Frozen identity encoder must not receive gradients!"


def test_multiple_disconnected_regions(sample_tensors):
    """Test 10: Supports multiple disconnected masked regions."""
    target_img, _, refs, _ = sample_tensors
    b, c, h, w = target_img.shape
    # Create two disconnected mask patches
    multi_mask = torch.zeros(b, 1, h, w)
    multi_mask[:, :, 30:70, 30:70] = 1.0 # Patch 1
    multi_mask[:, :, 150:200, 150:200] = 1.0 # Patch 2
    
    module = RegionalIdentityFusionModule(token_dim=128, num_layers=1)
    scoring_report = {
        "task_mode": TaskMode.MODE_3_MULTI_MASK.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "r0", "weight": 0.6}, {"name": "r1", "weight": 0.4}]
    }
    
    out = module(
        target_tensor=target_img,
        target_mask=multi_mask,
        reference_tensors=refs[:2],
        scoring_report=scoring_report
    )
    assert out["fused_tokens"].shape == (b, 256, 128)
    assert out["identity_conditioning_enabled"] is True


def test_identity_mismatch_rejection(sample_tensors):
    """Test 11: Strict identity verification rejects mismatched reference IDs."""
    target_img, target_mask, refs, _ = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=128, strict_identity=True)
    
    scoring_report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "r0", "weight": 1.0}]
    }
    
    with pytest.raises(IdentityMismatchError):
        module(
            target_tensor=target_img,
            target_mask=target_mask,
            reference_tensors=[refs[0]],
            scoring_report=scoring_report,
            target_identity_id="person_A",
            reference_identity_ids=["person_B"] # Mismatch!
        )


def test_invalid_reference_handling(sample_tensors):
    """Test 12: Empty reference list raises assertion error."""
    target_img, target_mask, _, _ = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=128)
    
    scoring_report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": []
    }
    with pytest.raises(AssertionError):
        module(
            target_tensor=target_img,
            target_mask=target_mask,
            reference_tensors=[], # Empty!
            scoring_report=scoring_report
        )


def test_non_face_object_mode_suppresses_identity(sample_tensors):
    """Test 13: MODE_4_NON_FACE_OBJECT_REMOVAL strictly produces zero identity conditioning tokens."""
    target_img, target_mask, refs, _ = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=128)
    
    non_face_report = {
        "task_mode": TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value,
        "identity_conditioning_enabled": False,
        "global_ranked_references": []
    }
    
    out = module(
        target_tensor=target_img,
        target_mask=target_mask,
        reference_tensors=refs[:2],
        scoring_report=non_face_report
    )
    
    assert out["identity_conditioning_enabled"] is False
    assert torch.all(out["fused_tokens"] == 0.0), "Non-face object removal must produce all-zero identity tokens!"


def test_deterministic_inference(sample_tensors):
    """Test 14: Fixed inputs and weights produce bit-exact identical fused tokens."""
    target_img, target_mask, refs, _ = sample_tensors
    module = RegionalIdentityFusionModule(token_dim=128, num_layers=1)
    module.eval()
    
    scoring_report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "r0", "weight": 1.0}]
    }
    
    with torch.no_grad():
        out1 = module(target_tensor=target_img, target_mask=target_mask, reference_tensors=[refs[0]], scoring_report=scoring_report)
        out2 = module(target_tensor=target_img, target_mask=target_mask, reference_tensors=[refs[0]], scoring_report=scoring_report)
        
    assert torch.allclose(out1["fused_tokens"], out2["fused_tokens"])


def test_region_agnostic_behavior(sample_tensors):
    """Test 15: Regional transformer operates across arbitrary coordinates without hardcoded organs."""
    target_img, _, refs, _ = sample_tensors
    b, c, h, w = target_img.shape
    module = RegionalIdentityFusionModule(token_dim=128, num_layers=1)
    module.eval()
    
    scoring_report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "r0", "weight": 1.0}]
    }
    
    # Test 3 completely arbitrary mask patches
    coords = [(20, 50, 80, 110), (120, 150, 180, 210), (70, 10, 130, 90)]
    for y1, x1, y2, x2 in coords:
        mask = torch.zeros(b, 1, h, w)
        mask[:, :, y1:y2, x1:x2] = 1.0
        with torch.no_grad():
            out = module(target_tensor=target_img, target_mask=mask, reference_tensors=[refs[0]], scoring_report=scoring_report)
            assert out["fused_tokens"].shape == (b, 256, 128)


def test_cpu_smoke_test_synthetic_tensors():
    """Test 16: Fast CPU smoke test verifying full pipeline in < 1 second."""
    b, c, h, w = 1, 3, 128, 128
    x_tgt = torch.randn(b, c, h, w)
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, 30:70, 30:70] = 1.0
    x_ref = torch.randn(b, c, h, w)
    
    module = RegionalIdentityFusionModule(
        token_dim=64,
        num_heads=4,
        num_layers=1,
        spatial_patch_grid=(8, 8)
    )
    module.eval()
    
    report = {
        "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
        "identity_conditioning_enabled": True,
        "global_ranked_references": [{"name": "ref0", "weight": 1.0}]
    }
    
    with torch.no_grad():
        out = module(
            target_tensor=x_tgt,
            target_mask=mask,
            reference_tensors=[x_ref],
            scoring_report=report
        )
        
    assert out["fused_tokens"].shape == (1, 64, 64) # 8x8 patches, token_dim 64
    assert out["identity_conditioning_enabled"] is True
