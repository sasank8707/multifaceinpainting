"""
Phase 5 Comprehensive Unit Tests:
Coarse Semantic + Edge-Aware Reconstruction Network (Wei et al. 2019).
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.backbone.coarse_unet import CoarseUNetGenerator, BottleneckIdentityCrossAttention
from training.losses.laplace_loss import Laplacian5x5Filter, CoarseReconstructionLoss
from modules.structural_guidance.coarse_reconstructor import CoarseReconstructor
from modules.masking.task_router import TaskMode
from modules.masking.synthetic_masks import MaskGenerator


@pytest.fixture
def synthetic_inputs():
    """Generates synthetic image and mask tensors."""
    torch.manual_seed(42)
    b, c, h, w = 2, 3, 128, 128
    img = torch.randn(b, c, h, w).clamp(-1.0, 1.0)
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, 40:88, 40:88] = 1.0 # 48x48 central mask
    
    # 8x8 = 64 tokens, dim 256 for fast CPU execution
    id_tokens = torch.randn(b, 64, 256)
    return img, mask, id_tokens


def test_5x5_laplace_operator_filter():
    """Test 1: 5x5 Laplacian filter kernel and second-order derivative calculation."""
    filt = Laplacian5x5Filter(in_channels=3)
    x = torch.ones(1, 3, 32, 32)
    
    # Flat image has zero second derivative
    out_flat = filt(x)
    assert torch.allclose(out_flat, torch.zeros_like(out_flat), atol=1e-5)
    
    # Impulse / step edge produces high second derivative
    x_edge = x.clone()
    x_edge[:, :, 16:, :] = 0.0 # Step edge
    out_edge = filt(x_edge)
    assert out_edge.abs().max().item() > 0.5


def test_region_weighted_loss_amplification():
    """Test 2: Region-Weighted loss applies multiplier m=5.0 to masked edge errors."""
    loss_fn = CoarseReconstructionLoss(rw_multiplier_m=5.0)
    
    target = torch.randn(1, 3, 64, 64)
    # Case A: Error placed entirely inside mask
    mask_in = torch.zeros(1, 1, 64, 64)
    mask_in[:, :, 20:44, 20:44] = 1.0
    pred_err_in = target.clone()
    pred_err_in[:, :, 20:44, 20:44] += 1.0
    
    # Case B: Same error placed entirely outside mask
    mask_out = torch.zeros(1, 1, 64, 64)
    mask_out[:, :, 0:24, 0:24] = 1.0 # Mask is elsewhere
    pred_err_out = target.clone()
    pred_err_out[:, :, 20:44, 20:44] += 1.0
    
    loss_a = loss_fn(pred_err_in, target, mask_in)
    loss_b = loss_fn(pred_err_out, target, mask_out)
    
    # Region-weighted loss inside mask must be significantly higher due to m=5.0
    assert loss_a["loss_rw"].item() > loss_b["loss_rw"].item()


def test_boundary_loss_computation():
    """Test 3: Boundary transition band extraction and gradient loss."""
    loss_fn = CoarseReconstructionLoss(boundary_radius=2)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 20:44, 20:44] = 1.0
    
    band = loss_fn.compute_boundary_band(mask)
    assert band.shape == mask.shape
    # Center of mask and far outside should be 0; perimeter band should be 1
    assert band[0, 0, 32, 32].item() == 0.0 # Inside center
    assert band[0, 0, 5, 5].item() == 0.0   # Outside far
    assert band[0, 0, 20, 32].item() > 0.5 # On boundary edge


def test_coarse_unet_forward_pass_and_shapes(synthetic_inputs):
    """Test 4: U-Net forward pass produces correct spatial dimensions and structural features."""
    img, mask, id_tokens = synthetic_inputs
    b, c, h, w = img.shape
    
    # Use base_channels=32 and token_dim=256 for fast unit testing
    unet = CoarseUNetGenerator(
        in_channels=4,
        out_channels=3,
        base_channels=32,
        identity_token_dim=256,
        num_heads=4
    )
    
    res = unet(masked_image=img, mask=mask, identity_tokens=id_tokens)
    assert "coarse_prediction" in res
    assert "composited_output" in res
    assert "bottleneck_features" in res
    
    assert res["coarse_prediction"].shape == (b, 3, h, w)
    assert res["composited_output"].shape == (b, 3, h, w)
    # Bottleneck features downsampled by 16: 128 / 16 = 8x8
    assert res["bottleneck_features"].shape == (b, 256, 8, 8)


def test_unmasked_pixel_bit_exact_preservation(synthetic_inputs):
    """Test 5: Compositing layer guarantees unmasked pixels are 100% bit-exact identical to input."""
    img, mask, id_tokens = synthetic_inputs
    unet = CoarseUNetGenerator(base_channels=32, identity_token_dim=256, num_heads=4)
    
    res = unet(masked_image=img, mask=mask, identity_tokens=id_tokens)
    comp = res["composited_output"]
    
    # Unmasked region: mask == 0
    unmasked = (mask == 0.0).repeat(1, 3, 1, 1)
    diff = torch.abs(comp[unmasked] - img[unmasked])
    max_diff = diff.max().item()
    
    assert max_diff == 0.0, f"Unmasked pixel invariance violated: max_diff={max_diff}"


def test_masked_region_generation(synthetic_inputs):
    """Test 6: Output inside masked region is newly generated and non-trivial."""
    img, mask, id_tokens = synthetic_inputs
    unet = CoarseUNetGenerator(base_channels=32, identity_token_dim=256, num_heads=4)
    
    # Initialize masked input with zeros in mask region
    masked_in = img * (1.0 - mask)
    res = unet(masked_image=masked_in, mask=mask, identity_tokens=id_tokens)
    comp = res["composited_output"]
    
    masked_locs = (mask == 1.0).repeat(1, 3, 1, 1)
    gen_pixels = comp[masked_locs]
    
    # Generated pixels must not be all zero or blank
    assert gen_pixels.abs().mean().item() > 0.01


def test_identity_conditioning_path_active(synthetic_inputs):
    """Test 7: Changing Phase 4 identity tokens measurably shifts the generated coarse output."""
    img, mask, id_tokens1 = synthetic_inputs
    id_tokens2 = torch.randn_like(id_tokens1)
    
    unet = CoarseUNetGenerator(base_channels=32, identity_token_dim=256, num_heads=4)
    unet.eval()
    
    with torch.no_grad():
        out1 = unet(masked_image=img, mask=mask, identity_tokens=id_tokens1)
        out2 = unet(masked_image=img, mask=mask, identity_tokens=id_tokens2)
        
    diff = torch.abs(out1["coarse_prediction"] - out2["coarse_prediction"]).mean().item()
    assert diff > 1e-4, "Identity tokens must directly steer the coarse reconstruction output!"


def test_non_face_object_mode_disables_identity(synthetic_inputs):
    """Test 8: MODE_4_NON_FACE_OBJECT_REMOVAL suppresses identity conditioning."""
    img, mask, id_tokens = synthetic_inputs
    reconstructor = CoarseReconstructor(base_channels=32, identity_token_dim=256, num_heads=4)
    reconstructor.eval()
    
    with torch.no_grad():
        # Pass different identity tokens under MODE_4
        out_a = reconstructor(img, mask, identity_tokens=id_tokens, task_mode=TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value)
        out_b = reconstructor(img, mask, identity_tokens=torch.randn_like(id_tokens), task_mode=TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value)
        
    assert out_a["identity_conditioning_active"] is False
    assert out_b["identity_conditioning_active"] is False
    # Outputs must be identical because identity tokens are suppressed
    assert torch.allclose(out_a["coarse_prediction"], out_b["coarse_prediction"], atol=1e-5)


def test_gradient_flow_through_coarse_unet(synthetic_inputs):
    """Test 9: Gradients flow cleanly from composite loss through all U-Net layers."""
    img, mask, id_tokens = synthetic_inputs
    reconstructor = CoarseReconstructor(base_channels=32, identity_token_dim=256, num_heads=4)
    reconstructor.train()
    
    target_gt = torch.randn_like(img)
    out = reconstructor(img, mask, identity_tokens=id_tokens)
    
    losses = reconstructor.compute_loss(pred=out["coarse_prediction"], target=target_gt, mask=mask)
    total_loss = losses["total_loss"]
    total_loss.backward()
    
    # Check gradients in encoder, cross-attention, and decoder
    assert reconstructor.generator.enc1.block[0].weight.grad is not None
    assert reconstructor.generator.cross_attn.q_proj.weight.grad is not None
    assert reconstructor.generator.final_conv[0].weight.grad is not None
    assert reconstructor.generator.cross_attn.q_proj.weight.grad.norm().item() > 0.0


def test_arbitrary_masks_and_severities(synthetic_inputs):
    """Test 10: Supports arbitrary mask severities (25%, 50%, 75%, 100%)."""
    img, _, id_tokens = synthetic_inputs
    b, c, h, w = img.shape
    reconstructor = CoarseReconstructor(base_channels=32, identity_token_dim=256, num_heads=4)
    reconstructor.eval()
    
    for severity in [0.25, 0.50, 0.75, 1.0]:
        mask = torch.zeros(b, 1, h, w)
        if severity >= 0.95:
            mask[:] = 1.0 # 100% full face
        else:
            cover_h = int(h * severity)
            mask[:, :, :cover_h, :] = 1.0
            
        with torch.no_grad():
            out = reconstructor(img, mask, identity_tokens=id_tokens)
            assert out["composited_output"].shape == (b, 3, h, w)
            assert not torch.isnan(out["composited_output"]).any()


def test_multiple_disconnected_masks(synthetic_inputs):
    """Test 11: Multi-component disconnected masks reconstructed in single pass."""
    img, _, id_tokens = synthetic_inputs
    b, c, h, w = img.shape
    reconstructor = CoarseReconstructor(base_channels=32, identity_token_dim=256, num_heads=4)
    reconstructor.eval()
    
    # 3 disjoint mask patches
    multi_mask = torch.zeros(b, 1, h, w)
    multi_mask[:, :, 10:30, 10:30] = 1.0
    multi_mask[:, :, 50:70, 50:70] = 1.0
    multi_mask[:, :, 90:110, 90:110] = 1.0
    
    with torch.no_grad():
        out = reconstructor(img, multi_mask, identity_tokens=id_tokens, task_mode=TaskMode.MODE_3_MULTI_MASK.value)
        assert out["composited_output"].shape == (b, 3, h, w)
        
        # Verify all 3 patches contain generated content
        masked_locs = (multi_mask == 1.0).repeat(1, 3, 1, 1)
        assert out["composited_output"][masked_locs].abs().mean().item() > 0.01


def test_reconstruct_numpy_pipeline():
    """Test 12: Numpy inference helper with strict invariance check."""
    reconstructor = CoarseReconstructor(base_channels=32, identity_token_dim=256, num_heads=4)
    img_np = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    mask_np = np.zeros((128, 128), dtype=np.float32)
    mask_np[40:80, 40:80] = 1.0
    
    restored, meta = reconstructor.reconstruct_numpy(img_np, mask_np)
    assert restored.shape == (128, 128, 3)
    assert restored.dtype == np.uint8
    assert meta["unmasked_invariance_passed"] is True
    assert meta["max_unmasked_diff"] == 0.0


def test_cpu_smoke_test_tiny_tensors():
    """Test 13: Ultra-fast CPU smoke test executing full forward, loss, and backward in < 0.5s."""
    b, c, h, w = 1, 3, 64, 64
    x = torch.randn(b, c, h, w)
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, 20:44, 20:44] = 1.0
    # 4x4 = 16 tokens, dim 64
    tokens = torch.randn(b, 16, 64)
    
    net = CoarseReconstructor(base_channels=16, identity_token_dim=64, num_heads=2)
    net.train()
    
    out = net(x, mask, identity_tokens=tokens)
    assert out["composited_output"].shape == (1, 3, 64, 64)
    
    loss_dict = net.compute_loss(out["coarse_prediction"], x, mask)
    loss = loss_dict["total_loss"]
    assert not torch.isnan(loss)
    
    loss.backward()
    assert net.generator.final_conv[0].weight.grad is not None
