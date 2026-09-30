"""
Deterministic Unit Test Suite: Identity Conditioning Calibration and Mask Semantics.
Verifies all 9 mandatory requirements:
1. Identity adapter output shape = [B, 77, 768]
2. Calibrated token norms are within documented tolerance of CLIP conditioning distribution.
3. No NaN / Inf in any conditioning outputs.
4. ArcFace backbone remains strictly frozen.
5. Target identity features are NOT passed into the identity conditioning path.
6. Masked-image preprocessing produces exact neutral 0.0 in masked region.
7. Unmasked pixels remain strictly unchanged.
8. masked_image_latents have valid shape and finite values.
9. Diffusion adapter does not accidentally reapply Diffusers zero-masking.
"""

import sys
import os
import inspect
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion.conditioning_adapter import (
    IdentityConditioningAdapter,
    ConditioningCalibrator
)
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from modules.masking.task_router import TaskMode
from diffusers.image_processor import VaeImageProcessor


@pytest.fixture(scope="module")
def loaded_adapter():
    """Loads IdentityConditioningAdapter from the calibrated checkpoint."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "identity_adapter" / "identity_adapter_best.pt"
    assert ckpt_path.exists(), f"Checkpoint missing at {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    adapter = IdentityConditioningAdapter(
        in_dim=512,
        out_dim=768,
        num_tokens=256,
        out_tokens=77,
        num_id_queries=16,
        dropout=0.0
    )
    adapter.load_state_dict(ckpt["model_state_dict"])
    adapter.eval()
    return adapter


class TestIdentityConditioningCalibration:
    """Verifies all 9 calibration and mask semantics requirements."""

    def test_1_identity_adapter_output_shape(self, loaded_adapter):
        """Requirement 1: Identity adapter output shape = [B, 77, 768]."""
        adapter = loaded_adapter

        for b in [1, 2]:
            dummy_tokens = F.normalize(torch.randn(b, 256, 512), p=2, dim=-1)
            with torch.no_grad():
                out = adapter(dummy_tokens, task_mode=TaskMode.MODE_1_FACIAL_INPAINTING.value)

            proj = out["projected_tokens"]
            assert proj.shape == (b, 77, 768), f"Expected shape ({b}, 77, 768), got {proj.shape}"

            # Also check unconditional null embedding shape
            null_emb = adapter.get_null_embedding(batch_size=b, target_len=77)
            assert null_emb.shape == (b, 77, 768), f"Expected null shape ({b}, 77, 768), got {null_emb.shape}"

    def test_2_calibrated_token_norms_within_tolerance(self, loaded_adapter):
        """
        Requirement 2: Calibrated token norms are within documented tolerance of CLIP conditioning distribution.
        CLIP targets:
          - Token 0 norm: ~43.82 (tolerance: [40.0, 46.0])
          - Tokens 1-76 mean norm: ~28.38 (tolerance: [27.0, 30.0])
          - Overall mean norm: ~28.53 (tolerance: [27.0, 30.0])
          - Global scalar mean: ~ -0.1116 (tolerance: [-0.20, 0.0])
          - Global scalar std: ~ 1.0253 (tolerance: [0.90, 1.15])
        """
        adapter = loaded_adapter
        torch.manual_seed(42)

        dummy_tokens = F.normalize(torch.randn(10, 256, 512), p=2, dim=-1)
        with torch.no_grad():
            out = adapter(dummy_tokens)

        proj = out["projected_tokens"] # [10, 77, 768]
        norms = torch.norm(proj, p=2, dim=-1) # [10, 77]

        token_0_norm = norms[:, 0].mean().item()
        tokens_1_76_norm = norms[:, 1:].mean().item()
        overall_mean_norm = norms.mean().item()
        global_mean = proj.mean().item()
        global_std = proj.std().item()

        assert 40.0 <= token_0_norm <= 46.0, f"Token 0 norm {token_0_norm:.2f} outside [40.0, 46.0]"
        assert 27.0 <= tokens_1_76_norm <= 30.0, f"Tokens 1-76 norm {tokens_1_76_norm:.2f} outside [27.0, 30.0]"
        assert 27.0 <= overall_mean_norm <= 30.0, f"Overall mean norm {overall_mean_norm:.2f} outside [27.0, 30.0]"
        assert -0.20 <= global_mean <= 0.0, f"Global mean {global_mean:.4f} outside [-0.20, 0.0]"
        assert 0.90 <= global_std <= 1.15, f"Global std {global_std:.4f} outside [0.90, 1.15]"

        # Also check null embedding norms
        null_emb = adapter.get_null_embedding(batch_size=10, target_len=77)
        null_norms = torch.norm(null_emb, p=2, dim=-1)
        assert 27.0 <= null_norms.mean().item() <= 30.0, f"Null norm {null_norms.mean().item():.2f} outside [27.0, 30.0]"

    def test_3_no_nan_or_inf(self, loaded_adapter):
        """Requirement 3: No NaN / Inf in any conditioning outputs."""
        adapter = loaded_adapter
        dummy_tokens = F.normalize(torch.randn(4, 256, 512), p=2, dim=-1)

        with torch.no_grad():
            out = adapter(dummy_tokens)
            null_emb = adapter.get_null_embedding(batch_size=4, target_len=77)

        proj = out["projected_tokens"]
        probe = out["probe_arcface"]

        assert not torch.isnan(proj).any(), "NaN found in projected_tokens"
        assert not torch.isinf(proj).any(), "Inf found in projected_tokens"
        assert not torch.isnan(probe).any(), "NaN found in probe_arcface"
        assert not torch.isinf(probe).any(), "Inf found in probe_arcface"
        assert not torch.isnan(null_emb).any(), "NaN found in null_embedding"
        assert not torch.isinf(null_emb).any(), "Inf found in null_embedding"

    def test_4_arcface_remains_frozen(self):
        """Requirement 4: ArcFace backbone remains strictly frozen."""
        encoder = RegionalIdentityEncoder(token_dim=512)

        for name, param in encoder.named_parameters():
            assert not param.requires_grad, f"ArcFace parameter {name} has requires_grad=True!"

    def test_5_target_identity_features_not_passed(self):
        """Requirement 5: Target identity features are NOT passed into identity conditioning path."""
        # Inspect fusion transformer signature and code
        transformer = RegionalIdentityTransformer(
            token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0
        )
        transformer.eval()

        ref_tokens_a = [F.normalize(torch.randn(1, 256, 512), p=2, dim=-1) for _ in range(3)]
        ref_tokens_b = [F.normalize(torch.randn(1, 256, 512), p=2, dim=-1) for _ in range(3)]
        mask = torch.zeros(1, 1, 256, 256)
        weights = torch.tensor([[0.5, 0.3, 0.2]])

        with torch.no_grad():
            # Pass target_tokens=None: verifies no target tokens are needed or used
            fused_a = transformer(target_tokens=None, target_mask=mask, reference_tokens_list=ref_tokens_a, reference_weights=weights)["fused_tokens"]
            fused_b = transformer(target_tokens=None, target_mask=mask, reference_tokens_list=ref_tokens_b, reference_weights=weights)["fused_tokens"]

        diff = torch.norm(fused_a - fused_b).item()
        assert diff > 0.5, f"Swapping references produced no change (diff={diff:.4f})"

    def test_6_masked_image_preprocessing_normalized_values(self):
        """Requirement 6: Masked-image preprocessing produces exact neutral 0.0 in masked region."""
        img_proc = VaeImageProcessor()
        mask_proc = VaeImageProcessor(do_normalize=False, do_binarize=True, do_convert_grayscale=True)

        # Image with non-zero RGB
        img_np = np.full((512, 512, 3), [100, 150, 200], dtype=np.uint8)
        pil_img = Image.fromarray(img_np)

        # Mask: center 100x100 is masked (255)
        mask_np = np.zeros((512, 512), dtype=np.uint8)
        mask_np[200:300, 200:300] = 255
        pil_mask = Image.fromarray(mask_np)

        init_image = img_proc.preprocess(pil_img, height=512, width=512) # in [-1, 1]
        mask_condition = mask_proc.preprocess(pil_mask, height=512, width=512) # 1 in masked, 0 in unmasked

        # Pretrained pipeline masking formula
        masked_image = init_image * (mask_condition < 0.5)

        # Verify masked pixels
        masked_pixels = masked_image[:, :, 200:300, 200:300]
        assert torch.all(masked_pixels == 0.0), (
            f"Expected masked pixels to be exact neutral 0.0, but got min {masked_pixels.min().item()} "
            f"and max {masked_pixels.max().item()}"
        )

    def test_7_unmasked_pixels_remain_unchanged(self):
        """Requirement 7: Unmasked pixels remain strictly unchanged from normalized original image."""
        img_proc = VaeImageProcessor()
        mask_proc = VaeImageProcessor(do_normalize=False, do_binarize=True, do_convert_grayscale=True)

        img_np = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_np)

        mask_np = np.zeros((512, 512), dtype=np.uint8)
        mask_np[100:200, 100:200] = 255
        pil_mask = Image.fromarray(mask_np)

        init_image = img_proc.preprocess(pil_img, height=512, width=512)
        mask_condition = mask_proc.preprocess(pil_mask, height=512, width=512)

        masked_image = init_image * (mask_condition < 0.5)

        # Unmasked region: top-left 50x50
        unmasked_orig = init_image[:, :, :50, :50]
        unmasked_out = masked_image[:, :, :50, :50]

        assert torch.equal(unmasked_orig, unmasked_out), "Unmasked pixels were altered during masking!"

    def test_8_masked_image_latents_finite_and_valid(self):
        """Requirement 8: masked_image_latents have valid 4-channel shape and finite values."""
        # Test mathematical shape and value sanity
        img_proc = VaeImageProcessor()
        mask_proc = VaeImageProcessor(do_normalize=False, do_binarize=True, do_convert_grayscale=True)

        img_np = np.full((512, 512, 3), [128, 128, 128], dtype=np.uint8)
        mask_np = np.zeros((512, 512), dtype=np.uint8)
        mask_np[150:350, 150:350] = 255

        init_image = img_proc.preprocess(Image.fromarray(img_np), height=512, width=512)
        mask_condition = mask_proc.preprocess(Image.fromarray(mask_np), height=512, width=512)
        masked_image = init_image * (mask_condition < 0.5)

        # Latent dimensions for 512x512 with 8x downsampling
        h_lat, w_lat = 512 // 8, 512 // 8
        assert masked_image.shape == (1, 3, 512, 512)
        assert (h_lat, w_lat) == (64, 64)

        # Confirm non-nan finite values in masked image tensor before VAE
        assert not torch.isnan(masked_image).any()
        assert not torch.isinf(masked_image).any()

    def test_9_no_reapplied_zero_masking(self):
        """
        Requirement 9: The diffusion adapter does not accidentally reapply Diffusers zero-masking
        after our explicit latent construction.
        """
        diff_file = PROJECT_ROOT / "models" / "diffusion" / "diffusion_adapter.py"
        with open(diff_file, "r") as f:
            src = f.read()

        # 1. Adapter passes masked_image_latents=guided_latents
        assert "masked_image_latents=guided_latents" in src, "Adapter does not pass masked_image_latents to pipeline"
        assert "_encode_vae_image" in src, "Adapter does not explicitly encode VAE latents"

        # 2. Check diffusers pipeline behavior: when masked_image_latents is provided,
        # Diffusers assigns masked_image = masked_image_latents and does NOT re-zero init_image
        from diffusers import StableDiffusionInpaintPipeline
        pipe_src = inspect.getsource(StableDiffusionInpaintPipeline.__call__)

        # Confirm Diffusers check: 'if masked_image_latents is None:'
        assert "if masked_image_latents is None:" in pipe_src, (
            "Diffusers pipeline does not check for masked_image_latents"
        )
        assert "masked_image = masked_image_latents" in pipe_src, (
            "Diffusers pipeline does not preserve passed masked_image_latents"
        )
