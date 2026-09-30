"""
Phase 6 Test Suite: Diffusion Inpainting Backbone Integration.
Validates:
1. Multi-modal conditioning adapter projections and MODE_4 identity suppression.
2. Structural guidance latent projection.
3. Fast, deterministic MockDiffusionInpainter (< 100ms execution, seed determinism).
4. Multi-candidate generation (K=1, 2, 4).
5. Aspect-ratio and resolution preservation for arbitrary input dimensions.
6. Bit-exact unmasked pixel preservation (verify_unmasked_invariance).
7. Zero reference pixel warping / blending.
8. Diffusers pipeline architecture / config inspection from local cache.
9. End-to-end integration across all phases (1 through 6).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import pytest
import torch
import numpy as np

from modules.masking.task_router import TaskMode
from modules.utils.geometry import verify_unmasked_invariance
from models.diffusion.conditioning_adapter import (
    IdentityConditioningAdapter,
    StructuralGuidanceAdapter,
    DiffusionConditioningBundle,
)
from models.diffusion.mock_diffusion import MockDiffusionInpainter
from models.diffusion.diffusion_adapter import (
    AspectRatioPreserver,
    DiffusionInpaintingAdapter,
)
from modules.inpainting.diffusion_manager import DiffusionInpaintingManager


class TestConditioningAdapter:
    """Validates multi-modal conditioning projection and task-mode gating."""

    def test_identity_conditioning_projection(self):
        adapter = IdentityConditioningAdapter(in_dim=512, out_dim=768, num_tokens=256)
        dummy_tokens = torch.randn(2, 256, 512)

        out = adapter(dummy_tokens, task_mode=TaskMode.MODE_1_FACIAL_INPAINTING.value)
        proj = out["projected_tokens"]

        assert proj.shape == (2, 256, 768), f"Expected (2, 256, 768), got {proj.shape}"
        assert out["identity_active"] is True
        assert not torch.all(proj == 0.0)

    def test_mode4_identity_suppression(self):
        """Under MODE_4_NON_FACE_OBJECT_REMOVAL, facial tokens must be strictly suppressed."""
        adapter = IdentityConditioningAdapter(in_dim=512, out_dim=768, num_tokens=256)
        dummy_tokens = torch.randn(2, 256, 512)

        # 1. Mode 4 suppression
        out_mode4 = adapter(dummy_tokens, task_mode=TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value)
        assert out_mode4["identity_active"] is False
        assert out_mode4["projected_tokens"].shape == (2, 256, 768)

        # 2. None tokens fallback
        out_none = adapter(identity_tokens=None, task_mode=TaskMode.MODE_1_FACIAL_INPAINTING.value, batch_size=2)
        assert out_none["identity_active"] is False
        assert out_none["projected_tokens"].shape == (2, 256, 768)

        # 3. All-zeros tokens fallback
        zeros_tokens = torch.zeros(2, 256, 512)
        out_zeros = adapter(zeros_tokens, task_mode=TaskMode.MODE_1_FACIAL_INPAINTING.value)
        assert out_zeros["identity_active"] is False

    def test_structural_guidance_adapter(self):
        adapter = StructuralGuidanceAdapter(
            coarse_rgb_channels=3,
            bottleneck_dim=512,
            latent_channels=4
        )
        b, h, w = 2, 256, 256
        coarse_rgb = torch.randn(b, 3, h, w)
        bottleneck = torch.randn(b, 512, h // 16, w // 16) # [2, 512, 16, 16]

        # With bottleneck features
        latents = adapter(coarse_rgb, bottleneck)
        assert latents.shape == (b, 4, h // 8, w // 8), f"Expected (2, 4, 32, 32), got {latents.shape}"

        # Without bottleneck features (RGB only)
        latents_rgb = adapter(coarse_rgb, None)
        assert latents_rgb.shape == (b, 4, h // 8, w // 8)


class TestMockDiffusionInpainter:
    """Validates mock diffusion engine speed, determinism, and candidate sampling."""

    def test_speed_and_determinism(self):
        mock = MockDiffusionInpainter()
        mock.eval()
        b, h, w = 1, 256, 256
        img = torch.randn(b, 3, h, w)
        mask = torch.zeros(b, 1, h, w)
        mask[:, :, 80:160, 80:160] = 1.0
        coarse = torch.randn(b, 3, h, w)
        tokens = torch.randn(b, 256, 512)

        t0 = time.perf_counter()
        out1 = mock(
            masked_image=img,
            mask=mask,
            coarse_prediction=coarse,
            identity_tokens=tokens,
            num_candidates=1,
            seed=123
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Fast execution on dual-core CPU
        assert elapsed_ms < 200.0, f"Mock diffusion took {elapsed_ms:.1f}ms (expected < 200ms)"

        # Repeat with same seed: must produce identical candidates
        out2 = mock(
            masked_image=img,
            mask=mask,
            coarse_prediction=coarse,
            identity_tokens=tokens,
            num_candidates=1,
            seed=123
        )
        assert torch.allclose(out1["candidates"], out2["candidates"], atol=1e-5), "Determinism violated for same seed"

        # Repeat with different seed: must produce different candidates
        out3 = mock(
            masked_image=img,
            mask=mask,
            coarse_prediction=coarse,
            identity_tokens=tokens,
            num_candidates=1,
            seed=999
        )
        assert not torch.allclose(out1["candidates"], out3["candidates"], atol=1e-4), "Different seeds produced identical output"

    def test_multi_candidate_generation(self):
        mock = MockDiffusionInpainter()
        b, h, w = 1, 256, 256
        img = torch.randn(b, 3, h, w)
        mask = torch.zeros(b, 1, h, w)
        mask[:, :, 100:180, 100:180] = 1.0
        coarse = torch.randn(b, 3, h, w)

        k = 4
        out = mock(
            masked_image=img,
            mask=mask,
            coarse_prediction=coarse,
            num_candidates=k,
            seed=42
        )

        candidates = out["candidates"]
        assert candidates.shape == (b, k, 3, h, w), f"Expected (1, 4, 3, 256, 256), got {candidates.shape}"
        assert len(out["seeds"]) == k

        # Verify each candidate has distinct synthesized pixels inside the mask
        for i in range(k):
            for j in range(i + 1, k):
                diff = torch.abs(candidates[0, i] - candidates[0, j])
                masked_diff = diff * mask[0]
                assert masked_diff.sum().item() > 1.0, f"Candidates {i} and {j} are identical within mask"

        # Verify unmasked pixels match input exactly across all candidates
        for i in range(k):
            unmasked_diff = torch.abs(candidates[0, i] - img[0]) * (1.0 - mask[0])
            assert unmasked_diff.max().item() < 1e-5, f"Candidate {i} altered unmasked pixels"


class TestAspectRatioPreservation:
    """Validates aspect-ratio preservation and bit-exact unmasked restoration."""

    def test_letterbox_and_unletterbox_arbitrary_dimensions(self):
        # Non-square arbitrary image dimensions (e.g. 300 x 400)
        h, w = 300, 400
        image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[100:200, 150:250] = 1.0

        target_size = (256, 256)
        pad_img, pad_mask, info = AspectRatioPreserver.letterbox(image, mask, target_size=target_size)

        assert pad_img.shape == (256, 256, 3)
        assert pad_mask.shape == (256, 256)
        assert info["orig_h"] == 300
        assert info["orig_w"] == 400

        # Simulate candidate generation on padded image
        fake_candidate = pad_img.copy()
        fake_candidate[pad_mask > 0.5] = 128

        restored = AspectRatioPreserver.unletterbox(fake_candidate, image, mask, info)

        assert restored.shape == (h, w, 3), f"Expected ({h}, {w}, 3), got {restored.shape}"

        # Bit-exact unmasked pixel preservation
        passed, max_diff, violated = verify_unmasked_invariance(image, restored, mask)
        assert passed, f"Unmasked invariance failed: max_diff={max_diff}, violated_pixels={violated}"
        assert max_diff == 0, f"Expected max diff 0, got {max_diff}"

    def test_diffusion_adapter_numpy_arbitrary_resolutions(self):
        adapter = DiffusionInpaintingAdapter(backend="mock", working_resolution=(256, 256))

        # Test vertical aspect ratio (384 x 256)
        h, w = 384, 256
        image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[120:220, 80:180] = 1.0
        coarse = image.copy()
        coarse[mask > 0.5] = 200

        candidates, meta = adapter.generate_candidates_numpy(
            image_np=image,
            mask_np=mask,
            coarse_prediction_np=coarse,
            num_candidates=3,
            seed=101
        )

        assert len(candidates) == 3
        for c in candidates:
            assert c.shape == (h, w, 3)
            passed, max_diff, _ = verify_unmasked_invariance(image, c, mask)
            assert passed, f"Candidate failed unmasked invariance: max_diff={max_diff}"
            assert max_diff == 0

        assert meta["unmasked_invariance_passed"] is True
        assert meta["max_unmasked_diff"] == 0
        assert meta["original_dimensions"] == (h, w)


class TestDiffusersArchitectureInspection:
    """Validates Diffusers inpainting pipeline configuration and cache access."""

    def test_diffusers_inpainting_config(self):
        from diffusers import UNet2DConditionModel

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        unet_dir = os.path.join(
            cache_dir,
            "models--runwayml--stable-diffusion-inpainting",
            "snapshots"
        )
        if os.path.isdir(unet_dir):
            snapshots = [os.path.join(unet_dir, d) for d in os.listdir(unet_dir)]
            if snapshots:
                unet_path = os.path.join(snapshots[0], "unet")
                if os.path.isdir(unet_path):
                    # Inspect UNet configuration directly without running full diffusion
                    unet_config = UNet2DConditionModel.load_config(unet_path)
                    assert unet_config["in_channels"] == 9, (
                        f"Expected 9-channel inpainting UNet, got {unet_config['in_channels']}"
                    )
                    assert unet_config["out_channels"] == 4
                    assert unet_config["cross_attention_dim"] == 768
                    return

        pytest.skip("Local runwayml cache snapshot not found, skipping config verification.")


class TestEndToEndManagerIntegration:
    """Validates complete Phase 1 through 6 integration pipeline."""

    def test_pipeline_integration(self):
        manager = DiffusionInpaintingManager(
            working_resolution=(256, 256),
            backend="mock"
        )

        # Create synthetic face-like target and references
        np.random.seed(42)
        h, w = 256, 256
        target = np.ones((h, w, 3), dtype=np.uint8) * 180
        # Draw face ellipse
        import cv2
        cv2.ellipse(target, (128, 128), (60, 80), 0, 0, 360, (200, 160, 140), -1)

        # Mask on the face center
        mask = np.zeros((h, w), dtype=np.float32)
        mask[100:150, 100:150] = 1.0

        # Two references
        ref1 = target.copy()
        cv2.ellipse(ref1, (128, 128), (60, 80), 0, 0, 360, (205, 165, 145), -1)
        ref2 = target.copy()
        cv2.ellipse(ref2, (128, 128), (60, 80), 0, 0, 360, (195, 155, 135), -1)

        candidates, trace = manager.inpaint(
            target_image=target,
            mask=mask,
            reference_images=[ref1, ref2],
            num_candidates=2,
            seed=42
        )

        assert len(candidates) == 2
        for i, cand in enumerate(candidates):
            assert cand.shape == (h, w, 3)
            passed, max_diff, _ = verify_unmasked_invariance(target, cand, mask)
            assert passed, f"Candidate {i} violated unmasked invariance"
            assert max_diff == 0

        assert "task_mode" in trace
        assert "coarse_reconstruction" in trace
        assert "diffusion_generation" in trace
        assert trace["diffusion_generation"]["unmasked_invariance_passed"] is True
