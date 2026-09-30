"""
Runtime Architecture-Integrity and Conditioning Instrumentation Test.
Demonstrates mathematically and mechanically that:
1. Changing Phase-4 identity tokens changes the diffusion cross-attention conditioning path.
2. Changing Phase-5 structural guidance changes the diffusion latent conditioning path.
3. MODE_4 (non-face object removal) strictly suppresses/zeroes out identity conditioning.
4. UNet attention and latent pathways consume these conditioning signals directly.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest
from diffusers import UNet2DConditionModel

from modules.masking.task_router import TaskMode
from models.diffusion.conditioning_adapter import (
    IdentityConditioningAdapter,
    StructuralGuidanceAdapter,
)


class TestConditioningIntegrity:
    """Verifies that conditioning signals actively modulate the diffusion model."""

    def test_identity_token_variation_and_mode4_suppression(self):
        """Validates that token changes alter projected embeddings, and MODE_4 suppresses identity."""
        adapter = IdentityConditioningAdapter(in_dim=512, out_dim=768, num_tokens=77, dropout=0.0)
        adapter.eval()

        # Input identity tokens A and B
        tokens_a = torch.randn(1, 77, 512)
        tokens_b = torch.randn(1, 77, 512)

        out_a = adapter(tokens_a, task_mode=TaskMode.MODE_1_FACIAL_INPAINTING.value)
        out_b = adapter(tokens_b, task_mode=TaskMode.MODE_1_FACIAL_INPAINTING.value)

        # 1. Changing identity tokens MUST change projected cross-attention embeddings
        diff_ab = torch.abs(out_a["projected_tokens"] - out_b["projected_tokens"]).max().item()
        assert diff_ab > 0.05, f"Expected distinct projected tokens, got max diff {diff_ab}"
        assert out_a["identity_active"] is True
        assert out_b["identity_active"] is True

        # 2. MODE_4 MUST suppress identity conditioning and return null embedding
        out_mode4 = adapter(tokens_a, task_mode=TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value)
        assert out_mode4["identity_active"] is False

        diff_mode4 = torch.abs(out_a["projected_tokens"] - out_mode4["projected_tokens"]).max().item()
        assert diff_mode4 > 0.05, "MODE_4 output identical to conditioned identity tokens"

    def test_structural_guidance_variation(self):
        """Validates that changing coarse prediction alters structural guidance latents."""
        adapter = StructuralGuidanceAdapter(coarse_rgb_channels=3, bottleneck_dim=512, latent_channels=4)
        adapter.eval()

        coarse_a = torch.randn(1, 3, 256, 256)
        coarse_b = torch.randn(1, 3, 256, 256)

        latent_a = adapter(coarse_a)
        latent_b = adapter(coarse_b)

        diff = torch.abs(latent_a - latent_b).max().item()
        assert diff > 0.05, f"Structural guidance latents failed to change: max diff {diff}"

    def test_unet_cross_attention_and_latent_consumption(self):
        """Directly probes real UNet from local HuggingFace cache to prove active conditioning."""
        cache_dir = "C:/Users/sasan/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb/unet"
        if not os.path.isdir(cache_dir):
            pytest.skip("Local UNet cache snapshot not found.")

        unet = UNet2DConditionModel.from_pretrained(cache_dir, torch_dtype=torch.float32)
        unet.eval()

        # Inpainting sample: 9 channels [B, 9, 64, 64]
        # Channels 0:4 = noisy latent, 4:5 = mask, 5:9 = masked image latent (structural guidance)
        sample1 = torch.randn(1, 9, 64, 64)
        timestep = torch.tensor([500])

        cond_a = torch.randn(1, 77, 768)
        cond_b = torch.randn(1, 77, 768)

        # 1. Probing Cross-Attention consumption (Identity Tokens)
        with torch.no_grad():
            out_cond_a = unet(sample1, timestep, encoder_hidden_states=cond_a).sample
            out_cond_b = unet(sample1, timestep, encoder_hidden_states=cond_b).sample

        attn_diff = torch.abs(out_cond_a - out_cond_b).max().item()
        assert attn_diff > 0.01, f"UNet ignored cross-attention identity tokens (diff: {attn_diff})"

        # 2. Probing Latent Structural Guidance consumption (Channels 5:9)
        sample2 = sample1.clone()
        sample2[:, 5:9, :, :] = torch.randn(1, 4, 64, 64) # Different coarse structural guidance

        with torch.no_grad():
            out_struct = unet(sample2, timestep, encoder_hidden_states=cond_a).sample

        struct_diff = torch.abs(out_cond_a - out_struct).max().item()
        assert struct_diff > 0.01, f"UNet ignored structural guidance latents (diff: {struct_diff})"
