"""
High-Speed Deterministic Mock Diffusion Inpainter for CPU Testing.
Simulates latent-diffusion candidate generation in milliseconds without heavy U-Net iterations,
validating conditioning contracts, multi-candidate sampling, seed determinism,
and strict bit-exact unmasked pixel preservation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from modules.masking.task_router import TaskMode
from modules.utils.geometry import verify_unmasked_invariance
from models.diffusion.conditioning_adapter import (
    IdentityConditioningAdapter,
    StructuralGuidanceAdapter,
    DiffusionConditioningBundle
)


class MockDiffusionInpainter(nn.Module):
    """
    Deterministic CPU-friendly mock of the diffusion inpainting backbone.
    Enables instant unit testing and full-pipeline verification on dual-core CPUs.
    """

    def __init__(
        self,
        token_dim: int = 512,
        cross_attention_dim: int = 768,
        latent_channels: int = 4,
        noise_strength: float = 0.15
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.noise_strength = noise_strength

        # Use the real conditioning adapters to validate tensor shapes and gating
        self.identity_adapter = IdentityConditioningAdapter(
            in_dim=token_dim,
            out_dim=cross_attention_dim,
            dropout=0.0
        )
        self.structural_adapter = StructuralGuidanceAdapter(
            coarse_rgb_channels=3,
            bottleneck_dim=512,
            latent_channels=latent_channels
        )

        # Lightweight synthesis projection mapping structural latents + identity back to RGB residual
        self.candidate_head = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.eval()

    def forward(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        coarse_prediction: torch.Tensor,
        bottleneck_features: Optional[torch.Tensor] = None,
        identity_tokens: Optional[torch.Tensor] = None,
        task_mode: str = TaskMode.MODE_1_FACIAL_INPAINTING.value,
        num_candidates: int = 1,
        seed: Optional[int] = 42
    ) -> Dict[str, Any]:
        """
        Generates K candidate reconstructions.
        
        Args:
            masked_image: [B, 3, H, W] in [-1, 1]
            mask: [B, 1, H, W] in [0, 1] (1 = masked hole, 0 = keep)
            coarse_prediction: [B, 3, H, W] in [-1, 1]
            bottleneck_features: [B, 512, H/16, W/16] or None
            identity_tokens: [B, M, D] or None
            task_mode: operational task mode string
            num_candidates: number of candidate outputs (K >= 1)
            seed: base random seed for reproducibility
            
        Returns:
            dict containing:
                "candidates": [B, K, 3, H, W] in [-1, 1]
                "conditioning_bundle": DiffusionConditioningBundle
                "seeds": list of seeds
                "identity_active": bool
                "task_mode": str
        """
        b, c, h, w = masked_image.shape
        device = masked_image.device
        dtype = masked_image.dtype

        # 1. Process identity conditioning with task-mode gating
        id_out = self.identity_adapter(
            identity_tokens=identity_tokens,
            task_mode=task_mode,
            batch_size=b
        )
        cross_tokens = id_out["projected_tokens"]
        identity_active = id_out["identity_active"]

        # 2. Process structural guidance
        struct_latent = self.structural_adapter(
            coarse_prediction=coarse_prediction,
            bottleneck_features=bottleneck_features
        )

        # Downsample mask to latent resolution
        mask_latent = nn.functional.interpolate(
            mask, size=struct_latent.shape[-2:], mode="nearest"
        )

        bundle = DiffusionConditioningBundle(
            cross_attention_tokens=cross_tokens,
            structural_guidance=struct_latent,
            mask_latent=mask_latent,
            task_mode=task_mode,
            identity_active=identity_active,
            num_tokens=cross_tokens.shape[1]
        )

        # 3. Generate K candidates
        candidates_list = []
        seeds_used = []
        base_seed = seed if seed is not None else 42

        for k in range(num_candidates):
            cand_seed = base_seed + k * 10007
            seeds_used.append(cand_seed)
            gen = torch.Generator(device=device).manual_seed(cand_seed)

            # Generate latent perturbation
            noise = torch.randn(
                struct_latent.shape,
                generator=gen,
                device=device,
                dtype=dtype
            )

            # Modulate perturbed latent by mask and conditioning
            perturbed_latent = struct_latent + self.noise_strength * noise * mask_latent
            if identity_active:
                # Modulate slightly by mean identity energy
                id_mod = cross_tokens.mean(dim=1).unsqueeze(-1).unsqueeze(-1) # [B, D_cross, 1, 1]
                id_scalar = torch.sigmoid(id_mod[:, :struct_latent.shape[1]]) # [B, 4, 1, 1]
                perturbed_latent = perturbed_latent * (0.9 + 0.2 * id_scalar)

            # Decode through lightweight head
            rgb_delta = self.candidate_head(perturbed_latent)

            # Refined synthesis: coarse prediction + modulated delta
            synth = coarse_prediction + (0.3 * rgb_delta)
            synth = torch.clamp(synth, -1.0, 1.0)

            # Strict unmasked pixel preservation
            # Output = synth * mask + masked_image * (1 - mask)
            composited = synth * mask + masked_image * (1.0 - mask)
            candidates_list.append(composited.unsqueeze(1)) # [B, 1, 3, H, W]

        # [B, K, 3, H, W]
        candidates = torch.cat(candidates_list, dim=1)

        return {
            "candidates": candidates,
            "conditioning_bundle": bundle,
            "seeds": seeds_used,
            "identity_active": identity_active,
            "task_mode": task_mode,
            "num_candidates": num_candidates
        }

    def generate_candidates_numpy(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        coarse_prediction_np: np.ndarray,
        identity_tokens: Optional[torch.Tensor] = None,
        task_mode: str = TaskMode.MODE_1_FACIAL_INPAINTING.value,
        num_candidates: int = 1,
        seed: Optional[int] = 42,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Numpy inference helper with strict bit-exact unmasked pixel preservation.
        
        Args:
            image_np: [H, W, 3] uint8
            mask_np: [H, W] float32 in [0, 1]
            coarse_prediction_np: [H, W, 3] uint8 in [0, 255]
            identity_tokens: [1, M, D] or None
            task_mode: operational task mode string
            num_candidates: K >= 1
            seed: random seed
            device: torch device
            
        Returns:
            candidates: list of K [H, W, 3] uint8 numpy arrays
            metadata: execution stats and invariance checks
        """
        self.eval()
        with torch.no_grad():
            h, w = image_np.shape[:2]
            img_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
            coarse_t = torch.from_numpy(coarse_prediction_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

            img_t = img_t.to(device)
            mask_t = mask_t.to(device)
            coarse_t = coarse_t.to(device)
            if identity_tokens is not None:
                identity_tokens = identity_tokens.to(device)

            out = self.forward(
                masked_image=img_t,
                mask=mask_t,
                coarse_prediction=coarse_t,
                identity_tokens=identity_tokens,
                task_mode=task_mode,
                num_candidates=num_candidates,
                seed=seed
            )

            candidates_np = []
            invariance_passed = True
            max_diff = 0

            mask_bool = (mask_np > 0.5)

            for k in range(num_candidates):
                cand_t = out["candidates"][0, k].permute(1, 2, 0).cpu().numpy()
                cand_uint8 = np.clip((cand_t + 1.0) * 127.5, 0, 255).astype(np.uint8)

                # Bit-exact unmasked replacement
                final_np = image_np.copy()
                final_np[mask_bool] = cand_uint8[mask_bool]

                # Verify invariance
                passed, diff, _ = verify_unmasked_invariance(image_np, final_np, mask_np)
                if not passed:
                    invariance_passed = False
                if diff > max_diff:
                    max_diff = diff

                candidates_np.append(final_np)

            meta = {
                "num_candidates": num_candidates,
                "seeds": out["seeds"],
                "identity_active": out["identity_active"],
                "task_mode": task_mode,
                "unmasked_invariance_passed": invariance_passed,
                "max_unmasked_diff": max_diff
            }

            return candidates_np, meta
