"""
High-Level Coarse Reconstructor Coordinator.
Integrates CoarseUNetGenerator with Phase 4 regional identity tokens,
Phase 2 task routing, and the 5x5 Laplace loss / Region Weighting from Wei et al. (2019).
Guarantees bit-exact unmasked pixel preservation and exports structural guidance features.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from models.backbone.coarse_unet import CoarseUNetGenerator
from training.losses.laplace_loss import CoarseReconstructionLoss
from modules.masking.task_router import TaskMode
from modules.utils.geometry import verify_unmasked_invariance


class CoarseReconstructor(nn.Module):
    """
    Coordinating module for semantic coarse layout generation and edge preservation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        identity_token_dim: int = 512,
        num_heads: int = 8,
        rw_multiplier_m: float = 5.0
    ):
        super().__init__()
        self.generator = CoarseUNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            identity_token_dim=identity_token_dim,
            num_heads=num_heads
        )
        self.loss_fn = CoarseReconstructionLoss(rw_multiplier_m=rw_multiplier_m)

    def forward(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        identity_tokens: Optional[torch.Tensor] = None,
        task_mode: str = TaskMode.MODE_1_FACIAL_INPAINTING.value
    ) -> Dict[str, Any]:
        """
        Executes coarse reconstruction forward pass.
        
        Args:
            masked_image: [B, 3, H, W] in [-1, 1]
            mask: [B, 1, H, W] binary mask in [0, 1]
            identity_tokens: [B, M, D] fused regional identity tokens (or None)
            task_mode: operational task mode from routing
            
        Returns:
            Dict containing coarse_prediction, composited_output, and bottleneck_features.
        """
        # If task mode is NON_FACE_OBJECT_REMOVAL, strictly zero out any identity tokens
        if task_mode == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value:
            effective_tokens = None
        else:
            effective_tokens = identity_tokens

        gen_out = self.generator(
            masked_image=masked_image,
            mask=mask,
            identity_tokens=effective_tokens
        )
        
        return {
            "coarse_prediction": gen_out["coarse_prediction"],
            "composited_output": gen_out["composited_output"],
            "bottleneck_features": gen_out["bottleneck_features"],
            "mask": gen_out["mask"],
            "task_mode": task_mode,
            "identity_conditioning_active": effective_tokens is not None and not torch.all(effective_tokens == 0.0)
        }

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculates reconstruction, 5x5 Laplace, and Region-Weighted losses."""
        return self.loss_fn(pred=pred, target=target, mask=mask)

    def reconstruct_numpy(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        identity_tokens: Optional[torch.Tensor] = None,
        task_mode: str = TaskMode.MODE_1_FACIAL_INPAINTING.value,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inference helper for numpy arrays with strict unmasked pixel preservation.
        
        Args:
            image_np: [H, W, 3] uint8 in [0, 255]
            mask_np: [H, W] float32 in [0, 1]
            
        Returns:
            restored_image: [H, W, 3] uint8
            metadata: execution stats
        """
        self.eval()
        with torch.no_grad():
            h, w = image_np.shape[:2]
            # Normalize to [-1, 1]
            img_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
            
            img_t = img_t.to(device)
            mask_t = mask_t.to(device)
            if identity_tokens is not None:
                identity_tokens = identity_tokens.to(device)
                
            out = self.forward(img_t, mask_t, identity_tokens, task_mode=task_mode)
            comp_t = out["composited_output"].squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Convert reconstructed patch to [0, 255] uint8
            recon_uint8 = np.clip((comp_t + 1.0) * 127.5, 0, 255).astype(np.uint8)
            
            # Bit-exact unmasked pixel preservation: copy original image and replace only masked pixels
            restored_np = image_np.copy()
            mask_bool = (mask_np > 0.5)
            restored_np[mask_bool] = recon_uint8[mask_bool]
            
            # Verify unmasked invariance bit-exact
            passed, max_diff, violated = verify_unmasked_invariance(image_np, restored_np, mask_np)
            
            meta = {
                "task_mode": task_mode,
                "unmasked_invariance_passed": passed,
                "max_unmasked_diff": max_diff,
                "violated_pixels": violated,
                "identity_conditioning_active": out["identity_conditioning_active"]
            }
            return restored_np, meta
