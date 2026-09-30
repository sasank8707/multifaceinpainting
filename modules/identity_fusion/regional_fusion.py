"""
High-Level Regional Identity Fusion Coordinator (O2 Fuse).
Connects frozen feature extraction, Phase-3 O1 retrieval scoring weights,
and the trainable Mask-Conditioned Regional Transformer.
Enforces strict identity verification and non-face object removal identity suppression.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from models.identity.arcface_backbone import RegionalIdentityEncoder, build_frozen_identity_encoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from modules.masking.task_router import TaskMode
from modules.reference_scoring.scorer import IdentityMismatchError


class RegionalIdentityFusionModule(nn.Module):
    """
    Coordinates regional feature extraction and transformer fusion.
    """

    def __init__(
        self,
        token_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        spatial_patch_grid: Tuple[int, int] = (16, 16),
        dropout: float = 0.1,
        strict_identity: bool = True
    ):
        super().__init__()
        self.token_dim = token_dim
        self.spatial_patch_grid = spatial_patch_grid
        self.strict_identity = strict_identity
        
        # 1. Frozen Identity Feature Extractor (no gradients)
        self.identity_encoder = RegionalIdentityEncoder(
            token_dim=token_dim,
            spatial_patch_grid=spatial_patch_grid
        )
        
        # 2. Trainable Regional Transformer Fusion
        self.transformer = RegionalIdentityTransformer(
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            spatial_patch_grid=spatial_patch_grid,
            dropout=dropout
        )

    def forward(
        self,
        target_tensor: torch.Tensor,
        target_mask: torch.Tensor,
        reference_tensors: List[torch.Tensor],
        scoring_report: Dict[str, Any],
        target_identity_id: Optional[str] = None,
        reference_identity_ids: Optional[List[str]] = None,
        target_landmarks: Optional[torch.Tensor] = None,
        reference_landmarks_list: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Executes full regional identity fusion.
        
        Args:
            target_tensor: [B, 3, H, W] target image in [-1, 1]
            target_mask: [B, 1, H, W] target binary mask
            reference_tensors: List of K image tensors, each [B, 3, H, W] in [-1, 1]
            scoring_report: Output from MaskAwareReferenceScorer (Phase 3)
            target_identity_id: Ground truth subject identity ID
            reference_identity_ids: List of reference identity IDs
            target_landmarks: [B, N_lm, 2] normalized coordinates (optional)
            reference_landmarks_list: List of K landmark tensors (optional)
            
        Returns:
            Dict containing fused_tokens, attention_maps, and metadata.
        """
        b = target_tensor.shape[0]
        device = target_tensor.device
        h_p, w_p = self.spatial_patch_grid
        num_tokens = h_p * w_p

        # 1. Check task mode: Suppress identity conditioning for non-face object removal
        task_mode = scoring_report.get("task_mode", TaskMode.MODE_1_FACIAL_INPAINTING.value)
        id_enabled = scoring_report.get("identity_conditioning_enabled", True)

        if task_mode == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value or not id_enabled:
            # STRICT REQUIREMENT: Zero facial identity conditioning tokens for non-face regions
            zero_tokens = torch.zeros((b, num_tokens, self.token_dim), device=device, dtype=target_tensor.dtype)
            return {
                "fused_tokens": zero_tokens,
                "identity_conditioning_enabled": False,
                "task_mode": task_mode,
                "reason": "Facial identity conditioning suppressed for non-face object removal.",
                "attention_maps": [],
                "num_references": 0,
                "reference_weights_used": None
            }

        # 2. Strict Identity Verification
        if self.strict_identity and target_identity_id is not None and reference_identity_ids is not None:
            for idx, r_id in enumerate(reference_identity_ids):
                if r_id != target_identity_id:
                    raise IdentityMismatchError(
                        f"Identity mismatch in fusion: Target ID '{target_identity_id}' != Reference ID '{r_id}' at index {idx}."
                    )

        # 3. Resolve Reference Weights from Phase-3 Scoring Report
        global_ranked = scoring_report.get("global_ranked_references", [])
        num_refs = len(reference_tensors)
        assert num_refs > 0, "At least one reference must be supplied."

        if global_ranked:
            weights = [r.get("weight", 1.0 / num_refs) for r in global_ranked[:num_refs]]
            # Pad or truncate if mismatch
            while len(weights) < num_refs:
                weights.append(1.0 / num_refs)
            weights = weights[:num_refs]
            total_w = sum(weights)
            weights = [w / total_w for w in weights] if total_w > 0 else [1.0 / num_refs] * num_refs
        else:
            weights = [1.0 / num_refs] * num_refs

        weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32).unsqueeze(0).repeat(b, 1) # [B, K]

        # 4. Extract Spatial Identity Tokens using Frozen Encoder
        with torch.no_grad():
            # Target queries extracted from target input
            target_feat = self.identity_encoder(target_tensor)
            target_tokens = target_feat["spatial_tokens"] # [B, M, D]

            # Reference tokens
            ref_tokens_list = []
            for ref_tensor in reference_tensors:
                ref_feat = self.identity_encoder(ref_tensor)
                ref_tokens_list.append(ref_feat["spatial_tokens"]) # [B, P, D]

        # 5. Execute Trainable Regional Transformer Fusion
        fusion_out = self.transformer(
            target_tokens=target_tokens,
            target_mask=target_mask,
            reference_tokens_list=ref_tokens_list,
            reference_weights=weights_tensor,
            target_landmarks=target_landmarks,
            reference_landmarks_list=reference_landmarks_list
        )

        return {
            "fused_tokens": fusion_out["fused_tokens"],
            "identity_conditioning_enabled": True,
            "task_mode": task_mode,
            "attention_maps": fusion_out["attention_maps"],
            "num_references": num_refs,
            "reference_weights_used": weights_tensor,
            "token_dim": self.token_dim,
            "num_tokens": num_tokens
        }
