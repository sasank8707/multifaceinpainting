"""
Trainable Regional Transformer Fusion Module (O2 Fuse).
Performs mask-conditioned cross-attention between target spatial queries and multi-reference keys/values,
modulated directly by Phase-3 O1 retrieval weights.
Produces spatially aligned regional identity tokens for generative inpainting.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from models.transformer.positional_encoding import (
    Sinusoidal2DPositionalEncoding,
    LandmarkProximityEncoding
)


class MaskConditionedCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention with spatial mask bias.
    """
    def __init__(self, token_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        assert token_dim % num_heads == 0, "token_dim must be divisible by num_heads"

        self.w_q = nn.Linear(token_dim, token_dim, bias=False)
        self.w_k = nn.Linear(token_dim, token_dim, bias=False)
        self.w_v = nn.Linear(token_dim, token_dim, bias=False)
        self.w_out = nn.Linear(token_dim, token_dim)

        self.dropout = nn.Dropout(dropout)
        self.mask_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_spatial: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Target query tokens [B, M, D]
            k: Reference key tokens [B, P, D]
            v: Reference value tokens [B, P, D]
            mask_spatial: [B, M, 1] spatial mask values (1 = occluded, 0 = unmasked)
            
        Returns:
            attended_tokens: [B, M, D]
            attention_weights: [B, num_heads, M, P]
        """
        b, m, d = q.shape
        _, p, _ = k.shape

        # Linear projections & split into heads
        q_proj = self.w_q(q).view(b, m, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, M, d_h]
        k_proj = self.w_k(k).view(b, p, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, P, d_h]
        v_proj = self.w_v(v).view(b, p, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, P, d_h]

        # Scaled dot-product attention
        attn_logits = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale # [B, H, M, P]

        # Apply mask conditioning bias:
        # In masked regions (mask > 0.5), query tokens receive a learned attention bias
        if mask_spatial is not None:
            # mask_spatial: [B, M, 1] -> [B, 1, M, 1]
            m_bias = mask_spatial.unsqueeze(1) * self.mask_scale # [B, 1, M, 1]
            attn_logits = attn_logits + m_bias

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # Context aggregation
        out = torch.matmul(attn_weights_drop, v_proj) # [B, H, M, d_h]
        out = out.transpose(1, 2).contiguous().view(b, m, d) # [B, M, D]
        out = self.w_out(out) # [B, M, D]

        return out, attn_weights


class TransformerSelfAttentionBlock(nn.Module):
    """Standard Transformer Self-Attention Block with LayerNorm and FFN."""
    def __init__(self, token_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(token_dim)
        mlp_hidden = int(token_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class RegionalIdentityTransformer(nn.Module):
    """
    Regional Identity Fusion Transformer.
    Integrates Top-K multi-reference identity tokens with mask-conditioned cross-attention
    and Phase-3 reference weight modulation.
    """

    def __init__(
        self,
        token_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        spatial_patch_grid: Tuple[int, int] = (16, 16),
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.spatial_patch_grid = spatial_patch_grid
        self.num_patches = spatial_patch_grid[0] * spatial_patch_grid[1]
        
        # Positional and landmark encoders (spatial WHERE conditioning)
        self.pos_encoder = Sinusoidal2DPositionalEncoding(token_dim)
        self.landmark_encoder = LandmarkProximityEncoding(token_dim)

        # Base spatial query embeddings for the M=256 grid patches
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, token_dim) * 0.02)

        # Mask geometric encoding projection and region-gating (localizes masked vs visible zones)
        self.mask_proj = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim)
        )
        self.mask_gate = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.Sigmoid()
        )
        self.query_norm = nn.LayerNorm(token_dim)

        # Cross-Attention per reference
        self.cross_attention = MaskConditionedCrossAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fusion_norm = nn.LayerNorm(token_dim)

        # Inter-patch contextualization blocks
        self.blocks = nn.ModuleList([
            TransformerSelfAttentionBlock(token_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        target_tokens: Optional[torch.Tensor],
        target_mask: torch.Tensor,
        reference_tokens_list: List[torch.Tensor],
        reference_weights: torch.Tensor,
        target_landmarks: Optional[torch.Tensor] = None,
        reference_landmarks_list: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for Regional Transformer Fusion with strict target identity anti-leakage.
        
        =============================================================================
        ARCHITECTURAL DECOUPLING: WHERE vs. WHO
        - The TARGET provides spatial and geometric context (WHERE identity is needed):
          - Mask geometry (which regions are occluded vs visible)
          - 2D coordinate grid positions (Sinusoidal2DPositionalEncoding)
          - Normalized facial landmarks (LandmarkProximityEncoding)
          - Learnable spatial queries (base grid patch embeddings)
          * STRICT REQUIREMENT: Target ArcFace identity embeddings are NEVER used as
            attention values, residuals, or identity representations.
        
        - The REFERENCES provide the identity representations (WHO the subject is):
          - Dense regional ArcFace spatial tokens from verified same-subject photos
          - Multi-reference retrieval weights from Phase 3 (pose/quality/visibility)
          - Mask-conditioned cross-attention aggregating identity features into query locations
          - All residuals and final fused tokens originate 100% from reference features.
        =============================================================================
        
        Args:
            target_tokens: [B, M, D] optional target queries (visual identity tokens unused to prevent leakage)
            target_mask: [B, 1, H, W] or [B, M, 1] target binary mask
            reference_tokens_list: List of K tensors, each [B, P, D]
            reference_weights: [B, K] Phase-3 normalized retrieval weights (sum to 1.0)
            target_landmarks: [B, N_lm, 2] target normalized landmark coordinates (optional)
            reference_landmarks_list: List of K tensors [B, N_lm, 2] reference landmarks (optional)
            
        Returns:
            Dict containing:
                "fused_tokens": [B, M, D] spatially fused identity tokens (100% reference-derived)
                "attention_maps": List of K attention weight tensors [B, H, M, P]
                "raw_reference_tokens": List of individual attended tokens
        """
        num_refs = len(reference_tokens_list)
        assert num_refs > 0, "At least one reference token tensor must be provided"
        assert reference_weights.shape[-1] == num_refs, f"Weight count ({reference_weights.shape[-1]}) != ref count ({num_refs})"

        b, p, d = reference_tokens_list[0].shape
        device = reference_tokens_list[0].device
        m = self.num_patches

        # Ensure spatial mask is of shape [B, M, 1]
        if target_mask.ndim == 4:
            down_mask = F.interpolate(
                target_mask.float(),
                size=self.spatial_patch_grid,
                mode="nearest"
            ) # [B, 1, H_p, W_p]
            mask_spatial = down_mask.flatten(2).transpose(1, 2) # [B, M, 1]
        elif target_mask.ndim == 3 and target_mask.shape[-1] == 1:
            mask_spatial = target_mask
        else:
            mask_spatial = target_mask.unsqueeze(-1)

        # 1. Construct Spatial Queries (WHERE identity information is needed on the canvas):
        # Combines learnable patch queries, 2D coordinates, landmark proximity, and mask geometry.
        # STRICT ANTI-LEAKAGE: Zero target ArcFace identity embeddings are injected.
        pos_enc = self.pos_encoder(self.spatial_patch_grid, device=device) # [1, M, D]
        lm_enc = self.landmark_encoder(self.spatial_patch_grid, target_landmarks, device=device, batch_size=b) # [B, M, D]
        mask_enc = self.mask_proj(mask_spatial) # [B, M, D]
        base_q = self.spatial_queries.expand(b, -1, -1) # [B, M, D]

        q = self.query_norm(base_q + pos_enc + lm_enc + mask_enc) # [B, M, D]

        # 2. Multi-Reference Mask-Conditioned Cross-Attention
        attended_list = []
        attn_maps = []

        for k in range(num_refs):
            ref_k = reference_tokens_list[k] # [B, P, D]
            ref_lm = reference_landmarks_list[k] if reference_landmarks_list is not None and k < len(reference_landmarks_list) else None
            ref_lm_enc = self.landmark_encoder(self.spatial_patch_grid, ref_lm, device=device, batch_size=b)
            
            # Key = Ref_Tokens + Spatial_Pos + Landmark_Pos
            # Value = Ref_Tokens (WHO)
            ref_k_pos = ref_k + pos_enc + ref_lm_enc
            
            t_k, attn_k = self.cross_attention(
                q=q,
                k=ref_k_pos,
                v=ref_k,
                mask_spatial=mask_spatial
            )
            attended_list.append(t_k)
            attn_maps.append(attn_k)

        # 3. Weighted Multi-Reference Aggregation using Phase-3 Weights
        stacked_attended = torch.stack(attended_list, dim=1) # [B, K, M, D]
        weights_expanded = reference_weights.unsqueeze(-1).unsqueeze(-1) # [B, K, 1, 1]
        weighted_tokens = (stacked_attended * weights_expanded).sum(dim=1) # [B, M, D]

        # 4. Reference-Derived Residual Connection with Mask-Conditioned Regional Modulation
        # In occluded/masked target regions, cross-attended reference features are prioritized.
        # Residual is computed strictly from reference spatial tokens (STRICTLY ZERO TARGET RESIDUAL).
        stacked_refs = torch.stack(reference_tokens_list, dim=1) # [B, K, P, D]
        ref_base = (stacked_refs * weights_expanded).sum(dim=1) # [B, P, D]

        gate = self.mask_gate(mask_spatial) # [B, M, D]
        fused_ref = weighted_tokens * (0.5 + gate) + ref_base * (1.0 - 0.5 * gate)

        fused = self.fusion_norm(fused_ref) # [B, M, D]

        # 5. Contextual Self-Attention Blocks over reference features
        for blk in self.blocks:
            fused = blk(fused)

        fused_tokens = self.final_norm(fused) # [B, M, D]

        return {
            "fused_tokens": fused_tokens,
            "attention_maps": attn_maps,
            "num_references": num_refs,
            "reference_weights_used": reference_weights,
            "token_dim": d,
            "num_tokens": m
        }
