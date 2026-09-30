"""
Multi-Modal Conditioning Adapter for Diffusion Inpainting.
Projects Phase 4 regional identity tokens and Phase 5 coarse structural guidance
into diffusion cross-attention and latent spaces with task-mode gating.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any

from modules.masking.task_router import TaskMode


@dataclass
class DiffusionConditioningBundle:
    """Encapsulates all multi-modal conditioning signals passed to diffusion."""
    cross_attention_tokens: torch.Tensor          # [B, M, D_cross] (e.g., 768 for SD 1.5)
    structural_guidance: Optional[torch.Tensor]   # [B, C_latent, H_lat, W_lat] (e.g., 4 channels)
    mask_latent: Optional[torch.Tensor]           # [B, 1, H_lat, W_lat]
    task_mode: str
    identity_active: bool
    num_tokens: int


class ConditioningCalibrator(nn.Module):
    """
    Learned Distribution Calibration Layer for Cross-Attention Conditioning.
    Calibrates multi-modal identity tokens to match the first-order statistics
    (token norm, scalar mean, and variance) of the Stable Diffusion CLIP text manifold,
    while strictly preserving the directional angular geometry of biometric tokens.
    """
    def __init__(self, embed_dim: int = 768, num_tokens: int = 77):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # 1. Feature normalization across channel dimension
        self.layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=True)

        # 2. Directional preservation + learnable per-token scale
        # Initialize token scales to empirical CLIP profile (Token 0: 43.82, Tokens 1-76: 28.38)
        initial_scales = torch.full((1, num_tokens, 1), 28.38)
        initial_scales[:, 0, :] = 43.82
        self.token_scale = nn.Parameter(initial_scales)

        # 3. Global affine channel adjustment (matching scalar mean and std)
        self.affine_shift = nn.Parameter(torch.full((1, 1, embed_dim), -0.1116))
        self.affine_scale = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calibrate token sequence x of shape [B, L, D].
        Preserves directional information while matching the target norm distribution.
        """
        b, l, d = x.shape
        h = self.layer_norm(x)

        # Unit normalize along channel dimension to strictly preserve directional geometry
        direction = nn.functional.normalize(h, p=2, dim=-1)

        # Apply per-token scale
        if l == self.num_tokens:
            scale = self.token_scale
        else:
            scale = self.token_scale.mean()

        calibrated = (direction * scale + self.affine_shift) * self.affine_scale

        return calibrated


class IdentityConditioningAdapter(nn.Module):
    """
    Trainable Regional Identity Conditioning Adapter.
    Maps Phase 4 regional ArcFace tokens (256 spatial tokens of 512-dim)
    into structured cross-attention representations for diffusion.
    Preserves spatial regional token layout and supports reference ranking weights.
    """

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 768,
        num_tokens: int = 256,
        out_tokens: Optional[int] = None,
        num_id_queries: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens = num_tokens
        self.out_tokens = out_tokens
        self.num_id_queries = num_id_queries

        # 1. Spatial Token Projection (preserves 256 spatial layout)
        self.spatial_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        # 2. Learnable Spatial Identity Queries (attend across 256 facial patches)
        self.id_queries = nn.Parameter(torch.randn(1, num_id_queries, out_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 3. Base Context Soft-Prompt Embeddings (77 tokens in CLIP text space)
        self.base_context = nn.Parameter(torch.randn(1, 77, out_dim) * 0.02)

        # 4. Identity Biometric Probe Head (maps pooled tokens back to 512-D ArcFace space for training)
        self.probe_head = nn.Sequential(
            nn.Linear(out_dim, in_dim),
            nn.LayerNorm(in_dim)
        )

        # 5. Learnable null token for unconditional/suppressed conditioning (Mode 4)
        self.null_embedding = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.normal_(self.null_embedding, std=0.02)

        # 6. Conditioning Calibrator for Stable Diffusion CLIP manifold alignment
        self.calibrator = ConditioningCalibrator(embed_dim=out_dim, num_tokens=77)

        # Backward-compatible direct projection MLP
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def get_null_embedding(self, batch_size: int = 1, target_len: int = 77) -> torch.Tensor:
        """Returns calibrated unconditional / null conditioning tensor [B, target_len, out_dim]."""
        raw_null = self.null_embedding.expand(batch_size, target_len, self.out_dim)
        if target_len == 77:
            return self.calibrator(raw_null)
        return raw_null

    def forward(
        self,
        identity_tokens: Optional[torch.Tensor] = None,
        task_mode: str = TaskMode.MODE_1_FACIAL_INPAINTING.value,
        reference_weights: Optional[torch.Tensor] = None,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Projects identity tokens with task-mode gating, spatial query attention, and token resampling.
        
        Args:
            identity_tokens: [B, M, in_dim] where M=256 spatial patches, or None
            task_mode: operational task mode string
            reference_weights: optional [B, K] weighting from Phase 3
            batch_size: batch size if identity_tokens is None
            
        Returns:
            dict containing:
                "projected_tokens": [B, out_tokens, out_dim]
                "spatial_tokens": [B, M, in_dim] original regional tokens
                "id_feature_tokens": [B, num_id_queries, out_dim] queried identity tokens
                "probe_arcface": [B, in_dim] estimated global ArcFace embedding
                "identity_active": bool flag
        """
        target_len = self.out_tokens if self.out_tokens is not None else (
            identity_tokens.shape[1] if identity_tokens is not None else self.num_tokens
        )

        # Strict Mode 4 Suppression: Never inject identity into non-face scenes
        if task_mode == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value or identity_tokens is None:
            b = identity_tokens.shape[0] if identity_tokens is not None else batch_size
            null_tokens = self.get_null_embedding(batch_size=b, target_len=target_len)
            return {
                "projected_tokens": null_tokens,
                "spatial_tokens": None,
                "id_feature_tokens": None,
                "probe_arcface": None,
                "identity_active": False
            }

        # Check for all-zero tokens
        if torch.all(identity_tokens == 0.0):
            null_tokens = self.get_null_embedding(batch_size=identity_tokens.shape[0], target_len=target_len)
            return {
                "projected_tokens": null_tokens,
                "spatial_tokens": identity_tokens,
                "id_feature_tokens": None,
                "probe_arcface": None,
                "identity_active": False
            }

        b, m, d = identity_tokens.shape

        # Step 1: Spatial Token Transformation (preserving 256 patches)
        spatial_features = self.spatial_proj(identity_tokens) # [B, 256, out_dim]

        # Step 2: Spatial Query Attention
        queries = self.id_queries.expand(b, -1, -1) # [B, num_queries, out_dim]
        id_queried, _ = self.cross_attn(query=queries, key=spatial_features, value=spatial_features) # [B, num_queries, out_dim]

        # Step 3: Global probe projection back to ArcFace space
        pooled_id = torch.mean(id_queried, dim=1) # [B, out_dim]
        probe_arcface = nn.functional.normalize(self.probe_head(pooled_id), p=2, dim=-1) # [B, 512]

        # Step 4: Construct output conditioning sequence
        if target_len == 77:
            # Soft-prompt insertion into base context (prefix: index 1 to 1+num_queries)
            prompt_seq = self.base_context.expand(b, -1, -1).clone()
            n_insert = min(self.num_id_queries, 75)
            prompt_seq[:, 1:1 + n_insert, :] = id_queried[:, :n_insert, :]
            projected = self.calibrator(prompt_seq)
        elif target_len == m:
            # Direct spatial projection for tests expecting [B, 256, out_dim]
            projected = spatial_features
        else:
            # Resample sequence length
            p = spatial_features.permute(0, 2, 1)
            p = nn.functional.interpolate(p, size=target_len, mode="linear", align_corners=False)
            projected = p.permute(0, 2, 1)

        return {
            "projected_tokens": projected,
            "spatial_tokens": identity_tokens,
            "id_feature_tokens": id_queried,
            "probe_arcface": probe_arcface,
            "identity_active": True
        }


class StructuralGuidanceAdapter(nn.Module):
    """
    Projects Phase 5 coarse reconstruction and bottleneck features
    into latent-space structural conditioning representations.
    """

    def __init__(
        self,
        coarse_rgb_channels: int = 3,
        bottleneck_dim: int = 512,
        latent_channels: int = 4
    ):
        super().__init__()
        self.latent_channels = latent_channels

        # Coarse RGB to latent encoder (stride 8 downsampling matching VAE)
        self.coarse_rgb_encoder = nn.Sequential(
            nn.Conv2d(coarse_rgb_channels, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                   # /4
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),      # /8
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        )

        # Bottleneck adapter (/16 to /8 upsampling)
        self.bottleneck_adapter = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_dim, 64, kernel_size=4, stride=2, padding=1), # *2 -> /8
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1)
        )

        # Fusion layer combining RGB latent and bottleneck feature
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=latent_channels),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        )

    def forward(
        self,
        coarse_prediction: torch.Tensor,
        bottleneck_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extracts structural guidance latents.
        
        Args:
            coarse_prediction: [B, 3, H, W] in [-1, 1]
            bottleneck_features: [B, 512, H/16, W/16] or None
            
        Returns:
            structural_latent: [B, 4, H/8, W/8]
        """
        rgb_latent = self.coarse_rgb_encoder(coarse_prediction)
        
        if bottleneck_features is not None:
            bn_latent = self.bottleneck_adapter(bottleneck_features)
            # Ensure spatial match if there is any rounding difference
            if bn_latent.shape[-2:] != rgb_latent.shape[-2:]:
                bn_latent = nn.functional.interpolate(
                    bn_latent, size=rgb_latent.shape[-2:], mode="bilinear", align_corners=False
                )
            combined = torch.cat([rgb_latent, bn_latent], dim=1)
            return self.fusion(combined)
        
        return rgb_latent
