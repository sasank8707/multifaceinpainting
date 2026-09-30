"""
Coarse Semantic and Edge-Aware U-Net Reconstruction Generator.
Synthesizes initial semantic layout and organ edge contours using:
1. Multi-scale U-Net encoder-decoder with skip connections
2. Bottleneck regional identity cross-attention conditioning (Phase 4 tokens)
3. Compositing layer ensuring strict bit-exact unmasked pixel preservation
4. Structural guidance feature export for future diffusion backbone
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class UNetConvBlock(nn.Module):
    """Dual convolution block with BatchNorm and LeakyReLU."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BottleneckIdentityCrossAttention(nn.Module):
    """
    Cross-attention conditioning at the U-Net bottleneck.
    Injects regional identity tokens into the coarse structural representation.
    """
    def __init__(self, feature_dim: int = 512, token_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, z: torch.Tensor, identity_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z: [B, C, H_b, W_b] bottleneck feature map
            identity_tokens: [B, M, D] fused regional identity tokens (or None)
            
        Returns:
            conditioned_z: [B, C, H_b, W_b]
        """
        if identity_tokens is None or torch.all(identity_tokens == 0.0):
            # Non-face object removal or zero conditioning: bypass cross-attention
            return z

        b, c, h_b, w_b = z.shape
        m_tokens = h_b * w_b

        # Flatten bottleneck features to [B, M, C]
        z_flat = z.flatten(2).transpose(1, 2) # [B, M, C]
        
        # Project Queries, Keys, Values
        q = self.q_proj(z_flat).view(b, m_tokens, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, M, d_h]
        k = self.k_proj(identity_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, P, d_h]
        v = self.v_proj(identity_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, P, d_h]

        # Cross-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B, H, M, P]
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v) # [B, H, M, d_h]
        out = out.transpose(1, 2).contiguous().view(b, m_tokens, c) # [B, M, C]
        out = self.out_proj(out)

        # Residual connection + LayerNorm
        z_cond = self.norm(z_flat + out) # [B, M, C]
        
        # Reshape back to [B, C, H_b, W_b]
        return z_cond.transpose(1, 2).contiguous().view(b, c, h_b, w_b)


class CoarseUNetGenerator(nn.Module):
    """
    Coarse Semantic Layout & Organ Edge Inpainting Network.
    Inspired by Wei et al. (IEEE Access 2019).
    """

    def __init__(
        self,
        in_channels: int = 4, # 3 (RGB masked image) + 1 (binary mask)
        out_channels: int = 3,
        base_channels: int = 64,
        identity_token_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Encoder Stages
        self.enc1 = UNetConvBlock(in_channels, base_channels)             # 256x256 -> 64
        self.down1 = nn.MaxPool2d(2)                                      # -> 128x128
        
        self.enc2 = UNetConvBlock(base_channels, base_channels * 2)       # 128x128 -> 128
        self.down2 = nn.MaxPool2d(2)                                      # -> 64x64
        
        self.enc3 = UNetConvBlock(base_channels * 2, base_channels * 4)   # 64x64 -> 256
        self.down3 = nn.MaxPool2d(2)                                      # -> 32x32
        
        self.enc4 = UNetConvBlock(base_channels * 4, base_channels * 8)   # 32x32 -> 512
        self.down4 = nn.MaxPool2d(2)                                      # -> 16x16
        
        # Bottleneck (16x16 for 256x256 input)
        self.bottleneck = UNetConvBlock(base_channels * 8, base_channels * 8)
        self.cross_attn = BottleneckIdentityCrossAttention(
            feature_dim=base_channels * 8,
            token_dim=identity_token_dim,
            num_heads=num_heads
        )

        # Decoder Stages with Skip Connections
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2) # 16x16 -> 32x32
        self.dec4 = UNetConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2) # 32x32 -> 64x64
        self.dec3 = UNetConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)     # 64x64 -> 128x128
        self.dec2 = UNetConvBlock(base_channels * 2 + base_channels, base_channels)
        
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=2, stride=2)    # 128x128 -> 256x256
        self.dec1 = UNetConvBlock(base_channels + base_channels // 2, base_channels // 2)

        # Output projection head to RGB [-1, 1]
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Apply Kaiming initialization to preserve signal variance through encoder/decoder
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        identity_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for coarse semantic reconstruction.
        
        Args:
            masked_image: [B, 3, H, W] target image with mask applied in [-1, 1]
            mask: [B, 1, H, W] binary mask (1.0 = occluded, 0.0 = unmasked)
            identity_tokens: [B, M, D] fused regional identity tokens from Phase 4
            
        Returns:
            Dict containing:
                "coarse_prediction": [B, 3, H, W] raw neural network output in [-1, 1]
                "composited_output": [B, 3, H, W] with bit-exact unmasked pixels preserved
                "bottleneck_features": [B, C, H_b, W_b] structural guidance for diffusion
        """
        # Ensure mask is [B, 1, H, W]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
            
        x_in = torch.cat([masked_image, mask], dim=1) # [B, 4, H, W]

        # Encoder forward pass with skip connections
        s1 = self.enc1(x_in)       # [B, 64, H, W]
        d1 = self.down1(s1)        # [B, 64, H/2, W/2]
        
        s2 = self.enc2(d1)         # [B, 128, H/2, W/2]
        d2 = self.down2(s2)        # [B, 128, H/4, W/4]
        
        s3 = self.enc3(d2)         # [B, 256, H/4, W/4]
        d3 = self.down3(s3)        # [B, 256, H/8, W/8]
        
        s4 = self.enc4(d3)         # [B, 512, H/8, W/8]
        d4 = self.down4(s4)        # [B, 512, H/16, W/16]

        # Bottleneck + Regional Identity Conditioning
        b_feat = self.bottleneck(d4) # [B, 512, H/16, W/16]
        b_cond = self.cross_attn(b_feat, identity_tokens) # [B, 512, H/16, W/16]

        # Decoder forward pass
        u4 = self.up4(b_cond)                         # [B, 256, H/8, W/8]
        cat4 = torch.cat([u4, s4], dim=1)             # [B, 768, H/8, W/8]
        dec4 = self.dec4(cat4)

        u3 = self.up3(dec4)                           # [B, 128, H/4, W/4]
        cat3 = torch.cat([u3, s3], dim=1)             # [B, 384, H/4, W/4]
        dec3 = self.dec3(cat3)

        u2 = self.up2(dec3)                           # [B, 64, H/2, W/2]
        cat2 = torch.cat([u2, s2], dim=1)             # [B, 192, H/2, W/2]
        dec2 = self.dec2(cat2)

        u1 = self.up1(dec2)                           # [B, 32, H, W]
        cat1 = torch.cat([u1, s1], dim=1)             # [B, 96, H, W]
        dec1 = self.dec1(cat1)

        raw_pred = self.final_conv(dec1) # [B, 3, H, W]

        # Strict Compositing Layer: Preserve unmasked pixels exactly
        # mask = 1.0 (reconstructed), mask = 0.0 (original masked_image)
        composited = mask * raw_pred + (1.0 - mask) * masked_image

        return {
            "coarse_prediction": raw_pred,
            "composited_output": composited,
            "bottleneck_features": b_cond,
            "mask": mask
        }
