"""
Laplacian Edge Loss and Region-Weighted Loss Module.
Implements the 5x5 second-order differential Laplacian operator and Region Weighting (RW)
strategy from Wei et al. (IEEE Access 2019) to accelerate facial edge convergence and sharpen boundaries.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class Laplacian5x5Filter(nn.Module):
    """
    Fixed 5x5 Laplacian operator convolution filter.
    Preserves edges and computes second-order spatial derivatives.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        
        # 5x5 Discrete Laplacian Kernel from Wei et al. (2019)
        kernel = torch.tensor([
            [ 0.0,  0.0, -1.0,  0.0,  0.0],
            [ 0.0, -1.0, -2.0, -1.0,  0.0],
            [-1.0, -2.0, 16.0, -2.0, -1.0],
            [ 0.0, -1.0, -2.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0,  0.0,  0.0]
        ], dtype=torch.float32)
        
        # Shape: [in_channels, 1, 5, 5] for depthwise channel-independent convolution
        kernel = kernel.view(1, 1, 5, 5).repeat(in_channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies depthwise 5x5 Laplacian filter.
        Input: [B, C, H, W]
        Output: [B, C, H, W] second-derivative edge map
        """
        # Reflection padding ensures boundaries do not create artificial edge artifacts
        padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, self.kernel, padding=0, groups=self.in_channels)


class SobelGradientFilter(nn.Module):
    """
    Fixed 3x3 Sobel filter for first-order gradient extraction along boundary bands.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        
        kx = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        ky = torch.tensor([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes gradient magnitude sqrt(gx^2 + gy^2)."""
        gx = F.conv2d(x, self.kx, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.ky, padding=1, groups=self.in_channels)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


class CoarseReconstructionLoss(nn.Module):
    """
    Composite loss for coarse semantic reconstruction.
    Combines L1 reconstruction, masked-region L1, 5x5 Laplace loss,
    Region-Weighted edge loss (Wei et al. 2019), and boundary continuity loss.
    """

    def __init__(
        self,
        weight_rec: float = 1.0,
        weight_mask: float = 5.0,
        weight_laplace: float = 1.0,
        weight_rw: float = 1.0,
        weight_boundary: float = 2.0,
        rw_multiplier_m: float = 5.0,
        boundary_radius: int = 3
    ):
        super().__init__()
        self.w_rec = weight_rec
        self.w_mask = weight_mask
        self.w_laplace = weight_laplace
        self.w_rw = weight_rw
        self.w_boundary = weight_boundary
        self.m = rw_multiplier_m
        self.boundary_radius = boundary_radius

        self.laplacian = Laplacian5x5Filter(in_channels=3)
        self.sobel = SobelGradientFilter(in_channels=3)

    def compute_boundary_band(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts boundary transition band around mask edge via morphological dilation - erosion.
        """
        k_size = 2 * self.boundary_radius + 1
        # Max pooling acts as morphological dilation
        dilated = F.max_pool2d(mask, kernel_size=k_size, stride=1, padding=self.boundary_radius)
        # Negative max pooling acts as morphological erosion
        eroded = -F.max_pool2d(-mask, kernel_size=k_size, stride=1, padding=self.boundary_radius)
        boundary_band = (dilated - eroded).clamp(0.0, 1.0)
        return boundary_band

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates all loss components.
        
        Args:
            pred: [B, 3, H, W] predicted coarse image
            target: [B, 3, H, W] ground truth target image
            mask: [B, 1, H, W] binary mask (1 = missing region, 0 = unmasked)
            
        Returns:
            Dict containing total_loss and individual scalar loss components.
        """
        b, c, h, w = pred.shape
        eps = 1e-6

        # 1. Global L1 Reconstruction Loss
        loss_rec = F.l1_loss(pred, target)

        # 2. Masked Region Specific L1 Loss
        diff_abs = torch.abs(pred - target)
        mask_expanded = mask.repeat(1, c, 1, 1)
        loss_mask = (diff_abs * mask_expanded).sum() / (mask_expanded.sum() + eps)

        # 3. 5x5 Laplace Edge Loss (Wei et al. 2019)
        # Second derivative features
        lap_pred = self.laplacian(pred)
        lap_target = self.laplacian(target)
        lap_diff = torch.abs(lap_pred - lap_target)
        loss_laplace = lap_diff.mean()

        # 4. Region-Weighted (RW) Loss (Wei et al. 2019)
        # Weights: 1.0 outside mask, m in masked blurry region (amplifying edge penalty)
        rw_map = 1.0 + (self.m - 1.0) * mask_expanded
        loss_rw = (lap_diff * rw_map).sum() / (rw_map.sum() + eps)

        # 5. Boundary Continuity Loss
        boundary_band = self.compute_boundary_band(mask)
        boundary_expanded = boundary_band.repeat(1, c, 1, 1)
        grad_pred = self.sobel(pred)
        grad_target = self.sobel(target)
        grad_diff = torch.abs(grad_pred - grad_target)
        loss_boundary = (grad_diff * boundary_expanded).sum() / (boundary_expanded.sum() + eps)

        # Total Composite Loss
        total_loss = (
            self.w_rec * loss_rec +
            self.w_mask * loss_mask +
            self.w_laplace * loss_laplace +
            self.w_rw * loss_rw +
            self.w_boundary * loss_boundary
        )

        return {
            "total_loss": total_loss,
            "loss_rec": loss_rec,
            "loss_mask": loss_mask,
            "loss_laplace": loss_laplace,
            "loss_rw": loss_rw,
            "loss_boundary": loss_boundary
        }
