"""
Spatial and Facial Landmark Positional Encoding Module.
Combines 2D sinusoidal spatial grid encodings with continuous landmark proximity representations
to associate patch identity tokens with exact facial anatomy and coordinates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    Computes 2D sinusoidal positional encodings for a grid of size (H_p, W_p).
    """
    def __init__(self, token_dim: int):
        super().__init__()
        self.token_dim = token_dim
        assert token_dim % 4 == 0, "token_dim must be divisible by 4 for 2D sinusoidal encoding"

    def forward(self, grid_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """
        Returns:
            Positional encoding tensor of shape [1, P, D] where P = H_p * W_p.
        """
        h_p, w_p = grid_size
        d_half = self.token_dim // 2
        d_quarter = d_half // 2

        # Generate coordinates normalized in [0, 1]
        y_coords = torch.linspace(0, 1, h_p, device=device)
        x_coords = torch.linspace(0, 1, w_p, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        grid_y = grid_y.flatten().unsqueeze(1) # [P, 1]
        grid_x = grid_x.flatten().unsqueeze(1) # [P, 1]

        # Frequency bands
        omega = torch.arange(d_quarter, dtype=torch.float32, device=device) / d_quarter
        omega = 1.0 / (10000.0 ** omega) # [d_quarter]
        omega = omega.unsqueeze(0)        # [1, d_quarter]

        # Compute sin/cos for X and Y
        out_x = grid_x * omega # [P, d_quarter]
        out_y = grid_y * omega # [P, d_quarter]

        pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1) # [P, d_half]
        pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1) # [P, d_half]

        pos = torch.cat([pos_x, pos_y], dim=-1) # [P, D]
        return pos.unsqueeze(0) # [1, P, D]


class LandmarkProximityEncoding(nn.Module):
    """
    Encodes spatial relationship between patch centers and facial landmarks.
    Computes Euclidean distance vectors to landmark clusters and projects to token_dim.
    """
    def __init__(self, token_dim: int, num_landmark_anchors: int = 5):
        super().__init__()
        self.token_dim = token_dim
        self.num_anchors = num_landmark_anchors
        
        # MLP projecting distance vectors into landmark embedding space
        self.proj = nn.Sequential(
            nn.Linear(num_landmark_anchors, token_dim // 2),
            nn.SiLU(),
            nn.Linear(token_dim // 2, token_dim),
            nn.LayerNorm(token_dim)
        )

    def forward(
        self,
        grid_size: Tuple[int, int],
        landmarks: Optional[torch.Tensor],
        device: torch.device,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Args:
            grid_size: (H_p, W_p) of patch tokens
            landmarks: [B, N_lm, 2] in normalized [0, 1] coords (or None)
            device: torch device
            batch_size: batch size
            
        Returns:
            Landmark encoding tensor of shape [B, P, D].
        """
        h_p, w_p = grid_size
        p = h_p * w_p

        # Patch centers in [0, 1]
        y_coords = (torch.arange(h_p, device=device, dtype=torch.float32) + 0.5) / h_p
        x_coords = (torch.arange(w_p, device=device, dtype=torch.float32) + 0.5) / w_p
        gy, gx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        patch_centers = torch.stack([gx.flatten(), gy.flatten()], dim=-1) # [P, 2]
        patch_centers = patch_centers.unsqueeze(0) # [1, P, 2]

        if landmarks is None or landmarks.shape[1] == 0:
            # Neutral zero landmark encoding if no landmarks detected
            return torch.zeros((batch_size, p, self.token_dim), device=device)

        # Subsample to canonical anchors if full dense landmarks provided:
        # e.g. Nose tip (0), Left eye (1), Right eye (2), Left mouth (3), Right mouth (4)
        n_lm = landmarks.shape[1]
        if n_lm >= 5:
            # Sample 5 key anchor landmarks
            anchor_indices = [n_lm // 2, n_lm // 6, (5 * n_lm) // 6, n_lm // 4, (3 * n_lm) // 4][:self.num_anchors]
            anchors = landmarks[:, anchor_indices, :2] # [B, 5, 2]
        else:
            anchors = landmarks[:, :min(n_lm, self.num_anchors), :2] # [B, K, 2]
            if anchors.shape[1] < self.num_anchors:
                # Pad with center (0.5, 0.5)
                pad_k = self.num_anchors - anchors.shape[1]
                pad_pts = torch.full((batch_size, pad_k, 2), 0.5, device=device)
                anchors = torch.cat([anchors, pad_pts], dim=1)

        # Compute pairwise distance: [B, 1, 5, 2] vs [1, P, 1, 2]
        # dist: [B, P, 5]
        anchors_exp = anchors.unsqueeze(1) # [B, 1, 5, 2]
        centers_exp = patch_centers.unsqueeze(2) # [1, P, 1, 2]
        distances = torch.norm(centers_exp - anchors_exp, p=2, dim=-1) # [B, P, 5]

        # Project to token_dim
        landmark_tokens = self.proj(distances) # [B, P, D]
        return landmark_tokens
