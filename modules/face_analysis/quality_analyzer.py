"""
Image and Face Quality Analysis Module.
Computes Laplacian sharpness/clarity (inspired by Wei et al. 2019), illumination metrics,
and mask-conditioned landmark visibility scores.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional


class QualityAnalyzer:
    """
    Evaluates image sharpness, exposure balance, and landmark visibility.
    """

    def __init__(self, laplacian_kernel_size: int = 5):
        self.laplacian_kernel_size = laplacian_kernel_size

    def compute_sharpness(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """
        Computes Laplacian variance and mean absolute gradient (Clarity_Laplace).
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.dtype == np.uint8 else image[:, :, 0]
        else:
            gray = image.copy()
            
        # If face_bbox is provided, compute on face crop
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            crop = gray[y1:y2, x1:x2]
            if crop.size > 0:
                gray = crop

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.laplacian_kernel_size)
        laplacian_var = float(np.var(laplacian))
        laplacian_mean_abs = float(np.mean(np.abs(laplacian)))
        
        # Normalized clarity score in [0, 1] using Laplacian clarity (Wei et al., 2019)
        # Scaled by reference half-saturation constant C=60.0
        norm_sharpness = float(laplacian_mean_abs / (laplacian_mean_abs + 60.0))

        return {
            "laplacian_variance": round(laplacian_var, 2),
            "laplacian_clarity": round(laplacian_mean_abs, 2),
            "normalized_sharpness": round(norm_sharpness, 4),
            "is_blurry": laplacian_mean_abs < 40.0
        }

    def compute_illumination(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """
        Computes luminance mean, contrast, and exposure distribution.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.dtype == np.uint8 else image[:, :, 0]
        else:
            gray = image.copy()
            
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            crop = gray[y1:y2, x1:x2]
            if crop.size > 0:
                gray = crop

        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        
        # Underexposure (<20) and overexposure (>235) fraction
        underexposed_frac = float(np.mean(gray < 20))
        overexposed_frac = float(np.mean(gray > 235))
        
        # Balanced illumination score in [0, 1]
        exposure_balance = max(0.0, 1.0 - (underexposed_frac + overexposed_frac) * 2.0)
        
        return {
            "mean_luminance": round(mean_lum, 2),
            "contrast_std": round(std_lum, 2),
            "underexposed_ratio": round(underexposed_frac, 4),
            "overexposed_ratio": round(overexposed_frac, 4),
            "exposure_balance_score": round(exposure_balance, 4)
        }

    def evaluate_landmark_visibility(
        self,
        landmarks_pixel: List[List[float]],
        mask: np.ndarray,
        radius: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluates which landmarks fall inside the mask (occluded) vs outside (visible).
        
        Args:
            landmarks_pixel: List of (x, y) coordinates.
            mask: 2D numpy array [0, 1] or [0, 255].
            radius: Neighborhood radius around landmark to check occlusion.
            
        Returns:
            Dict containing visibility ratio, visible_indices, occluded_indices.
        """
        h, w = mask.shape[:2]
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        visible_indices = []
        occluded_indices = []
        
        for idx, pt in enumerate(landmarks_pixel):
            x, y = int(round(pt[0])), int(round(pt[1]))
            if x < 0 or x >= w or y < 0 or y >= h:
                occluded_indices.append(idx)
                continue
                
            # Sample local neighborhood around landmark
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(w, x + radius + 1)
            y2 = min(h, y + radius + 1)
            
            patch = binary_mask[y1:y2, x1:x2]
            if np.mean(patch) > 0.5:
                occluded_indices.append(idx)
            else:
                visible_indices.append(idx)
                
        total_lm = len(landmarks_pixel)
        vis_ratio = float(len(visible_indices) / total_lm) if total_lm > 0 else 0.0
        
        return {
            "total_landmarks": total_lm,
            "visible_count": len(visible_indices),
            "occluded_count": len(occluded_indices),
            "visibility_ratio": round(vis_ratio, 4),
            "occlusion_ratio": round(1.0 - vis_ratio, 4),
            "visible_indices": visible_indices,
            "occluded_indices": occluded_indices
        }

    def compute_composite_quality_score(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Combines normalized sharpness and illumination balance into overall quality score in [0, 1].
        """
        sharpness_info = self.compute_sharpness(image, face_bbox)
        illum_info = self.compute_illumination(image, face_bbox)
        
        composite = (
            0.6 * sharpness_info["normalized_sharpness"] +
            0.4 * illum_info["exposure_balance_score"]
        )
        return round(float(np.clip(composite, 0.0, 1.0)), 4)
