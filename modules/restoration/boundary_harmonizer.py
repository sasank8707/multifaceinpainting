"""
Modular Boundary Harmonization and Gradient Compositing Module.
Eliminates boundary haloing, edge seams, and color discontinuity in diffusion inpainting
while guaranteeing strict bit-exact canvas invariance outside the mask (Delta = 0.0).

Key Capabilities:
1. Inward Feathered Alpha Blending (C1 smooth cosine transition strictly inside mask)
2. Poisson Gradient Compositing (Dirichlet boundary condition matching)
3. Local Skin Tone Harmonization (Reinhard LAB transfer from surrounding skin ring)
4. Bit-Exact Canvas Invariance Enforcement
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
from modules.utils.geometry import verify_unmasked_invariance


class BoundaryHarmonizer:
    """
    Precision Boundary Harmonization Engine for Inpainted / Restored Facial Images.
    Operates at the compositing stage after inverse letterbox mapping.
    """

    def __init__(
        self,
        feather_radius: float = 5.0,
        enable_color_transfer: bool = False,
        ring_width: int = 15,
        blend_mode: str = "feathered"  # "feathered", "poisson", "mixed"
    ):
        """
        Args:
            feather_radius: Width of inward transition ramp in pixels (default 5.0).
            enable_color_transfer: If True, matches LAB skin tone of surrounding ring.
            ring_width: Width of outer healthy skin ring used for local color reference.
            blend_mode: Blending strategy:
                - 'feathered': Inward distance-transform cosine alpha blending.
                - 'poisson': Gradient-domain Poisson blending (Dirichlet boundary).
                - 'color_feathered': LAB color matching + inward feathering.
        """
        self.feather_radius = feather_radius
        self.enable_color_transfer = enable_color_transfer
        self.ring_width = ring_width
        self.blend_mode = blend_mode

    def compute_inward_alpha(
        self,
        mask: np.ndarray,
        radius: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes C1-smooth inward alpha ramp strictly inside the mask.
        alpha = 0.0 at mask boundary and outside (mask == 0).
        alpha = 1.0 deep inside the mask (distance >= radius).
        """
        r = radius or self.feather_radius
        bin_mask = (mask > 0).astype(np.uint8) * 255
        
        if np.sum(bin_mask) == 0:
            return np.zeros_like(mask, dtype=np.float32)

        # Distance transform from zeros (outer boundary) into the mask
        dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)

        # Linear ramp [0, 1]
        alpha_linear = np.clip(dist / max(r, 1e-3), 0.0, 1.0)

        # C1 cosine smoothing: 0.5 * (1 - cos(pi * alpha))
        alpha_smooth = 0.5 * (1.0 - np.cos(np.pi * alpha_linear))

        # Strictly enforce 0 outside mask
        alpha_smooth[bin_mask == 0] = 0.0
        return alpha_smooth.astype(np.float32)

    def harmonize_color(
        self,
        target: np.ndarray,
        restored: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Transfers LAB color distribution from surrounding unmasked skin ring to restored patch.
        Prevents desaturated or mismatched skin tones inside inpainting mask.
        """
        bin_mask = (mask > 0).astype(np.uint8) * 255
        h, w = target.shape[:2]

        # Extract surrounding healthy skin ring
        k_size = self.ring_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(bin_mask, kernel)
        ring_mask = (dilated > 0) & (bin_mask == 0)

        if np.sum(ring_mask) < 20:
            # Fallback if ring is too small (e.g. at edge of image)
            return restored.copy()

        # Convert to LAB float32
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        restored_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB).astype(np.float32)

        mask_bool = (bin_mask > 0)
        ring_mean = np.mean(target_lab[ring_mask], axis=0)
        ring_std = np.std(target_lab[ring_mask], axis=0) + 1e-5

        recon_mean = np.mean(restored_lab[mask_bool], axis=0)
        recon_std = np.std(restored_lab[mask_bool], axis=0) + 1e-5

        # Reinhard color transfer on the patch
        harmonized_lab = restored_lab.copy()
        for c in range(3):
            # Scale and shift
            harmonized_lab[mask_bool, c] = (
                (restored_lab[mask_bool, c] - recon_mean[c]) * (ring_std[c] / recon_std[c])
            ) + ring_mean[c]

        harmonized_lab = np.clip(harmonized_lab, 0, 255).astype(np.uint8)
        harmonized_rgb = cv2.cvtColor(harmonized_lab, cv2.COLOR_LAB2RGB)

        # Ensure outside mask is unchanged
        output = restored.copy()
        output[mask_bool] = harmonized_rgb[mask_bool]
        return output

    def composite(
        self,
        target_image: np.ndarray,
        restored_image: np.ndarray,
        mask: np.ndarray,
        blend_mode: Optional[str] = None,
        feather_radius: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes boundary-harmonized compositing.

        Args:
            target_image: Original uncorrupted/unmasked RGB image [H, W, 3] uint8.
            restored_image: Full restored candidate RGB image [H, W, 3] uint8.
            mask: Binary mask [H, W] where >0 indicates damaged/inpainted area.
            blend_mode: Override default blend mode.
            feather_radius: Override default feather radius.

        Returns:
            final_composite: [H, W, 3] uint8 composite.
            metrics: Dictionary with invariance and blending metadata.
        """
        assert target_image.shape == restored_image.shape, "Shape mismatch between target and restored image"
        h, w = target_image.shape[:2]
        bin_mask = (mask > 0).astype(np.uint8) * 255
        mode = blend_mode or self.blend_mode
        radius = feather_radius or self.feather_radius

        if np.sum(bin_mask) == 0:
            # Empty mask: return original target
            return target_image.copy(), {"invariance_passed": True, "max_diff": 0.0, "mode": "noop"}

        working_restored = restored_image.copy()

        # Step 1: Optional local color harmonization
        if self.enable_color_transfer or mode in ("color_feathered", "color_poisson"):
            working_restored = self.harmonize_color(target_image, working_restored, bin_mask)

        # Step 2: Blending execution
        if mode in ("feathered", "color_feathered"):
            alpha = self.compute_inward_alpha(bin_mask, radius=radius)
            alpha_3c = np.expand_dims(alpha, -1)
            composite_float = (
                target_image.astype(np.float32) * (1.0 - alpha_3c) +
                working_restored.astype(np.float32) * alpha_3c
            )
            composite = np.clip(np.round(composite_float), 0, 255).astype(np.uint8)

        elif mode == "poisson":
            ys, xs = np.where(bin_mask > 0)
            center = (int(np.mean(xs)), int(np.mean(ys)))
            try:
                # Pass bin_mask.copy() because cv2.seamlessClone mutates mask in-place in OpenCV C++
                cloned_bgr = cv2.seamlessClone(rst_bgr, tgt_bgr, bin_mask.copy(), center, cv2.NORMAL_CLONE)
                composite = cv2.cvtColor(cloned_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                # Fallback to feathered if seamlessClone fails (e.g. at canvas edge)
                alpha = self.compute_inward_alpha(bin_mask, radius=radius)
                alpha_3c = np.expand_dims(alpha, -1)
                composite_float = (
                    target_image.astype(np.float32) * (1.0 - alpha_3c) +
                    working_restored.astype(np.float32) * alpha_3c
                )
                composite = np.clip(np.round(composite_float), 0, 255).astype(np.uint8)

        elif mode == "hard":
            composite = target_image.copy()
            composite[bin_mask > 0] = working_restored[bin_mask > 0]

        else:
            raise ValueError(f"Unknown blend_mode: {mode}")

        # Step 3: Enforce strict Bit-Exact Canvas Invariance (Delta = 0.0 outside mask)
        # Any mathematical rounding outside the mask is reset to exact original pixels
        final_image = target_image.copy()
        final_image[bin_mask > 0] = composite[bin_mask > 0]

        # Verify invariance
        passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(
            target_image, final_image, bin_mask
        )
        assert passed_inv, f"Invariance violation in BoundaryHarmonizer: max_diff={max_diff}, violated={violated_pixels}"

        metrics = {
            "blend_mode": mode,
            "feather_radius": radius,
            "invariance_passed": passed_inv,
            "max_unmasked_diff": max_diff,
            "violated_unmasked_pixels": violated_pixels,
            "total_masked_pixels": int(np.sum(bin_mask > 0))
        }
        return final_image, metrics
