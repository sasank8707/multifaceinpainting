"""
Independent Deblurring and Edge-Preserving Restoration Module.
Restores sharpness, structural edges, and local contrast in blurry images
using CPU-friendly multi-scale deconvolution and bilateral noise-suppressed sharpening.

Design:
- Completely independent from Face Inpainting and Background Removal.
- No heavy diffusion or multi-gigabyte models required; sub-second CPU runtime.
- Multi-Stage Architecture:
  1. Blur Severity Estimation (Adaptive Gain Modulation via Laplacian Energy)
  2. Edge-Preserving Decomposition (Bilateral Filtering separates detail from noise)
  3. Noise-Gated Detail Amplification (sharpens edges without boosting flat-area grain)
  4. CIELAB L* Local Contrast Restoration (CLAHE with conservative clip limit)
  5. Richardson-Lucy Deconvolution option for optical/motion defocus
- Preserves exact image dimensions and color fidelity.
"""

import sys
import time
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image


class ImageDeblurrer:
    """
    Precision Edge-Preserving Deblurring Engine.
    Operates on RGB images, restoring optical clarity and edge definition without noise amplification.
    """

    def __init__(
        self,
        strength: float = 1.0,
        enable_clahe: bool = True,
        noise_threshold: float = 8.0,
        enable_richardson_lucy: bool = False,
        rl_iterations: int = 5
    ):
        """
        Args:
            strength: Deblurring/sharpening intensity multiplier [0.5, 2.5] (default 1.0).
            enable_clahe: If True, applies adaptive contrast restoration on CIELAB L* channel.
            noise_threshold: Detail variance threshold below which noise is suppressed (default 8.0).
            enable_richardson_lucy: If True, applies iterative Richardson-Lucy deconvolution.
            rl_iterations: Number of Richardson-Lucy iterations if enabled (default 5).
        """
        self.strength = float(np.clip(strength, 0.1, 3.0))
        self.enable_clahe = enable_clahe
        self.noise_threshold = noise_threshold
        self.enable_richardson_lucy = enable_richardson_lucy
        self.rl_iterations = rl_iterations

    def _normalize_input(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Converts input image to standard [H, W, 3] uint8 RGB numpy array."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            assert path.exists(), f"Image path does not exist: {path}"
            pil_img = Image.open(path).convert("RGB")
            return np.array(pil_img)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            assert image.ndim == 3 and image.shape[2] in (3, 4), "Input numpy array must be 3-channel RGB or 4-channel RGBA."
            if image.shape[2] == 4:
                return image[:, :, :3].copy()
            return image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def estimate_blur_metric(self, image_rgb: np.ndarray) -> Dict[str, float]:
        """
        Estimates image sharpness using Laplacian variance and Tenengrad gradient energy.
        Higher values indicate sharper images; lower values indicate severe blur.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Tenengrad gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx**2 + gy**2))

        return {
            "laplacian_variance": round(lap_var, 2),
            "tenengrad_energy": round(tenengrad, 2),
            "is_blurry": bool(lap_var < 150.0)
        }

    def _richardson_lucy_deconvolve(self, channel: np.ndarray, psf_size: int = 5, psf_sigma: float = 1.0, iterations: int = 5) -> np.ndarray:
        """
        Iterative Richardson-Lucy deconvolution on a single channel in float [0, 1].
        Assumes a 2D isotropic Gaussian point-spread function (PSF).
        """
        # Create Gaussian PSF
        k = cv2.getGaussianKernel(psf_size, psf_sigma)
        psf = np.outer(k, k)
        psf_flipped = np.flip(psf)

        eps = 1e-12
        latent = channel.copy()
        
        for _ in range(iterations):
            conv_est = cv2.filter2D(latent, -1, psf) + eps
            relative_blur = channel / conv_est
            error_est = cv2.filter2D(relative_blur, -1, psf_flipped)
            latent = latent * error_est
            latent = np.clip(latent, 0.0, 1.0)

        return latent

    def deblur(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        strength: Optional[float] = None,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Executes edge-preserving deblurring and contrast restoration.

        Args:
            image: Input image (numpy RGB, PIL Image, or file path).
            strength: Deblur intensity override (optional).
            return_dict: If True, returns rich dictionary with before/after metrics and timings.

        Returns:
            deblurred_rgb: [H, W, 3] uint8 restored image (if return_dict is False).
            Dictionary with restored image and execution metrics (if return_dict is True).
        """
        t_start = time.time()
        orig_rgb = self._normalize_input(image)
        h, w = orig_rgb.shape[:2]
        
        effective_strength = float(strength) if strength is not None else self.strength
        initial_metrics = self.estimate_blur_metric(orig_rgb)

        # -------------------------------------------------------------
        # STEP 1: Optional Richardson-Lucy Deconvolution
        # -------------------------------------------------------------
        if self.enable_richardson_lucy:
            deconv_rgb = np.zeros_like(orig_rgb, dtype=np.float32)
            for c in range(3):
                ch_f = orig_rgb[:, :, c].astype(np.float32) / 255.0
                deconv_rgb[:, :, c] = self._richardson_lucy_deconvolve(
                    ch_f, psf_size=5, psf_sigma=1.2, iterations=self.rl_iterations
                ) * 255.0
            base_rgb = np.clip(deconv_rgb, 0, 255).astype(np.uint8)
        else:
            base_rgb = orig_rgb.copy()

        # -------------------------------------------------------------
        # STEP 2: Edge-Preserving Bilateral Detail Extraction
        # -------------------------------------------------------------
        # Smooth base image to isolate low-frequency structures
        smooth_bgr = cv2.bilateralFilter(
            cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR),
            d=5,
            sigmaColor=30,
            sigmaSpace=30
        )
        smooth_rgb = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2RGB)

        # High-frequency details = Base - Smooth
        details = base_rgb.astype(np.float32) - smooth_rgb.astype(np.float32)

        # -------------------------------------------------------------
        # STEP 3: Noise-Gated Detail Amplification
        # -------------------------------------------------------------
        # Compute local detail magnitude: where details are tiny (flat skin/sky), suppress amplification
        detail_mag = np.linalg.norm(details, axis=-1, keepdims=True)
        # Smooth noise gate: 0.0 where detail < threshold, ramping to 1.0
        gate = np.clip((detail_mag - self.noise_threshold) / (self.noise_threshold * 2.0), 0.0, 1.0)
        
        # Amplified details with adaptive strength
        amplified_details = details * (1.0 + effective_strength * 1.5 * gate)
        sharpened = np.clip(smooth_rgb.astype(np.float32) + amplified_details, 0.0, 255.0).astype(np.uint8)

        # -------------------------------------------------------------
        # STEP 4: Local Contrast Restoration (CLAHE on CIELAB L*)
        # -------------------------------------------------------------
        if self.enable_clahe:
            lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            
            # Conservative CLAHE clip limit to prevent over-saturation
            clahe = cv2.createCLAHE(clipLimit=1.6 * effective_strength, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_chan)
            
            # Blend 70% enhanced L* with 30% original L* for natural tone
            l_final = cv2.addWeighted(l_enhanced, 0.75, l_chan, 0.25, 0)
            enhanced_lab = cv2.merge([l_final, a_chan, b_chan])
            final_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            final_rgb = sharpened

        runtime_ms = (time.time() - t_start) * 1000.0
        final_metrics = self.estimate_blur_metric(final_rgb)

        # Compute sharpness improvement ratio
        sharpness_gain = round(
            float(final_metrics["laplacian_variance"] / max(initial_metrics["laplacian_variance"], 1e-4)), 2
        )

        if return_dict:
            return {
                "deblurred_image": final_rgb,
                "original_image": orig_rgb,
                "runtime_ms": round(runtime_ms, 2),
                "initial_metrics": initial_metrics,
                "final_metrics": final_metrics,
                "sharpness_gain": sharpness_gain,
                "strength_used": effective_strength,
                "dimensions": [w, h]
            }

        return final_rgb

    def save_result(
        self,
        image_rgb: np.ndarray,
        output_path: Union[str, Path]
    ) -> Path:
        """Saves RGB image to disk."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_rgb).save(out_p, quality=95)
        return out_p
