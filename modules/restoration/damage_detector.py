"""
Modular Facial Damage & Defect Mask Detection Module.
Identifies natural facial damage regions (burns, scars, scratches, abrasions, lesions,
or missing facial areas) when no user mask is provided.

Design Principles:
1. Modular Architecture: BaseDamageDetector abstract interface allowing detector upgrades.
2. Facial Constraint: Automatically bounds detection within facial skin boundaries,
   preventing background or hair from accidental masking.
3. Multi-Cue Anomaly Modeling: Combines CIELAB photometric outlier modeling with
   high-frequency morphological texture residual filtering.
4. Reference Consistency: Optionally checks anomalies against reference gallery skin tones.
5. Strict Mask Validation: Returns binary uint8 mask {0, 255}, bounding box, coverage %,
   and confidence score.
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

from modules.face_analysis.face_detector import FaceDetector


@dataclass
class DamageDetectionResult:
    """Encapsulates the output of a damage mask detector."""
    mask: np.ndarray                      # 2D uint8 array {0, 255} matching target (H, W)
    confidence: float                     # Confidence score in [0.0, 1.0]
    bbox: Tuple[int, int, int, int]       # Bounding box (x1, y1, x2, y2) of damaged region
    num_pixels: int                       # Total number of damaged pixels detected
    face_coverage_pct: float              # Percentage of detected face area affected
    image_coverage_pct: float             # Percentage of total image area affected
    detection_method: str                 # Identifier of the detector used
    face_detected: bool                   # Whether a face was successfully identified
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    visualization: Optional[np.ndarray] = None # RGB image with overlay for debugging
    # Diagnostic anomaly and constraint maps (Task 5)
    color_anomaly_map: Optional[np.ndarray] = None      # 2D uint8 [H, W] normalized color distance
    texture_anomaly_map: Optional[np.ndarray] = None    # 2D uint8 [H, W] normalized texture residual
    facial_constraint_mask: Optional[np.ndarray] = None # 2D uint8 [H, W] {0, 255} skin mask boundary
    # Safety and review metrics (Task 2 & Task 6)
    num_faces_detected: int = 1                         # Total faces found in image
    landmarks_reliable: bool = True                     # Landmark quality indicator
    safety_flags: List[str] = field(default_factory=list) # Safety flags triggered (if any)
    safety_passed: bool = True                          # True if all safety checks pass
    status: str = "AUTO_MASK_REVIEW_REQUIRED"           # Human review status

    def save_diagnostics(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, str]:
        """
        Saves the 5 required detector diagnostic maps plus a composite breakdown figure.
        Outputs:
        1. raw color anomaly map
        2. raw texture anomaly map
        3. facial-region constraint
        4. final mask
        5. final overlay
        6. composite diagnostic breakdown figure
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        # 1. Raw color anomaly map
        if self.color_anomaly_map is not None:
            p = out_path / f"{prefix}01_raw_color_anomaly.png"
            cv2.imwrite(str(p), self.color_anomaly_map)
            saved_paths["color_anomaly"] = str(p)

        # 2. Raw texture anomaly map
        if self.texture_anomaly_map is not None:
            p = out_path / f"{prefix}02_raw_texture_anomaly.png"
            cv2.imwrite(str(p), self.texture_anomaly_map)
            saved_paths["texture_anomaly"] = str(p)

        # 3. Facial-region constraint mask
        if self.facial_constraint_mask is not None:
            p = out_path / f"{prefix}03_facial_region_constraint.png"
            cv2.imwrite(str(p), self.facial_constraint_mask)
            saved_paths["facial_constraint"] = str(p)

        # 4. Final binary mask
        if self.mask is not None:
            p = out_path / f"{prefix}04_final_mask.png"
            cv2.imwrite(str(p), self.mask)
            saved_paths["final_mask"] = str(p)

        # 5. Final overlay
        if self.visualization is not None:
            p = out_path / f"{prefix}05_final_overlay.png"
            cv2.imwrite(str(p), cv2.cvtColor(self.visualization, cv2.COLOR_RGB2BGR))
            saved_paths["final_overlay"] = str(p)

        # 6. Composite diagnostic breakdown figure
        p_comp = out_path / f"{prefix}diagnostic_breakdown.png"
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        if self.color_anomaly_map is not None:
            axes[0].imshow(self.color_anomaly_map, cmap="jet")
        axes[0].set_title("1. Raw Color Anomaly\n(CIELAB Distance)")
        axes[0].axis("off")

        if self.texture_anomaly_map is not None:
            axes[1].imshow(self.texture_anomaly_map, cmap="magma")
        axes[1].set_title("2. Raw Texture Anomaly\n(Morph Residual)")
        axes[1].axis("off")

        if self.facial_constraint_mask is not None:
            axes[2].imshow(self.facial_constraint_mask, cmap="gray")
        axes[2].set_title("3. Facial Constraint\n(Hull & Eyes Excluded)")
        axes[2].axis("off")

        if self.mask is not None:
            axes[3].imshow(self.mask, cmap="gray")
        axes[3].set_title(f"4. Detected Mask\n({self.num_pixels:,} px, {self.face_coverage_pct}%)")
        axes[3].axis("off")

        if self.visualization is not None:
            axes[4].imshow(self.visualization)
        axes[4].set_title(f"5. Final Overlay\n(Conf: {self.confidence:.2f}, {self.status})")
        axes[4].axis("off")

        fig.suptitle(f"Detector Diagnostics: {self.status} | Safety Passed: {self.safety_passed}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(p_comp), dpi=150)
        plt.close(fig)
        saved_paths["diagnostic_breakdown"] = str(p_comp)

        return saved_paths


class BaseDamageDetector(ABC):
    """Abstract base class for all facial damage detection implementations."""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        reference_images: Optional[List[np.ndarray]] = None,
        face_info: Optional[Dict[str, Any]] = None
    ) -> DamageDetectionResult:
        """
        Detects damaged or abnormal regions on a face.

        Args:
            image: [H, W, 3] RGB uint8 image.
            reference_images: Optional list of same-identity reference photos.
            face_info: Optional precomputed face detection dictionary.

        Returns:
            DamageDetectionResult containing binary mask and diagnostic metadata.
        """
        pass


class DamageMaskDetector(BaseDamageDetector):
    """
    Robust unsupervised facial damage detector.
    Combines facial landmark boundary isolation, CIELAB skin tone outlier modeling,
    morphological texture residual analysis, and spatial clustering.
    """

    def __init__(
        self,
        color_threshold_sigma: float = 2.2,
        texture_kernel_size: int = 9,
        min_damage_pixels: int = 50,
        max_face_coverage: float = 0.65,
        dilation_iterations: int = 2,
        face_detector: Optional[FaceDetector] = None
    ):
        """
        Args:
            color_threshold_sigma: Multiplier for standard deviation in skin color outlier scoring.
            texture_kernel_size: Kernel size for morphological top-hat / black-hat residual filtering.
            min_damage_pixels: Minimum pixel count to register as a valid damage region.
            max_face_coverage: Safety upper bound on face area coverage (default 65%).
            dilation_iterations: Number of morphological dilation steps for lesion boundary coverage.
            face_detector: Optional shared instance of FaceDetector.
        """
        self.color_threshold_sigma = color_threshold_sigma
        self.texture_kernel_size = texture_kernel_size
        self.min_damage_pixels = min_damage_pixels
        self.max_face_coverage = max_face_coverage
        self.dilation_iterations = dilation_iterations

        if face_detector is not None:
            self.face_detector = face_detector
        else:
            try:
                self.face_detector = FaceDetector()
            except Exception:
                self.face_detector = None

    def _get_face_skin_mask(
        self,
        image: np.ndarray,
        face_info: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Creates a binary mask isolating the facial skin surface using detected landmarks.
        """
        h, w = image.shape[:2]
        skin_mask = np.zeros((h, w), dtype=np.uint8)

        if face_info is not None and face_info.get("face_detected", False):
            bbox = face_info["bbox"]
            landmarks = (
                face_info.get("canonical_68_pixel") or
                face_info.get("landmarks_pixel") or
                []
            )

            if len(landmarks) >= 30:
                # Use facial boundary landmarks to form convex hull of face
                jaw_indices = list(range(0, 17))
                eyebrow_indices = list(range(17, 27))
                boundary_pts = [landmarks[i][:2] for i in jaw_indices + eyebrow_indices if i < len(landmarks)]
                hull = cv2.convexHull(np.array(boundary_pts, dtype=np.int32))
                cv2.fillConvexPoly(skin_mask, hull, 255)

                # Carve out eye orbits to avoid flagging normal pupils/eyelashes as lesions
                for eye_range in [range(36, 42), range(42, 48)]:
                    eye_pts = [landmarks[i][:2] for i in eye_range if i < len(landmarks)]
                    if eye_pts:
                        eye_hull = cv2.convexHull(np.array(eye_pts, dtype=np.int32))
                        cv2.fillConvexPoly(skin_mask, eye_hull, 0)
            else:
                # Fallback to inner face bounding box with margin
                fx1, fy1, fx2, fy2 = bbox
                fw, fh = fx2 - fx1, fy2 - fy1
                skin_mask[int(fy1 + 0.1 * fh):int(fy2 - 0.05 * fh), int(fx1 + 0.1 * fw):int(fx2 - 0.1 * fw)] = 255
            return skin_mask, bbox

        # Fallback if no face was detected: use central elliptical face prior
        cx, cy = w // 2, h // 2
        axes = (int(w * 0.28), int(h * 0.38))
        cv2.ellipse(skin_mask, (cx, cy), axes, 0, 0, 360, 255, -1)
        fallback_bbox = (max(0, cx - axes[0]), max(0, cy - axes[1]), min(w, cx + axes[0]), min(h, cy + axes[1]))
        return skin_mask, fallback_bbox

    def detect(
        self,
        image: np.ndarray,
        reference_images: Optional[List[np.ndarray]] = None,
        face_info: Optional[Dict[str, Any]] = None
    ) -> DamageDetectionResult:
        """
        Executes automatic damage mask estimation on target image with safety checks.
        """
        assert image is not None and image.ndim == 3, "Input image must be a 3-channel RGB image."
        h, w, c = image.shape
        total_image_pixels = h * w

        # 1. Face detection & landmark acquisition
        if face_info is None and self.face_detector is not None:
            try:
                face_info = self.face_detector.detect(image)
            except Exception:
                face_info = None

        has_face = face_info is not None and face_info.get("face_detected", False)
        num_faces = face_info.get("num_faces_detected", 1) if has_face else 0
        landmarks = (
            face_info.get("landmarks_pixel") or
            face_info.get("dense_landmarks_pixel") or
            face_info.get("canonical_68_pixel") or []
        ) if has_face else []
        num_landmarks = len(landmarks)

        # Landmark reliability assessment
        landmarks_reliable = True
        if has_face:
            f_bbox = face_info.get("bbox", (0, 0, 0, 0))
            f_w = f_bbox[2] - f_bbox[0]
            f_h = f_bbox[3] - f_bbox[1]
            if num_landmarks < 30 or f_w < 40 or f_h < 40:
                landmarks_reliable = False
        else:
            landmarks_reliable = False

        safety_flags: List[str] = []
        if not has_face:
            safety_flags.append("NO_FACE_DETECTED")
        if num_faces > 1:
            safety_flags.append("MULTIPLE_FACES_DETECTED")
        if has_face and not landmarks_reliable:
            safety_flags.append("LANDMARKS_UNRELIABLE")

        # Early exit if no face detected: do not hallucinate damage on arbitrary non-face images
        if not has_face:
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            safety_flags.append("EMPTY_MASK")
            return DamageDetectionResult(
                mask=empty_mask,
                confidence=0.0,
                bbox=(0, 0, 0, 0),
                num_pixels=0,
                face_coverage_pct=0.0,
                image_coverage_pct=0.0,
                detection_method="SkinColorTextureResidual_v1",
                face_detected=False,
                face_bbox=None,
                visualization=image.copy(),
                color_anomaly_map=np.zeros((h, w), dtype=np.uint8),
                texture_anomaly_map=np.zeros((h, w), dtype=np.uint8),
                facial_constraint_mask=np.zeros((h, w), dtype=np.uint8),
                num_faces_detected=0,
                landmarks_reliable=False,
                safety_flags=safety_flags,
                safety_passed=False,
                status="AUTO_MASK_REJECTED"
            )

        skin_mask, face_bbox = self._get_face_skin_mask(image, face_info)
        face_pixel_count = int(np.sum(skin_mask > 0))

        if face_pixel_count < 50:
            if "LANDMARKS_UNRELIABLE" not in safety_flags:
                safety_flags.append("LANDMARKS_UNRELIABLE")
            if "EMPTY_MASK" not in safety_flags:
                safety_flags.append("EMPTY_MASK")
            return DamageDetectionResult(
                mask=np.zeros((h, w), dtype=np.uint8),
                confidence=0.0,
                bbox=(0, 0, 0, 0),
                num_pixels=0,
                face_coverage_pct=0.0,
                image_coverage_pct=0.0,
                detection_method="SkinColorTextureResidual_v1",
                face_detected=has_face,
                face_bbox=face_bbox,
                visualization=image.copy(),
                color_anomaly_map=np.zeros((h, w), dtype=np.uint8),
                texture_anomaly_map=np.zeros((h, w), dtype=np.uint8),
                facial_constraint_mask=skin_mask,
                num_faces_detected=num_faces,
                landmarks_reliable=False,
                safety_flags=safety_flags,
                safety_passed=False,
                status="AUTO_MASK_REJECTED"
            )

        # 2. Color Space Anomaly Analysis (CIELAB)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        skin_indices = np.where(skin_mask > 0)
        l_skin = l_channel[skin_indices]
        a_skin = a_channel[skin_indices]
        b_skin = b_channel[skin_indices]

        # Robust median and MAD / IQR statistics for healthy skin
        med_l, std_l = np.median(l_skin), np.std(l_skin) + 1e-6
        med_a, std_a = np.median(a_skin), np.std(a_skin) + 1e-6
        med_b, std_b = np.median(b_skin), np.std(b_skin) + 1e-6

        # Standardized Mahalanobis-like color distance from median skin
        dist_color = np.sqrt(
            ((l_channel - med_l) / std_l) ** 2 +
            ((a_channel - med_a) / std_a) ** 2 +
            ((b_channel - med_b) / std_b) ** 2
        )

        # 3. High-Frequency Morphological Texture Residual Analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        k_size = self.texture_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        # Top-hat highlights bright scratch/scar lines; black-hat highlights dark crusts/scabs/tears
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        texture_anomaly = top_hat + black_hat

        tex_skin = texture_anomaly[skin_indices]
        med_tex, std_tex = np.median(tex_skin), np.std(tex_skin) + 1e-6
        dist_texture = (texture_anomaly - med_tex) / std_tex

        # Raw normalized diagnostic anomaly maps for inspection
        norm_color_map = np.clip((dist_color / 6.0) * 255.0, 0, 255).astype(np.uint8)
        norm_texture_map = np.clip((np.maximum(0, dist_texture) / 6.0) * 255.0, 0, 255).astype(np.uint8)

        # 4. Joint Damage Saliency Map
        saliency = 0.65 * dist_color + 0.35 * dist_texture
        saliency_skin = saliency[skin_indices]
        threshold = np.percentile(saliency_skin, 92.0)
        threshold = max(threshold, float(np.mean(saliency_skin) + self.color_threshold_sigma * np.std(saliency_skin)))

        raw_damage = (saliency > threshold).astype(np.uint8) * 255
        raw_damage = cv2.bitwise_and(raw_damage, skin_mask)

        # 5. Reference Gallery Calibration (if references provided)
        if reference_images:
            ref_med_a = []
            ref_med_b = []
            for ref_img in reference_images:
                if ref_img is not None and ref_img.size > 0:
                    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB).astype(np.float32)
                    ref_med_a.append(np.median(ref_lab[:, :, 1]))
                    ref_med_b.append(np.median(ref_lab[:, :, 2]))
            if ref_med_a and ref_med_b:
                ref_a = np.mean(ref_med_a)
                ref_b = np.mean(ref_med_b)
                # Check chroma deviation against reference gallery
                chroma_diff = np.sqrt((a_channel - ref_a) ** 2 + (b_channel - ref_b) ** 2)
                chroma_thresh = np.percentile(chroma_diff[skin_indices], 85.0)
                chroma_mask = (chroma_diff > chroma_thresh).astype(np.uint8) * 255
                # Weight into damage mask
                raw_damage = cv2.bitwise_or(raw_damage, cv2.bitwise_and(chroma_mask, skin_mask))

        # 6. Spatial Filtering & Morphological Refinement
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Open to remove isolated single-pixel noise
        cleaned = cv2.morphologyEx(raw_damage, cv2.MORPH_OPEN, cleanup_kernel)
        # Close to join closely fragmented damage patches
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

        # Remove tiny connected components smaller than min_damage_pixels
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        final_mask = np.zeros_like(closed)
        retained_components = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_damage_pixels:
                final_mask[labels == i] = 255
                retained_components += 1

        # Dilate slightly so inpainting covers lesion perimeter smoothly
        if self.dilation_iterations > 0 and retained_components > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=self.dilation_iterations)
            # Re-constrain strictly inside skin mask
            final_mask = cv2.bitwise_and(final_mask, skin_mask)

        # 7. Compute Statistics & Safety Checks
        num_pixels = int(np.sum(final_mask > 0))
        face_coverage_pct = float(num_pixels / face_pixel_count) * 100.0
        image_coverage_pct = float(num_pixels / total_image_pixels) * 100.0

        # Safety clamp: if detection exceeded max_face_coverage, prune lowest saliency pixels
        if face_coverage_pct > (self.max_face_coverage * 100.0):
            excess_factor = (self.max_face_coverage * 100.0) / max(face_coverage_pct, 1.0)
            target_keep = int(face_pixel_count * self.max_face_coverage)
            mask_indices = np.where(final_mask > 0)
            sal_values = saliency[mask_indices]
            thresh_sal = np.percentile(sal_values, (1.0 - excess_factor) * 100.0)
            final_mask = np.logical_and(final_mask > 0, saliency >= thresh_sal).astype(np.uint8) * 255
            num_pixels = int(np.sum(final_mask > 0))
            face_coverage_pct = float(num_pixels / face_pixel_count) * 100.0
            image_coverage_pct = float(num_pixels / total_image_pixels) * 100.0

        # Compute bounding box and confidence
        if num_pixels > 0:
            ys, xs = np.where(final_mask > 0)
            bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
            mean_anomaly_sal = float(np.mean(saliency[ys, xs]))
            baseline_sal = float(np.mean(saliency_skin))
            confidence = min(1.0, max(0.2, (mean_anomaly_sal - baseline_sal) / (np.std(saliency_skin) * 3.0 + 1e-6)))
        else:
            bbox = (0, 0, 0, 0)
            confidence = 0.0

        # Safety Check 1: Empty mask
        if num_pixels == 0:
            safety_flags.append("EMPTY_MASK")

        # Safety Check 2: Excessively large mask (>65% face or >50% canvas)
        if face_coverage_pct > (self.max_face_coverage * 100.0) or image_coverage_pct > 50.0:
            safety_flags.append("MASK_EXCESSIVELY_LARGE")

        # Safety Check 3: Mask mostly outside the face
        if num_pixels > 0:
            outside_pixels = int(np.sum((final_mask > 0) & (skin_mask == 0)))
            if (outside_pixels / num_pixels) > 0.05:
                safety_flags.append("MASK_OUTSIDE_FACE")

        # Safety Check 4: Detector confidence too low
        if confidence < 0.20:
            safety_flags.append("CONFIDENCE_TOO_LOW")

        safety_passed = (len(safety_flags) == 0)
        status = "AUTO_MASK_REVIEW_REQUIRED" if safety_passed else "AUTO_MASK_REJECTED"

        # 8. Create Debug Visualization Overlay
        vis_img = image.copy()
        if num_pixels > 0:
            overlay = vis_img.copy()
            # Red highlight if safety passed; amber highlight if safety failed/rejected
            color_mask = [230, 40, 40] if safety_passed else [240, 140, 20]
            overlay[final_mask > 0] = color_mask
            vis_img = cv2.addWeighted(overlay, 0.45, vis_img, 0.55, 0)
            x1, y1, x2, y2 = bbox
            box_color = (0, 255, 255) if safety_passed else (0, 165, 255)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)
            label_text = f"Auto-Damage ({face_coverage_pct:.1f}% face, conf: {confidence:.2f})"
            if not safety_passed:
                label_text += f" [{','.join(safety_flags)}]"
            cv2.putText(
                vis_img,
                label_text,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

        return DamageDetectionResult(
            mask=final_mask,
            confidence=round(confidence, 3),
            bbox=bbox,
            num_pixels=num_pixels,
            face_coverage_pct=round(face_coverage_pct, 2),
            image_coverage_pct=round(image_coverage_pct, 2),
            detection_method="SkinColorTextureResidual_v1",
            face_detected=has_face,
            face_bbox=face_bbox,
            visualization=vis_img,
            color_anomaly_map=norm_color_map,
            texture_anomaly_map=norm_texture_map,
            facial_constraint_mask=skin_mask.copy(),
            num_faces_detected=num_faces,
            landmarks_reliable=landmarks_reliable,
            safety_flags=safety_flags,
            safety_passed=safety_passed,
            status=status
        )


# Backward compatibility alias
DamageMaskDetectorV1 = DamageMaskDetector


class DamageMaskDetectorV2(BaseDamageDetector):
    """
    Precision Facial Damage & Trauma Segmentation Detector (v2).
    Addresses v1 failure modes:
    1. Reference-Aware Photometric Discrepancy: Compares target to landmark-aligned clean reference in CIELAB space.
    2. Semantic Structure Protection: Strictly shields eye orbits (eyes, upper eyelids, eyebrows), lips, and nostrils from false positives.
    3. Forehead & Temple Hull Coverage: Uses 478 dense landmarks to cover upper temple lacerations up to the hairline.
    4. Dual-Cue Gating: Requires both high-frequency texture discontinuity (scabs/tears) and color anomaly.
    5. Spatial Coherence: Prunes disconnected low-saliency peripheral noise clusters into a unified damage region.
    6. Strict Safety Bounds: Enforces <=65% face coverage limit.
    """

    def __init__(
        self,
        min_damage_pixels: int = 80,
        max_face_coverage: float = 0.65,
        saliency_threshold: float = 1.05,
        dilation_iterations: int = 2,
        face_detector: Optional[FaceDetector] = None
    ):
        self.min_damage_pixels = min_damage_pixels
        self.max_face_coverage = max_face_coverage
        self.saliency_threshold = saliency_threshold
        self.dilation_iterations = dilation_iterations

        if face_detector is not None:
            self.face_detector = face_detector
        else:
            try:
                self.face_detector = FaceDetector()
            except Exception:
                self.face_detector = None

    def _align_reference_to_target(
        self,
        target_img: np.ndarray,
        ref_img: np.ndarray,
        lms_t: np.ndarray,
        ref_info: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Aligns clean reference face to target face using 5-point affine transform."""
        lms_r = np.array(ref_info.get("canonical_68_pixel") or ref_info.get("landmarks_pixel") or [])
        if len(lms_r) < 68 or len(lms_t) < 68:
            return None

        h, w = target_img.shape[:2]
        src_pts = np.float32([
            lms_r[36:42].mean(axis=0), # right eye center
            lms_r[42:48].mean(axis=0), # left eye center
            lms_r[30],                 # nose tip
            lms_r[48],                 # right mouth corner
            lms_r[54]                  # left mouth corner
        ])
        dst_pts = np.float32([
            lms_t[36:42].mean(axis=0),
            lms_t[42:48].mean(axis=0),
            lms_t[30],
            lms_t[48],
            lms_t[54]
        ])

        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if M is None:
            return None
        warped = cv2.warpAffine(ref_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return warped

    def detect(
        self,
        image: np.ndarray,
        reference_images: Optional[List[np.ndarray]] = None,
        face_info: Optional[Dict[str, Any]] = None
    ) -> DamageDetectionResult:
        assert image is not None and image.ndim == 3, "Input image must be a 3-channel RGB image."
        h, w, c = image.shape
        total_image_pixels = h * w

        # 1. Face Detection & Landmark Extraction
        if face_info is None and self.face_detector is not None:
            try:
                face_info = self.face_detector.detect(image)
            except Exception:
                face_info = None

        has_face = face_info is not None and face_info.get("face_detected", False)
        num_faces = face_info.get("num_faces_detected", 1) if has_face else 0
        lms68 = np.array(face_info.get("canonical_68_pixel") or face_info.get("landmarks_pixel") or []) if has_face else np.empty((0, 2))
        dense = np.array(face_info.get("dense_landmarks_pixel") or []) if has_face else np.empty((0, 3))

        safety_flags: List[str] = []
        if not has_face:
            safety_flags.append("NO_FACE_DETECTED")
            safety_flags.append("EMPTY_MASK")
            return DamageDetectionResult(
                mask=np.zeros((h, w), dtype=np.uint8),
                confidence=0.0,
                bbox=(0, 0, 0, 0),
                num_pixels=0,
                face_coverage_pct=0.0,
                image_coverage_pct=0.0,
                detection_method="LocalizedDamageDetector_v2",
                face_detected=False,
                face_bbox=None,
                visualization=image.copy(),
                color_anomaly_map=np.zeros((h, w), dtype=np.uint8),
                texture_anomaly_map=np.zeros((h, w), dtype=np.uint8),
                facial_constraint_mask=np.zeros((h, w), dtype=np.uint8),
                num_faces_detected=0,
                landmarks_reliable=False,
                safety_flags=safety_flags,
                safety_passed=False,
                status="AUTO_MASK_REJECTED"
            )

        face_bbox = face_info["bbox"]
        fx1, fy1, fx2, fy2 = face_bbox
        landmarks_reliable = len(lms68) >= 60 and (fx2 - fx1) >= 40 and (fy2 - fy1) >= 40

        # 2. Build Complete Facial Hull (including forehead and temples)
        face_hull = np.zeros((h, w), dtype=np.uint8)
        if len(dense) >= 100:
            hull_dense = cv2.convexHull(dense[:, :2].astype(np.int32))
            cv2.fillConvexPoly(face_hull, hull_dense, 255)
        elif len(lms68) >= 27:
            jaw_idx = list(range(0, 17))
            eb_idx = list(range(17, 27))
            boundary_pts = [lms68[i][:2] for i in jaw_idx + eb_idx]
            hull_68 = cv2.convexHull(np.array(boundary_pts, dtype=np.int32))
            cv2.fillConvexPoly(face_hull, hull_68, 255)
            top_y = max(0, int(fy1))
            eb_min_y = int(np.min(lms68[17:27, 1]))
            if eb_min_y > top_y:
                forehead_box = np.zeros((h, w), dtype=np.uint8)
                forehead_box[top_y:eb_min_y, int(fx1):int(fx2)] = 255
                face_hull = cv2.bitwise_or(face_hull, forehead_box)
        else:
            cx, cy = w // 2, h // 2
            axes = (int(w * 0.28), int(h * 0.38))
            cv2.ellipse(face_hull, (cx, cy), axes, 0, 0, 360, 255, -1)

        face_pixel_count = max(1, int(np.sum(face_hull > 0)))

        # 3. Construct Semantic Structure Protection Mask (Requirement B)
        # Complete Orbital Protection (Eyeball + Upper Eyelid + Eyebrow), Lips, Nostrils
        semantic_excl = np.zeros((h, w), dtype=np.uint8)
        if len(lms68) >= 68:
            right_orbit_pts = np.vstack([lms68[36:42], lms68[17:22]])
            cv2.fillConvexPoly(semantic_excl, cv2.convexHull(right_orbit_pts.astype(np.int32)), 255)

            left_orbit_pts = np.vstack([lms68[42:48], lms68[22:27]])
            cv2.fillConvexPoly(semantic_excl, cv2.convexHull(left_orbit_pts.astype(np.int32)), 255)

            mouth_hull = cv2.convexHull(lms68[48:68, :2].astype(np.int32))
            cv2.fillConvexPoly(semantic_excl, mouth_hull, 255)

            nostril_pts = lms68[[31, 32, 33, 34, 35, 30], :2].astype(np.int32)
            cv2.fillConvexPoly(semantic_excl, cv2.convexHull(nostril_pts), 255)

            excl_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            semantic_excl = cv2.dilate(semantic_excl, excl_kernel, iterations=1)

        valid_skin = cv2.bitwise_and(face_hull, cv2.bitwise_not(semantic_excl))
        skin_idx = np.where(valid_skin > 0)

        # 4. Multi-Scale Morphological Texture Residual Analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        b5 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k5).astype(np.float32)
        b9 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k9).astype(np.float32)
        t5 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k5).astype(np.float32)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(sobelx**2 + sobely**2)

        tex_raw = b5 * 0.40 + b9 * 0.30 + t5 * 0.10 + grad * 0.20
        med_tex = np.median(tex_raw[skin_idx])
        std_tex = np.std(tex_raw[skin_idx]) + 1e-6
        norm_tex = np.maximum(0, (tex_raw - med_tex) / std_tex)

        # 5. Intra-Face Color Anomaly (CIELAB)
        lab_t = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l, a, b = lab_t[:, :, 0], lab_t[:, :, 1], lab_t[:, :, 2]
        med_l, std_l = np.median(l[skin_idx]), np.std(l[skin_idx]) + 1e-6
        med_a, std_a = np.median(a[skin_idx]), np.std(a[skin_idx]) + 1e-6
        redness = np.maximum(0, (a - med_a) / std_a)
        darkness = np.maximum(0, (med_l - l) / std_l)
        intra_color = redness * 0.60 + darkness * 0.40

        # 6. Reference-Aware Alignment & Photometric Discrepancy (Requirement A)
        norm_de = np.zeros((h, w), dtype=np.float32)
        has_reference_delta = False
        if reference_images and self.face_detector is not None:
            for ref_img in reference_images:
                try:
                    ref_info = self.face_detector.detect(ref_img)
                    if ref_info and ref_info.get("face_detected", False):
                        warped_ref = self._align_reference_to_target(image, ref_img, lms68, ref_info)
                        if warped_ref is not None:
                            lab_r = cv2.cvtColor(warped_ref, cv2.COLOR_RGB2LAB).astype(np.float32)
                            delta_e = np.sqrt(
                                (lab_t[:, :, 0] - lab_r[:, :, 0])**2 +
                                (lab_t[:, :, 1] - lab_r[:, :, 1])**2 +
                                (lab_t[:, :, 2] - lab_r[:, :, 2])**2
                            )
                            med_de = np.median(delta_e[skin_idx])
                            std_de = np.std(delta_e[skin_idx]) + 1e-6
                            norm_de = np.maximum(0, (delta_e - med_de) / std_de)
                            has_reference_delta = True
                            break
                except Exception:
                    continue

        # 7. Saliency Combination & Dual-Cue Gating
        if has_reference_delta:
            saliency = 0.40 * norm_tex + 0.35 * norm_de + 0.25 * intra_color
            core_condition = np.logical_and(
                valid_skin > 0,
                np.logical_or(
                    norm_tex > 2.2,
                    np.logical_and(norm_tex > 0.8, norm_de > 1.2)
                )
            )
        else:
            saliency = 0.55 * norm_tex + 0.45 * intra_color
            core_condition = np.logical_and(
                valid_skin > 0,
                np.logical_or(
                    norm_tex > 2.2,
                    np.logical_and(norm_tex > 1.0, intra_color > 1.0)
                )
            )

        core_mask = core_condition.astype(np.uint8) * 255

        # 8. Morphological Dilation & Boundary Refinement strictly inside valid skin
        cleaned_core = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        dilated = cv2.dilate(cleaned_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=self.dilation_iterations)
        dilated = cv2.bitwise_and(dilated, valid_skin)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        closed = cv2.bitwise_and(closed, valid_skin)

        # 9. Spatial Coherence & Component Saliency Filtering (Requirement C)
        num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        final_mask = np.zeros_like(closed)
        retained_components = 0

        for i in range(1, num_lbl):
            area = stats[i, cv2.CC_STAT_AREA]
            comp_sal = saliency[lbls == i].mean()
            if area >= self.min_damage_pixels and comp_sal >= self.saliency_threshold:
                final_mask[lbls == i] = 255
                retained_components += 1

        num_pixels = int(np.sum(final_mask > 0))
        face_coverage_pct = float(num_pixels / face_pixel_count) * 100.0
        image_coverage_pct = float(num_pixels / total_image_pixels) * 100.0

        # Safety Check: Enforce <=65% face coverage
        if face_coverage_pct > (self.max_face_coverage * 100.0):
            excess_factor = (self.max_face_coverage * 100.0) / max(face_coverage_pct, 1.0)
            thresh_sal = np.percentile(saliency[final_mask > 0], (1.0 - excess_factor) * 100.0)
            final_mask = np.logical_and(final_mask > 0, saliency >= thresh_sal).astype(np.uint8) * 255
            num_pixels = int(np.sum(final_mask > 0))
            face_coverage_pct = float(num_pixels / face_pixel_count) * 100.0
            image_coverage_pct = float(num_pixels / total_image_pixels) * 100.0
            safety_flags.append("MASK_EXCESSIVELY_LARGE")

        if num_pixels > 0:
            ys, xs = np.where(final_mask > 0)
            bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
            mean_sal = float(np.mean(saliency[ys, xs]))
            confidence = min(1.0, max(0.20, (mean_sal - 0.5) / 1.5))
        else:
            bbox = (0, 0, 0, 0)
            confidence = 0.0
            safety_flags.append("EMPTY_MASK")

        safety_passed = (len(safety_flags) == 0)
        status = "AUTO_MASK_REVIEW_REQUIRED" if safety_passed else "AUTO_MASK_REJECTED"

        # Diagnostic normalized maps (uint8)
        norm_color_map = np.clip((intra_color / 4.0) * 255.0, 0, 255).astype(np.uint8)
        norm_texture_map = np.clip((norm_tex / 4.0) * 255.0, 0, 255).astype(np.uint8)

        # Overlay image
        vis_img = image.copy()
        if num_pixels > 0:
            overlay = vis_img.copy()
            overlay[final_mask > 0] = [230, 40, 40]
            vis_img = cv2.addWeighted(overlay, 0.45, vis_img, 0.55, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                vis_img,
                f"Damage v2: {num_pixels:,} px ({face_coverage_pct:.1f}% face, conf: {confidence:.2f})",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        return DamageDetectionResult(
            mask=final_mask,
            confidence=round(confidence, 3),
            bbox=bbox,
            num_pixels=num_pixels,
            face_coverage_pct=round(face_coverage_pct, 2),
            image_coverage_pct=round(image_coverage_pct, 2),
            detection_method="LocalizedDamageDetector_v2",
            face_detected=has_face,
            face_bbox=face_bbox,
            visualization=vis_img,
            color_anomaly_map=norm_color_map,
            texture_anomaly_map=norm_texture_map,
            facial_constraint_mask=valid_skin.copy(),
            num_faces_detected=num_faces,
            landmarks_reliable=landmarks_reliable,
            safety_flags=safety_flags,
            safety_passed=safety_passed,
            status=status
        )


# Default to V2 implementation
DamageMaskDetector = DamageMaskDetectorV2
