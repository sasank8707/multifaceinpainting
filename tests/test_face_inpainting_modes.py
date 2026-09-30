"""
Unit and Integration Tests for Dual-Path Face Inpainting / Restoration.
Tests:
1. DamageMaskDetector functionality, face bounding, and anomaly segmentation.
2. Path A (User-Supplied Mask): Validation, routing, and bit-exact unmasked preservation.
3. Path B (Auto-Damage Mask, mask=None): Automatic detection, routing, and bit-exact unmasked preservation.
4. Validation Error Handling: Empty mask, shape mismatch, and full-canvas rejection.
5. Invariance Guarantee: Delta_max == 0.0 on all unmasked pixels across both paths.
"""

import os
import sys
import copy
import pytest
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.restoration.damage_detector import DamageMaskDetector, DamageDetectionResult
from modules.restoration.face_inpainting_pipeline import FaceInpaintingPipeline


def create_mock_face_info():
    """Generates synthetic canonical 68 landmarks, dense landmarks, and bbox for 300x300 face."""
    cx, cy = 150, 150
    landmarks = []
    for angle_deg in np.linspace(0, 180, 17):
        rad = np.deg2rad(angle_deg)
        x = cx + 80 * np.cos(rad)
        y = cy + 110 * np.sin(rad)
        landmarks.append([float(x), float(y)])
    for x in np.linspace(100, 135, 5):
        landmarks.append([float(x), 110.0])
    for x in np.linspace(165, 200, 5):
        landmarks.append([float(x), 110.0])
    for y in np.linspace(120, 170, 9):
        landmarks.append([150.0, float(y)])
    for rad in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        landmarks.append([115.0 + 10 * np.cos(rad), 130.0 + 6 * np.sin(rad)])
    for rad in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        landmarks.append([185.0 + 10 * np.cos(rad), 130.0 + 6 * np.sin(rad)])
    for rad in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        landmarks.append([150.0 + 25 * np.cos(rad), 210.0 + 10 * np.sin(rad)])

    dense = [[float(p[0]), float(p[1]), 0.0] for p in landmarks]
    while len(dense) < 478:
        dense.append([150.0, 150.0, 0.0])

    return {
        "face_detected": True,
        "num_faces_detected": 1,
        "bbox": (60, 30, 240, 270),
        "face_area": 180 * 240,
        "num_landmarks": 478,
        "canonical_68_pixel": landmarks,
        "dense_landmarks_pixel": dense,
        "image_dimensions": {"height": 300, "width": 300}
    }


class StubFaceDetector:
    """Deterministic stub replacing heavy MediaPipe FaceLandmarker for fast unit testing."""
    def __init__(self, *args, **kwargs):
        self.face_info = create_mock_face_info()

    def detect(self, image):
        return copy.deepcopy(self.face_info)


@pytest.fixture(autouse=True)
def patch_face_detector(monkeypatch):
    """Intercepts all FaceDetector instantiations across modules to prevent MediaPipe loading."""
    monkeypatch.setattr("modules.face_analysis.face_detector.FaceDetector", StubFaceDetector)
    monkeypatch.setattr("modules.face_analysis.pipeline.FaceDetector", StubFaceDetector)
    monkeypatch.setattr("modules.restoration.damage_detector.FaceDetector", StubFaceDetector)


@pytest.fixture
def synthetic_face_and_refs():
    """Generates a synthetic portrait and 2 reference portraits for testing."""
    h, w = 300, 300
    # Clean skin face (base tone)
    target = np.full((h, w, 3), (170, 140, 120), dtype=np.uint8)
    # Add face elliptical structure
    cv2.ellipse(target, (150, 150), (90, 120), 0, 0, 360, (190, 155, 130), -1)
    # Add eyes
    cv2.circle(target, (115, 130), 12, (240, 240, 240), -1)
    cv2.circle(target, (185, 130), 12, (240, 240, 240), -1)
    cv2.circle(target, (115, 130), 5, (40, 30, 20), -1)
    cv2.circle(target, (185, 130), 5, (40, 30, 20), -1)
    # Add mouth
    cv2.ellipse(target, (150, 210), (25, 10), 0, 0, 360, (160, 60, 60), -1)

    # Reference 1
    ref1 = target.copy()
    cv2.ellipse(ref1, (150, 150), (90, 120), 0, 0, 360, (185, 150, 125), -1)

    # Reference 2
    ref2 = target.copy()
    cv2.ellipse(ref2, (150, 150), (90, 120), 0, 0, 360, (195, 160, 135), -1)

    return target, [ref1, ref2]


@pytest.fixture
def damaged_face(synthetic_face_and_refs):
    """Adds a dark scar/burn anomaly on the cheek region of the synthetic face."""
    target, refs = synthetic_face_and_refs
    damaged = target.copy()
    # Add a burn/scar lesion on right cheek
    cv2.ellipse(damaged, (180, 180), (20, 12), 25, 0, 360, (50, 25, 20), -1)
    # Add scratch lines
    cv2.line(damaged, (165, 175), (195, 185), (20, 10, 10), 2)
    return damaged, refs


class TestDamageMaskDetector:
    def test_detector_initialization(self):
        detector = DamageMaskDetector()
        assert detector.min_damage_pixels > 0
        assert hasattr(detector, "saliency_threshold") or hasattr(detector, "color_threshold_sigma")

    def test_detects_facial_damage(self, damaged_face):
        damaged, refs = damaged_face
        detector = DamageMaskDetector(min_damage_pixels=20)
        result = detector.detect(damaged, reference_images=refs)

        assert isinstance(result, DamageDetectionResult)
        assert result.mask.shape == damaged.shape[:2]
        assert result.mask.dtype == np.uint8
        assert set(np.unique(result.mask)).issubset({0, 255})
        assert result.num_pixels > 0, "Detector should detect the cheek lesion"
        assert result.face_coverage_pct > 0.0
        assert result.face_coverage_pct <= 65.0, "Should satisfy safety coverage bound"
        assert result.confidence > 0.0

        # Check lesion center is masked
        assert result.mask[180, 180] == 255, "Cheek lesion center must be masked"

    def test_unmasked_face_bounds(self, synthetic_face_and_refs):
        target, refs = synthetic_face_and_refs
        detector = DamageMaskDetector(min_damage_pixels=50)
        # On pristine face with zero anomaly, mask should not cover background
        result = detector.detect(target, reference_images=refs)
        assert result.mask[10, 10] == 0, "Background corner must not be masked"


class TestFaceInpaintingPipelineDualModes:
    @pytest.fixture
    def pipeline(self):
        # Use mock_mode=True for fast unit testing without loading 3.4GB SD pipeline
        pipe = FaceInpaintingPipeline(mock_mode=True)
        return pipe

    def test_path_a_supplied_mask(self, pipeline, synthetic_face_and_refs):
        target, refs = synthetic_face_and_refs
        h, w = target.shape[:2]
        # Create user-supplied square mask
        user_mask = np.zeros((h, w), dtype=np.uint8)
        user_mask[100:150, 100:150] = 255

        result = pipeline.restore(
            target_image=target,
            mask=user_mask,
            reference_images=refs,
            seed=42,
            save_debug_vis=False
        )

        assert result["input_mode"] == "MASK_PROVIDED"
        assert result["invariance"]["passed"] is True
        assert result["invariance"]["max_difference"] == 0.0
        assert result["invariance"]["violated_pixels"] == 0
        assert result["final_image"].shape == target.shape
        assert (result["normalized_mask"] > 0).sum() == (user_mask > 0).sum()

        # Strict bit-exact invariance check outside mask
        unmasked_bool = (user_mask == 0)
        np.testing.assert_array_equal(result["final_image"][unmasked_bool], target[unmasked_bool])

    def test_path_b_auto_mask_no_mask_provided(self, pipeline, damaged_face):
        damaged, refs = damaged_face

        result = pipeline.restore(
            target_image=damaged,
            mask=None, # Path B: Auto-Mask
            reference_images=refs,
            seed=42,
            save_debug_vis=False
        )

        assert result["input_mode"] == "AUTO_MASK"
        assert result["mask_metrics"]["num_masked_pixels"] > 0
        assert result["invariance"]["passed"] is True
        assert result["invariance"]["max_difference"] == 0.0
        assert result["final_image"].shape == damaged.shape

        # Verify bit-exact unmasked preservation
        auto_mask = result["normalized_mask"]
        unmasked_bool = (auto_mask == 0)
        np.testing.assert_array_equal(result["final_image"][unmasked_bool], damaged[unmasked_bool])

    def test_validation_rejects_empty_mask(self, pipeline, synthetic_face_and_refs):
        target, refs = synthetic_face_and_refs
        h, w = target.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)

        with pytest.raises(ValueError, match="empty"):
            pipeline.restore(target_image=target, mask=empty_mask, reference_images=refs)

    def test_validation_rejects_shape_mismatch(self, pipeline, synthetic_face_and_refs):
        target, refs = synthetic_face_and_refs
        mismatched_mask = np.ones((100, 100), dtype=np.uint8) * 255

        with pytest.raises(ValueError, match="does not match"):
            pipeline.restore(target_image=target, mask=mismatched_mask, reference_images=refs)

    def test_validation_rejects_full_canvas_mask(self, pipeline, synthetic_face_and_refs):
        target, refs = synthetic_face_and_refs
        h, w = target.shape[:2]
        full_mask = np.full((h, w), 255, dtype=np.uint8)

        with pytest.raises(ValueError, match="covers"):
            pipeline.restore(target_image=target, mask=full_mask, reference_images=refs)

    def test_normalization_converts_float_mask(self, pipeline, synthetic_face_and_refs):
        target, _ = synthetic_face_and_refs
        h, w = target.shape[:2]
        float_mask = np.zeros((h, w), dtype=np.float32)
        float_mask[50:100, 50:100] = 1.0

        normalized = pipeline.validate_and_normalize_mask(float_mask, target.shape[:2])
        assert normalized.dtype == np.uint8
        assert set(np.unique(normalized)).issubset({0, 255})
        assert normalized[75, 75] == 255
