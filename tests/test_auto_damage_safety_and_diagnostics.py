"""
Fast, Deterministic Unit Tests for Auto-Damage Safety Checks, Diagnostic Maps, and Review Status.
Uses lightweight StubFaceDetector to eliminate expensive MediaPipe model re-initialization.

Covers:
- Safety Check 1: Rejection of images with no face detected
- Safety Check 2: Flagging of multiple ambiguous faces
- Safety Check 3: Flagging of unreliable landmarks
- Safety Check 4: Flagging of empty damage masks on clean faces
- Diagnostics: Generation of raw color, texture, and facial constraint maps
- Diagnostics: File output of all 5 anomaly maps + composite breakdown figure
- Review Status: Default AUTO_MASK_REVIEW_REQUIRED status on successful detection
"""

import sys
import copy
from pathlib import Path
import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.restoration.damage_detector import DamageMaskDetector, DamageDetectionResult


class StubFaceDetector:
    """Lightweight stub replacing heavy MediaPipe FaceLandmarker for unit testing."""
    def __init__(self, face_info=None):
        self.face_info = face_info

    def detect(self, image):
        return self.face_info


@pytest.fixture
def mock_face_info():
    """Generates synthetic canonical 68 landmarks and bounding box for a 300x300 face."""
    cx, cy = 150, 150
    landmarks = []
    # 0-16: Jawline points around lower half of ellipse
    for angle_deg in np.linspace(0, 180, 17):
        rad = np.deg2rad(angle_deg)
        x = cx + 80 * np.cos(rad)
        y = cy + 110 * np.sin(rad)
        landmarks.append([float(x), float(y)])
    # 17-21: Right eyebrow
    for x in np.linspace(100, 135, 5):
        landmarks.append([float(x), 110.0])
    # 22-26: Left eyebrow
    for x in np.linspace(165, 200, 5):
        landmarks.append([float(x), 110.0])
    # 27-35: Nose bridge & tip (9 points)
    for y in np.linspace(120, 170, 9):
        landmarks.append([150.0, float(y)])
    # 36-41: Right eye (6 points)
    for rad in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        landmarks.append([115.0 + 10 * np.cos(rad), 130.0 + 6 * np.sin(rad)])
    # 42-47: Left eye (6 points)
    for rad in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        landmarks.append([185.0 + 10 * np.cos(rad), 130.0 + 6 * np.sin(rad)])
    # 48-67: Mouth (20 points)
    for rad in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        landmarks.append([150.0 + 25 * np.cos(rad), 210.0 + 10 * np.sin(rad)])

    return {
        "face_detected": True,
        "num_faces_detected": 1,
        "bbox": (60, 30, 240, 270),
        "face_area": 180 * 240,
        "num_landmarks": 68,
        "canonical_68_pixel": landmarks,
        "image_dimensions": {"height": 300, "width": 300}
    }


@pytest.fixture
def synthetic_face_with_damage():
    """Generates a synthetic face with a simulated dark damage lesion."""
    h, w = 300, 300
    target = np.full((h, w, 3), (170, 140, 120), dtype=np.uint8)
    cv2.ellipse(target, (150, 150), (90, 120), 0, 0, 360, (190, 155, 130), -1)
    cv2.circle(target, (115, 130), 12, (240, 240, 240), -1)
    cv2.circle(target, (185, 130), 12, (240, 240, 240), -1)
    cv2.circle(target, (115, 130), 5, (40, 30, 20), -1)
    cv2.circle(target, (185, 130), 5, (40, 30, 20), -1)
    cv2.ellipse(target, (150, 210), (25, 10), 0, 0, 360, (160, 60, 60), -1)

    # Damage patch on cheek
    cv2.ellipse(target, (180, 180), (20, 12), 25, 0, 360, (50, 25, 20), -1)
    cv2.line(target, (165, 175), (195, 185), (20, 10, 10), 2)
    return target


class TestAutoDamageSafetyAndDiagnostics:
    def test_safety_rejects_non_face_image(self):
        """Safety Check 1: Non-face image must be rejected and flagged."""
        noise_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        stub = StubFaceDetector(face_info=None)
        detector = DamageMaskDetector(face_detector=stub)
        res = detector.detect(noise_img)

        assert res.face_detected is False
        assert res.safety_passed is False
        assert res.status == "AUTO_MASK_REJECTED"
        assert "NO_FACE_DETECTED" in res.safety_flags
        assert res.num_pixels == 0

    def test_safety_flags_multiple_faces(self, synthetic_face_with_damage, mock_face_info):
        """Safety Check 2: Flag multiple faces when detected."""
        multi_face_info = copy.deepcopy(mock_face_info)
        multi_face_info["num_faces_detected"] = 2
        stub = StubFaceDetector(face_info=multi_face_info)

        detector = DamageMaskDetector(min_damage_pixels=20, face_detector=stub)
        res = detector.detect(synthetic_face_with_damage)

        assert "MULTIPLE_FACES_DETECTED" in res.safety_flags
        assert res.safety_passed is False
        assert res.status == "AUTO_MASK_REJECTED"

    def test_safety_flags_unreliable_landmarks(self, synthetic_face_with_damage, mock_face_info):
        """Safety Check 3: Flag when landmarks are degenerate or sparse (<30)."""
        sparse_face_info = copy.deepcopy(mock_face_info)
        sparse_face_info["canonical_68_pixel"] = [[150.0, 150.0]] * 15
        sparse_face_info["num_landmarks"] = 15
        stub = StubFaceDetector(face_info=sparse_face_info)

        detector = DamageMaskDetector(min_damage_pixels=20, face_detector=stub)
        res = detector.detect(synthetic_face_with_damage)

        assert "LANDMARKS_UNRELIABLE" in res.safety_flags
        assert res.landmarks_reliable is False
        assert res.safety_passed is False

    def test_safety_flags_empty_mask_on_clean_face(self, mock_face_info):
        """Safety Check 4: Pristine face must yield EMPTY_MASK and not hallucinate."""
        clean_face = np.full((300, 300, 3), (170, 140, 120), dtype=np.uint8)
        cv2.ellipse(clean_face, (150, 150), (90, 120), 0, 0, 360, (190, 155, 130), -1)
        stub = StubFaceDetector(face_info=mock_face_info)

        detector = DamageMaskDetector(min_damage_pixels=100, face_detector=stub)
        res = detector.detect(clean_face)

        assert res.num_pixels == 0
        assert "EMPTY_MASK" in res.safety_flags
        assert res.safety_passed is False
        assert res.status == "AUTO_MASK_REJECTED"

    def test_diagnostics_maps_structure(self, synthetic_face_with_damage, mock_face_info):
        """Task 5: Confirm color, texture, and constraint maps are generated correctly."""
        stub = StubFaceDetector(face_info=mock_face_info)
        detector = DamageMaskDetector(min_damage_pixels=20, face_detector=stub)
        res = detector.detect(synthetic_face_with_damage)

        h, w = synthetic_face_with_damage.shape[:2]
        assert res.color_anomaly_map is not None
        assert res.color_anomaly_map.shape == (h, w)
        assert res.color_anomaly_map.dtype == np.uint8

        assert res.texture_anomaly_map is not None
        assert res.texture_anomaly_map.shape == (h, w)
        assert res.texture_anomaly_map.dtype == np.uint8

        assert res.facial_constraint_mask is not None
        assert res.facial_constraint_mask.shape == (h, w)
        assert res.facial_constraint_mask.dtype == np.uint8

    def test_save_diagnostics_outputs_all_files(self, synthetic_face_with_damage, mock_face_info, tmp_path):
        """Task 5: Confirm save_diagnostics writes all 5 anomaly maps + breakdown figure."""
        stub = StubFaceDetector(face_info=mock_face_info)
        detector = DamageMaskDetector(min_damage_pixels=20, face_detector=stub)
        res = detector.detect(synthetic_face_with_damage)

        diag_paths = res.save_diagnostics(tmp_path)
        assert len(diag_paths) == 6
        assert (tmp_path / "01_raw_color_anomaly.png").exists()
        assert (tmp_path / "02_raw_texture_anomaly.png").exists()
        assert (tmp_path / "03_facial_region_constraint.png").exists()
        assert (tmp_path / "04_final_mask.png").exists()
        assert (tmp_path / "05_final_overlay.png").exists()
        assert (tmp_path / "diagnostic_breakdown.png").exists()

    def test_human_verification_status_on_valid_detection(self, synthetic_face_with_damage, mock_face_info):
        """Task 2: Successful detection must have status AUTO_MASK_REVIEW_REQUIRED."""
        stub = StubFaceDetector(face_info=mock_face_info)
        detector = DamageMaskDetector(min_damage_pixels=20, face_detector=stub)
        res = detector.detect(synthetic_face_with_damage)

        assert res.num_pixels > 0
        assert res.safety_passed is True
        assert res.status == "AUTO_MASK_REVIEW_REQUIRED"
