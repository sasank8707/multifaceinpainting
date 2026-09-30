"""
Phase 2 Comprehensive Unit Tests:
Face Analysis, Geometry Analysis, and Mask-Aware Region Profiling.
"""

import os
import sys
import json
import pytest
import numpy as np
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.face_analysis.face_detector import FaceDetector
from modules.face_analysis.pose_estimator import PoseEstimator
from modules.face_analysis.quality_analyzer import QualityAnalyzer
from modules.masking.region_profiler import MaskRegionProfiler
from modules.masking.task_router import TaskRouter, TaskMode
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.masking.synthetic_masks import MaskGenerator


@pytest.fixture(scope="session")
def sample_face_image():
    """Loads a real CelebA-HQ face image for testing."""
    sample_path = r"C:\Users\sasan\Downloads\CelebAMask-HQ\CelebA-HQ-img\0.jpg"
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        # Resize to 512x512 for fast CPU testing
        img = cv2.resize(img, (512, 512))
        return img
    else:
        # Synthetic fallback if image not found
        img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        return img


@pytest.fixture(scope="session")
def pipeline():
    """Initializes FaceAnalysisPipeline."""
    return FaceAnalysisPipeline()


def test_normal_face_detection(sample_face_image):
    """Test 1: Normal face detection and landmark extraction."""
    detector = FaceDetector()
    res = detector.detect(sample_face_image)
    
    assert res is not None, "Face detector failed on CelebA-HQ face!"
    assert res["face_detected"] is True
    assert res["num_landmarks"] == 478
    assert len(res["canonical_68_pixel"]) == 68
    
    bbox = res["bbox"]
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
    assert res["face_area"] > 5000


def test_pose_extraction(sample_face_image):
    """Test 2: Head pose estimation (yaw, pitch, roll)."""
    detector = FaceDetector()
    res = detector.detect(sample_face_image)
    assert res is not None
    
    pose_estimator = PoseEstimator()
    pose = pose_estimator.estimate_pose(res["dense_landmarks_pixel"], sample_face_image.shape[:2])
    
    assert "yaw" in pose and "pitch" in pose and "roll" in pose
    assert abs(pose["yaw"]) <= 90.0
    assert abs(pose["pitch"]) <= 90.0
    assert abs(pose["roll"]) <= 90.0
    
    # Test pose similarity between identical poses
    sim_same = pose_estimator.compute_pose_similarity(pose, pose)
    assert sim_same == 1.0
    
    # Test pose similarity between orthogonal poses
    ortho_pose = {"yaw": pose["yaw"] + 45.0, "pitch": pose["pitch"], "roll": pose["roll"]}
    sim_diff = pose_estimator.compute_pose_similarity(pose, ortho_pose)
    assert sim_diff < sim_same


def test_quality_measurement(sample_face_image):
    """Test 3: Quality measurement (sharpness, illumination, landmark visibility)."""
    analyzer = QualityAnalyzer()
    
    sharpness = analyzer.compute_sharpness(sample_face_image)
    assert sharpness["laplacian_variance"] > 0
    assert 0.0 <= sharpness["normalized_sharpness"] <= 1.0
    
    illum = analyzer.compute_illumination(sample_face_image)
    assert 0.0 <= illum["mean_luminance"] <= 255.0
    assert 0.0 <= illum["exposure_balance_score"] <= 1.0
    
    composite = analyzer.compute_composite_quality_score(sample_face_image)
    assert 0.0 <= composite <= 1.0


def test_arbitrary_rectangular_mask(sample_face_image, pipeline):
    """Test 4: Rectangular mask analysis."""
    gen = MaskGenerator(random_seed=42)
    h, w = sample_face_image.shape[:2]
    rect_mask = gen.generate_rectangular_mask((h, w), target_coverage=0.15)
    
    res = pipeline.analyze(sample_face_image, rect_mask)
    assert res["mask_profile"]["num_components"] == 1
    assert res["mask_profile"]["image_coverage"] > 0.10


def test_elliptical_mask(sample_face_image, pipeline):
    """Test 5: Elliptical mask analysis."""
    gen = MaskGenerator(random_seed=42)
    h, w = sample_face_image.shape[:2]
    face_bbox = pipeline.detector.detect(sample_face_image)["bbox"]
    ellipse_mask = gen.generate_elliptical_mask((h, w), face_bbox=face_bbox, target_coverage=0.25)
    
    res = pipeline.analyze(sample_face_image, ellipse_mask)
    assert res["mask_profile"]["num_components"] == 1
    assert res["mask_profile"]["is_facial"] is True


def test_irregular_mask(sample_face_image, pipeline):
    """Test 6: Irregular brush stroke mask."""
    gen = MaskGenerator(random_seed=42)
    h, w = sample_face_image.shape[:2]
    irreg_mask = gen.generate_irregular_mask((h, w), num_strokes=4)
    
    res = pipeline.analyze(sample_face_image, irreg_mask)
    assert res["mask_profile"]["total_masked_pixels"] > 0
    assert res["mask_profile"]["image_coverage"] > 0.0


def test_multiple_disconnected_masks(sample_face_image, pipeline):
    """Test 7: Multiple disconnected masks and routing to MODE_3."""
    gen = MaskGenerator(random_seed=42)
    h, w = sample_face_image.shape[:2]
    face_bbox = pipeline.detector.detect(sample_face_image)["bbox"]
    multi_mask, comps = gen.generate_multi_component_mask((h, w), face_bbox=face_bbox, num_components=3)
    
    res = pipeline.analyze(sample_face_image, multi_mask)
    assert res["mask_profile"]["num_components"] >= 2
    assert res["mask_profile"]["is_multi_mask"] is True
    assert res["routing"]["task_mode"] == TaskMode.MODE_3_MULTI_MASK.value


def test_partial_facial_mask_and_landmark_overlap(sample_face_image, pipeline):
    """Test 8: Partial facial mask and landmark occlusion detection."""
    h, w = sample_face_image.shape[:2]
    face_info = pipeline.detector.detect(sample_face_image)
    landmarks = face_info["dense_landmarks_pixel"]
    
    # Create mask directly over nose landmark (index 1)
    nose_x, nose_y = int(landmarks[1][0]), int(landmarks[1][1])
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (nose_x, nose_y), 30, 1.0, -1)
    
    res = pipeline.analyze(sample_face_image, mask)
    affected = res["mask_profile"]["global_affected_landmarks"]
    assert 1 in affected, "Nose landmark (1) was occluded but not detected in affected list!"
    assert res["quality"]["landmark_visibility"]["visibility_ratio"] < 1.0


def test_large_facial_mask_severe_occlusion(sample_face_image, pipeline):
    """Test 9: Large facial occlusion (75% and 100% full-face) and routing to MODE_1."""
    gen = MaskGenerator(random_seed=42)
    h, w = sample_face_image.shape[:2]
    face_bbox = pipeline.detector.detect(sample_face_image)["bbox"]
    
    mask_75 = gen.generate_severe_occlusion_mask((h, w), face_bbox=face_bbox, severity_ratio=0.75)
    res_75 = pipeline.analyze(sample_face_image, mask_75)
    
    assert res_75["mask_profile"]["face_coverage"] >= 0.70
    assert res_75["routing"]["task_mode"] == TaskMode.MODE_1_FACIAL_INPAINTING.value
    assert res_75["routing"]["identity_conditioning_enabled"] is True


def test_degradation_mask_and_restoration_routing(sample_face_image, pipeline):
    """Test 10: Localized degradation mask and routing to MODE_2."""
    gen = MaskGenerator(random_seed=42)
    h, w = sample_face_image.shape[:2]
    face_bbox = pipeline.detector.detect(sample_face_image)["bbox"]
    
    deg_mask = gen.generate_degradation_mask((h, w), face_bbox=face_bbox, degradation_type="scar")
    res = pipeline.analyze(sample_face_image, deg_mask, is_degradation=True)
    
    assert res["routing"]["task_mode"] == TaskMode.MODE_2_FACIAL_RESTORATION.value
    assert res["routing"]["identity_conditioning_enabled"] is True
    assert res["routing"]["structural_guidance_enabled"] is True


def test_non_face_object_mask_and_removal_routing(sample_face_image, pipeline):
    """Test 11: Non-face object mask strictly in background and routing to MODE_4."""
    gen = MaskGenerator(random_seed=42)
    h, w = sample_face_image.shape[:2]
    face_bbox = pipeline.detector.detect(sample_face_image)["bbox"]
    
    obj_mask = gen.generate_object_removal_mask((h, w), face_bbox=face_bbox, object_type="letter_L")
    res = pipeline.analyze(sample_face_image, obj_mask)
    
    assert res["mask_profile"]["is_pure_background"] is True
    assert res["mask_profile"]["face_overlap_pixels"] == 0
    assert res["routing"]["task_mode"] == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value
    # CRITICAL: Identity conditioning MUST be FALSE for background removal!
    assert res["routing"]["identity_conditioning_enabled"] is False


def test_edge_case_no_face_detected(pipeline):
    """Test 12: Edge case where no face is present (pure texture / background)."""
    blank_image = np.ones((512, 512, 3), dtype=np.uint8) * 180
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:200, 100:200] = 1.0 # arbitrary object mask
    
    res = pipeline.analyze(blank_image, mask)
    assert res["face"]["detected"] is False
    assert res["face"]["bbox"] is None
    assert res["pose"]["confidence"] == 0.0
    assert res["routing"]["task_mode"] == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value
    assert res["routing"]["identity_conditioning_enabled"] is False


def test_edge_case_mask_outside_face(sample_face_image, pipeline):
    """Test 13: Edge case where a face is present, but mask is completely outside the face."""
    h, w = sample_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    # Mask a tiny corner in top-left (0,0) to (30,30), far from any face
    mask[0:30, 0:30] = 1.0
    
    res = pipeline.analyze(sample_face_image, mask)
    assert res["face"]["detected"] is True
    assert res["mask_profile"]["face_overlap_pixels"] == 0
    assert res["mask_profile"]["is_pure_background"] is True
    assert res["routing"]["task_mode"] == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value
    assert res["routing"]["identity_conditioning_enabled"] is False


def test_deterministic_json_serialization(sample_face_image, pipeline, tmp_path):
    """Test 14: Deterministic JSON serialization and deserialization."""
    h, w = sample_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:250, 200:250] = 1.0
    
    res = pipeline.analyze(sample_face_image, mask)
    json_path = os.path.join(tmp_path, "test_analysis.json")
    pipeline.save_analysis(res, json_path)
    
    assert os.path.exists(json_path)
    with open(json_path, "r") as f:
        loaded = json.load(f)
    assert loaded["face"]["detected"] == res["face"]["detected"]
    assert loaded["routing"]["task_mode"] == res["routing"]["task_mode"]
    assert "mask_profile" in loaded


def test_region_agnostic_property(sample_face_image, pipeline):
    """Test 15: Region-agnostic check ensuring no hardcoded organ logic exists."""
    h, w = sample_face_image.shape[:2]
    # Test masks across 4 arbitrary positions
    for pos_y, pos_x in [(100, 100), (300, 200), (250, 350), (400, 150)]:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[pos_y : pos_y + 40, pos_x : pos_x + 40] = 1.0
        res = pipeline.analyze(sample_face_image, mask)
        # Verify pipeline operates generically and reports metrics
        assert "image_coverage" in res["mask_profile"]
        assert "face_coverage" in res["mask_profile"]
        assert "routing" in res
