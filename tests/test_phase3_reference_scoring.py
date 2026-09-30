"""
Phase 3 Comprehensive Unit Tests:
O1 Mask-Aware Multi-Reference Scoring Module.
"""

import os
import sys
import json
import pytest
import numpy as np
import cv2

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.reference_scoring.scorer import MaskAwareReferenceScorer, IdentityMismatchError
from modules.reference_scoring.metrics import (
    compute_pose_compatibility,
    compute_quality_score,
    compute_masked_landmark_visibility,
    compute_region_overlap_compatibility
)
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.masking.synthetic_masks import MaskGenerator
from modules.masking.task_router import TaskMode


@pytest.fixture(scope="session")
def real_face_image():
    """Loads CelebA-HQ image for real face analysis."""
    p = r"C:\Users\sasan\Downloads\CelebAMask-HQ\CelebA-HQ-img\0.jpg"
    if os.path.exists(p):
        img = cv2.imread(p)
        return cv2.resize(img, (512, 512))
    return np.ones((512, 512, 3), dtype=np.uint8) * 128


@pytest.fixture(scope="session")
def pipeline():
    return FaceAnalysisPipeline()


@pytest.fixture(scope="session")
def scorer():
    return MaskAwareReferenceScorer(
        pose_weight=0.30,
        quality_weight=0.25,
        visibility_weight=0.30,
        overlap_weight=0.15,
        top_k=2,
        strict_identity=True
    )


def test_one_reference(real_face_image, pipeline, scorer):
    """Test 1: Scoring and ranking with N=1 reference."""
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:260, 200:260] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    ref_analysis = pipeline.analyze(real_face_image)

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis],
        target_identity_id="ident_01",
        reference_identity_ids=["ident_01"],
        reference_names=["ref_0"]
    )

    assert res["num_references_evaluated"] == 1
    assert "ref_0" in res["global_scores"]
    assert res["global_scores"]["ref_0"] > 0.5
    assert len(res["global_top_k_selected"]) == 1
    assert res["global_top_k_selected"][0] == "ref_0"


def test_three_references(real_face_image, pipeline, scorer):
    """Test 2: Scoring and ranking with N=3 references."""
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[180:250, 180:250] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)

    # Simulate 3 references with varying properties
    ref1_analysis = pipeline.analyze(real_face_image) # Identical sharp face
    ref2_analysis = pipeline.analyze(cv2.GaussianBlur(real_face_image, (15, 15), 0)) # Blurry face
    ref3_analysis = pipeline.analyze(real_face_image)
    # Give ref3 a simulated yaw deviation
    ref3_analysis["pose"] = {"yaw": 40.0, "pitch": target_analysis["pose"]["pitch"], "roll": 0.0, "confidence": 1.0}

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref1_analysis, ref2_analysis, ref3_analysis],
        target_identity_id="ident_42",
        reference_identity_ids=["ident_42", "ident_42", "ident_42"],
        reference_names=["ref_sharp", "ref_blurry", "ref_rotated"]
    )

    assert res["num_references_evaluated"] == 3
    # Sharp, well-aligned ref_sharp should rank #1
    assert res["global_ranked_references"][0]["name"] == "ref_sharp"
    # Top-K (K=2) must contain top 2
    assert len(res["global_top_k_selected"]) == 2
    assert res["global_top_k_selected"][0] == "ref_sharp"


def test_pose_compatibility_metric():
    """Test 3: Head pose compatibility metric."""
    target_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 1.0}
    same_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 1.0}
    slight_pose = {"yaw": 5.0, "pitch": 2.0, "roll": 1.0, "confidence": 1.0}
    far_pose = {"yaw": 45.0, "pitch": 20.0, "roll": 10.0, "confidence": 1.0}

    s_same = compute_pose_compatibility(target_pose, same_pose)
    s_slight = compute_pose_compatibility(target_pose, slight_pose)
    s_far = compute_pose_compatibility(target_pose, far_pose)

    assert s_same == 1.0
    assert s_slight > 0.8
    assert s_far < 0.2
    assert s_same > s_slight > s_far


def test_image_quality_metric():
    """Test 4: Quality & sharpness sub-score."""
    sharp_quality = {
        "sharpness": {"normalized_sharpness": 0.95},
        "illumination": {"exposure_balance_score": 0.90}
    }
    blurry_quality = {
        "sharpness": {"normalized_sharpness": 0.10},
        "illumination": {"exposure_balance_score": 0.80}
    }

    q_sharp = compute_quality_score(sharp_quality)
    q_blurry = compute_quality_score(blurry_quality)

    assert q_sharp > 0.85
    assert q_blurry < 0.45
    assert q_sharp > q_blurry


def test_masked_landmark_visibility_metric():
    """Test 5: Landmark visibility within occluded target region."""
    ref_face_in_bounds = {
        "detected": True,
        "dense_landmarks_pixel": [[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]
    }
    # One landmark is outside reference bounds (clipping)
    ref_face_clipped = {
        "detected": True,
        "dense_landmarks_pixel": [[100.0, 100.0], [600.0, 600.0], [700.0, 700.0]]
    }

    affected = [0, 1]
    v_in = compute_masked_landmark_visibility(affected, ref_face_in_bounds, (512, 512))
    v_clip = compute_masked_landmark_visibility(affected, ref_face_clipped, (512, 512))

    assert v_in == 1.0
    assert v_clip == 0.5
    assert v_in > v_clip


def test_region_overlap_compatibility_metric():
    """Test 6: Geometric region overlap compatibility."""
    target_face = (100, 100, 400, 400)
    comp_bbox = (150, 150, 250, 250) # Inside face

    # Ref face matches target
    ref_centered = {"detected": True, "bbox": (100, 100, 400, 400)}
    # Ref face pushed against border causing overlap region to clip
    ref_shifted = {"detected": True, "bbox": (-150, 100, 150, 400)}

    o_centered = compute_region_overlap_compatibility(comp_bbox, target_face, ref_centered, (512, 512))
    o_shifted = compute_region_overlap_compatibility(comp_bbox, target_face, ref_shifted, (512, 512))

    assert o_centered == 1.0
    assert o_shifted < o_centered


def test_multiple_disconnected_masked_regions(real_face_image, pipeline, scorer):
    """Test 7: Multi-component mask: each component is scored independently."""
    gen = MaskGenerator(random_seed=42)
    h, w = real_face_image.shape[:2]
    face_bbox = pipeline.detector.detect(real_face_image)["bbox"]

    multi_mask, comps = gen.generate_multi_component_mask((h, w), face_bbox=face_bbox, num_components=3)
    target_analysis = pipeline.analyze(real_face_image, multi_mask)

    ref_analysis1 = pipeline.analyze(real_face_image)
    ref_analysis2 = pipeline.analyze(cv2.GaussianBlur(real_face_image, (11, 11), 0))

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis1, ref_analysis2],
        target_identity_id="id_multi",
        reference_identity_ids=["id_multi", "id_multi"],
        reference_names=["ref_clear", "ref_blur"]
    )

    assert len(res["components"]) >= 2
    # Check that each component has its own score breakdown
    for comp_res in res["components"]:
        assert "score_breakdown" in comp_res
        assert "normalized_weights" in comp_res
        assert comp_res["score_breakdown"]["ref_clear"]["composite_score"] >= comp_res["score_breakdown"]["ref_blur"]["composite_score"]


def test_ranking_correctness(real_face_image, pipeline, scorer):
    """Test 8: Ranked references list is strictly sorted in descending score order."""
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:280, 200:280] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    ref1 = pipeline.analyze(real_face_image)
    ref2 = pipeline.analyze(cv2.GaussianBlur(real_face_image, (21, 21), 0))
    ref3 = pipeline.analyze(cv2.GaussianBlur(real_face_image, (9, 9), 0))

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref1, ref2, ref3],
        target_identity_id="id_rank",
        reference_identity_ids=["id_rank", "id_rank", "id_rank"],
        reference_names=["best", "worst", "medium"]
    )

    ranked = res["global_ranked_references"]
    for i in range(len(ranked) - 1):
        assert ranked[i]["score"] >= ranked[i + 1]["score"], f"Ranking violation: {ranked[i]} vs {ranked[i+1]}"


def test_top_k_selection(real_face_image, pipeline):
    """Test 9: Top-K selection properly caps at K entries."""
    custom_scorer = MaskAwareReferenceScorer(top_k=2)
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:260, 200:260] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    refs = [pipeline.analyze(real_face_image) for _ in range(4)]

    res = custom_scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=refs,
        target_identity_id="id_k",
        reference_identity_ids=["id_k"] * 4,
        reference_names=[f"ref_{i}" for i in range(4)]
    )

    assert len(res["global_top_k_selected"]) == 2


def test_verified_identity_enforcement(real_face_image, pipeline, scorer):
    """Test 10: Verified matching identity passes successfully."""
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:250, 200:250] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    ref_analysis = pipeline.analyze(real_face_image)

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis],
        target_identity_id="verified_001",
        reference_identity_ids=["verified_001"],
        reference_names=["ref_0"]
    )
    assert res["identity_status"][0]["is_valid"] is True


def test_identity_mismatch_rejection_strict(real_face_image, pipeline, scorer):
    """Test 11a: Identity mismatch raises IdentityMismatchError in strict mode."""
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:250, 200:250] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    ref_analysis = pipeline.analyze(real_face_image)

    with pytest.raises(IdentityMismatchError):
        scorer.score_references(
            target_analysis=target_analysis,
            reference_analyses=[ref_analysis],
            target_identity_id="person_A",
            reference_identity_ids=["person_B"], # Mismatch!
            reference_names=["impostor"]
        )


def test_identity_mismatch_filtering_lenient(real_face_image, pipeline):
    """Test 11b: Identity mismatch scores 0.0 and is rejected in non-strict filter mode."""
    lenient_scorer = MaskAwareReferenceScorer(strict_identity=False)
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:250, 200:250] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    ref_same = pipeline.analyze(real_face_image)
    ref_diff = pipeline.analyze(real_face_image)

    res = lenient_scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_same, ref_diff],
        target_identity_id="person_A",
        reference_identity_ids=["person_A", "person_B"],
        reference_names=["valid_ref", "impostor_ref"]
    )

    assert res["global_scores"]["impostor_ref"] == 0.0
    assert "impostor_ref" not in res["global_top_k_selected"]
    assert res["identity_status"][1]["is_valid"] is False


def test_non_face_object_removal(real_face_image, pipeline, scorer):
    """Test 12: Non-face object removal disables facial identity conditioning and references."""
    gen = MaskGenerator(random_seed=42)
    h, w = real_face_image.shape[:2]
    face_bbox = pipeline.detector.detect(real_face_image)["bbox"]

    obj_mask = gen.generate_object_removal_mask((h, w), face_bbox=face_bbox, object_type="letter_L")
    target_analysis = pipeline.analyze(real_face_image, obj_mask)

    ref_analysis = pipeline.analyze(real_face_image)

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis],
        target_identity_id="id_obj",
        reference_identity_ids=["id_obj"],
        reference_names=["ref_0"]
    )

    assert res["identity_conditioning_enabled"] is False
    assert res["task_mode"] == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value
    assert res["global_scores"]["ref_0"] == 0.0
    assert len(res["global_top_k_selected"]) == 0


def test_no_face_edge_case(pipeline, scorer):
    """Test 13: Edge case where target has no face detected."""
    blank = np.ones((512, 512, 3), dtype=np.uint8) * 200
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:200, 100:200] = 1.0

    target_analysis = pipeline.analyze(blank, mask)
    ref_analysis = pipeline.analyze(blank)

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis],
        reference_names=["ref_blank"]
    )

    # Automatically routes to non-face mode with 0 facial identity conditioning
    assert res["identity_conditioning_enabled"] is False
    assert res["global_scores"]["ref_blank"] == 0.0


def test_deterministic_scoring(real_face_image, pipeline, scorer):
    """Test 14: Deterministic scoring produces bit-exact identical scores across runs."""
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:270, 200:270] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    ref_analysis = pipeline.analyze(real_face_image)

    res1 = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis],
        reference_names=["ref_det"]
    )
    res2 = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis],
        reference_names=["ref_det"]
    )

    assert res1["global_scores"] == res2["global_scores"]
    assert res1["global_normalized_weights"] == res2["global_normalized_weights"]


def test_region_agnostic_behavior(real_face_image, pipeline, scorer):
    """Test 15: Operates on arbitrary mask coordinates without hardcoding anatomical organs."""
    h, w = real_face_image.shape[:2]
    ref_analysis = pipeline.analyze(real_face_image)

    # Test masks across 3 arbitrary bounding boxes
    boxes = [(100, 120, 150, 170), (280, 200, 340, 260), (380, 150, 420, 300)]
    for y1, x1, y2, x2 in boxes:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        target_analysis = pipeline.analyze(real_face_image, mask)
        res = scorer.score_references(
            target_analysis=target_analysis,
            reference_analyses=[ref_analysis],
            reference_names=["ref_gen"]
        )
        assert res["global_scores"]["ref_gen"] > 0.0


def test_scoring_report_json_export(real_face_image, pipeline, scorer, tmp_path):
    """Test 16: JSON serialization of scoring report."""
    h, w = real_face_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    mask[200:250, 200:250] = 1.0

    target_analysis = pipeline.analyze(real_face_image, mask)
    ref_analysis = pipeline.analyze(real_face_image)

    res = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=[ref_analysis],
        reference_names=["ref_json"]
    )

    out_json = os.path.join(tmp_path, "scoring_report.json")
    scorer.save_scoring_report(res, out_json)

    assert os.path.exists(out_json)
    with open(out_json, "r") as f:
        loaded = json.load(f)
    assert loaded["task_mode"] == res["task_mode"]
    assert "ref_json" in loaded["global_scores"]
