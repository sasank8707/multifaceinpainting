"""
Phase 1 Unit Tests: Dataset Validation, Identity Grouping, and Synthetic Mask Generators.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data.identity_index import CelebAIdentityIndexer
from modules.masking.synthetic_masks import MaskGenerator


def test_identity_indexer_grouping():
    """Verifies that the indexer correctly parses files and forms multi-image identity sets."""
    indexer = CelebAIdentityIndexer()
    data = indexer.load_or_build_index()
    
    assert "id_to_hq" in data
    assert "splits" in data
    assert len(data["id_to_hq"]) > 0, "No multi-image identities found!"
    
    # Check that all identities in id_to_hq have >= 2 images
    for ident_id, images in data["id_to_hq"].items():
        assert len(images) >= 2, f"Identity {ident_id} has fewer than 2 images ({len(images)})"


def test_zero_identity_leakage_across_splits():
    """Verifies that train, val, and test splits have 0 common identities."""
    indexer = CelebAIdentityIndexer()
    data = indexer.load_or_build_index()
    splits = data["splits"]
    
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Identity overlap between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Identity overlap between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Identity overlap between Val and Test!"


def test_sample_pair_selection():
    """Verifies that get_sample_pair returns target and distinct references of same identity."""
    indexer = CelebAIdentityIndexer()
    indexer.load_or_build_index()
    
    sample_ident = list(indexer.id_to_hq.keys())[0]
    target_idx, refs = indexer.get_sample_pair(sample_ident, max_references=3, seed=123)
    
    assert target_idx is not None
    assert len(refs) >= 1
    assert target_idx not in refs, "Target image was included in reference list!"
    # Verify all belong to this identity
    all_images = set(indexer.id_to_hq[sample_ident])
    assert target_idx in all_images
    for r in refs:
        assert r in all_images


def test_mask_generator_rectangular():
    """Verifies rectangular mask generation and coverage."""
    gen = MaskGenerator(random_seed=42)
    shape = (256, 256)
    face_bbox = (50, 50, 200, 200)
    
    mask = gen.generate_rectangular_mask(shape, face_bbox=face_bbox, target_coverage=0.25)
    assert mask.shape == shape
    analysis = gen.analyze_mask(mask, face_bbox=face_bbox)
    
    assert analysis["face_coverage"] > 0.15 and analysis["face_coverage"] < 0.35
    assert analysis["num_components"] == 1


def test_mask_generator_elliptical():
    """Verifies elliptical mask generation."""
    gen = MaskGenerator(random_seed=42)
    shape = (256, 256)
    face_bbox = (40, 40, 210, 210)
    
    mask = gen.generate_elliptical_mask(shape, face_bbox=face_bbox, target_coverage=0.25)
    assert mask.shape == shape
    analysis = gen.analyze_mask(mask, face_bbox=face_bbox)
    assert analysis["num_components"] == 1
    assert analysis["face_overlap_pixels"] > 0


def test_mask_generator_irregular():
    """Verifies irregular free-form mask generation."""
    gen = MaskGenerator(random_seed=42)
    shape = (256, 256)
    
    mask = gen.generate_irregular_mask(shape, num_strokes=5)
    assert mask.shape == shape
    assert np.sum(mask) > 0


def test_mask_generator_multi_component():
    """Verifies multi-component disconnected mask generation."""
    gen = MaskGenerator(random_seed=42)
    shape = (256, 256)
    face_bbox = (30, 30, 220, 220)
    
    mask, comps = gen.generate_multi_component_mask(shape, face_bbox=face_bbox, num_components=3)
    assert mask.shape == shape
    analysis = gen.analyze_mask(mask, face_bbox=face_bbox)
    assert analysis["num_components"] >= 2


def test_mask_generator_severe_occlusion():
    """Verifies severe occlusion masks (50%, 75%, 100%)."""
    gen = MaskGenerator(random_seed=42)
    shape = (256, 256)
    face_bbox = (20, 20, 230, 230)
    
    # 75% occlusion
    mask_75 = gen.generate_severe_occlusion_mask(shape, face_bbox=face_bbox, severity_ratio=0.75)
    analysis_75 = gen.analyze_mask(mask_75, face_bbox=face_bbox)
    assert analysis_75["face_coverage"] >= 0.70
    
    # 100% full face
    mask_100 = gen.generate_severe_occlusion_mask(shape, face_bbox=face_bbox, severity_ratio=1.0)
    analysis_100 = gen.analyze_mask(mask_100, face_bbox=face_bbox)
    assert analysis_100["face_coverage"] >= 0.95


def test_mask_generator_object_removal():
    """Verifies non-face object removal mask strictly in background (face overlap == 0)."""
    gen = MaskGenerator(random_seed=42)
    shape = (256, 256)
    face_bbox = (80, 80, 180, 180) # Face centered in image
    
    mask_obj = gen.generate_object_removal_mask(shape, face_bbox=face_bbox, object_type="letter_L")
    analysis_obj = gen.analyze_mask(mask_obj, face_bbox=face_bbox)
    
    assert analysis_obj["is_pure_background"], "Object mask must not overlap face region!"
    assert analysis_obj["face_overlap_pixels"] == 0
