"""
Tests for Verified Pretrained ArcFace Backbone (Part A Verification).
Ensures:
1. Real pretrained ArcFace weights are loaded.
2. Embeddings are 512-dimensional and L2 normalized.
3. Same-identity pairs achieve significantly higher cosine similarity than different identities.
4. Randomly initialized / corrupted models explicitly fail.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.identity.arcface_backbone import RegionalIdentityEncoder, build_frozen_identity_encoder
from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset


@pytest.fixture
def arcface_encoder():
    return RegionalIdentityEncoder(token_dim=512)


def test_arcface_weights_loaded(arcface_encoder):
    """Verifies that the encoder loads real weights and flags is_pretrained as True."""
    assert arcface_encoder.is_pretrained is True
    assert os.path.isfile(arcface_encoder.model_path)
    assert os.path.getsize(arcface_encoder.model_path) > 100 * 1024 * 1024  # > 100MB


def test_arcface_embedding_properties(arcface_encoder):
    """Verifies embedding dimension and strict L2-normalization."""
    dummy_input = torch.randn(2, 3, 256, 256)
    out = arcface_encoder(dummy_input)

    # Global embedding checks
    global_emb = out["global_embedding"]
    assert global_emb.shape == (2, 512)
    norms = torch.norm(global_emb, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # Spatial token checks
    spatial_tokens = out["spatial_tokens"]
    assert spatial_tokens.shape == (2, 256, 512)


def test_arcface_same_vs_diff_identity_discrimination(arcface_encoder):
    """
    Verifies that real ArcFace achieves strong separation between
    verified same-identity vs. different-identity CelebA-HQ subjects.
    """
    ds = CelebAHQIdentityDataset(split="test", smoke_subset_size=1)

    # Subject ID 3111 images
    t_7367 = ds._load_image_tensor(7367).unsqueeze(0)
    t_13401 = ds._load_image_tensor(13401).unsqueeze(0)
    t_23058 = ds._load_image_tensor(23058).unsqueeze(0)

    # Subject ID 3330 image (from validation split)
    p_3616 = ds._get_image_path(3616)
    import cv2
    img_3616 = cv2.imread(str(p_3616))
    rgb_3616 = cv2.cvtColor(img_3616, cv2.COLOR_BGR2RGB)
    t_3616 = torch.from_numpy(cv2.resize(rgb_3616, (256, 256))).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

    # Extract embeddings
    e_7367 = arcface_encoder(t_7367)["global_embedding"]
    e_13401 = arcface_encoder(t_13401)["global_embedding"]
    e_23058 = arcface_encoder(t_23058)["global_embedding"]
    e_3616 = arcface_encoder(t_3616)["global_embedding"]

    # Same-identity cosine similarities
    sim_same_1 = F.cosine_similarity(e_7367, e_13401).item()
    sim_same_2 = F.cosine_similarity(e_7367, e_23058).item()
    sim_same_3 = F.cosine_similarity(e_13401, e_23058).item()

    # Different-identity cosine similarities
    sim_diff_1 = F.cosine_similarity(e_7367, e_3616).item()
    sim_diff_2 = F.cosine_similarity(e_13401, e_3616).item()

    print(f"\nPretrained ArcFace Verification:")
    print(f"Same-ID (7367 vs 13401): {sim_same_1:.4f}")
    print(f"Same-ID (7367 vs 23058): {sim_same_2:.4f}")
    print(f"Same-ID (13401 vs 23058): {sim_same_3:.4f}")
    print(f"Diff-ID (7367 vs 3616): {sim_diff_1:.4f}")
    print(f"Diff-ID (13401 vs 3616): {sim_diff_2:.4f}")

    # Assert strong discrimination
    assert sim_same_1 > 0.50, f"Expected same-ID sim > 0.50, got {sim_same_1:.4f}"
    assert sim_same_2 > 0.50, f"Expected same-ID sim > 0.50, got {sim_same_2:.4f}"
    assert sim_diff_1 < 0.15, f"Expected diff-ID sim < 0.15, got {sim_diff_1:.4f}"
    assert sim_diff_2 < 0.15, f"Expected diff-ID sim < 0.15, got {sim_diff_2:.4f}"
    assert (sim_same_1 - sim_diff_1) > 0.40, "Discrimination margin must exceed 0.40"


def test_random_weights_explicit_failure():
    """
    Verifies that a mock or corrupted encoder fails the pretrained weights verification check.
    """
    # Create encoder instance
    enc = RegionalIdentityEncoder(token_dim=512)

    # Corrupt verification signature to simulate uninitialized/random model
    from models.identity import arcface_backbone
    orig_sig = arcface_backbone.EXPECTED_ZERO_SIGNATURE_PREFIX
    try:
        arcface_backbone.EXPECTED_ZERO_SIGNATURE_PREFIX = np.ones(5, dtype=np.float32)
        with pytest.raises(RuntimeError, match="ArcFace backbone weights verification failed"):
            enc.verify_pretrained_weights()
    finally:
        arcface_backbone.EXPECTED_ZERO_SIGNATURE_PREFIX = orig_sig
