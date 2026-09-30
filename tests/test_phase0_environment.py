"""
Phase 0 Unit Tests: Environment, Hardware Audit, Device Dispatch, and Geometric Utilities.
"""

import os
import sys
import yaml
import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.utils.device import get_hardware_profile, get_optimal_device
from modules.utils.geometry import (
    resize_and_pad,
    unpad_and_restore,
    extract_roi,
    paste_roi,
    verify_unmasked_invariance
)


def test_python_and_packages():
    """Verifies that core deep learning and computer vision libraries are installed and operational."""
    import torch
    import torchvision
    import cv2
    import PIL
    import scipy
    
    # Verify PyTorch tensor creation
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert (t * 2).tolist() == [2.0, 4.0, 6.0]


def test_hardware_profile():
    """Verifies that hardware profile reports CPU, RAM, and GPU status cleanly."""
    profile = get_hardware_profile()
    assert "platform" in profile
    assert "cpu" in profile
    assert profile["cpu"]["physical_cores"] is not None and profile["cpu"]["physical_cores"] > 0
    assert profile["ram"]["total_gb"] > 0
    assert "cuda_available" in profile["gpu"]


def test_optimal_device_dispatch():
    """Verifies that optimal device selection works with fallbacks."""
    device_str, dtype = get_optimal_device(preferred="auto")
    assert device_str in ["cpu", "cuda:0", "cuda"]
    
    # Test explicit CPU request
    cpu_dev, cpu_dtype = get_optimal_device(preferred="cpu")
    assert cpu_dev == "cpu"


def test_config_files_validity():
    """Verifies that all YAML configuration files load cleanly."""
    config_paths = [
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/evaluation.yaml",
    ]
    for p in config_paths:
        assert os.path.exists(p), f"Missing config file: {p}"
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
            assert isinstance(cfg, dict) and len(cfg) > 0


def test_aspect_ratio_resize_and_unpad():
    """Verifies that resize_and_pad followed by unpad_and_restore reproduces original dimensions bit-exact."""
    # Test with non-square image (640 x 480)
    orig = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    target_size = (256, 256)
    
    padded, pad_info = resize_and_pad(orig, target_size=target_size)
    assert padded.shape == (256, 256, 3)
    
    restored = unpad_and_restore(padded, pad_info)
    assert restored.shape == (480, 640, 3), f"Shape mismatch: {restored.shape} vs (480, 640, 3)"


def test_roi_extraction_and_pasting():
    """Verifies ROI extraction around a mask and pasting back into base image."""
    h, w = 512, 512
    image = np.ones((h, w, 3), dtype=np.uint8) * 100
    mask = np.zeros((h, w), dtype=np.float32)
    # Mask a 50x50 region in center
    mask[200:250, 200:250] = 1.0
    
    roi_img, roi_mask, bbox = extract_roi(image, mask, margin_ratio=0.3)
    x1, y1, x2, y2 = bbox
    assert x1 <= 200 and x2 >= 250
    assert y1 <= 200 and y2 >= 250
    
    # Fill ROI with simulated reconstruction
    recon_roi = roi_img.copy()
    recon_roi[roi_mask > 0.5] = 255 # Reconstructed region becomes white
    
    pasted = paste_roi(image, recon_roi, bbox, mask_roi=roi_mask)
    assert pasted.shape == (h, w, 3)
    
    # Check that unmasked pixels are 100% untouched
    passed, max_diff, violated = verify_unmasked_invariance(image, pasted, mask)
    assert passed, f"Unmasked pixel invariance violated: max_diff={max_diff}, violated={violated}"
    assert max_diff == 0.0
