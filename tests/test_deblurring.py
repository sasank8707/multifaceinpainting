"""
Unit and Regression Tests for Mode 3: Deblurring.
Verifies independent execution, dimension preservation, sharpness enhancement,
and arbitrary dimension handling.
"""

import sys
import pytest
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.deblurring.deblurrer import ImageDeblurrer


@pytest.fixture(scope="module")
def deblurrer():
    """Shared ImageDeblurrer instance."""
    return ImageDeblurrer(strength=1.2, enable_clahe=True)


class TestImageDeblurring:
    """Test suite for ImageDeblurrer."""

    def test_deblurrer_initialization(self, deblurrer):
        """Verifies clean initialization and parameter bounds."""
        assert deblurrer is not None
        assert 0.1 <= deblurrer.strength <= 3.0

    def test_deblur_preserves_dimensions(self, deblurrer):
        """Strict requirement: image dimensions must remain identical."""
        h, w = 180, 240
        test_img = np.ones((h, w, 3), dtype=np.uint8) * 128
        out = deblurrer.deblur(test_img)
        assert out.shape == (h, w, 3)
        assert out.dtype == np.uint8

    def test_sharpness_increases_on_blurred_input(self, deblurrer):
        """Verifies that Laplacian variance increases after deblurring a blurry image."""
        # Create a synthetic image with geometric patterns
        base = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(base, (30, 30), (170, 170), (255, 255, 255), -1)
        cv2.circle(base, (100, 100), 40, (0, 0, 255), -1)

        # Apply synthetic Gaussian blur
        blurry = cv2.GaussianBlur(base, (11, 11), 3.0)
        res = deblurrer.deblur(blurry, return_dict=True)

        init_var = res["initial_metrics"]["laplacian_variance"]
        final_var = res["final_metrics"]["laplacian_variance"]

        assert final_var > init_var, f"Sharpness did not increase: init={init_var}, final={final_var}"
        assert res["sharpness_gain"] > 1.0

    def test_supports_arbitrary_dimensions(self, deblurrer):
        """Verifies handling of non-square and irregular dimensions."""
        dimensions = [(133, 217), (256, 128), (99, 99)]
        for h, w in dimensions:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            out = deblurrer.deblur(img)
            assert out.shape == (h, w, 3)

    def test_return_dict_metadata(self, deblurrer):
        """Verifies rich metadata dictionary return."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = deblurrer.deblur(img, return_dict=True)
        assert isinstance(res, dict)
        assert "deblurred_image" in res
        assert "runtime_ms" in res
        assert "sharpness_gain" in res
        assert res["runtime_ms"] > 0

    def test_real_image_deblurring(self, deblurrer):
        """Tests deblurring execution on actual Case 01 portrait image."""
        real_img_path = PROJECT_ROOT / "user_tests/natural_damage/case01/target.jpg"
        if real_img_path.exists():
            res = deblurrer.deblur(real_img_path, return_dict=True)
            assert res["deblurred_image"].shape[:2] == (285, 450)
            assert res["runtime_ms"] < 2000.0  # Sub-2-second CPU requirement
