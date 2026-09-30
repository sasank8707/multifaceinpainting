"""
Unit and Regression Tests for Mode 2: Background Removal.
Verifies independent execution, bit-exact RGB preservation, RGBA alpha generation,
and arbitrary dimension handling.
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.background_removal.background_remover import BackgroundRemover


@pytest.fixture(scope="module")
def remover():
    """Shared BackgroundRemover instance."""
    return BackgroundRemover(threshold=0.5, edge_refinement=True)


class TestBackgroundRemoval:
    """Test suite for BackgroundRemover."""

    def test_remover_initialization(self, remover):
        """Verifies clean initialization and model loading."""
        assert remover is not None
        assert remover.model is not None
        assert remover.threshold == 0.5

    def test_removes_background_returns_rgba(self, remover):
        """Verifies output format is [H, W, 4] RGBA uint8."""
        # Synthetic test portrait
        h, w = 240, 320
        test_img = np.ones((h, w, 3), dtype=np.uint8) * 100
        # Draw a synthetic bright central ellipse simulating a person
        import cv2
        cv2.ellipse(test_img, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, (220, 160, 130), -1)

        rgba, alpha = remover.remove_background(test_img)
        assert rgba.shape == (h, w, 4), f"Expected shape {(h, w, 4)}, got {rgba.shape}"
        assert rgba.dtype == np.uint8
        assert alpha.shape == (h, w)
        assert alpha.dtype == np.uint8

    def test_preserves_original_rgb_values(self, remover):
        """Strict requirement: original RGB pixels must be 100% bit-exact preserved."""
        rng = np.random.RandomState(42)
        test_rgb = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        rgba, _ = remover.remove_background(test_rgb)
        assert np.array_equal(rgba[:, :, :3], test_rgb), "RGB pixels were modified during background removal!"

    def test_supports_arbitrary_dimensions(self, remover):
        """Verifies handling of non-square and arbitrary dimensions."""
        dimensions = [(150, 200), (312, 175), (64, 64)]
        for h, w in dimensions:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            rgba, alpha = remover.remove_background(img)
            assert rgba.shape == (h, w, 4)
            assert alpha.shape == (h, w)

    def test_return_dict_metadata(self, remover):
        """Verifies rich dictionary return mode."""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = remover.remove_background(test_img, return_dict=True)
        assert isinstance(res, dict)
        assert "rgba_image" in res
        assert "alpha_mask" in res
        assert "runtime_ms" in res
        assert "foreground_pixels" in res
        assert res["runtime_ms"] > 0

    def test_real_image_execution(self, remover):
        """Tests background removal on actual Case 01 portrait image."""
        real_img_path = PROJECT_ROOT / "user_tests/natural_damage/case01/target.jpg"
        if real_img_path.exists():
            res = remover.remove_background(real_img_path, return_dict=True)
            assert res["foreground_pixels"] > 0
            assert res["foreground_percentage"] > 20.0  # Portrait fills majority of frame
            assert res["rgba_image"].shape[2] == 4
