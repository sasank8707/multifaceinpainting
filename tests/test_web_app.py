"""
Unit and Integration Tests for Flask Web Application (app.py).
Verifies route rendering, health checks, and API endpoints for all three modes.
"""

import io
import sys
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create_test_image_bytes(width=100, height=100, color=(200, 100, 100)):
    """Helper to generate in-memory PNG bytes for testing file uploads."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestWebApp:
    """Test suite for web server endpoints."""

    def test_index_page(self, client):
        """Verifies root route returns 200 and renders HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Multi-Mode AI Image Restoration" in response.data
        assert b"Mode 1: Face Inpainting &amp; Face Restoration" in response.data or b"Mode 1: Face Inpainting & Face Restoration" in response.data
        assert b"Mode 2: Background Removal" in response.data
        assert b"Mode 3: Edge-Preserving Deblurring" in response.data

    def test_health_endpoint(self, client):
        """Verifies health check returns healthy status and lists all 3 modes."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "mode_1_face_restoration" in data["modes"]
        assert "mode_2_background_removal" in data["modes"]
        assert "mode_3_deblurring" in data["modes"]

    def test_mode2_background_removal_api(self, client):
        """Verifies Mode 2 background removal API returns RGBA output."""
        img_bytes = create_test_image_bytes(120, 120)
        data = {
            "image": (img_bytes, "test.png"),
            "threshold": 0.5
        }
        response = client.post("/api/mode2/remove_background", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        res = response.get_json()
        assert res["status"] == "success"
        assert "result_base64" in res
        assert "alpha_base64" in res
        assert res["result_base64"].startswith("data:image/png;base64,")

    def test_mode3_deblur_api(self, client):
        """Verifies Mode 3 deblurring API returns enhanced image."""
        img_bytes = create_test_image_bytes(100, 100)
        data = {
            "image": (img_bytes, "blurry.png"),
            "strength": 1.2
        }
        response = client.post("/api/mode3/deblur", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        res = response.get_json()
        assert res["status"] == "success"
        assert "result_base64" in res
        assert "sharpness_gain" in res
        assert res["result_base64"].startswith("data:image/jpeg;base64,")

    def test_mode1_restore_preview_api(self, client):
        """Verifies Mode 1 fast preview restoration endpoint."""
        target_bytes = create_test_image_bytes(128, 128)
        # Create a valid non-empty mask covering < 98% of canvas (e.g., 30x30 central box)
        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[40:70, 40:70] = 255
        mask_pil = Image.fromarray(mask_arr)
        mask_buf = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")
        mask_buf.seek(0)

        data = {
            "target": (target_bytes, "target.png"),
            "mask": (mask_buf, "mask.png"),
            "execution_mode": "preview"
        }
        response = client.post("/api/mode1/restore", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        res = response.get_json()
        assert res["status"] == "success"
        assert res["mode_routed"] == "MASK_PROVIDED"
        assert res["invariance_passed"] is True
        assert "result_base64" in res

    def test_mode1_mask_dimension_mismatch_rejected(self, client):
        """Verifies strict mask dimension check: rejects mismatch with HTTP 400 and clear error."""
        target_bytes = create_test_image_bytes(128, 128)
        # Create mismatched 100x100 mask
        mismatch_mask_bytes = create_test_image_bytes(100, 100)

        data = {
            "target": (target_bytes, "target.png"),
            "mask": (mismatch_mask_bytes, "mask_mismatched.png"),
            "execution_mode": "preview"
        }
        response = client.post("/api/mode1/restore", data=data, content_type="multipart/form-data")
        assert response.status_code == 400
        res = response.get_json()
        assert res["status"] == "error"
        assert "Mask size (100x100) does not match the face image (128x128)" in res["error"]
        assert "Please upload a matching mask or create the mask using Draw Mask." in res["error"]

    def test_mode1_detect_damage_api(self, client):
        """Verifies Option B automatic damage detection endpoint."""
        target_bytes = create_test_image_bytes(128, 128)
        data = {
            "target": (target_bytes, "target.png")
        }
        response = client.post("/api/mode1/detect_damage", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        res = response.get_json()
        assert res["status"] == "success"
        assert "detected_pixels" in res
        assert "face_coverage_pct" in res
        assert "confidence" in res
        assert "overlay_base64" in res
        assert "mask_base64" in res

    def test_mode1_full_diffusion_queued_job_and_poll(self, client):
        """Verifies background job submission and status polling."""
        target_bytes = create_test_image_bytes(128, 128)
        mask_arr = np.zeros((128, 128), dtype=np.uint8)
        mask_arr[40:70, 40:70] = 255
        mask_pil = Image.fromarray(mask_arr)
        mask_buf = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")
        mask_buf.seek(0)

        data = {
            "target": (target_bytes, "target.png"),
            "mask": (mask_buf, "mask.png"),
            "execution_mode": "full_diffusion"
        }
        response = client.post("/api/mode1/restore", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        res = response.get_json()
        assert res["status"] == "queued"
        assert "job_id" in res
        job_id = res["job_id"]

        # Poll job status
        poll_resp = client.get(f"/api/mode1/job_status/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.get_json()
        assert poll_data["job_id"] == job_id
        assert poll_data["status"] in ["queued", "running", "completed", "failed_reconstruction", "error"]
        assert "progress_pct" in poll_data

    def test_mode1_job_status_not_found(self, client):
        """Verifies non-existent job ID returns HTTP 404."""
        response = client.get("/api/mode1/job_status/non-existent-uuid-12345")
        assert response.status_code == 404
        data = response.get_json()
        assert data["status"] == "error"

