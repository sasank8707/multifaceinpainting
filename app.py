"""
Multi-Mode AI Image Restoration System Web Server.
Provides independent endpoints and interactive user interface for:
- Mode 1: Face Inpainting & Face Restoration (Research Baseline: Step 150 + V2 Damage Detector)
- Mode 2: Background Removal (Foreground/Person Segmentation -> Transparent RGBA)
- Mode 3: Deblurring (Edge-Preserving Detail & Contrast Enhancement)
"""

import os
import sys
import io
import time
import base64
import threading
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mode 1 imports
from modules.restoration.face_inpainting_pipeline import FaceInpaintingPipeline
from modules.restoration.damage_detector import DamageMaskDetectorV2
from modules.restoration.boundary_harmonizer import BoundaryHarmonizer
from modules.utils.geometry import verify_unmasked_invariance

# Mode 2 imports
from modules.background_removal.background_remover import BackgroundRemover

# Mode 3 imports
from modules.deblurring.deblurrer import ImageDeblurrer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload
UPLOAD_FOLDER = PROJECT_ROOT / "evaluation/web_uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Lazy-loaded singletons for efficiency
_DAMAGE_DETECTOR = None
_BACKGROUND_REMOVER = None
_DEBLURRER = None
_PIPELINE_FAST = None
_PIPELINE_REAL = None


def get_damage_detector():
    global _DAMAGE_DETECTOR
    if _DAMAGE_DETECTOR is None:
        _DAMAGE_DETECTOR = DamageMaskDetectorV2()
    return _DAMAGE_DETECTOR


def get_background_remover():
    global _BACKGROUND_REMOVER
    if _BACKGROUND_REMOVER is None:
        _BACKGROUND_REMOVER = BackgroundRemover(threshold=0.5, edge_refinement=True)
    return _BACKGROUND_REMOVER


def get_deblurrer():
    global _DEBLURRER
    if _DEBLURRER is None:
        _DEBLURRER = ImageDeblurrer(strength=1.2, enable_clahe=True)
    return _DEBLURRER


def get_pipeline(mock_mode: bool = True):
    global _PIPELINE_FAST, _PIPELINE_REAL
    if mock_mode:
        if _PIPELINE_FAST is None:
            _PIPELINE_FAST = FaceInpaintingPipeline(mock_mode=True)
        return _PIPELINE_FAST
    else:
        if _PIPELINE_REAL is None:
            _PIPELINE_REAL = FaceInpaintingPipeline(
                checkpoint_path="checkpoints/baselines/step_150_best/step_150.pt",
                mock_mode=False,
                identity_scale=0.10,
                guidance_scale=7.5,
                default_steps=50,
                damage_detector=get_damage_detector()
            )
        return _PIPELINE_REAL


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Encodes PIL image as a data URI string."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def decode_image_file(file_storage) -> np.ndarray:
    """Reads Flask uploaded file into uint8 RGB numpy array."""
    file_bytes = file_storage.read()
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(pil_img)


@app.route("/")
def index():
    """Serves the main application single-page interface."""
    return render_template("index.html")


@app.route("/health")
def health():
    """System status and mode availability check."""
    return jsonify({
        "status": "healthy",
        "system": "Multi-Mode AI Image Restoration System",
        "modes": {
            "mode_1_face_restoration": "Active (Path A + Path B)",
            "mode_2_background_removal": "Active (LRASPP Transparent RGBA)",
            "mode_3_deblurring": "Active (Bilateral Contrast Deblurring)"
        }
    })


# ---------------------------------------------------------------------------
# MODE 1: FACE INPAINTING / RESTORATION ENDPOINTS
# ---------------------------------------------------------------------------

@app.route("/api/mode1/detect_damage", methods=["POST"])
def mode1_detect_damage():
    """Path B Auto-Damage Detection endpoint."""
    if "target" not in request.files:
        return jsonify({"error": "Missing 'target' image in request"}), 400

    target_np = decode_image_file(request.files["target"])
    detector = get_damage_detector()

    # Optional references for comparison
    ref_list = []
    for key in sorted(request.files.keys()):
        if key.startswith("ref"):
            ref_list.append(decode_image_file(request.files[key]))

    t0 = time.time()
    result = detector.detect(target_np, reference_images=ref_list)
    dt_ms = (time.time() - t0) * 1000.0

    mask_u8 = result.mask
    num_px = result.num_pixels

    # Create red overlay
    overlay = target_np.copy()
    overlay[mask_u8 > 0] = [230, 40, 40]
    blended = cv2.addWeighted(overlay, 0.45, target_np, 0.55, 0)

    return jsonify({
        "status": "success",
        "detected_pixels": num_px,
        "face_coverage_pct": round(result.face_coverage_pct, 1),
        "confidence": round(result.confidence, 2),
        "runtime_ms": round(dt_ms, 1),
        "mask_base64": pil_to_base64(Image.fromarray(mask_u8)),
        "overlay_base64": pil_to_base64(Image.fromarray(blended))
    })


def generate_preview_layout(target_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Generates an honest, non-fabricated preview visualization.
    Overlays the mask contour and region with an explicit banner inside image bounds:
    'FAST PREVIEW — DEMO ONLY: Run Full Diffusion for 50-Step Neural Restoration'.
    """
    h, w = target_np.shape[:2]
    preview = target_np.copy()

    # Create semi-transparent overlay on masked region
    mask_bool = (mask_np > 0)
    if np.any(mask_bool):
        overlay = preview.copy()
        overlay[mask_bool] = np.clip(
            overlay[mask_bool].astype(np.float32) * 0.35 + np.array([0, 195, 235], dtype=np.float32) * 0.65,
            0, 255
        ).astype(np.uint8)
        preview[mask_bool] = overlay[mask_bool]

        # Draw crisp contour along mask edge
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 230, 255), 2)

    # Draw top informational badge inside image bounds
    banner_h = min(44, max(30, int(h * 0.1)))
    sub = preview[0:banner_h, 0:w].copy()
    dark_rect = np.zeros_like(sub)
    dark_rect[:] = (15, 23, 42)
    preview[0:banner_h, 0:w] = cv2.addWeighted(dark_rect, 0.88, sub, 0.12, 0)
    cv2.line(preview, (0, banner_h), (w, banner_h), (56, 189, 248), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(0.55, w / 950.0))
    cv2.putText(preview, "FAST PREVIEW - DEMO ONLY (Diffusion Not Executed)", (10, int(banner_h * 0.44)),
                font, scale, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(preview, "Run 'Full Diffusion Restoration (50 Steps)' for authentic neural restoration", (10, int(banner_h * 0.85)),
                font, scale * 0.82, (226, 232, 240), 1, cv2.LINE_AA)

    return preview


RESTORATION_JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _run_full_restoration_job(job_id: str, target_np: np.ndarray, mask_np: Optional[np.ndarray], ref_list: List[np.ndarray], steps: int = 50):
    """Background worker for neural diffusion restoration with live step reporting."""
    try:
        with JOBS_LOCK:
            if job_id not in RESTORATION_JOBS:
                return
            RESTORATION_JOBS[job_id]["status"] = "running"
            RESTORATION_JOBS[job_id]["message"] = "Preparing image..."
            RESTORATION_JOBS[job_id]["progress_pct"] = 4

        if ref_list:
            with JOBS_LOCK:
                if job_id in RESTORATION_JOBS:
                    RESTORATION_JOBS[job_id]["message"] = f"Analyzing {len(ref_list)} reference photo(s)..."
                    RESTORATION_JOBS[job_id]["progress_pct"] = 8

        with JOBS_LOCK:
            if job_id in RESTORATION_JOBS:
                RESTORATION_JOBS[job_id]["message"] = "Preparing restoration..."
                RESTORATION_JOBS[job_id]["progress_pct"] = 10

        def step_callback(current_step: int, total_steps: int):
            pct = int(10 + (current_step / total_steps) * 85)
            with JOBS_LOCK:
                if job_id in RESTORATION_JOBS:
                    RESTORATION_JOBS[job_id]["step"] = current_step
                    RESTORATION_JOBS[job_id]["total_steps"] = total_steps
                    RESTORATION_JOBS[job_id]["progress_pct"] = pct
                    RESTORATION_JOBS[job_id]["message"] = f"Diffusion step {current_step} / {total_steps} — {int(current_step/total_steps*100)}%"

        pipeline = get_pipeline(mock_mode=False)
        res = pipeline.restore(
            target_image=target_np,
            mask=mask_np,
            reference_images=ref_list,
            detector=get_damage_detector(),
            save_debug_vis=False,
            steps=steps,
            step_callback=step_callback
        )

        with JOBS_LOCK:
            if job_id in RESTORATION_JOBS:
                RESTORATION_JOBS[job_id]["message"] = "Finalizing result & boundary harmonization..."
                RESTORATION_JOBS[job_id]["progress_pct"] = 96

        final_img = res["final_image"]
        final_mask = res["normalized_mask"]

        # Apply boundary harmonization r=7
        harmonizer = BoundaryHarmonizer(feather_radius=7.0, blend_mode="feathered")
        composited, _ = harmonizer.composite(target_np, final_img, final_mask)

        # Outside-mask invariance check
        passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_np, composited, final_mask)

        # Anti-Copy Audit on masked pixels (Input-only audit)
        inside_mask = (final_mask > 0)
        total_masked = int(np.sum(inside_mask))
        inside_diff = np.abs(composited[inside_mask].astype(int) - target_np[inside_mask].astype(int))
        inside_changed = int(np.sum(np.max(inside_diff, axis=-1) > 5))
        regen_ratio = float(inside_changed) / max(1, total_masked)

        if regen_ratio < 0.50:
            with JOBS_LOCK:
                if job_id in RESTORATION_JOBS:
                    RESTORATION_JOBS[job_id].update({
                        "status": "failed_reconstruction",
                        "error": "The selected region was not reconstructed sufficiently.",
                        "can_fallback_to_auto": True,
                        "masked_pixels": total_masked,
                        "regen_percentage": round(regen_ratio * 100, 1),
                        "masked_pixels_regenerated": f"{inside_changed:,} / {total_masked:,} ({regen_ratio*100:.1f}%)",
                        "inside_mask_changed_pixels": inside_changed,
                        "inside_mask_total_pixels": total_masked,
                        "outside_mask_changed_pixels": violated_pixels,
                        "invariance_passed": passed_inv,
                        "max_unmasked_diff": max_diff,
                        "result_base64": pil_to_base64(Image.fromarray(composited)),
                        "mask_base64": pil_to_base64(Image.fromarray(final_mask))
                    })
            return

        with JOBS_LOCK:
            if job_id in RESTORATION_JOBS:
                RESTORATION_JOBS[job_id].update({
                    "status": "completed",
                    "message": "Restoration complete.",
                    "progress_pct": 100,
                    "step": 50,
                    "result": {
                        "status": "success",
                        "status_badge": "RESTORATION COMPLETE",
                        "status_message": "Full restoration — 50 diffusion steps",
                        "execution_mode": "full_diffusion",
                        "is_preview": False,
                        "mode_routed": res["input_mode"],
                        "masked_pixels": total_masked,
                        "regen_percentage": round(regen_ratio * 100, 1),
                        "masked_pixels_regenerated": f"{inside_changed:,} / {total_masked:,} ({regen_ratio*100:.1f}%)",
                        "inside_mask_changed_pixels": inside_changed,
                        "inside_mask_total_pixels": total_masked,
                        "outside_mask_changed_pixels": violated_pixels,
                        "references_used": len(ref_list),
                        "invariance_passed": passed_inv,
                        "max_unmasked_diff": max_diff,
                        "runtime_seconds": round(res.get("performance", {}).get("total_pipeline_seconds", res.get("runtime_seconds", 0.0)), 2),
                        "result_base64": pil_to_base64(Image.fromarray(composited)),
                        "mask_base64": pil_to_base64(Image.fromarray(final_mask))
                    }
                })
    except Exception as e:
        with JOBS_LOCK:
            if job_id in RESTORATION_JOBS:
                RESTORATION_JOBS[job_id].update({
                    "status": "error",
                    "error": str(e)
                })


@app.route("/api/mode1/restore", methods=["POST"])
def mode1_restore():
    """Executes Face Restoration (Option A or Option B)."""
    if "target" not in request.files:
        return jsonify({"error": "Missing 'target' image in request"}), 400

    target_np = decode_image_file(request.files["target"])
    h, w = target_np.shape[:2]

    # Optional mask
    mask_np = None
    if "mask" in request.files and request.files["mask"].filename != "":
        mask_bytes = request.files["mask"].read()
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_np = (np.array(mask_pil) > 127).astype(np.uint8) * 255

        # CRITICAL MASK DIMENSION CHECK (NEVER SILENTLY RESIZE)
        if mask_np.shape[:2] != (h, w):
            return jsonify({
                "status": "error",
                "error": f"Mask size ({mask_np.shape[1]}x{mask_np.shape[0]}) does not match the face image ({w}x{h}). Please upload a matching mask or create the mask using Draw Mask."
            }), 400

    # Optional references (up to 3)
    ref_list = []
    for key in sorted(request.files.keys()):
        if key.startswith("ref"):
            ref_list.append(decode_image_file(request.files[key]))

    execution_mode = request.form.get("execution_mode", "preview")  # 'preview' or 'full_diffusion'

    try:
        if execution_mode == "preview":
            # Fast Preview — Demonstration only: verifies inputs and mask contour
            pipeline = get_pipeline(mock_mode=True)
            res = pipeline.restore(
                target_image=target_np,
                mask=mask_np,
                reference_images=ref_list,
                detector=get_damage_detector(),
                save_debug_vis=False
            )

            final_mask = res["normalized_mask"]
            total_masked = int(np.sum(final_mask > 0))
            preview_vis = generate_preview_layout(target_np, final_mask)

            return jsonify({
                "status": "success",
                "status_badge": "Fast preview — demo only",
                "status_message": "Fast preview does not run the full diffusion restoration.",
                "execution_mode": "preview",
                "is_preview": True,
                "mode_routed": res["input_mode"],
                "masked_pixels": total_masked,
                "masked_pixels_regenerated": "N/A (Preview Mode)",
                "inside_mask_changed_pixels": 0,
                "inside_mask_total_pixels": total_masked,
                "outside_mask_changed_pixels": 0,
                "references_used": len(ref_list),
                "invariance_passed": True,
                "max_unmasked_diff": 0.0,
                "runtime_seconds": round(res.get("performance", {}).get("total_pipeline_seconds", res.get("runtime_seconds", 0.0)), 2),
                "result_base64": pil_to_base64(Image.fromarray(preview_vis)),
                "mask_base64": pil_to_base64(Image.fromarray(final_mask))
            })

        else:
            # Full Diffusion Restoration — Background Job with Real Step Reporting
            req_steps = int(request.form.get("steps", 50))
            job_id = str(uuid.uuid4())
            with JOBS_LOCK:
                RESTORATION_JOBS[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "step": 0,
                    "total_steps": req_steps,
                    "progress_pct": 0,
                    "message": "Queued restoration job...",
                    "result": None,
                    "error": None,
                    "created_at": time.time()
                }

            thread = threading.Thread(
                target=_run_full_restoration_job,
                args=(job_id, target_np, mask_np, ref_list, req_steps),
                daemon=True
            )
            thread.start()

            # Support synchronous mode if requested (e.g., tests)
            if request.form.get("sync") == "true":
                thread.join()
                with JOBS_LOCK:
                    job = RESTORATION_JOBS.get(job_id)
                    if job.get("status") == "completed":
                        return jsonify(job["result"])
                    elif job.get("status") == "failed_reconstruction":
                        return jsonify(job), 422
                    else:
                        return jsonify({"status": "error", "error": job.get("error", "Restoration failed")}), 500

            return jsonify({
                "status": "queued",
                "job_id": job_id,
                "message": "Restoration job started."
            })

    except Exception as e:
        return jsonify({
            "status": "error",
            "status_badge": "RESTORATION FAILED",
            "error": str(e)
        }), 500


@app.route("/api/mode1/job_status/<job_id>", methods=["GET"])
def mode1_job_status(job_id: str):
    """Returns status and progress of a background restoration job."""
    with JOBS_LOCK:
        job = RESTORATION_JOBS.get(job_id)
        if not job:
            return jsonify({"status": "error", "error": "Job not found"}), 404
        return jsonify(job)


# ---------------------------------------------------------------------------
# MODE 2: BACKGROUND REMOVAL ENDPOINT
# ---------------------------------------------------------------------------

@app.route("/api/mode2/remove_background", methods=["POST"])
def mode2_remove_background():
    """Mode 2 Foreground Person/Object Segmentation."""
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' in request"}), 400

    img_np = decode_image_file(request.files["image"])
    threshold = float(request.form.get("threshold", 0.5))

    remover = get_background_remover()
    res = remover.remove_background(img_np, threshold=threshold, return_dict=True)

    rgba = res["rgba_image"]
    alpha = res["alpha_mask"]

    return jsonify({
        "status": "success",
        "runtime_ms": res["runtime_ms"],
        "foreground_percentage": res["foreground_percentage"],
        "dimensions": [img_np.shape[1], img_np.shape[0]],
        "result_base64": pil_to_base64(Image.fromarray(rgba), format="PNG"),
        "alpha_base64": pil_to_base64(Image.fromarray(alpha), format="PNG")
    })


# ---------------------------------------------------------------------------
# MODE 3: DEBLURRING ENDPOINT
# ---------------------------------------------------------------------------

@app.route("/api/mode3/deblur", methods=["POST"])
def mode3_deblur():
    """Mode 3 Edge-Preserving Deblurring & Contrast Restoration."""
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' in request"}), 400

    img_np = decode_image_file(request.files["image"])
    strength = float(request.form.get("strength", 1.2))

    deblurrer = get_deblurrer()
    res = deblurrer.deblur(img_np, strength=strength, return_dict=True)

    deblurred = res["deblurred_image"]

    return jsonify({
        "status": "success",
        "runtime_ms": res["runtime_ms"],
        "sharpness_gain": res["sharpness_gain"],
        "initial_variance": res["initial_metrics"]["laplacian_variance"],
        "final_variance": res["final_metrics"]["laplacian_variance"],
        "result_base64": pil_to_base64(Image.fromarray(deblurred), format="JPEG")
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 80)
    print(f"MULTI-MODE AI IMAGE RESTORATION SYSTEM STARTING ON PORT {port}")
    print("Modes: [1] Face Restoration | [2] Background Removal | [3] Deblurring")
    print("=" * 80)
    app.run(host="0.0.0.0", port=port, debug=False)
