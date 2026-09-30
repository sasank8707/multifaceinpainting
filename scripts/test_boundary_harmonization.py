"""
Comprehensive Boundary Harmonization Test Script for Case 01 V2 Restoration.
Evaluates inward feathered blending, Poisson compositing, and color harmonization
against the raw hard-mask baseline.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.restoration.boundary_harmonizer import BoundaryHarmonizer
from modules.utils.geometry import verify_unmasked_invariance


def main():
    base_dir = PROJECT_ROOT / "evaluation/visualizations/auto_damage_validation/case01_v2_restoration"
    assert base_dir.exists(), f"Directory not found: {base_dir}"

    target_path = base_dir / "target.jpg"
    recon_path = base_dir / "reconstruction.jpg"
    mask_path = base_dir / "mask.png"

    target_bgr = cv2.imread(str(target_path))
    recon_bgr = cv2.imread(str(recon_path))
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)

    h, w = target_rgb.shape[:2]
    print(f"Loaded inputs: target {target_rgb.shape}, mask {mask_gray.shape} ({np.sum(mask_gray > 0)} px)")

    # Test Configurations
    configs = [
        {"name": "baseline_hard", "mode": "hard", "radius": 0.0, "color": False, "title": "Hard Cut (Baseline)"},
        {"name": "feathered_r3", "mode": "feathered", "radius": 3.0, "color": False, "title": "Feathered (r=3px)"},
        {"name": "feathered_r5", "mode": "feathered", "radius": 5.0, "color": False, "title": "Feathered (r=5px)"},
        {"name": "feathered_r7", "mode": "feathered", "radius": 7.0, "color": False, "title": "Feathered (r=7px)"},
        {"name": "color_feathered_r5", "mode": "color_feathered", "radius": 5.0, "color": True, "title": "Color Harmonized + Feathered (r=5px)"},
        {"name": "poisson", "mode": "poisson", "radius": 5.0, "color": False, "title": "Poisson Seamless Clone"}
    ]

    out_dir = PROJECT_ROOT / "evaluation/visualizations/auto_damage_validation/case01_boundary_harmonization"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Define inner boundary for halo quantification
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inner_boundary = (mask_gray > 0) & (cv2.erode(mask_gray, kernel) == 0)
    outer_boundary = (mask_gray == 0) & (cv2.dilate(mask_gray, kernel) > 0)

    target_inner_mean = np.mean(target_rgb[inner_boundary], axis=0)
    target_outer_mean = np.mean(target_rgb[outer_boundary], axis=0)
    print(f"\nTarget Boundary Baseline: Outer RGB={target_outer_mean.round(1)}, Inner RGB={target_inner_mean.round(1)}")

    # Wound zoom crop coordinates with padding
    pad = 15
    bx1, by1, bx2, by2 = 255, 41, 329, 205
    cy1, cy2 = max(0, by1 - pad), min(h, by2 + pad)
    cx1, cx2 = max(0, bx1 - pad), min(w, bx2 + pad)

    crops = {"Target (Before)": target_rgb[cy1:cy2, cx1:cx2]}

    for cfg in configs:
        harmonizer = BoundaryHarmonizer(
            feather_radius=cfg["radius"],
            enable_color_transfer=cfg["color"],
            blend_mode=cfg["mode"]
        )
        comp_img, metrics = harmonizer.composite(
            target_image=target_rgb,
            restored_image=recon_rgb,
            mask=mask_gray
        )

        # Invariance check
        passed, max_diff, violated = verify_unmasked_invariance(target_rgb, comp_img, mask_gray)
        assert passed and max_diff == 0.0 and violated == 0, f"Invariance failed for {cfg['name']}"

        # Boundary jump quantification: |inner_RGB - outer_RGB|
        comp_inner_mean = np.mean(comp_img[inner_boundary], axis=0)
        boundary_jump = np.linalg.norm(comp_inner_mean - target_outer_mean)

        results[cfg["name"]] = {
            "image": comp_img,
            "passed_invariance": passed,
            "max_unmasked_diff": max_diff,
            "inner_rgb": comp_inner_mean.round(1).tolist(),
            "boundary_jump": round(float(boundary_jump), 2),
            "title": cfg["title"]
        }
        crops[cfg["title"]] = comp_img[cy1:cy2, cx1:cx2]

        print(f"[{cfg['name']}] Invariance: passed={passed}, Inner RGB={comp_inner_mean.round(1)}, Boundary Jump={boundary_jump:.2f}")

        # Save individual RGB
        Image.fromarray(comp_img).save(out_dir / f"{cfg['name']}.png")

    # Render Comparative Multi-Panel Figure: Full Faces
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panel_order = ["baseline_hard", "feathered_r3", "feathered_r5", "feathered_r7", "color_feathered_r5", "poisson"]
    
    for idx, key in enumerate(panel_order):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(results[key]["image"])
        jump = results[key]["boundary_jump"]
        ax.set_title(f"{results[key]['title']}\nBoundary Edge Jump: {jump:.1f} (Lower = Smoother)", fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "full_face_comparisons.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'full_face_comparisons.png'}")

    # Render Zoomed Damaged Region Comparison
    fig_zoom, axes_z = plt.subplots(1, 4, figsize=(20, 6))
    zoom_keys = ["Target (Before)", "Hard Cut (Baseline)", "Feathered (r=5px)", "Color Harmonized + Feathered (r=5px)"]
    for i, zkey in enumerate(zoom_keys):
        axes_z[i].imshow(crops[zkey])
        axes_z[i].set_title(zkey, fontsize=12, fontweight="bold")
        axes_z[i].axis("off")

    fig_zoom.tight_layout()
    fig_zoom.savefig(out_dir / "zoomed_lesion_comparison.png", dpi=150)
    plt.close(fig_zoom)
    print(f"Saved: {out_dir / 'zoomed_lesion_comparison.png'}")

    print("\nAll boundary harmonization tests completed successfully.")


if __name__ == "__main__":
    main()
