"""
Generate and verify canonical evaluation masks across 4 severity tiers:
- Tier 1: 25% face occlusion (canonical_eval_mask_25pct.png)
- Tier 2: 50% face occlusion (canonical_eval_mask_50pct.png)
- Tier 3: 75% face occlusion (canonical_eval_mask_75pct.png)
- Tier 4: 100% full-face occlusion (canonical_eval_mask_100pct.png)

Generated strictly using the project's MaskGenerator on the detected face bounding box of 7367.jpg.
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.masking.synthetic_masks import MaskGenerator

def generate_canonical_masks():
    masks_dir = PROJECT_ROOT / "data/masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    target_img_p = PROJECT_ROOT / "data/processed/dataset_cache/7367.jpg"
    target_img = cv2.imread(str(target_img_p))
    h, w = target_img.shape[:2]
    total_img_pixels = h * w

    # Canonical detected face bounding box on 7367.jpg: (fx1, fy1, fx2, fy2)
    face_bbox = (174, 256, 791, 944)
    fx1, fy1, fx2, fy2 = face_bbox
    face_area = (fx2 - fx1) * (fy2 - fy1)

    gen = MaskGenerator(random_seed=42)

    # 1. Existing verified 25% mask
    p25 = masks_dir / "canonical_eval_mask_25pct.png"
    assert p25.exists(), f"Missing baseline 25% mask: {p25}"
    m25 = cv2.imread(str(p25), cv2.IMREAD_GRAYSCALE)

    # 2. 50% mask
    m50 = gen.generate_elliptical_mask((h, w), face_bbox=face_bbox, target_coverage=0.50)
    m50_uint8 = (m50 * 255.0).astype(np.uint8)
    p50 = masks_dir / "canonical_eval_mask_50pct.png"
    cv2.imwrite(str(p50), m50_uint8)

    # 3. 75% mask
    m75 = gen.generate_elliptical_mask((h, w), face_bbox=face_bbox, target_coverage=0.75)
    m75_uint8 = (m75 * 255.0).astype(np.uint8)
    p75 = masks_dir / "canonical_eval_mask_75pct.png"
    cv2.imwrite(str(p75), m75_uint8)

    # 4. 100% full-face mask
    m100 = gen.generate_severe_occlusion_mask((h, w), face_bbox=face_bbox, severity_ratio=1.0)
    m100_uint8 = (m100 * 255.0).astype(np.uint8)
    p100 = masks_dir / "canonical_eval_mask_100pct.png"
    cv2.imwrite(str(p100), m100_uint8)

    tiers = {
        "25pct": {"path": p25, "mask": m25 > 127, "target_ratio": 0.25, "name": "Tier 1 (25% Occlusion)"},
        "50pct": {"path": p50, "mask": m50 > 0.5, "target_ratio": 0.50, "name": "Tier 2 (50% Occlusion)"},
        "75pct": {"path": p75, "mask": m75 > 0.5, "target_ratio": 0.75, "name": "Tier 3 (75% Occlusion)"},
        "100pct": {"path": p100, "mask": m100 > 0.5, "target_ratio": 1.00, "name": "Tier 4 (100% Full-Face)"},
    }

    manifest = {
        "target_image": "7367.jpg",
        "image_size": [h, w],
        "total_image_pixels": total_img_pixels,
        "face_bbox": list(face_bbox),
        "face_bbox_area": face_area,
        "tiers": {}
    }

    print("=" * 80)
    print("CANONICAL OCCLUSION-SEVERITY MASK MANIFEST")
    print("=" * 80)
    print(f"{'Tier':<10} | {'File Name':<30} | {'Masked Pixels':<14} | {'% Face Area':<12} | {'% Image Area':<12}")
    print("-" * 80)

    for k, v in tiers.items():
        masked_pixels = int(np.sum(v["mask"]))
        pct_face = float((masked_pixels / face_area) * 100)
        pct_img = float((masked_pixels / total_img_pixels) * 100)

        manifest["tiers"][k] = {
            "name": v["name"],
            "filename": v["path"].name,
            "target_ratio": v["target_ratio"],
            "masked_pixels": masked_pixels,
            "pct_face_area": round(pct_face, 2),
            "pct_image_area": round(pct_img, 2)
        }
        print(f"{k:<10} | {v['path'].name:<30} | {masked_pixels:<14,d} | {pct_face:<11.2f}% | {pct_img:<11.2f}%")

    print("=" * 80)
    manifest_p = masks_dir / "severity_masks_manifest.json"
    with open(manifest_p, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to: {manifest_p}")

if __name__ == "__main__":
    generate_canonical_masks()
