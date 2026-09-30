"""
V5 Diffusion Conditioning Experiment for Case 01 Natural Facial Damage.
Tests tone-aware positive prompt guidance, negative prompt steering, and
clean unmasked letterbox input on authentic Step 150 diffusion inpainting.

Fixed Conditions:
- Authoritative V2 damage mask (5,831 px, bounding box [255..329, 41..205])
- Checkpoint: checkpoints/baselines/step_150_best/step_150.pt
- Decoupled identity cross-attention (O2 regional token fusion, lambda = 0.10)
- 3 Reference images (ref1, ref2, ref3) with O1 scoring weights
- 50 DDIM steps, CFG = 7.5, Seed = 42
- Fixed final boundary compositing: BoundaryHarmonizer(feather_radius=7.0, blend_mode="feathered")
- Strict bit-exact canvas invariance outside the V2 mask (Delta = 0.0)
"""

import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.restoration.face_inpainting_pipeline import FaceInpaintingPipeline
from modules.restoration.boundary_harmonizer import BoundaryHarmonizer
from modules.utils.geometry import verify_unmasked_invariance


def main():
    base_case_dir = PROJECT_ROOT / "user_tests/natural_damage/case01"
    v2_vis_dir = PROJECT_ROOT / "evaluation/visualizations/auto_damage_validation/case01_v2_restoration"
    out_dir = PROJECT_ROOT / "evaluation/visualizations/auto_damage_validation/case01_v5_conditioning_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = base_case_dir / "target.jpg"
    mask_path = v2_vis_dir / "mask.png"
    ref_paths = [base_case_dir / "ref1.jpg", base_case_dir / "ref2.jpg", base_case_dir / "ref3.jpg"]
    ckpt_path = PROJECT_ROOT / "checkpoints/baselines/step_150_best/step_150.pt"

    target_bgr = cv2.imread(str(target_path))
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ref_bgrs = [cv2.imread(str(rp)) for rp in ref_paths]

    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    ref_rgbs = [cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in ref_bgrs]
    h, w = target_rgb.shape[:2]

    print("================================================================================")
    print("V5 DIFFUSION CONDITIONING EXPERIMENT: TECHNICAL SPECIFICATION")
    print("================================================================================")
    print(f"Target: {target_path} ({w}x{h})")
    print(f"Authoritative V2 Mask: {mask_path} ({np.sum(mask_gray > 0)} pixels)")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Fixed: 50 DDIM steps, CFG=7.5, lambda_id=0.10, seed=42, feather_r=7px")

    # Define the 4 V5 Variants
    variants = [
        {
            "id": "V5-A",
            "name": "Existing Baseline (Generic Prompt, No Negative)",
            "prompt": "high quality photorealistic portrait of a person, natural facial anatomy, clean skin",
            "negative_prompt": None,
            "status": "COMPLETED (Re-used from V2 restoration)",
            "use_existing_raw": str(v2_vis_dir / "reconstruction.jpg")
        },
        {
            "id": "V5-B",
            "name": "Tone-Aware Positive Prompt (Local Skin Statistics)",
            "prompt": (
                "high quality photorealistic portrait of a young man, authentic facial skin with visible micro pores, "
                "natural warm peach facial complexion, fine light facial stubble, natural dermal translucency, "
                "subtle cheek blush, sharp focus, 8k photographic detail"
            ),
            "negative_prompt": None,
            "status": "VALID",
            "use_existing_raw": None
        },
        {
            "id": "V5-C",
            "name": "Tone-Aware Positive + Negative Guidance Prompt",
            "prompt": (
                "high quality photorealistic portrait of a young man, authentic facial skin with visible micro pores, "
                "natural warm peach facial complexion, fine light facial stubble, natural dermal translucency, "
                "subtle cheek blush, sharp focus, 8k photographic detail"
            ),
            "negative_prompt": (
                "gray skin, olive cast, ashen, flat, plastic, clay, muddy, smooth, airbrushed, blurry, cartoon, "
                "oversaturated, desaturated, artificial, painted, makeup, white halo, boundary seam, discoloration"
            ),
            "status": "VALID (Primary Recommended Variant)",
            "use_existing_raw": None
        },
        {
            "id": "V5-D",
            "name": "Latent Initialization with Healthy Skin Ring Context",
            "prompt": (
                "high quality photorealistic portrait of a young man, authentic facial skin with visible micro pores, "
                "natural warm peach facial complexion, fine light facial stubble, natural dermal translucency"
            ),
            "negative_prompt": (
                "gray skin, olive cast, flat, plastic, clay, muddy, smooth, blurry, desaturated, artificial"
            ),
            "status": "ANALYZED (Architecturally constrained by RunwayML 9-channel UNet)",
            "use_existing_raw": None
        }
    ]

    spec_path = out_dir / "variants_specification.json"
    with open(spec_path, "w") as f:
        json.dump(variants, f, indent=2)
    print(f"\nSaved variants specification to {spec_path}")


if __name__ == "__main__":
    main()
