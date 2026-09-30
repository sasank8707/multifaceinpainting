"""
Execution Script: Case 01 V5-C Diffusion Conditioning Restoration.
Executes authentic 50-step DDIM diffusion on CPU with tone-aware positive prompt
and anti-gray/anti-flat negative guidance prompt, using Step 150 adapter weights.

Preserves:
- Exact V2 authoritative mask (5,831 px)
- Step 150 checkpoint (checkpoints/baselines/step_150_best/step_150.pt)
- O1 mask-aware reference scoring
- O2 regional identity token fusion
- Decoupled identity cross-attention (scale = 0.10)
- CFG = 7.5, seed = 42, 50 DDIM steps on CPU
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
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance


def main():
    base_case_dir = PROJECT_ROOT / "user_tests/natural_damage/case01"
    v2_dir = PROJECT_ROOT / "evaluation/visualizations/auto_damage_validation/case01_v2_restoration"
    v4_dir = PROJECT_ROOT / "evaluation/visualizations/auto_damage_validation/case01_v4_texture_experiment"
    out_dir = PROJECT_ROOT / "evaluation/visualizations/auto_damage_validation/case01_v5c_restoration"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = base_case_dir / "target.jpg"
    mask_path = v2_dir / "mask.png"
    ref_paths = [base_case_dir / "ref1.jpg", base_case_dir / "ref2.jpg", base_case_dir / "ref3.jpg"]
    ckpt_path = PROJECT_ROOT / "checkpoints/baselines/step_150_best/step_150.pt"

    assert target_path.exists(), f"Target not found: {target_path}"
    assert mask_path.exists(), f"Mask not found: {mask_path}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    target_bgr = cv2.imread(str(target_path))
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ref_bgrs = [cv2.imread(str(p)) for p in ref_paths]

    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    ref_rgbs = [cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in ref_bgrs]
    h, w = target_rgb.shape[:2]

    # Verify authoritative mask
    num_masked_pixels = int(np.sum(mask_gray > 0))
    print("================================================================================")
    print("STARTING AUTHENTIC V5-C DIFFUSION RESTORATION (50 DDIM STEPS ON CPU)")
    print("================================================================================")
    print(f"Target Image: {w}x{h} ({target_rgb.size} values)")
    print(f"Authoritative V2 Mask: {num_masked_pixels:,} pixels ({num_masked_pixels/(h*w)*100:.2f}% of canvas)")
    print(f"References: {len(ref_rgbs)} images")
    print(f"Checkpoint: {ckpt_path.name}")
    print("Parameters: 50 DDIM steps, CFG=7.5, identity_scale=0.10, seed=42")

    # Positive & Negative Prompts for V5-C
    pos_prompt = (
        "high quality photorealistic portrait of a young man, authentic facial skin with visible micro pores, "
        "natural warm peach facial complexion, fine light facial stubble, natural dermal translucency, "
        "subtle cheek blush, sharp focus, 8k photographic detail"
    )
    neg_prompt = (
        "gray skin, olive cast, ashen, flat, plastic, clay, muddy, smooth, airbrushed, blurry, cartoon, "
        "oversaturated, desaturated, artificial, painted, makeup, white halo, boundary seam, discoloration"
    )

    print(f"\nPositive Prompt: \"{pos_prompt}\"")
    print(f"Negative Prompt: \"{neg_prompt}\"")

    # 1. Initialize Pipeline (with real diffusion backend)
    print("\nInitializing FaceInpaintingPipeline backend...")
    pipeline = FaceInpaintingPipeline(
        checkpoint_path=str(ckpt_path),
        mock_mode=False,
        identity_scale=0.10,
        guidance_scale=7.5,
        default_steps=50
    )

    # 2. O1 Reference Scoring
    print("Computing O1 Reference Scoring...")
    target_features = pipeline.face_pipeline.analyze(target_rgb, mask_gray)
    ref_features_list = [pipeline.face_pipeline.analyze(r) for r in ref_rgbs]
    scoring_result = pipeline.reference_scorer.score_references(target_features, ref_features_list)
    if "global_normalized_weights" in scoring_result:
        ref_weights = [round(float(v), 4) for v in scoring_result["global_normalized_weights"].values()]
    else:
        ref_weights = [round(float(v), 4) for v in scoring_result.get("weights", [1.0/len(ref_rgbs)]*len(ref_rgbs))]
    print(f"O1 Weights: {ref_weights}")

    # 3. O2 Regional Identity Token Fusion
    print("Computing O2 Regional Token Fusion...")
    def prep_256(img):
        r = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(pipeline.device) / 127.5 - 1.0

    mask_256 = cv2.resize(mask_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(pipeline.device)

    with torch.no_grad():
        ref_tokens = []
        for r_img in ref_rgbs:
            t_tok = pipeline.arcface_encoder(prep_256(r_img))["spatial_tokens"].to(pipeline.device)
            ref_tokens.append(t_tok)

        w_tensor = torch.tensor([ref_weights], dtype=torch.float32, device=pipeline.device)
        t_fused = pipeline.fusion_transformer(
            target_tokens=None,
            target_mask=mask_tensor_256,
            reference_tokens_list=ref_tokens,
            reference_weights=w_tensor
        )["fused_tokens"]

    # 4. AspectRatioPreserver Letterbox
    pil_img, pil_mask, transform_info = AspectRatioPreserver.letterbox(
        image=target_rgb,
        mask=mask_gray,
        target_size=(512, 512)
    )

    # 5. Configure Decoupled Attention Manager
    pipeline.manager.set_scale(0.10)
    pipeline.manager.set_identity_tokens(t_fused.to(device=pipeline.device, dtype=pipeline.pipe.unet.dtype))

    gen = (
        torch.Generator(device="cpu").manual_seed(42)
        if pipeline.device.type == "cpu"
        else torch.Generator(device=pipeline.device).manual_seed(42)
    )

    # 6. Execute 50-step DDIM Diffusion
    print("\nExecuting 50 DDIM Denoising Steps with Decoupled Identity Cross-Attention & Negative Guidance...")
    t_start = time.time()
    diff_out = pipeline.pipe(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=gen
    ).images[0]
    diffusion_time = time.time() - t_start
    print(f"Diffusion generation completed in {diffusion_time:.2f}s ({diffusion_time/60:.2f} minutes).")

    # 7. Unletterbox to exact original resolution
    padded_cand = np.array(diff_out, dtype=np.uint8)
    raw_reconstructed = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_cand,
        original_image=target_rgb,
        original_mask=mask_gray,
        transform_info=transform_info
    )

    # Save raw diffusion output before post-processing/feathering
    raw_out_path = out_dir / "01_raw_diffusion_output.png"
    Image.fromarray(raw_reconstructed).save(raw_out_path)
    print(f"Saved raw diffusion output: {raw_out_path}")

    # 8. Fixed Boundary Harmonization (Inward Feather r=7px)
    print("\nApplying fixed boundary harmonization (inward feather r=7px)...")
    harmonizer = BoundaryHarmonizer(feather_radius=7.0, blend_mode="feathered")
    final_composite, comp_metrics = harmonizer.composite(
        target_image=target_rgb,
        restored_image=raw_reconstructed,
        mask=mask_gray
    )

    final_out_path = out_dir / "02_final_r7_composite.png"
    Image.fromarray(final_composite).save(final_out_path)
    print(f"Saved final r=7 composite: {final_out_path}")

    # 9. Verify Strict Canvas Invariance
    passed_inv, max_diff_out, violated_out = verify_unmasked_invariance(
        target_rgb, final_composite, mask_gray
    )
    print(f"\nCanvas Invariance: passed={passed_inv}, Delta_max={max_diff_out:.5f}, violated_pixels={violated_out}")
    assert passed_inv and max_diff_out == 0.0 and violated_out == 0, (
        f"Invariance failed: max_diff={max_diff_out}, violated={violated_out}"
    )

    # 10. Quantitative Measurements
    # Surrounding healthy skin ring (20px width)
    k_size = 20 * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(mask_gray, kernel)
    ring_mask = (dilated > 0) & (mask_gray == 0)

    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring_lab_mean = np.mean(target_lab[ring_mask], axis=0)

    # Deep interior core (eroded by 7px from mask boundary)
    k_erode7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    deep_interior = (cv2.erode(mask_gray, k_erode7) > 0)

    # Measurements on RAW output
    raw_inside_diff = np.abs(raw_reconstructed.astype(np.float32) - target_rgb.astype(np.float32))[mask_gray > 0]
    raw_mean_delta = float(np.mean(raw_inside_diff))
    raw_max_delta = float(np.max(raw_inside_diff))

    raw_gray = cv2.cvtColor(raw_reconstructed, cv2.COLOR_RGB2GRAY).astype(np.float32)
    raw_lap = cv2.Laplacian(raw_gray, cv2.CV_32F)
    raw_deep_tex_var = float(np.var(raw_lap[deep_interior]))

    raw_lab = cv2.cvtColor(raw_reconstructed, cv2.COLOR_RGB2LAB).astype(np.float32)
    raw_deep_lab_mean = np.mean(raw_lab[deep_interior], axis=0)
    raw_deep_lab_std = np.std(raw_lab[deep_interior], axis=0)
    raw_delta_e = float(np.linalg.norm(raw_deep_lab_mean - ring_lab_mean))

    # Measurements on FINAL r=7 composite
    final_inside_diff = np.abs(final_composite.astype(np.float32) - target_rgb.astype(np.float32))[mask_gray > 0]
    final_mean_delta = float(np.mean(final_inside_diff))
    final_max_delta = float(np.max(final_inside_diff))

    final_gray = cv2.cvtColor(final_composite, cv2.COLOR_RGB2GRAY).astype(np.float32)
    final_lap = cv2.Laplacian(final_gray, cv2.CV_32F)
    final_deep_tex_var = float(np.var(final_lap[deep_interior]))

    final_lab = cv2.cvtColor(final_composite, cv2.COLOR_RGB2LAB).astype(np.float32)
    final_deep_lab_mean = np.mean(final_lab[deep_interior], axis=0)
    final_deep_lab_std = np.std(final_lab[deep_interior], axis=0)
    final_delta_e = float(np.linalg.norm(final_deep_lab_mean - ring_lab_mean))

    # Boundary metrics
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inner_boundary = (mask_gray > 0) & (cv2.erode(mask_gray, kernel_rect) == 0)
    outer_boundary = (mask_gray == 0) & (cv2.dilate(mask_gray, kernel_rect) > 0)
    target_outer_rgb = np.mean(target_rgb[outer_boundary], axis=0)
    final_inner_rgb = np.mean(final_composite[inner_boundary], axis=0)
    boundary_jump = float(np.linalg.norm(final_inner_rgb - target_outer_rgb))

    lap_tgt = cv2.Laplacian(target_rgb, cv2.CV_32F)
    lap_final = cv2.Laplacian(final_composite, cv2.CV_32F)
    lap_step_delta = float(np.mean(np.abs(lap_tgt[inner_boundary] - lap_final[inner_boundary])))

    # Load V5-A Baseline for Direct Comparison
    v5a_raw_path = v2_dir / "reconstruction.jpg"
    v5a_r7_path = v4_dir / "00_baseline_feather_r7.png"
    v5a_raw_rgb = cv2.cvtColor(cv2.imread(str(v5a_raw_path)), cv2.COLOR_BGR2RGB) if v5a_raw_path.exists() else None
    v5a_r7_rgb = cv2.cvtColor(cv2.imread(str(v5a_r7_path)), cv2.COLOR_BGR2RGB) if v5a_r7_path.exists() else None

    # Save Zoomed Comparison Panel
    pad = 15
    bx1, by1, bx2, by2 = 255, 41, 329, 205
    cy1, cy2 = max(0, by1 - pad), min(h, by2 + pad)
    cx1, cx2 = max(0, bx1 - pad), min(w, bx2 + pad)

    crop_target = target_rgb[cy1:cy2, cx1:cx2]
    crop_v5a_raw = v5a_raw_rgb[cy1:cy2, cx1:cx2] if v5a_raw_rgb is not None else np.zeros_like(crop_target)
    crop_v5a_r7 = v5a_r7_rgb[cy1:cy2, cx1:cx2] if v5a_r7_rgb is not None else np.zeros_like(crop_target)
    crop_v5c_raw = raw_reconstructed[cy1:cy2, cx1:cx2]
    crop_v5c_r7 = final_composite[cy1:cy2, cx1:cx2]

    fig_zoom, axes_z = plt.subplots(1, 5, figsize=(25, 6))
    crops_list = [
        ("1. Target (Before)", crop_target),
        ("2. V5-A Baseline (Raw)", crop_v5a_raw),
        ("3. V5-A Baseline (r=7)", crop_v5a_r7),
        ("4. V5-C Denoised (Raw)", crop_v5c_raw),
        ("5. V5-C Denoised (r=7)", crop_v5c_r7)
    ]
    for idx, (title, c_img) in enumerate(crops_list):
        axes_z[idx].imshow(c_img)
        axes_z[idx].set_title(title, fontsize=11, fontweight="bold")
        axes_z[idx].axis("off")

    fig_zoom.tight_layout()
    zoom_save_path = out_dir / "zoomed_v5a_vs_v5c_comparison.png"
    fig_zoom.savefig(zoom_save_path, dpi=150)
    plt.close(fig_zoom)
    print(f"Saved zoomed comparison: {zoom_save_path}")

    # Save Full-Face Comparison Panel
    fig_full, axes_f = plt.subplots(1, 4, figsize=(24, 6))
    full_list = [
        ("1. Original Target", target_rgb),
        ("2. V5-A Baseline (r=7)", v5a_r7_rgb if v5a_r7_rgb is not None else target_rgb),
        ("3. V5-C Raw Diffusion", raw_reconstructed),
        ("4. V5-C Restored (r=7)", final_composite)
    ]
    for idx, (title, f_img) in enumerate(full_list):
        axes_f[idx].imshow(f_img)
        axes_f[idx].set_title(title, fontsize=11, fontweight="bold")
        axes_f[idx].axis("off")

    fig_full.tight_layout()
    full_save_path = out_dir / "full_face_v5a_vs_v5c_comparison.png"
    fig_full.savefig(full_save_path, dpi=150)
    plt.close(fig_full)
    print(f"Saved full-face comparison: {full_save_path}")

    # Metadata & Results Dictionary
    metadata = {
        "experiment_id": "case01_v5c_restoration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(diffusion_time, 2),
        "prompts": {
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt
        },
        "fixed_parameters": {
            "checkpoint": str(ckpt_path),
            "adapter": "DecoupledIdentityCrossAttention (Step 150)",
            "identity_scale": 0.10,
            "guidance_scale": 7.5,
            "steps": 50,
            "seed": 42,
            "device": str(pipeline.device),
            "feather_radius": 7.0
        },
        "canvas_invariance": {
            "passed": passed_inv,
            "max_difference_outside_mask": max_diff_out,
            "violated_pixels_outside_mask": violated_out
        },
        "raw_diffusion_output_metrics": {
            "inside_mask_mean_delta": round(raw_mean_delta, 2),
            "inside_mask_max_delta": round(raw_max_delta, 2),
            "deep_interior_texture_variance": round(raw_deep_tex_var, 2),
            "deep_interior_lab_mean": [round(float(v), 2) for v in raw_deep_lab_mean],
            "deep_interior_lab_std": [round(float(v), 2) for v in raw_deep_lab_std],
            "delta_e_to_healthy_ring": round(raw_delta_e, 2)
        },
        "final_r7_composite_metrics": {
            "inside_mask_mean_delta": round(final_mean_delta, 2),
            "inside_mask_max_delta": round(final_max_delta, 2),
            "boundary_jump": round(boundary_jump, 2),
            "boundary_laplacian_step_delta": round(lap_step_delta, 2),
            "deep_interior_texture_variance": round(final_deep_tex_var, 2),
            "deep_interior_lab_mean": [round(float(v), 2) for v in final_deep_lab_mean],
            "deep_interior_lab_std": [round(float(v), 2) for v in final_deep_lab_std],
            "delta_e_to_healthy_ring": round(final_delta_e, 2)
        }
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    print("\n================================================================================")
    print("V5-C DIFFUSION RESTORATION EXECUTION COMPLETED")
    print("================================================================================")


if __name__ == "__main__":
    main()
