"""
Multi-Subject Statistical Benchmark (Step 150 vs Step 500).
Evaluates across a curated, balanced subset of 10 CelebA-HQ subjects:
- Exactly 3 verified same-identity references per subject
- 1 cross-identity impostor per subject
- Standardized canonical 25% occlusion mask
- Deterministic per-subject seeds (seed = 42 + subject_index)
- Full statistical aggregation: Mean, Std Dev, Median, Min, Max, Paired Difference, p-value
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
import skimage.metrics
from PIL import Image

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from diffusers import StableDiffusionInpaintPipeline


# Curated 10 Diverse Benchmark Subjects (from verified CelebA-HQ metadata and cache)
DEFAULT_SUBJECTS = [
    {"subject_id": 1, "identity": "3111", "target": 7367, "references": [13401, 23058, 23414], "diff_id": 3616},
    {"subject_id": 2, "identity": "6995", "target": 51,   "references": [1964, 2241, 3117],   "diff_id": 69},
    {"subject_id": 3, "identity": "1657", "target": 69,   "references": [199, 329, 11896],    "diff_id": 81},
    {"subject_id": 4, "identity": "3298", "target": 81,   "references": [5035, 6562, 22285],  "diff_id": 84},
    {"subject_id": 5, "identity": "7883", "target": 84,   "references": [1538, 2074, 9056],   "diff_id": 90},
    {"subject_id": 6, "identity": "1985", "target": 90,   "references": [6079, 6089, 11032],  "diff_id": 93},
    {"subject_id": 7, "identity": "7789", "target": 93,   "references": [1358, 2139, 3461],   "diff_id": 164},
    {"subject_id": 8, "identity": "5551", "target": 164,  "references": [2854, 3129, 16511],  "diff_id": 183},
    {"subject_id": 9, "identity": "861",  "target": 183,  "references": [1183, 1539, 10088],  "diff_id": 242},
    {"subject_id": 10, "identity": "7006", "target": 242, "references": [1837, 7893, 22231],  "diff_id": 7367}
]


def find_sd_model():
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
    if cache_base.exists():
        dirs = [d for d in cache_base.iterdir() if d.is_dir()]
        if dirs:
            return str(dirs[0])
    return "runwayml/stable-diffusion-inpainting"


def compute_paired_statistics(vals_150: List[float], vals_500: List[float]) -> Dict[str, Any]:
    """Computes mean, std, median, min, max, delta, and paired t-test p-value."""
    a = np.array(vals_150, dtype=np.float64)
    b = np.array(vals_500, dtype=np.float64)
    deltas = b - a

    stat_summary = {
        "step_150": {
            "mean": round(float(np.mean(a)), 4),
            "std": round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else 0.0,
            "median": round(float(np.median(a)), 4),
            "min": round(float(np.min(a)), 4),
            "max": round(float(np.max(a)), 4)
        },
        "step_500": {
            "mean": round(float(np.mean(b)), 4),
            "std": round(float(np.std(b, ddof=1)), 4) if len(b) > 1 else 0.0,
            "median": round(float(np.median(b)), 4),
            "min": round(float(np.min(b)), 4),
            "max": round(float(np.max(b)), 4)
        },
        "paired_delta_step500_minus_step150": {
            "mean": round(float(np.mean(deltas)), 4),
            "std": round(float(np.std(deltas, ddof=1)), 4) if len(deltas) > 1 else 0.0,
            "median": round(float(np.median(deltas)), 4),
            "min": round(float(np.min(deltas)), 4),
            "max": round(float(np.max(deltas)), 4)
        }
    }

    if len(a) > 2:
        try:
            t_res = scipy.stats.ttest_rel(b, a)
            stat_summary["p_value_paired_ttest"] = round(float(t_res.pvalue), 5)
            stat_summary["statistically_significant_05"] = bool(t_res.pvalue < 0.05)
        except Exception:
            stat_summary["p_value_paired_ttest"] = None
    return stat_summary


def run_multi_subject_benchmark(
    subjects: List[Dict[str, Any]] = DEFAULT_SUBJECTS,
    mask_path: str = "data/masks/canonical_eval_mask_25pct.png",
    checkpoints: List[str] = [
        "checkpoints/baselines/step_150_best/step_150.pt",
        "checkpoints/cloud_training/step_500.pt"
    ],
    device_str: str = "auto",
    output_dir: str = "evaluation/reports",
    steps: int = 50,
    base_seed: int = 42,
    smoke_test: bool = False
):
    print("=" * 95)
    print(f"MULTI-SUBJECT STATISTICAL BENCHMARK ({len(subjects)} SUBJECTS)")
    print("=" * 95)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    print(f"Compute Device: {device}")

    if smoke_test:
        print("[SMOKE TEST ACTIVE: 1 subject, 1 step]")
        subjects = subjects[:1]
        steps = 1

    mask_file = PROJECT_ROOT / mask_path
    assert mask_file.exists(), f"Missing mask: {mask_file}"
    mask_gray = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)
    mask_bool = (mask > 0.5)
    ys, xs = np.where(mask_bool)
    crop_coords = (int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs)))
    min_y, max_y, min_x, max_x = crop_coords

    def prep_256(img):
        r = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    def prep_112(img):
        r = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    # Models
    print("\n[1/3] Loading Pretrained Feature Extractors & Diffusion Pipeline...")
    diff_arcface_path = str(PROJECT_ROOT / "models/pretrained/w600k_r50.onnx")
    diff_arcface = DifferentiableArcFaceBackbone(diff_arcface_path).to(device)
    diff_arcface.eval()
    arcface = RegionalIdentityEncoder(token_dim=512)
    arcface.eval()
    phase4 = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(device)
    phase4.eval()

    sd_model = find_sd_model()
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float32,
            safety_checker=None,
            local_files_only=True
        ).to(device)
    except Exception:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(device)

    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=0.10)
    manager.install(zero_init_v=False)
    manager.to(device=device, dtype=pipe.unet.dtype)

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    results_per_subject = []
    print("\n[2/3] Evaluating Subjects...")

    for subj in subjects:
        s_id = subj["subject_id"]
        identity = subj["identity"]
        tgt_img_id = subj["target"]
        ref_ids = subj["references"]
        diff_id = subj["diff_id"]
        subj_seed = base_seed + s_id

        print(f"\n--- Subject {s_id}/{len(subjects)} (Identity {identity}) | Target: {tgt_img_id} | Refs: {ref_ids} ---")

        tgt_p = PROJECT_ROOT / f"data/processed/dataset_cache/{tgt_img_id}.jpg"
        diff_p = PROJECT_ROOT / f"data/processed/dataset_cache/{diff_id}.jpg"
        ref_ps = [PROJECT_ROOT / f"data/processed/dataset_cache/{r}.jpg" for r in ref_ids]

        assert tgt_p.exists(), f"Missing target: {tgt_p}"
        assert diff_p.exists(), f"Missing impostor: {diff_p}"
        for rp in ref_ps: assert rp.exists(), f"Missing ref: {rp}"

        target_rgb = cv2.cvtColor(cv2.imread(str(tgt_p)), cv2.COLOR_BGR2RGB)
        diff_rgb = cv2.cvtColor(cv2.imread(str(diff_p)), cv2.COLOR_BGR2RGB)
        refs_rgb = [cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB) for rp in ref_ps]
        crop_gt = target_rgb[min_y:max_y, min_x:max_x]

        # Extract tokens
        with torch.no_grad():
            ref_tokens = [arcface(prep_256(r))["spatial_tokens"].to(device) for r in refs_rgb]
            w_multi = torch.tensor([[1.0/len(ref_tokens)] * len(ref_tokens)], dtype=torch.float32, device=device)
            t_fused = phase4(
                target_tokens=None,
                target_mask=mask_tensor_256,
                reference_tokens_list=ref_tokens,
                reference_weights=w_multi
            )["fused_tokens"]

            e_refs = [diff_arcface.extract_embedding(prep_112(r)) for r in refs_rgb]
            e_ref_mean = F.normalize(torch.stack(e_refs).mean(dim=0), p=2, dim=-1)
            e_diff = diff_arcface.extract_embedding(prep_112(diff_rgb))
            e_target = diff_arcface.extract_embedding(prep_112(target_rgb))

        padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(target_rgb, mask, target_size=(512, 512))
        pil_img = Image.fromarray(padded_img)
        pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

        prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"

        subj_eval = {
            "subject_id": s_id,
            "identity": identity,
            "target": tgt_img_id,
            "references": ref_ids,
            "diff_id": diff_id,
            "seed": subj_seed,
            "runs": {}
        }

        for ckpt_p_str in checkpoints:
            ckpt_p = (PROJECT_ROOT / ckpt_p_str).resolve()
            ckpt_name = ckpt_p.stem

            ckpt_data = manager.load_checkpoint(str(ckpt_p), strict=True)
            manager.to(device=device, dtype=pipe.unet.dtype)
            manager.set_scale(0.10)
            manager.set_identity_tokens(t_fused.to(device=device, dtype=pipe.unet.dtype))

            gen = torch.Generator(device="cpu").manual_seed(subj_seed) if device.type == "cpu" else torch.Generator(device=device).manual_seed(subj_seed)

            t0 = time.time()
            diff_out = pipe(
                prompt=prompt,
                image=pil_img,
                mask_image=pil_mask,
                num_inference_steps=steps,
                guidance_scale=7.5,
                generator=gen
            ).images[0]
            gen_dur = time.time() - t0

            padded_cand = np.array(diff_out, dtype=np.uint8)
            final_recon = AspectRatioPreserver.unletterbox(
                padded_candidate=padded_cand,
                original_image=target_rgb,
                original_mask=mask,
                transform_info=transform_info
            )

            passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, final_recon, mask)
            crop_recon = final_recon[min_y:max_y, min_x:max_x]

            mae_diff = np.abs(final_recon[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
            recon_mae = float(np.mean(mae_diff))
            recon_rmse = float(np.sqrt(np.mean(mae_diff**2)))
            recon_psnr = float(20 * np.log10(255.0 / (recon_rmse + 1e-9)))
            recon_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_recon, channel_axis=2))

            with torch.no_grad():
                e_recon = diff_arcface.extract_embedding(prep_112(final_recon))
                cos_ref = float(F.cosine_similarity(e_recon, e_ref_mean, dim=-1).item())
                cos_tgt = float(F.cosine_similarity(e_recon, e_target, dim=-1).item())
                cos_diff = float(F.cosine_similarity(e_recon, e_diff, dim=-1).item())
                bio_gap = cos_ref - cos_diff

            subj_eval["runs"][ckpt_name] = {
                "sim_ref": round(cos_ref, 4),
                "sim_gt": round(cos_tgt, 4),
                "sim_diff": round(cos_diff, 4),
                "bio_gap": round(bio_gap, 4),
                "psnr": round(recon_psnr, 2),
                "ssim": round(recon_ssim, 4),
                "mae": round(recon_mae, 2),
                "rmse": round(recon_rmse, 2),
                "invariance_passed": passed_inv,
                "runtime_seconds": round(gen_dur, 2)
            }
            print(f"  {ckpt_name:<8} -> SimRef: {cos_ref:.4f} | SimGT: {cos_tgt:.4f} | Bio Gap: {bio_gap:+.4f} | PSNR: {recon_psnr:.2f} dB | SSIM: {recon_ssim:.4f}")

        # Compute subject paired delta
        if "step_150" in subj_eval["runs"] and "step_500" in subj_eval["runs"]:
            r150 = subj_eval["runs"]["step_150"]
            r500 = subj_eval["runs"]["step_500"]
            subj_eval["delta_step500_minus_step150"] = {
                "sim_ref": round(r500["sim_ref"] - r150["sim_ref"], 4),
                "sim_gt": round(r500["sim_gt"] - r150["sim_gt"], 4),
                "bio_gap": round(r500["bio_gap"] - r150["bio_gap"], 4),
                "psnr": round(r500["psnr"] - r150["psnr"], 2),
                "ssim": round(r500["ssim"] - r150["ssim"], 4)
            }
        results_per_subject.append(subj_eval)

    # Statistical Aggregation
    print("\n[3/3] Aggregating Benchmark Statistics across Subjects...")
    metrics_to_aggregate = ["sim_ref", "sim_gt", "sim_diff", "bio_gap", "psnr", "ssim", "mae", "rmse"]
    aggregated_stats = {
        "num_subjects": len(results_per_subject),
        "metrics": {}
    }

    for m in metrics_to_aggregate:
        v150 = [s["runs"]["step_150"][m] for s in results_per_subject if "step_150" in s["runs"]]
        v500 = [s["runs"]["step_500"][m] for s in results_per_subject if "step_500" in s["runs"]]
        if v150 and v500:
            aggregated_stats["metrics"][m] = compute_paired_statistics(v150, v500)

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"{'Metric':<12} | {'Step 150 Mean (Std)':<22} | {'Step 500 Mean (Std)':<22} | {'Delta Mean':<12} | {'p-value':<10} | {'Sig? (p<0.05)':<12}")
    print("-" * 105)
    for m, stats in aggregated_stats["metrics"].items():
        s150_str = f"{stats['step_150']['mean']:.4f} (+/- {stats['step_150']['std']:.4f})"
        s500_str = f"{stats['step_500']['mean']:.4f} (+/- {stats['step_500']['std']:.4f})"
        delta_str = f"{stats['paired_delta_step500_minus_step150']['mean']:+.4f}"
        pval = stats.get("p_value_paired_ttest")
        pval_str = f"{pval:.4f}" if pval is not None else "N/A"
        sig_str = "YES" if stats.get("statistically_significant_05") else "NO"
        print(f"{m:<12} | {s150_str:<22} | {s500_str:<22} | {delta_str:<12} | {pval_str:<10} | {sig_str:<12}")
    print("=" * 105)

    # Save outputs
    out_dir_p = Path(output_dir)
    if not out_dir_p.is_absolute():
        out_dir_p = (PROJECT_ROOT / out_dir_p).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    json_p = out_dir_p / "multi_subject_benchmark.json"
    with open(json_p, "w") as f:
        json.dump({
            "summary_statistics": aggregated_stats,
            "subjects_detail": results_per_subject
        }, f, indent=2)

    csv_p = out_dir_p / "multi_subject_benchmark.csv"
    with open(csv_p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id", "identity", "target", "checkpoint",
            "sim_ref", "sim_gt", "sim_diff", "bio_gap", "psnr", "ssim", "mae", "rmse"
        ])
        for s in results_per_subject:
            for ckpt_k, r in s["runs"].items():
                writer.writerow([
                    s["subject_id"], s["identity"], s["target"], ckpt_k,
                    r["sim_ref"], r["sim_gt"], r["sim_diff"], r["bio_gap"], r["psnr"], r["ssim"], r["mae"], r["rmse"]
                ])

    print(f"\nOutputs saved:")
    print(f"  - Detailed JSON: {json_p}")
    print(f"  - Summary CSV:   {csv_p}")
    print("=" * 95)
    return aggregated_stats, results_per_subject


def main():
    parser = argparse.ArgumentParser(description="Multi-Subject Statistical Benchmark (Step 150 vs Step 500)")
    parser.add_argument("--checkpoints", nargs="+", default=[
        "checkpoints/baselines/step_150_best/step_150.pt",
        "checkpoints/cloud_training/step_500.pt"
    ])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="evaluation/reports")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 subject with 1 step to verify execution flow")
    args = parser.parse_args()

    run_multi_subject_benchmark(
        checkpoints=args.checkpoints,
        device_str=args.device,
        output_dir=args.output_dir,
        steps=args.steps,
        base_seed=args.seed,
        smoke_test=args.smoke_test
    )


if __name__ == "__main__":
    main()
