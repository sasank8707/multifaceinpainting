"""
Evaluation of Clean O3 Verifier-Aware Sampling / Reranking.
Evaluates Step 150 and Step 500 checkpoints using zero-leakage O3 candidate selection.

Protocol:
1. Candidate Generation:
   - Target: CelebA-HQ 7367 (Identity 3111)
   - References: 13401, 23058, 23414 (Same-ID); Impostor: 3616 (Diff-ID)
   - Canonical 25% occlusion mask (data/masks/canonical_eval_mask_25pct.png)
   - K=3 independent diffusion candidates (seeds: [42, 43, 44])
   - CFG = 7.5, Identity Scale = 0.10, Steps = 50 (or 1 with --smoke)
2. Clean Reranking:
   - Zero access to ground-truth target pixels or evaluation ArcFace.
   - Independent matcher: InceptionResnetV1 (FaceNet) on references.
   - Blind quality: BRISQUE (piq) on candidate image.
   - Joint score: 0.60 * S_id + 0.40 * S_qual.
3. Held-Out Evaluation:
   - Evaluates selected winner on held-out ArcFace (w600k_r50.onnx) and ground-truth PSNR/SSIM.
   - Compares winning candidate against single-shot baseline (seed 42).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage.metrics
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from modules.reranking.clean_verifier import CleanO3Verifier


def compute_laplacian_error(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    laplace_5x5 = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=np.float32)

    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0
    lap_gt = np.zeros_like(gt_f)
    lap_pred = np.zeros_like(pred_f)
    for c in range(3):
        lap_gt[:, :, c] = cv2.filter2D(gt_f[:, :, c], -1, laplace_5x5)
        lap_pred[:, :, c] = cv2.filter2D(pred_f[:, :, c], -1, laplace_5x5)
    mask_bool = (mask > 0.5)
    return float(np.mean(np.abs(lap_gt - lap_pred)[mask_bool]))


def compute_sobel_error(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx_gt = cv2.Sobel(gt_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_gt = cv2.Sobel(gt_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_gt = np.sqrt(gx_gt**2 + gy_gt**2)
    gx_pred = cv2.Sobel(pred_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_pred = cv2.Sobel(pred_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_pred = np.sqrt(gx_pred**2 + gy_pred**2)
    mask_bool = (mask > 0.5)
    return float(np.mean(np.abs(mag_gt - mag_pred)[mask_bool]))


def find_sd_model():
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
    if cache_base.exists():
        dirs = [d for d in cache_base.iterdir() if d.is_dir()]
        if dirs:
            return str(dirs[0])
    return "runwayml/stable-diffusion-inpainting"


def evaluate_held_out_metrics(
    recon_img: np.ndarray,
    target_rgb: np.ndarray,
    mask: np.ndarray,
    crop_coords: Tuple[int, int, int, int],
    diff_arcface: DifferentiableArcFaceBackbone,
    e_ref_mean: torch.Tensor,
    e_target: torch.Tensor,
    e_diff: torch.Tensor,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates held-out metrics against ground truth.
    This function is ONLY called after candidate selection is complete.
    """
    passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, recon_img, mask)

    min_y, max_y, min_x, max_x = crop_coords
    mask_bool = (mask > 0.5)
    crop_gt = target_rgb[min_y:max_y, min_x:max_x]
    crop_recon = recon_img[min_y:max_y, min_x:max_x]

    mae_diff = np.abs(recon_img[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
    recon_mae = float(np.mean(mae_diff))
    recon_rmse = float(np.sqrt(np.mean(mae_diff**2)))
    recon_psnr = float(20 * np.log10(255.0 / (recon_rmse + 1e-9)))
    recon_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_recon, channel_axis=2))
    recon_sobel = compute_sobel_error(target_rgb, recon_img, mask)
    recon_lap = compute_laplacian_error(target_rgb, recon_img, mask)

    def prep_112(rgb_img):
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    with torch.no_grad():
        e_recon = diff_arcface.extract_embedding(prep_112(recon_img))
        cos_ref = float(F.cosine_similarity(e_recon, e_ref_mean, dim=-1).item())
        cos_tgt = float(F.cosine_similarity(e_recon, e_target, dim=-1).item())
        cos_diff = float(F.cosine_similarity(e_recon, e_diff, dim=-1).item())
        biometric_gap = cos_ref - cos_diff

    return {
        "invariance": {
            "passed": passed_inv,
            "max_difference": max_diff,
            "violated_pixels": violated_pixels
        },
        "biometrics": {
            "cos_recon_vs_ref_mean": round(cos_ref, 4),
            "cos_recon_vs_target_gt": round(cos_tgt, 4),
            "cos_recon_vs_diff_id": round(cos_diff, 4),
            "biometric_identity_gap": round(biometric_gap, 4)
        },
        "reconstruction": {
            "psnr": round(recon_psnr, 2),
            "ssim": round(recon_ssim, 4),
            "mae": round(recon_mae, 4),
            "rmse": round(recon_rmse, 4),
            "sobel_error": round(recon_sobel, 4),
            "laplacian_error": round(recon_lap, 4)
        }
    }


def run_clean_o3_evaluation(
    checkpoints: List[str] = [
        "checkpoints/baselines/step_150_best/step_150.pt",
        "checkpoints/cloud_training/step_500.pt"
    ],
    device_str: str = "auto",
    steps: int = 50,
    candidate_seeds: List[int] = [42, 43, 44],
    guidance_scale: float = 7.5,
    lambda_scale: float = 0.10,
    output_dir: str = "evaluation/reports/clean_o3_reranker",
    smoke_test: bool = False,
    mock_mode: bool = False
):
    print("=" * 80)
    print("CLEAN O3 VERIFIER-AWARE RERANKER BENCHMARK (ZERO-LEAKAGE)")
    print(f"Candidate Pool K: {len(candidate_seeds)} | Seeds: {candidate_seeds}")
    print("=" * 80)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    print(f"Compute Device: {device}")

    if smoke_test:
        print("[SMOKE TEST MODE: 1 step, K=2 candidates]")
        steps = 1
        candidate_seeds = candidate_seeds[:2]

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical assets
    target_p = PROJECT_ROOT / "data/processed/dataset_cache/7367.jpg"
    ref_a_p = PROJECT_ROOT / "data/processed/dataset_cache/13401.jpg"
    ref_b_p = PROJECT_ROOT / "data/processed/dataset_cache/23058.jpg"
    ref_c_p = PROJECT_ROOT / "data/processed/dataset_cache/23414.jpg"
    diff_p = PROJECT_ROOT / "data/processed/dataset_cache/3616.jpg"
    mask_p = PROJECT_ROOT / "data/masks/canonical_eval_mask_25pct.png"

    target_rgb = cv2.cvtColor(cv2.imread(str(target_p)), cv2.COLOR_BGR2RGB)
    ref_a_rgb = cv2.cvtColor(cv2.imread(str(ref_a_p)), cv2.COLOR_BGR2RGB)
    ref_b_rgb = cv2.cvtColor(cv2.imread(str(ref_b_p)), cv2.COLOR_BGR2RGB)
    ref_c_rgb = cv2.cvtColor(cv2.imread(str(ref_c_p)), cv2.COLOR_BGR2RGB)
    diff_rgb = cv2.cvtColor(cv2.imread(str(diff_p)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != target_rgb.shape[:2]:
        mask = cv2.resize(mask, (target_rgb.shape[1], target_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    reference_gallery = [ref_a_rgb, ref_b_rgb, ref_c_rgb]

    # Masked crop coords
    mask_y, mask_x = np.where(mask > 127)
    pad = 16
    min_y = max(0, int(np.min(mask_y)) - pad)
    max_y = min(target_rgb.shape[0], int(np.max(mask_y)) + pad)
    min_x = max(0, int(np.min(mask_x)) - pad)
    max_x = min(target_rgb.shape[1], int(np.max(mask_x)) + pad)
    crop_coords = (min_y, max_y, min_x, max_x)

    # Letterbox input
    masked_target = target_rgb.copy()
    masked_target[mask > 127] = 0
    pil_img, pil_mask, transform_info = AspectRatioPreserver.letterbox(
        image=masked_target,
        mask=mask,
        target_size=(512, 512)
    )

    # 2. Initialize Models
    print("\n[Step 1/4] Initializing Models & Matchers...")
    pipe = None
    manager = None
    if not mock_mode:
        sd_model_path = find_sd_model()
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        manager = DecoupledIdentityCrossAttentionManager(
            pipe.unet,
            identity_token_dim=512,
            scale=lambda_scale
        )
    else:
        print("[MOCK MODE ACTIVE: Simulating fast candidate diffusion generation]")

    # Held-out evaluation ArcFace
    diff_arcface_path = str(PROJECT_ROOT / "models/pretrained/w600k_r50.onnx")
    diff_arcface = DifferentiableArcFaceBackbone(diff_arcface_path).to(device)
    diff_arcface.eval()

    def prep_112(img):
        r = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    def prep_256(img):
        r = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    # Compute held-out ArcFace reference targets
    with torch.no_grad():
        e_ref_a = diff_arcface.extract_embedding(prep_112(ref_a_rgb))
        e_ref_b = diff_arcface.extract_embedding(prep_112(ref_b_rgb))
        e_ref_c = diff_arcface.extract_embedding(prep_112(ref_c_rgb))
        e_ref_mean = F.normalize((e_ref_a + e_ref_b + e_ref_c) / 3.0, p=2, dim=-1)
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))
        e_diff = diff_arcface.extract_embedding(prep_112(diff_rgb))

    # Regional Identity Encoder & Transformer
    arcface = RegionalIdentityEncoder(token_dim=512).to(device)
    arcface.eval()
    phase4 = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(device)
    phase4.eval()

    # Pre-compute fused identity tokens from references
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        tok_A = arcface(prep_256(ref_a_rgb))["spatial_tokens"].to(device)
        tok_B = arcface(prep_256(ref_b_rgb))["spatial_tokens"].to(device)
        tok_C = arcface(prep_256(ref_c_rgb))["spatial_tokens"].to(device)
        w_multi = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32, device=device)
        t_fused = phase4(
            target_tokens=None,
            target_mask=mask_tensor_256,
            reference_tokens_list=[tok_A, tok_B, tok_C],
            reference_weights=w_multi
        )["fused_tokens"]

    # 3. Initialize Clean O3 Verifier
    print("\n[Step 2/4] Initializing Clean O3 Verifier (FaceNet + BRISQUE)...")
    clean_verifier = CleanO3Verifier(device=device, weight_identity=0.60, weight_quality=0.40)

    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"

    results = {
        "benchmark": "Clean O3 Verifier-Aware Sampling Reranker",
        "protocol": {
            "target_subject": "7367",
            "references": ["13401", "23058", "23414"],
            "occlusion_severity": "25pct",
            "candidate_seeds": candidate_seeds,
            "inference_steps": steps,
            "guidance_scale": guidance_scale,
            "identity_scale": lambda_scale,
            "verifier_matcher": clean_verifier.matcher_name,
            "verifier_quality": clean_verifier.quality_metric_name,
            "w_identity": clean_verifier.w_id,
            "w_quality": clean_verifier.w_qual
        },
        "checkpoints": {}
    }

    # 4. Evaluate Checkpoints
    print("\n[Step 3/4] Generating Candidates and Reranking across Checkpoints...")
    for ckpt_rel in checkpoints:
        ckpt_path = PROJECT_ROOT / ckpt_rel
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        ckpt_name = ckpt_path.stem

        print(f"\nEvaluating Checkpoint: {ckpt_name} ({ckpt_path.name})")
        if not mock_mode:
            manager.load_checkpoint(str(ckpt_path))
            manager.to(device=device, dtype=pipe.unet.dtype)
            manager.set_scale(lambda_scale)
            manager.set_identity_tokens(t_fused.to(device=device, dtype=pipe.unet.dtype))
        else:
            # Validate checkpoint load integrity
            ckpt_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            assert "step" in ckpt_dict or "model_state_dict" in ckpt_dict or "to_k_id.weight" in ckpt_dict or len(ckpt_dict) > 0

        candidates_rgb = []
        gen_times = []

        print(f"  Generating {len(candidate_seeds)} independent diffusion candidates...")
        for c_idx, s in enumerate(candidate_seeds):
            t0 = time.time()
            if not mock_mode:
                gen = (
                    torch.Generator(device="cpu").manual_seed(s)
                    if device.type == "cpu"
                    else torch.Generator(device=device).manual_seed(s)
                )
                diff_out = pipe(
                    prompt=prompt,
                    image=pil_img,
                    mask_image=pil_mask,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=gen
                ).images[0]
                elapsed = time.time() - t0
                gen_times.append(elapsed)

                padded_cand = np.array(diff_out, dtype=np.uint8)
                final_cand = AspectRatioPreserver.unletterbox(
                    padded_candidate=padded_cand,
                    original_image=target_rgb,
                    original_mask=mask,
                    transform_info=transform_info
                )
            else:
                # Fast mock generation: blend reference face and deterministic seed variation into masked region
                rng = np.random.RandomState(s)
                ref_pick = reference_gallery[c_idx % len(reference_gallery)]
                mock_recon = target_rgb.copy()
                mask_bool = (mask > 127)
                noise = rng.normal(0, 5 + c_idx * 3, mock_recon.shape).astype(np.float32)
                blended = 0.85 * ref_pick.astype(np.float32) + 0.15 * target_rgb.astype(np.float32) + noise
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                mock_recon[mask_bool] = blended[mask_bool]
                elapsed = time.time() - t0
                gen_times.append(elapsed)
                final_cand = mock_recon

            candidates_rgb.append(final_cand)
            print(f"    Candidate {c_idx+1} (Seed {s}): generated in {elapsed:.2f}s")

            # Save individual candidate image
            cand_save_p = out_path / f"{ckpt_name}_candidate_{c_idx+1}_seed_{s}.png"
            cv2.imwrite(str(cand_save_p), cv2.cvtColor(final_cand, cv2.COLOR_RGB2BGR))

        # Perform Clean Reranking (ZERO ground-truth access!)
        print("  Running Clean O3 Verifier Reranking...")
        t_rerank_start = time.time()
        rerank_result = clean_verifier.rerank(
            candidates=candidates_rgb,
            candidate_seeds=candidate_seeds,
            reference_rgbs=reference_gallery
        )
        rerank_time = time.time() - t_rerank_start

        winning_idx = rerank_result["winning_index"] - 1 # 0-indexed
        winning_seed = rerank_result["winning_seed"]
        winning_recon = candidates_rgb[winning_idx]
        print(f"  >>> Winner Selected: Candidate {winning_idx+1} (Seed {winning_seed}) with Score={rerank_result['winning_score']:.4f} <<<")
        for c_info in rerank_result["all_candidates"]:
            print(f"      Cand {c_info['candidate_index']} (Seed {c_info['seed']}): "
                  f"S_id={c_info['s_identity_norm']:.4f} (raw {c_info['s_identity_raw']:.4f}) | "
                  f"S_qual={c_info['s_quality_norm']:.4f} (BRISQUE {c_info['brisque_raw']:.2f}) -> "
                  f"S_joint={c_info['s_joint']:.4f}")

        # Post-selection held-out evaluation
        print("  Evaluating winning candidate with held-out ArcFace & Ground Truth...")
        winning_eval = evaluate_held_out_metrics(
            recon_img=winning_recon,
            target_rgb=target_rgb,
            mask=mask,
            crop_coords=crop_coords,
            diff_arcface=diff_arcface,
            e_ref_mean=e_ref_mean,
            e_target=e_target,
            e_diff=e_diff,
            device=device
        )

        # Baseline single-shot evaluation (Seed 42, Candidate 1)
        baseline_recon = candidates_rgb[0]
        baseline_eval = evaluate_held_out_metrics(
            recon_img=baseline_recon,
            target_rgb=target_rgb,
            mask=mask,
            crop_coords=crop_coords,
            diff_arcface=diff_arcface,
            e_ref_mean=e_ref_mean,
            e_target=e_target,
            e_diff=e_diff,
            device=device
        )

        # Save winning reconstruction image
        win_save_p = out_path / f"{ckpt_name}_clean_o3_winner.png"
        cv2.imwrite(str(win_save_p), cv2.cvtColor(winning_recon, cv2.COLOR_RGB2BGR))

        # Save side-by-side figure
        fig, axes = plt.subplots(1, len(candidate_seeds) + 2, figsize=(4 * (len(candidate_seeds) + 2), 4))
        axes[0].imshow(target_rgb)
        axes[0].set_title("Ground Truth (7367)")
        axes[0].axis("off")

        axes[1].imshow(masked_target)
        axes[1].set_title("Masked Input (25%)")
        axes[1].axis("off")

        for idx, (c_img, s) in enumerate(zip(candidates_rgb, candidate_seeds)):
            is_win = (idx == winning_idx)
            title = f"Cand {idx+1} (s={s})\nScore: {rerank_result['all_candidates'][idx]['s_joint']:.4f}"
            if is_win:
                title += " [WINNER]"
            axes[idx + 2].imshow(c_img)
            axes[idx + 2].set_title(title, color="green" if is_win else "black", fontweight="bold" if is_win else "normal")
            axes[idx + 2].axis("off")

        fig.tight_layout()
        sbs_p = out_path / f"{ckpt_name}_candidates_panel.png"
        fig.savefig(str(sbs_p), dpi=150)
        plt.close(fig)

        # Collect metrics
        ckpt_summary = {
            "checkpoint_name": ckpt_name,
            "candidate_generation_runtimes": [round(t, 2) for t in gen_times],
            "rerank_runtime_seconds": round(rerank_time, 4),
            "rerank_selection": rerank_result,
            "winning_candidate_held_out_metrics": winning_eval,
            "baseline_single_shot_metrics": baseline_eval,
            "delta_clean_o3_vs_baseline": {
                "delta_cos_ref": round(winning_eval["biometrics"]["cos_recon_vs_ref_mean"] - baseline_eval["biometrics"]["cos_recon_vs_ref_mean"], 4),
                "delta_biometric_gap": round(winning_eval["biometrics"]["biometric_identity_gap"] - baseline_eval["biometrics"]["biometric_identity_gap"], 4),
                "delta_psnr": round(winning_eval["reconstruction"]["psnr"] - baseline_eval["reconstruction"]["psnr"], 2),
                "delta_ssim": round(winning_eval["reconstruction"]["ssim"] - baseline_eval["reconstruction"]["ssim"], 4)
            }
        }
        results["checkpoints"][ckpt_name] = ckpt_summary

    # Save complete JSON
    json_p = out_path / "clean_o3_evaluation_results.json"
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("CLEAN O3 EVALUATION COMPLETE")
    print(f"Results saved to: {json_p}")
    print("=" * 80)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Clean O3 Verifier-Aware Sampling Benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion inference steps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Candidate seeds")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=[
        "checkpoints/baselines/step_150_best/step_150.pt",
        "checkpoints/cloud_training/step_500.pt"
    ], help="Checkpoint relative paths")
    parser.add_argument("--smoke", action="store_true", help="Fast 1-step smoke test")
    parser.add_argument("--mock", action="store_true", help="High-speed mock candidate generation for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_clean_o3_evaluation(
        checkpoints=args.checkpoints,
        device_str=args.device,
        steps=args.steps,
        candidate_seeds=args.seeds,
        smoke_test=args.smoke,
        mock_mode=args.mock
    )
