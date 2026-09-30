"""
Unified Checkpoint Evaluation Benchmark Workflow.
Evaluates and compares model checkpoints across training milestones (150, 500, 1000, 2500, 5000, 10000).

Strict Protocol Adherence:
- Preserves identical evaluation protocol to Phase-7 / Tier-1 research evaluation
- Target: CelebA-HQ 7367 (Identity 3111)
- References: 13401, 23058, 23414 (Same-ID); Impostor: 3616 (Diff-ID)
- Mask Severity: Canonical 25% face occlusion
- Always includes checkpoints/baselines/step_150_best/step_150.pt as immutable baseline
- Evaluates:
  1. Biometric ArcFace Cosine Similarity (vs Reference Mean, Target GT, Impostor)
  2. Biometric Identity Gap (Sim_Ref - Sim_Diff)
  3. Perceptual and Pixel Reconstruction Quality (PSNR, SSIM, MAE, RMSE, Sobel, Laplacian)
  4. Bit-Exact Unmasked Invariance (Delta_max == 0.0)
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

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.reference_scoring.scorer import MaskAwareReferenceScorer
from diffusers import StableDiffusionInpaintPipeline


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


def evaluate_checkpoint_single(
    ckpt_path: Path,
    pipe: StableDiffusionInpaintPipeline,
    manager: DecoupledIdentityCrossAttentionManager,
    diff_arcface: DifferentiableArcFaceBackbone,
    target_rgb: np.ndarray,
    mask: np.ndarray,
    crop_coords: Tuple[int, int, int, int],
    pil_img: Image.Image,
    pil_mask: Image.Image,
    transform_info: Dict[str, Any],
    t_fused: torch.Tensor,
    e_ref_mean: torch.Tensor,
    e_target: torch.Tensor,
    e_diff: torch.Tensor,
    device: torch.device,
    seed: int = 42,
    steps: int = 50,
    guidance_scale: float = 7.5,
    lambda_scale: float = 0.10
) -> Dict[str, Any]:
    ckpt_file = Path(ckpt_path).resolve()
    assert ckpt_file.exists(), f"Checkpoint not found: {ckpt_file}"
    ckpt_data = manager.load_checkpoint(str(ckpt_file))
    manager.to(device=device, dtype=pipe.unet.dtype)
    step_num = ckpt_data.get("step", "unknown")
    manager.set_scale(lambda_scale)
    manager.set_identity_tokens(t_fused.to(device=device, dtype=pipe.unet.dtype))

    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    gen = torch.Generator(device="cpu").manual_seed(seed) if device.type == "cpu" else torch.Generator(device=device).manual_seed(seed)

    t0 = time.time()
    diff_output = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=gen
    ).images[0]
    gen_time = time.time() - t0

    padded_cand = np.array(diff_output, dtype=np.uint8)
    final_recon = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_cand,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )

    # Invariance check
    passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, final_recon, mask)

    # Masked crops
    min_y, max_y, min_x, max_x = crop_coords
    mask_bool = (mask > 0.5)
    crop_gt = target_rgb[min_y:max_y, min_x:max_x]
    crop_recon = final_recon[min_y:max_y, min_x:max_x]

    # Metrics
    mae_diff = np.abs(final_recon[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
    recon_mae = float(np.mean(mae_diff))
    recon_rmse = float(np.sqrt(np.mean(mae_diff**2)))
    recon_psnr = float(20 * np.log10(255.0 / (recon_rmse + 1e-9)))
    recon_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_recon, channel_axis=2))
    recon_sobel = compute_sobel_error(target_rgb, final_recon, mask)
    recon_lap = compute_laplacian_error(target_rgb, final_recon, mask)

    def prep_112(rgb_img):
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    with torch.no_grad():
        e_recon = diff_arcface.extract_embedding(prep_112(final_recon))
        cos_ref = float(F.cosine_similarity(e_recon, e_ref_mean, dim=-1).item())
        cos_tgt = float(F.cosine_similarity(e_recon, e_target, dim=-1).item())
        cos_diff = float(F.cosine_similarity(e_recon, e_diff, dim=-1).item())
        biometric_gap = cos_ref - cos_diff

    return {
        "checkpoint_path": str(ckpt_file),
        "checkpoint_name": ckpt_file.name,
        "step": step_num,
        "runtime_seconds": gen_time,
        "invariance": {
            "passed": passed_inv,
            "max_difference": max_diff,
            "violated_pixels": violated_pixels
        },
        "biometrics": {
            "cos_recon_vs_ref_mean": cos_ref,
            "cos_recon_vs_target_gt": cos_tgt,
            "cos_recon_vs_diff_id": cos_diff,
            "biometric_identity_gap": biometric_gap
        },
        "reconstruction": {
            "psnr": recon_psnr,
            "ssim": recon_ssim,
            "mae": recon_mae,
            "rmse": recon_rmse,
            "sobel_error": recon_sobel,
            "laplacian_error": recon_lap
        },
        "final_image": final_recon,
        "crop_image": crop_recon
    }


def resolve_eval_asset(candidates: List[Path], desc: str) -> Path:
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    checked_str = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Missing canonical evaluation asset for {desc}!\n"
        f"Checked the following locations:\n{checked_str}"
    )


def run_evaluation(
    checkpoints: List[str],
    device_str: str = "auto",
    output_dir: str = "evaluation/reports",
    test_inputs_only: bool = False,
    smoke_test: bool = False,
    steps: int = 50
):
    print("=" * 85)
    print("CHECKPOINT EVALUATION & PROGRESSION BENCHMARK WORKFLOW")
    print("=" * 85)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"Evaluation Compute Device: {device}")

    # Resolve standard evaluation assets (CelebA-HQ 7367 with 25% canonical occlusion)
    target_path = resolve_eval_asset([
        PROJECT_ROOT / "data/processed/dataset_cache/7367.jpg",
        PROJECT_ROOT / "data/raw/CelebA-HQ-img/7367.jpg"
    ], "Target Image (CelebA-HQ 7367)")

    mask_path = resolve_eval_asset([
        PROJECT_ROOT / "data/masks/canonical_eval_mask_25pct.png",
        PROJECT_ROOT / "data/masks/02_mask.png"
    ], "Canonical 25% Face Occlusion Mask")

    ref_a_path = resolve_eval_asset([
        PROJECT_ROOT / "data/processed/dataset_cache/13401.jpg",
        PROJECT_ROOT / "data/raw/CelebA-HQ-img/13401.jpg"
    ], "Reference A (CelebA-HQ 13401, Identity 3111)")

    ref_b_path = resolve_eval_asset([
        PROJECT_ROOT / "data/processed/dataset_cache/23058.jpg",
        PROJECT_ROOT / "data/raw/CelebA-HQ-img/23058.jpg"
    ], "Reference B (CelebA-HQ 23058, Identity 3111)")

    ref_c_path = resolve_eval_asset([
        PROJECT_ROOT / "data/processed/dataset_cache/23414.jpg",
        PROJECT_ROOT / "data/raw/CelebA-HQ-img/23414.jpg"
    ], "Reference C (CelebA-HQ 23414, Identity 3111)")

    diff_id_path = resolve_eval_asset([
        PROJECT_ROOT / "data/processed/dataset_cache/3616.jpg",
        PROJECT_ROOT / "data/raw/CelebA-HQ-img/3616.jpg"
    ], "Different Identity Impostor (CelebA-HQ 3616, Identity 3616)")

    target_bgr = cv2.imread(str(target_path))
    assert target_bgr is not None, f"Failed to load target image: {target_path}"
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask_gray is not None, f"Failed to load mask image: {mask_path}"
    mask = (mask_gray > 127).astype(np.float32)

    ref_a_bgr = cv2.imread(str(ref_a_path))
    assert ref_a_bgr is not None, f"Failed to load reference A: {ref_a_path}"
    ref_a_rgb = cv2.cvtColor(ref_a_bgr, cv2.COLOR_BGR2RGB)

    ref_b_bgr = cv2.imread(str(ref_b_path))
    assert ref_b_bgr is not None, f"Failed to load reference B: {ref_b_path}"
    ref_b_rgb = cv2.cvtColor(ref_b_bgr, cv2.COLOR_BGR2RGB)

    ref_c_bgr = cv2.imread(str(ref_c_path))
    assert ref_c_bgr is not None, f"Failed to load reference C: {ref_c_path}"
    ref_c_rgb = cv2.cvtColor(ref_c_bgr, cv2.COLOR_BGR2RGB)

    diff_id_bgr = cv2.imread(str(diff_id_path))
    assert diff_id_bgr is not None, f"Failed to load diff-ID impostor: {diff_id_path}"
    diff_id_rgb = cv2.cvtColor(diff_id_bgr, cv2.COLOR_BGR2RGB)

    h, w = target_rgb.shape[:2]
    num_masked_pixels = int(mask.sum())
    mask_bool = (mask > 0.5)
    ys, xs = np.where(mask_bool)
    crop_coords = (int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs)))

    print("\nCanonical Evaluation Assets Verified:")
    print(f"  Target Image:  {target_path} ({w}x{h})")
    print(f"  Mask:          {mask_path} ({num_masked_pixels} masked pixels, {num_masked_pixels/(h*w)*100:.2f}%)")
    print(f"  Reference A:   {ref_a_path}")
    print(f"  Reference B:   {ref_b_path}")
    print(f"  Reference C:   {ref_c_path}")
    print(f"  Diff-ID:       {diff_id_path}")

    # If test_inputs_only is requested, validate checkpoints and exit early without running diffusion
    if test_inputs_only:
        print("\n[VALIDATION TEST: Inputs & Checkpoint Deserialization]")
        all_ok = True
        for ckpt_p_str in checkpoints:
            ckpt_p = Path(ckpt_p_str)
            if not ckpt_p.is_absolute():
                ckpt_p = (PROJECT_ROOT / ckpt_p).resolve()
            if not ckpt_p.exists():
                print(f"  [MISSING ON LOCAL FILESYSTEM] {ckpt_p_str}")
                print(f"    -> Must be generated or downloaded from cloud training output before running full evaluation.")
                all_ok = False
            else:
                try:
                    ckpt_data = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
                    step = ckpt_data.get("step", "unknown")
                    sd = ckpt_data.get("state_dict", {})
                    print(f"  [PASS] Successfully loaded {ckpt_p_str}: step={step}, {len(sd)} state_dict keys.")
                except Exception as e:
                    print(f"  [FAIL] Error loading {ckpt_p_str}: {e}")
                    all_ok = False
        print("=" * 85)
        return all_ok

    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(target_rgb, mask, target_size=(512, 512))
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    def prep_img(rgb_img, size=(256, 256)):
        resized = cv2.resize(rgb_img, size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    def prep_112(rgb_img):
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    diff_arcface = DifferentiableArcFaceBackbone("models/pretrained/w600k_r50.onnx").to(device)
    diff_arcface.eval()
    arcface = RegionalIdentityEncoder(token_dim=512)
    arcface.eval()
    phase4 = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(device)
    phase4.eval()

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        tok_A = arcface(prep_img(ref_a_rgb))["spatial_tokens"].to(device)
        tok_B = arcface(prep_img(ref_b_rgb))["spatial_tokens"].to(device)
        tok_C = arcface(prep_img(ref_c_rgb))["spatial_tokens"].to(device)
        w_multi = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32, device=device)

        t_fused = phase4(
            target_tokens=None,
            target_mask=mask_tensor_256,
            reference_tokens_list=[tok_A, tok_B, tok_C],
            reference_weights=w_multi
        )["fused_tokens"]

        e_ref_A = diff_arcface.extract_embedding(prep_112(ref_a_rgb))
        e_ref_B = diff_arcface.extract_embedding(prep_112(ref_b_rgb))
        e_ref_C = diff_arcface.extract_embedding(prep_112(ref_c_rgb))
        e_ref_mean = F.normalize((e_ref_A + e_ref_B + e_ref_C) / 3.0, p=2, dim=-1)
        e_diff = diff_arcface.extract_embedding(prep_112(diff_id_rgb))
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))

    # Load SD inpainting pipeline
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
    load_path = str(list(cache_base.iterdir())[0]) if cache_base.exists() else "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)

    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=0.1)
    manager.install(zero_init_v=False)
    manager.to(device=device, dtype=pipe.unet.dtype)

    # Pre-flight Attention Tensor Device & Dtype Alignment Verification
    validate_attention_device_alignment(manager, device, pipe.unet.dtype)

    if smoke_test:
        print("[SMOKE TEST MODE ACTIVE: Evaluating first checkpoint with 1 step to verify GPU execution]")
        checkpoints = checkpoints[:1]
        steps = 1

    results = []
    print(f"\nEvaluating {len(checkpoints)} checkpoints across the canonical evaluation benchmark...\n")

    for ckpt_p_str in checkpoints:
        ckpt_p = Path(ckpt_p_str)
        if not ckpt_p.is_absolute():
            ckpt_p = (PROJECT_ROOT / ckpt_p).resolve()
        if not ckpt_p.exists():
            print(f"Skipping non-existent checkpoint: {ckpt_p}")
            continue
        print(f"Evaluating: {ckpt_p.name} ...")
        res = evaluate_checkpoint_single(
            ckpt_path=ckpt_p,
            pipe=pipe,
            manager=manager,
            diff_arcface=diff_arcface,
            target_rgb=target_rgb,
            mask=mask,
            crop_coords=crop_coords,
            pil_img=pil_img,
            pil_mask=pil_mask,
            transform_info=transform_info,
            t_fused=t_fused,
            e_ref_mean=e_ref_mean,
            e_target=e_target,
            e_diff=e_diff,
            device=device,
            steps=steps
        )
        results.append(res)
        print(f"  Step {res['step']} -> cos(Ref): {res['biometrics']['cos_recon_vs_ref_mean']:.4f} | Gap: {res['biometrics']['biometric_identity_gap']:+.4f} | PSNR: {res['reconstruction']['psnr']:.2f} dB | SSIM: {res['reconstruction']['ssim']:.4f}")

    # Print Formatted Comparison Table
    print("\n" + "=" * 95)
    print(f"| {'Checkpoint':<18} | {'Step':<6} | {'Sim(Ref)':<10} | {'Sim(GT)':<10} | {'Sim(Diff)':<10} | {'Bio Gap':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} |")
    print("=" * 95)
    for r in results:
        b = r["biometrics"]
        q = r["reconstruction"]
        print(f"| {r['checkpoint_name']:<18} | {str(r['step']):<6} | {b['cos_recon_vs_ref_mean']:<10.4f} | {b['cos_recon_vs_target_gt']:<10.4f} | {b['cos_recon_vs_diff_id']:<10.4f} | {b['biometric_identity_gap']:<+10.4f} | {q['psnr']:<10.2f} | {q['ssim']:<8.4f} |")
    print("=" * 95)

    # Save JSON report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = [{k: v for k, v in r.items() if k not in ["final_image", "crop_image"]} for r in results]
    report_file = out_dir / "checkpoints_progression_evaluation.json"
    with open(report_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved evaluation JSON report to: {report_file}")
    return results


def validate_attention_device_alignment(manager: DecoupledIdentityCrossAttentionManager, device: torch.device, dtype: torch.dtype):
    """Validates that query, ip_key, and ip_value are strictly on the same device and compatible dtype."""
    print("=" * 85)
    print("PRE-FLIGHT ATTENTION TENSOR DEVICE & DTYPE ALIGNMENT VERIFICATION")
    print("=" * 85)

    # Normalize device (e.g. resolve generic "cuda" to concrete active "cuda:0")
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    def match_device(d1: torch.device, d2: torch.device) -> bool:
        if d1.type != d2.type:
            return False
        if d1.index is not None and d2.index is not None and d1.index != d2.index:
            return False
        return True

    sample_proc = next(iter(manager.processors.values()))
    heads = sample_proc.heads
    head_dim = sample_proc.hidden_size // heads
    batch_size = 2
    dummy_query = torch.randn(batch_size, heads, 4096, head_dim, device=device, dtype=dtype)
    dummy_tokens = torch.randn(batch_size, 256, sample_proc.id_dim, device=device, dtype=dtype)

    # Ensure processor projection weights match device and dtype
    if not match_device(sample_proc.to_k_id.weight.device, device) or sample_proc.to_k_id.weight.dtype != dtype:
        sample_proc.to_k_id.to(device=device, dtype=dtype)
        sample_proc.to_v_id.to(device=device, dtype=dtype)

    ip_k = sample_proc.to_k_id(dummy_tokens).view(batch_size, -1, heads, head_dim).transpose(1, 2)
    ip_v = sample_proc.to_v_id(dummy_tokens).view(batch_size, -1, heads, head_dim).transpose(1, 2)

    attn_res = F.scaled_dot_product_attention(dummy_query, ip_k, ip_v)

    print(f"  Target Compute Device:   {device} ({dtype})")
    print(f"  Sample Query Tensor:     device={dummy_query.device}, dtype={dummy_query.dtype}, shape={list(dummy_query.shape)}")
    print(f"  Identity Key Tensor:     device={ip_k.device}, dtype={ip_k.dtype}, shape={list(ip_k.shape)}")
    print(f"  Identity Value Tensor:   device={ip_v.device}, dtype={ip_v.dtype}, shape={list(ip_v.shape)}")
    print(f"  Attention Result Tensor: device={attn_res.device}, dtype={attn_res.dtype}, shape={list(attn_res.shape)}")

    assert dummy_query.device == ip_k.device == ip_v.device == attn_res.device, (
        f"Device mismatch among attention tensors: query={dummy_query.device}, k={ip_k.device}, v={ip_v.device}, out={attn_res.device}"
    )
    assert dummy_query.dtype == ip_k.dtype == ip_v.dtype == attn_res.dtype, (
        f"Dtype mismatch among attention tensors: query={dummy_query.dtype}, k={ip_k.dtype}, v={ip_v.dtype}, out={attn_res.dtype}"
    )
    assert match_device(attn_res.device, device), (
        f"Attention output device {attn_res.device} does not match target compute device {device}!"
    )
    assert torch.isfinite(attn_res).all(), "Attention test computation produced non-finite values!"

    print(f"  Scaled Dot-Product:      SUCCESS (output shape: {list(attn_res.shape)}, device={attn_res.device})")
    print("  [PASS] Attention tensors are strictly co-located on identical device and dtype.")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Inpainting Checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "checkpoints/baselines/step_150_best/step_150.pt",
            "checkpoints/cloud_training/step_500.pt"
        ],
        help="List of checkpoint paths to evaluate (default: step_150_best and step_500)"
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="evaluation/reports")
    parser.add_argument("--test_inputs_only", action="store_true", help="Validate evaluation inputs and checkpoint availability without running diffusion")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-step smoke test on first checkpoint to prove device alignment")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM inference steps (default: 50)")
    args = parser.parse_args()

    run_evaluation(
        checkpoints=args.checkpoints,
        device_str=args.device,
        output_dir=args.output_dir,
        test_inputs_only=args.test_inputs_only,
        smoke_test=args.smoke_test,
        steps=args.steps
    )


if __name__ == "__main__":
    main()
