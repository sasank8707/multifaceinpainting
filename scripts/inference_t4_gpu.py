"""
NVIDIA T4 / CUDA High-Performance Inference Pipeline
Multi-Reference Facial Inpainting with Decoupled Identity Cross-Attention

Hardware Profile:
- Optimized for NVIDIA T4 (16GB VRAM) on Google Colab and Kaggle
- Automatic mixed precision (FP16 on CUDA / FP32 on CPU fallback)
- Typical T4 generation time: ~0.5 - 0.7s for 5 DDIM steps (~4.0 GB VRAM)

Pipeline Configuration (Matches Local Experiments Exactly):
- CFG: 7.5
- Identity Coupling Scale (lambda): 0.10
- Scorer: Dynamic MaskAwareReferenceScorer (pose, quality, visibility, overlap)
- Multi-Reference Regional Fusion: 3-reference RegionalIdentityTransformer (Anti-leakage target=None)
- Attention Injection: DecoupledIdentityCrossAttention (step_150.pt)
- Verification: Bit-Exact Invariance + Biometric ArcFace + Visual Reconstruction Metrics
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
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
    """Computes high-frequency edge transition error across the inpainting mask."""
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
    """Computes first-order gradient magnitude error across the inpainting mask."""
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


def resolve_path(path_str: str) -> Path:
    """Resolves relative or absolute paths against project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Check current directory first, then project root
    if p.exists():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def find_sd_inpaint_model() -> str:
    """Locates cached Stable Diffusion Inpainting weights or returns HuggingFace ID."""
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
    if cache_base.exists():
        snapshots = [d for d in cache_base.iterdir() if d.is_dir()]
        if snapshots:
            return str(snapshots[0])
    return "runwayml/stable-diffusion-inpainting"


def run_t4_inference(
    target_path: str,
    mask_path: str,
    ref_paths: List[str],
    checkpoint_path: str = "checkpoints/decoupled_identity_composite_loss/step_150.pt",
    output_dir: str = "user_tests/test1/output_t4",
    steps: int = 5,
    guidance_scale: float = 7.5,
    lambda_scale: float = 0.10,
    seed: int = 42,
    device_str: str = "auto",
    use_fp16: Optional[bool] = None,
    prompt: str = "high quality photorealistic portrait of a young man, natural facial anatomy, natural skin texture"
) -> Dict[str, Any]:
    """
    Executes full multi-reference facial inpainting on CUDA (T4 GPU) or CPU fallback.
    """
    print("=" * 80)
    print("MULTI-REFERENCE FACIAL INPAINTING: CUDA / T4 GPU INFERENCE PIPELINE")
    print("=" * 80)
    start_total_time = time.time()

    # 1. Device and Precision Configuration
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Compute Device: CUDA ({gpu_name}, {gpu_vram:.2f} GB VRAM)")
        torch_dtype = torch.float16 if (use_fp16 is None or use_fp16) else torch.float32
        torch.backends.cudnn.benchmark = True
    else:
        print("Compute Device: CPU (Fallback Mode)")
        torch_dtype = torch.float32

    print(f"Inference Dtype: {torch_dtype}")
    print(f"Diffusion Steps: {steps}, CFG: {guidance_scale}, Lambda: {lambda_scale}, Seed: {seed}")
    print(f"Checkpoint: {checkpoint_path}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Resolve output directory
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Input Image & Binary Mask Validation
    print("\n[1/6] Loading Target Image, Binary Mask, and 3 Reference Images...")
    resolved_target = resolve_path(target_path)
    resolved_mask = resolve_path(mask_path)
    assert resolved_target.exists(), f"Target image not found: {resolved_target}"
    assert resolved_mask.exists(), f"Mask image not found: {resolved_mask}"

    target_bgr = cv2.imread(str(resolved_target))
    assert target_bgr is not None, f"Could not decode target image: {resolved_target}"
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    h, w = target_rgb.shape[:2]

    mask_gray = cv2.imread(str(resolved_mask), cv2.IMREAD_GRAYSCALE)
    assert mask_gray is not None, f"Could not decode mask image: {resolved_mask}"
    if mask_gray.shape != (h, w):
        print(f"Resizing mask from {mask_gray.shape[1]}x{mask_gray.shape[0]} to match target {w}x{h}")
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask_gray > 127).astype(np.float32)
    num_masked_pixels = int(mask.sum())
    mask_ratio = num_masked_pixels / (h * w) * 100.0
    print(f"  Target Canvas: {w}x{h} px")
    print(f"  Mask Area: {num_masked_pixels:,} pixels ({mask_ratio:.2f}% of canvas)")

    assert len(ref_paths) == 3, f"Expected exactly 3 reference images, got {len(ref_paths)}"
    ref_rgbs = []
    for i, rp in enumerate(ref_paths):
        res_p = resolve_path(rp)
        assert res_p.exists(), f"Reference image not found: {res_p}"
        ref_bgr = cv2.imread(str(res_p))
        assert ref_bgr is not None, f"Could not decode ref {res_p}"
        ref_rgbs.append(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
        print(f"  Reference {i+1}: {res_p.name} ({ref_rgbs[-1].shape[1]}x{ref_rgbs[-1].shape[0]} px)")

    # 3. Dynamic Mask-Aware Reference Scoring
    print("\n[2/6] Executing Phase-2 Face Analysis & Phase-3 Dynamic Reference Scoring...")
    face_pipeline = FaceAnalysisPipeline()
    target_analysis = face_pipeline.analyze(target_rgb, mask)
    ref_analyses = [face_pipeline.analyze(r) for r in ref_rgbs]
    ref_names = [Path(rp).stem for rp in ref_paths]

    scorer = MaskAwareReferenceScorer(
        pose_weight=0.30,
        quality_weight=0.25,
        visibility_weight=0.30,
        overlap_weight=0.15,
        top_k=3,
        strict_identity=False
    )
    scoring_report = scorer.score_references(
        target_analysis=target_analysis,
        reference_analyses=ref_analyses,
        target_identity_id="query_target",
        reference_identity_ids=["query_target"] * 3,
        reference_names=ref_names
    )

    comp_0 = scoring_report["components"][0]
    norm_weights = comp_0["normalized_weights"]
    ref_weights = [norm_weights[name] for name in ref_names]
    sum_w = sum(ref_weights)
    ref_weights = [w_i / sum_w for w_i in ref_weights] if sum_w > 0 else [1.0/3, 1.0/3, 1.0/3]

    print("-" * 70)
    print("COMPUTED DYNAMIC REFERENCE WEIGHTS:")
    for i, name in enumerate(ref_names):
        bd = comp_0["score_breakdown"].get(name, {})
        print(f"  {name:8s}: weight={ref_weights[i]:.4f} | Composite={bd.get('composite_score',0):.4f} (Pose={bd.get('pose_score',0):.4f}, Qual={bd.get('quality_score',0):.4f}, Vis={bd.get('visibility_score',0):.4f})")
    print("-" * 70)

    # 4. Regional Identity Extraction & Transformer Fusion
    print("\n[3/6] Extracting Biometric Representations & Regional Transformer Fusion...")
    arcface = RegionalIdentityEncoder(token_dim=512)
    arcface.eval()
    for p in arcface.parameters():
        p.requires_grad = False

    phase4 = RegionalIdentityTransformer(
        token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0
    ).to(device=device)
    phase4.eval()
    for p in phase4.parameters():
        p.requires_grad = False

    diff_arcface = DifferentiableArcFaceBackbone().to(device=device)
    diff_arcface.eval()
    for p in diff_arcface.parameters():
        p.requires_grad = False

    def prep_img(rgb_img, size=(256, 256)):
        resized = cv2.resize(rgb_img, size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

    def prep_112(rgb_img):
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        tok_1 = arcface(prep_img(ref_rgbs[0]))["spatial_tokens"].to(device=device)
        tok_2 = arcface(prep_img(ref_rgbs[1]))["spatial_tokens"].to(device=device)
        tok_3 = arcface(prep_img(ref_rgbs[2]))["spatial_tokens"].to(device=device)

        w_multi = torch.tensor([ref_weights], dtype=torch.float32, device=device)

        # STRICT TARGET ANTI-LEAKAGE: target_tokens is None
        t_fused = phase4(
            target_tokens=None,
            target_mask=mask_tensor_256,
            reference_tokens_list=[tok_1, tok_2, tok_3],
            reference_weights=w_multi
        )["fused_tokens"]

        e_ref1 = diff_arcface.extract_embedding(prep_112(ref_rgbs[0]))
        e_ref2 = diff_arcface.extract_embedding(prep_112(ref_rgbs[1]))
        e_ref3 = diff_arcface.extract_embedding(prep_112(ref_rgbs[2]))
        e_ref_mean = F.normalize((e_ref1 + e_ref2 + e_ref3) / 3.0, p=2, dim=-1)
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))

    print(f"  Fused Identity Tokens Shape: {t_fused.shape} | Token Norm: {t_fused.norm(dim=-1).mean():.4f}")

    # 5. Load Pretrained Diffusion & Decoupled Cross-Attention
    print("\n[4/6] Loading Stable Diffusion Inpainting & Decoupled Identity Cross-Attention...")
    sd_model_id = find_sd_inpaint_model()
    print(f"  Loading Diffusion Pipeline from: {sd_model_id}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=torch_dtype,
        safety_checker=None
    ).to(device)

    # Install Decoupled Identity Cross-Attention Processors
    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=lambda_scale)
    manager.install(zero_init_v=False)

    ckpt_file = resolve_path(checkpoint_path)
    assert ckpt_file.exists(), f"Trained checkpoint not found: {ckpt_file}"
    manager.load_checkpoint(str(ckpt_file))
    manager.to(device=device, dtype=torch_dtype)
    manager.set_scale(lambda_scale)

    # Transfer identity conditioning tokens
    t_fused_cast = t_fused.to(device=device, dtype=torch_dtype)
    manager.set_identity_tokens(t_fused_cast)
    print(f"  Loaded step_150 weights successfully into {len(manager.processors)} cross-attention layers.")

    # 6. AspectRatioPreserver Letterbox & Generative Inpainting
    print("\n[5/6] Executing Aspect-Ratio Preserving Diffusion Inpainting...")
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
        image=target_rgb, mask=mask, target_size=(512, 512)
    )
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    gen = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_diff_start = time.time()

    diff_output = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=gen
    ).images[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_diff = time.time() - t_diff_start
    print(f"  Diffusion inpainting completed in {t_diff:.3f} seconds ({t_diff/steps*1000:.1f} ms/step).")

    # 7. Unletterbox, Strict Invariance Verification & Metrics
    print("\n[6/6] Unletterboxing, Compositing, and Verifying Invariance...")
    padded_cand = np.array(diff_output, dtype=np.uint8)
    final_reconstruction = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_cand,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )

    passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, final_reconstruction, mask)
    print(f"  Bit-Exact Unmasked Invariance: Passed={passed_inv} (Max Delta={max_diff}, Violated Pixels={violated_pixels})")
    assert passed_inv, f"Unmasked invariance violated: {max_diff}"

    # Extract masked crops
    mask_bool = (mask > 0.5)
    ys, xs = np.where(mask_bool)
    min_y, max_y, min_x, max_x = int(np.min(ys)), int(np.max(ys)), int(np.min(xs)), int(np.max(xs))
    crop_gt = target_rgb[min_y:max_y, min_x:max_x]
    crop_recon = final_reconstruction[min_y:max_y, min_x:max_x]

    # Compute Reconstruction Metrics
    mae_diff = np.abs(final_reconstruction[mask_bool].astype(float) - target_rgb[mask_bool].astype(float))
    recon_mae = float(np.mean(mae_diff))
    recon_rmse = float(np.sqrt(np.mean(mae_diff**2)))
    recon_psnr = float(20 * np.log10(255.0 / (recon_rmse + 1e-9)))
    recon_ssim = float(skimage.metrics.structural_similarity(crop_gt, crop_recon, channel_axis=2))
    recon_sobel = compute_sobel_error(target_rgb, final_reconstruction, mask)
    recon_lap = compute_laplacian_error(target_rgb, final_reconstruction, mask)

    # Compute Biometric ArcFace Cosine Similarities
    with torch.no_grad():
        e_recon = diff_arcface.extract_embedding(prep_112(final_reconstruction))
        cos_ref_mean = float(F.cosine_similarity(e_recon, e_ref_mean, dim=-1).item())
        cos_ref1 = float(F.cosine_similarity(e_recon, e_ref1, dim=-1).item())
        cos_ref2 = float(F.cosine_similarity(e_recon, e_ref2, dim=-1).item())
        cos_ref3 = float(F.cosine_similarity(e_recon, e_ref3, dim=-1).item())
        cos_target = float(F.cosine_similarity(e_recon, e_target, dim=-1).item())

    # Save Output Artifacts
    final_recon_path = out_dir / "final_reconstruction.png"
    crop_recon_path = out_dir / "masked_crop.png"
    Image.fromarray(final_reconstruction).save(final_recon_path)
    Image.fromarray(crop_recon).save(crop_recon_path)

    # Generate 5-Panel Side-by-Side Comparison
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(target_rgb)
    axes[0].set_title(f"Target Ground Truth\n({w}x{h})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    masked_vis = target_rgb.copy()
    masked_vis[mask_bool] = [255, 60, 60]
    axes[1].imshow(masked_vis)
    axes[1].set_title(f"Masked Input\n({num_masked_pixels:,} px, {mask_ratio:.1f}%)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(final_reconstruction)
    axes[2].set_title(
        f"Reconstruction (Step 150)\ncos(Ref Mean): {cos_ref_mean:.4f}\nPSNR: {recon_psnr:.2f} dB | SSIM: {recon_ssim:.4f}",
        fontsize=11, fontweight="bold", color="darkgreen"
    )
    axes[2].axis("off")

    ref_tile = np.hstack([
        cv2.resize(ref_rgbs[0], (160, 200)),
        cv2.resize(ref_rgbs[1], (160, 200)),
        cv2.resize(ref_rgbs[2], (160, 200))
    ])
    axes[3].imshow(ref_tile)
    axes[3].set_title(f"References (w={ref_weights[0]:.2f}, {ref_weights[1]:.2f}, {ref_weights[2]:.2f})", fontsize=11)
    axes[3].axis("off")

    crop_comp = np.hstack([cv2.resize(crop_gt, (140, 160)), cv2.resize(crop_recon, (140, 160))])
    axes[4].imshow(crop_comp)
    axes[4].set_title("GT vs Reconstructed Organ", fontsize=11)
    axes[4].axis("off")

    plt.tight_layout()
    comp_path = out_dir / "side_by_side_comparison.png"
    plt.savefig(comp_path, dpi=160)
    plt.close()

    total_time = time.time() - start_total_time

    # Construct Metrics Dictionary
    metrics = {
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "dtype": str(torch_dtype)
        },
        "timing": {
            "diffusion_runtime_seconds": t_diff,
            "diffusion_step_latency_ms": (t_diff / steps) * 1000.0,
            "total_pipeline_runtime_seconds": total_time
        },
        "reference_scoring": {
            "ref_names": ref_names,
            "dynamic_weights": ref_weights,
            "score_breakdown": comp_0.get("score_breakdown", {})
        },
        "unmasked_invariance": {
            "passed": passed_inv,
            "max_difference": max_diff,
            "violated_pixels": violated_pixels
        },
        "biometric_identity": {
            "cos_recon_vs_ref_mean": cos_ref_mean,
            "cos_recon_vs_ref1": cos_ref1,
            "cos_recon_vs_ref2": cos_ref2,
            "cos_recon_vs_ref3": cos_ref3,
            "cos_recon_vs_target_gt": cos_target
        },
        "reconstruction_quality": {
            "psnr": recon_psnr,
            "ssim": recon_ssim,
            "mae": recon_mae,
            "rmse": recon_rmse,
            "sobel_error": recon_sobel,
            "laplacian_error": recon_lap
        }
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("INFERENCE RUN SUMMARY & RESULTS")
    print("=" * 80)
    print(f"Device:               {metrics['hardware']['device_name']} ({metrics['hardware']['dtype']})")
    print(f"Diffusion Runtime:    {t_diff:.3f} s ({t_diff/steps*1000:.1f} ms/step)")
    print(f"Total Pipeline Time:  {total_time:.3f} s")
    print(f"Biometric Cosine Sim: {cos_ref_mean:.4f} (vs Ref Mean), {cos_target:.4f} (vs Target GT)")
    print(f"Reconstruction PSNR:  {recon_psnr:.2f} dB")
    print(f"Reconstruction SSIM:  {recon_ssim:.4f}")
    print(f"Reconstruction MAE:   {recon_mae:.2f}")
    print(f"Bit-Exact Invariance: {'PASSED' if passed_inv else 'FAILED'} (Max Delta: {max_diff})")
    print(f"Output Artifacts:     {out_dir}")
    print("  - final_reconstruction.png")
    print("  - masked_crop.png")
    print("  - side_by_side_comparison.png")
    print("  - metrics.json")
    print("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-Reference Inpainting CUDA/T4 Inference")
    parser.add_argument("--target", type=str, default="user_tests/test1/target.jpg", help="Path to target image")
    parser.add_argument("--mask", type=str, default="user_tests/test1/mask.png", help="Path to binary mask image")
    parser.add_argument("--ref1", type=str, default="user_tests/test1/ref1.jpg", help="Path to reference 1")
    parser.add_argument("--ref2", type=str, default="user_tests/test1/ref2.jpg", help="Path to reference 2")
    parser.add_argument("--ref3", type=str, default="user_tests/test1/ref3.jpg", help="Path to reference 3")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/decoupled_identity_composite_loss/step_150.pt", help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="user_tests/test1/output_t4", help="Output directory")
    parser.add_argument("--steps", type=int, default=5, help="Diffusion inference steps (default: 5)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--lambda_scale", type=float, default=0.10, help="Identity coupling scale (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--fp32", action="store_true", help="Force FP32 precision on CUDA")
    parser.add_argument("--prompt", type=str, default="high quality photorealistic portrait of a young man, natural facial anatomy, natural skin texture")

    args = parser.parse_args()
    use_fp16 = False if args.fp32 else None

    run_t4_inference(
        target_path=args.target,
        mask_path=args.mask,
        ref_paths=[args.ref1, args.ref2, args.ref3],
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        lambda_scale=args.lambda_scale,
        seed=args.seed,
        device_str=args.device,
        use_fp16=use_fp16,
        prompt=args.prompt
    )


if __name__ == "__main__":
    main()
