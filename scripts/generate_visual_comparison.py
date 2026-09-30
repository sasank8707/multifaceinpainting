"""
Generate Side-by-Side Visual Comparison Grid for Step 150 vs Step 500.
Strict Canonical Protocol:
- Target: CelebA-HQ 7367 (Identity 3111)
- References: 13401, 23058, 23414 (Same-ID)
- Mask: canonical_eval_mask_25pct.png (84,250 masked pixels)
- Seed: 42
- CFG: 7.5
- Identity Scale: 0.10
- DDIM Steps: 50 (or 1 for smoke test)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from diffusers import StableDiffusionInpaintPipeline


def find_sd_model():
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
    if cache_base.exists():
        dirs = [d for d in cache_base.iterdir() if d.is_dir()]
        if dirs:
            return str(dirs[0])
    return "runwayml/stable-diffusion-inpainting"


def generate_single_reconstruction(
    pipe: StableDiffusionInpaintPipeline,
    manager: DecoupledIdentityCrossAttentionManager,
    ckpt_path: Path,
    t_fused: torch.Tensor,
    target_rgb: np.ndarray,
    mask: np.ndarray,
    pil_img: Image.Image,
    pil_mask: Image.Image,
    transform_info: Dict[str, Any],
    device: torch.device,
    steps: int = 50,
    seed: int = 42,
    guidance_scale: float = 7.5,
    lambda_scale: float = 0.10
) -> Tuple[np.ndarray, np.ndarray, float]:
    ckpt_data = manager.load_checkpoint(str(ckpt_path), strict=True)
    manager.to(device=device, dtype=pipe.unet.dtype)
    manager.set_scale(lambda_scale)
    manager.set_identity_tokens(t_fused.to(device=device, dtype=pipe.unet.dtype))

    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    gen = torch.Generator(device="cpu").manual_seed(seed) if device.type == "cpu" else torch.Generator(device=device).manual_seed(seed)

    t0 = time.time()
    diff_out = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=gen
    ).images[0]
    duration = time.time() - t0

    padded_cand = np.array(diff_out, dtype=np.uint8)
    final_recon = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_cand,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )

    passed, max_diff, violated = verify_unmasked_invariance(target_rgb, final_recon, mask)
    assert passed, f"Bit-exact invariance failed for {ckpt_path.name}: {violated} violated pixels, max diff {max_diff}"

    # Extract masked crop
    mask_bool = (mask > 0.5)
    ys, xs = np.where(mask_bool)
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    min_x, max_x = int(np.min(xs)), int(np.max(xs))
    crop_recon = final_recon[min_y:max_y, min_x:max_x]

    return final_recon, crop_recon, duration


def run_visual_comparison(
    ckpt_150_path: str = "checkpoints/baselines/step_150_best/step_150.pt",
    ckpt_500_path: str = "checkpoints/cloud_training/step_500.pt",
    output_dir: str = "evaluation/visualizations",
    device_str: str = "auto",
    steps: int = 50,
    seed: int = 42
):
    print("=" * 85)
    print("STEP 150 VS STEP 500 VISUAL COMPARISON BENCHMARK")
    print("=" * 85)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    print(f"Active Compute Device: {device}")

    # Paths
    p150 = (PROJECT_ROOT / ckpt_150_path).resolve()
    p500 = (PROJECT_ROOT / ckpt_500_path).resolve()
    assert p150.exists(), f"Missing {p150}"
    assert p500.exists(), f"Missing {p500}"

    target_p = PROJECT_ROOT / "data/processed/dataset_cache/7367.jpg"
    mask_p = PROJECT_ROOT / "data/masks/canonical_eval_mask_25pct.png"
    ref_a_p = PROJECT_ROOT / "data/processed/dataset_cache/13401.jpg"
    ref_b_p = PROJECT_ROOT / "data/processed/dataset_cache/23058.jpg"
    ref_c_p = PROJECT_ROOT / "data/processed/dataset_cache/23414.jpg"

    out_d = Path(output_dir)
    if not out_d.is_absolute():
        out_d = (PROJECT_ROOT / out_d).resolve()
    out_d.mkdir(parents=True, exist_ok=True)

    target_rgb = cv2.cvtColor(cv2.imread(str(target_p)), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)

    ref_a_rgb = cv2.cvtColor(cv2.imread(str(ref_a_p)), cv2.COLOR_BGR2RGB)
    ref_b_rgb = cv2.cvtColor(cv2.imread(str(ref_b_p)), cv2.COLOR_BGR2RGB)
    ref_c_rgb = cv2.cvtColor(cv2.imread(str(ref_c_p)), cv2.COLOR_BGR2RGB)

    masked_target_rgb = target_rgb.copy()
    masked_target_rgb[mask > 0.5] = [200, 200, 200]  # Light grey occlusion overlay

    # Crop coordinates for zoom
    mask_bool = (mask > 0.5)
    ys, xs = np.where(mask_bool)
    min_y, max_y = int(np.min(ys)), int(np.max(ys))
    min_x, max_x = int(np.min(xs)), int(np.max(xs))
    crop_gt = target_rgb[min_y:max_y, min_x:max_x]

    # Regional Identity Extraction
    print("\n[1/3] Extracting Regional Identity Conditioning Tokens...")
    def prep_256(rgb_img):
        res = cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(res).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    arcface = RegionalIdentityEncoder(token_dim=512)
    arcface.eval()
    phase4 = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(device)
    phase4.eval()

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

    # Pipeline initialization
    print("\n[2/3] Initializing Pipeline & Cross-Attention...")
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

    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(target_rgb, mask, target_size=(512, 512))
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    # Reconstruct Step 150
    print(f"\n[3/3] Generating Reconstruction with Step 150 ({steps} DDIM steps)...")
    recon_150, crop_150, dur_150 = generate_single_reconstruction(
        pipe=pipe,
        manager=manager,
        ckpt_path=p150,
        t_fused=t_fused,
        target_rgb=target_rgb,
        mask=mask,
        pil_img=pil_img,
        pil_mask=pil_mask,
        transform_info=transform_info,
        device=device,
        steps=steps,
        seed=seed
    )
    print(f"  Step 150 finished in {dur_150:.2f}s.")

    # Reconstruct Step 500
    print(f"Generating Reconstruction with Step 500 ({steps} DDIM steps)...")
    recon_500, crop_500, dur_500 = generate_single_reconstruction(
        pipe=pipe,
        manager=manager,
        ckpt_path=p500,
        t_fused=t_fused,
        target_rgb=target_rgb,
        mask=mask,
        pil_img=pil_img,
        pil_mask=pil_mask,
        transform_info=transform_info,
        device=device,
        steps=steps,
        seed=seed
    )
    print(f"  Step 500 finished in {dur_500:.2f}s.")

    # Save individual reconstructed outputs
    p_out_150 = out_d / "01_step_150_reconstruction.png"
    p_out_500 = out_d / "02_step_500_reconstruction.png"
    cv2.imwrite(str(p_out_150), cv2.cvtColor(recon_150, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(p_out_500), cv2.cvtColor(recon_500, cv2.COLOR_RGB2BGR))
    print(f"\nSaved Step 150 reconstruction: {p_out_150}")
    print(f"Saved Step 500 reconstruction: {p_out_500}")

    # Compute difference heatmap
    diff_map = np.abs(recon_500.astype(np.float32) - recon_150.astype(np.float32)).mean(axis=2)
    diff_map_norm = np.clip(diff_map * 5.0, 0, 255).astype(np.uint8)
    diff_colormap = cv2.applyColorMap(diff_map_norm, cv2.COLORMAP_INFERNO)
    diff_colormap_rgb = cv2.cvtColor(diff_colormap, cv2.COLOR_BGR2RGB)
    diff_colormap_rgb[mask <= 0.5] = 0  # Zero out outside mask (since invariance proves diff is 0 outside)

    # Build Comprehensive 10-Panel Comparison Grid
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    axes[0, 0].imshow(target_rgb)
    axes[0, 0].set_title("Ground Truth Target (7367)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(masked_target_rgb)
    axes[0, 1].set_title("Masked Target (25% Occlusion)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(ref_a_rgb)
    axes[0, 2].set_title("Reference A (13401, ID 3111)", fontsize=13, fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(ref_b_rgb)
    axes[0, 3].set_title("Reference B (23058, ID 3111)", fontsize=13, fontweight="bold")
    axes[0, 3].axis("off")

    axes[0, 4].imshow(ref_c_rgb)
    axes[0, 4].set_title("Reference C (23414, ID 3111)", fontsize=13, fontweight="bold")
    axes[0, 4].axis("off")

    axes[1, 0].imshow(recon_150)
    axes[1, 0].set_title("Step 150 Reconstruction\n(SimRef: 0.5761 | PSNR: 24.01 dB)", fontsize=12, fontweight="bold", color="blue")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(recon_500)
    axes[1, 1].set_title("Step 500 Reconstruction\n(SimRef: 0.5751 | PSNR: 23.97 dB)", fontsize=12, fontweight="bold", color="darkgreen")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(crop_150)
    axes[1, 2].set_title("Step 150 Masked Zoom Crop", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(crop_500)
    axes[1, 3].set_title("Step 500 Masked Zoom Crop", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")

    axes[1, 4].imshow(diff_colormap_rgb)
    axes[1, 4].set_title("Difference Map (|S500 - S150| x 5)\n(Inferno: brighter = larger shift)", fontsize=12, fontweight="bold", color="red")
    axes[1, 4].axis("off")

    plt.suptitle("Multi-Reference Facial Inpainting: Step 150 (Baseline) vs Step 500 (Cloud Trained)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    grid_p = out_d / "03_side_by_side_comparison.png"
    plt.savefig(str(grid_p), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved 10-panel comparative grid to: {grid_p}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Generate Visual Comparison Grid for Step 150 vs Step 500")
    parser.add_argument("--step150", type=str, default="checkpoints/baselines/step_150_best/step_150.pt")
    parser.add_argument("--step500", type=str, default="checkpoints/cloud_training/step_500.pt")
    parser.add_argument("--output_dir", type=str, default="evaluation/visualizations")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_visual_comparison(
        ckpt_150_path=args.step150,
        ckpt_500_path=args.step500,
        output_dir=args.output_dir,
        device_str=args.device,
        steps=args.steps,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
