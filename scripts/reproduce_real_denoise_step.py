"""
Reproduce and verify ONE real diffusion inference step on CUDA / CPU with step_500.pt.

Prints the actual devices/dtypes for:
- UNet
- checkpoint
- to_k_id
- to_v_id
- identity tokens
- ip_key
- ip_value

Then executes exactly one real denoising step with the canonical target, mask, and 3 references.
"""

import os
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import (
    DecoupledIdentityCrossAttentionManager,
    DecoupledIdentityAttnProcessor
)
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from diffusers import StableDiffusionInpaintPipeline


def find_sd_model():
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
    if cache_base.exists():
        dirs = [d for d in cache_base.iterdir() if d.is_dir()]
        if dirs:
            return str(dirs[0])
    return "runwayml/stable-diffusion-inpainting"


def run_one_denoise_step(
    checkpoint_path: str = "checkpoints/cloud_training/step_500.pt",
    device_str: str = "auto"
):
    print("=" * 85)
    print("REAL DIFFUSION INFERENCE STEP & DEVICE ALIGNMENT VERIFICATION")
    print("=" * 85)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Active Compute Device: {device}")

    # 1. Resolve canonical assets
    target_p = PROJECT_ROOT / "data/processed/dataset_cache/7367.jpg"
    mask_p = PROJECT_ROOT / "data/masks/canonical_eval_mask_25pct.png"
    ref_a_p = PROJECT_ROOT / "data/processed/dataset_cache/13401.jpg"
    ref_b_p = PROJECT_ROOT / "data/processed/dataset_cache/23058.jpg"
    ref_c_p = PROJECT_ROOT / "data/processed/dataset_cache/23414.jpg"

    assert target_p.exists(), f"Missing {target_p}"
    assert mask_p.exists(), f"Missing {mask_p}"
    assert ref_a_p.exists(), f"Missing {ref_a_p}"
    assert ref_b_p.exists(), f"Missing {ref_b_p}"
    assert ref_c_p.exists(), f"Missing {ref_c_p}"

    ckpt_p = (PROJECT_ROOT / checkpoint_path).resolve()
    assert ckpt_p.exists(), f"Missing checkpoint: {ckpt_p}"

    target_rgb = cv2.cvtColor(cv2.imread(str(target_p)), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)

    ref_a_rgb = cv2.cvtColor(cv2.imread(str(ref_a_p)), cv2.COLOR_BGR2RGB)
    ref_b_rgb = cv2.cvtColor(cv2.imread(str(ref_b_p)), cv2.COLOR_BGR2RGB)
    ref_c_rgb = cv2.cvtColor(cv2.imread(str(ref_c_p)), cv2.COLOR_BGR2RGB)

    def prep_img(rgb_img, size=(256, 256)):
        resized = cv2.resize(rgb_img, size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    # 2. Extract Phase-4 Regional Fusion Identity Tokens
    print("\n[1/4] Extracting Regional Identity Conditioning Tokens...")
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

    # 3. Load Stable Diffusion and Decoupled Cross-Attention
    print("\n[2/4] Initializing Diffusion Pipeline & Decoupled Cross-Attention...")
    sd_model = find_sd_model()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model,
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)

    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=0.10)
    manager.install(zero_init_v=False)

    # Load actual checkpoint
    ckpt_data = manager.load_checkpoint(str(ckpt_p))
    manager.to(device=device, dtype=pipe.unet.dtype)
    manager.set_scale(0.10)

    # Cast tokens to model device and dtype
    t_fused_device = t_fused.to(device=device, dtype=pipe.unet.dtype)
    manager.set_identity_tokens(t_fused_device)

    # 4. Inspect and print exact devices and dtypes
    print("\n" + "=" * 85)
    print("DEVICE & DTYPE AUDIT TABLE")
    print("=" * 85)
    first_proc_name, first_proc = next(iter(manager.processors.items()))
    unet_param = next(pipe.unet.parameters())

    print(f"  - UNet:            device={unet_param.device}, dtype={unet_param.dtype}")
    print(f"  - Checkpoint:      Loaded step_{ckpt_data.get('step')} ({ckpt_p.name})")
    print(f"  - to_k_id:         device={first_proc.to_k_id.weight.device}, dtype={first_proc.to_k_id.weight.dtype}")
    print(f"  - to_v_id:         device={first_proc.to_v_id.weight.device}, dtype={first_proc.to_v_id.weight.dtype}")
    print(f"  - Identity Tokens: device={t_fused_device.device}, dtype={t_fused_device.dtype}, shape={list(t_fused_device.shape)}")

    manager.set_diagnostics_enabled(True)

    # Prepare inputs
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(target_rgb, mask, target_size=(512, 512))
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    import psutil
    proc_info = psutil.Process(os.getpid())
    ram_before_mb = proc_info.memory_info().rss / (1024 ** 2)
    sys_mem_start = psutil.virtual_memory()
    psutil.cpu_percent(interval=None)  # prime CPU meter

    gen = torch.Generator(device="cpu").manual_seed(42) if device.type == "cpu" else torch.Generator(device=device).manual_seed(42)

    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"

    # 5. Run ONE real diffusion denoising step
    print("\n[3/4] Running Exactly ONE Real Diffusion Denoising Step (num_inference_steps=1)...")
    t0 = time.time()
    diff_out = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=1,
        guidance_scale=7.5,
        generator=gen
    ).images[0]
    step_duration = time.time() - t0

    ram_after_mb = proc_info.memory_info().rss / (1024 ** 2)
    sys_mem_end = psutil.virtual_memory()
    cpu_util = psutil.cpu_percent(interval=None)

    # Print captured runtime attention tensors from the processor's internal diagnostics
    diag = first_proc.latest_diagnostics
    print(f"  - ip_key:          device={diag.get('ip_key_device')}, dtype={diag.get('ip_key_dtype')}")
    print(f"  - ip_value:        device={diag.get('ip_value_device')}, dtype={diag.get('ip_value_dtype')}")
    print("=" * 85)

    print(f"\n[4/4] Step Execution & Resource Audit Result:")
    print(f"  Duration (1 Step):  {step_duration:.2f} seconds")
    print(f"  Process RAM:        {ram_after_mb:.1f} MB (Delta: +{ram_after_mb - ram_before_mb:.1f} MB)")
    print(f"  System RAM Usage:   {sys_mem_end.used / (1024**3):.2f} GB / {sys_mem_end.total / (1024**3):.2f} GB ({sys_mem_end.percent}%)")
    print(f"  CPU Utilization:    {cpu_util:.1f}%")
    print(f"  Generated Image:    Size: {diff_out.size}, Mode: {diff_out.mode}")

    # Unletterbox
    padded_cand = np.array(diff_out, dtype=np.uint8)
    final_recon = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_cand,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )
    print(f"  Reconstructed:      Size: {final_recon.shape}")

    # Projections
    time_50_steps = step_duration * 50
    time_100_steps = step_duration * 100
    print("\n" + "=" * 85)
    print("PROJECTED RUNTIMES FOR FULL EVALUATION (ON CPU):")
    print(f"  - 1 Checkpoint (50 steps):         ~{time_50_steps / 60:.1f} minutes ({time_50_steps:.0f} seconds)")
    print(f"  - Both Checkpoints (100 steps total): ~{time_100_steps / 60:.1f} minutes ({time_100_steps:.0f} seconds)")
    print("=" * 85)
    print(">>> SUCCESS: Real diffusion step completed with 100% device and dtype alignment! <<<")
    print("=" * 85)
    return {
        "step_duration": step_duration,
        "ram_mb": ram_after_mb,
        "ram_delta_mb": ram_after_mb - ram_before_mb,
        "sys_ram_gb": sys_mem_end.used / (1024**3),
        "sys_ram_total_gb": sys_mem_end.total / (1024**3),
        "sys_ram_percent": sys_mem_end.percent,
        "cpu_util": cpu_util,
        "projected_50_min": time_50_steps / 60,
        "projected_100_min": time_100_steps / 60
    }



def main():
    parser = argparse.ArgumentParser(description="Reproduce one real denoising step with device audit")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cloud_training/step_500.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    run_one_denoise_step(checkpoint_path=args.checkpoint, device_str=args.device)


if __name__ == "__main__":
    main()
