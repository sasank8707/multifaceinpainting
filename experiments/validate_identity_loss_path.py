"""
Validation and Calibration Suite for Differentiable Identity Loss Path.

Investigates, implements, and validates the end-to-end differentiable identity loss:
L_total = L_diffusion + lambda_id * L_identity

Components:
1. Native Differentiable ArcFace Backbone (130 ops interpreted directly from w600k_r50.onnx).
2. Clean x0 reconstruction from diffusion epsilon prediction:
   x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta) / sqrt(alpha_bar_t)
3. Frozen VAE decoding of x0_pred -> I_pred [1, 3, 512, 512].
4. Differentiable ArcFace extraction: e_pred = ArcFace(I_pred_112).
5. Identity Loss: L_identity = 1 - cosine(e_pred, e_ref).
6. Gradient-flow verification through ArcFace -> VAE -> UNet -> to_k_id / to_v_id.
7. Frozen parameter integrity audit (UNet, VAE, CLIP, ArcFace, Phase 4).
8. Reference consistency & identity gap analysis.
9. Identity loss sanity test across Vanilla, Step 100, Target, and References.
10. Analytical lambda_id sensitivity test (0.01, 0.05, 0.10, 0.25).
11. CPU runtime cost profiling per component.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import psutil
import cv2
import numpy as np
import torch
import torch.nn as nn
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
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler


def main():
    print("=" * 80)
    print("DIFFERENTIABLE IDENTITY LOSS PATH: DESIGN, VALIDATION & CALIBRATION")
    print("=" * 80)
    start_total_time = time.time()

    # Optimal CPU threading
    torch.set_num_threads(4)
    device = torch.device("cpu")

    out_dir = PROJECT_ROOT / "experiments/exp_identity_loss_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load Pretrained Models & Feature Extractors
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Loading Pretrained Stable Diffusion Inpainting & Differentiable ArcFace...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0] if snapshots else "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    ).to(device)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = DDPMScheduler.from_pretrained(load_path, subfolder="scheduler")

    # Native Differentiable ArcFace Backbone
    diff_arcface = DifferentiableArcFaceBackbone("models/pretrained/w600k_r50.onnx").to(device)
    diff_arcface.eval()

    # Regional ArcFace (Phase 4 input)
    arcface = RegionalIdentityEncoder(token_dim=512).to(device)
    arcface.eval()

    # Phase 4 Regional Fusion Transformer
    phase4 = RegionalIdentityTransformer(
        token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0
    ).to(device)
    phase4.eval()

    # Freeze non-target parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    for p in diff_arcface.parameters():
        p.requires_grad = False
    for p in arcface.parameters():
        p.requires_grad = False
    for p in phase4.parameters():
        p.requires_grad = False

    # Attach Decoupled Identity Manager
    pristine_vanilla_processors = dict(unet.attn_processors)
    manager = DecoupledIdentityCrossAttentionManager(unet, id_dim=512, default_scale=0.1)
    manager.original_processors = dict(pristine_vanilla_processors)
    manager.install(zero_init_v=False)

    trainable_params = manager.get_trainable_parameters()
    for p in trainable_params:
        p.requires_grad = True

    print("Architecture loaded and frozen backbones verified.")

    # -------------------------------------------------------------------------
    # 2. Load Controlled Data: Target 7367, Mask, Refs A/B/C, Diff ID 3616
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Loading Controlled Evaluation Data (Target 7367, Mask, Refs A/B/C, Diff ID 3616)...")
    phase7_artifacts = PROJECT_ROOT / "experiments/exp_phase7_first_real_reconstruction/artifacts"
    phase7_data = PROJECT_ROOT / "experiments/exp_phase7_first_real_reconstruction/data"

    target_path = phase7_artifacts / "01_original_target.png"
    mask_path = phase7_artifacts / "02_mask.png"
    ref_a_path = phase7_data / "ref1_13401.jpg"
    ref_b_path = phase7_data / "ref2_23058.jpg"
    ref_c_path = phase7_data / "ref3_23414.jpg"
    diff_id_path = phase7_data / "diff_id_3616.jpg"

    target_rgb = cv2.cvtColor(cv2.imread(str(target_path)), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)

    ref_a_rgb = cv2.cvtColor(cv2.imread(str(ref_a_path)), cv2.COLOR_BGR2RGB)
    ref_b_rgb = cv2.cvtColor(cv2.imread(str(ref_b_path)), cv2.COLOR_BGR2RGB)
    ref_c_rgb = cv2.cvtColor(cv2.imread(str(ref_c_path)), cv2.COLOR_BGR2RGB)
    diff_id_rgb = cv2.cvtColor(cv2.imread(str(diff_id_path)), cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------------------
    # 3. Reference Consistency & Identity Gap Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Computing Reference Identity Consistency & Identity Gap...")
    def prep_arcface_input(rgb_img):
        # Resize to 112x112, convert to [-1, 1] tensor
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        return t.to(device)

    t_in_A = prep_arcface_input(ref_a_rgb)
    t_in_B = prep_arcface_input(ref_b_rgb)
    t_in_C = prep_arcface_input(ref_c_rgb)
    t_in_diff = prep_arcface_input(diff_id_rgb)
    t_in_target = prep_arcface_input(target_rgb)

    with torch.no_grad():
        e_ref_A = diff_arcface.extract_embedding(t_in_A)
        e_ref_B = diff_arcface.extract_embedding(t_in_B)
        e_ref_C = diff_arcface.extract_embedding(t_in_C)
        e_diff = diff_arcface.extract_embedding(t_in_diff)
        e_target = diff_arcface.extract_embedding(t_in_target)

        # Fused reference embedding: normalized mean of the 3 verified references
        e_ref_mean = F.normalize((e_ref_A + e_ref_B + e_ref_C) / 3.0, p=2, dim=-1)

    cos_AB = float(F.cosine_similarity(e_ref_A, e_ref_B, dim=-1).item())
    cos_AC = float(F.cosine_similarity(e_ref_A, e_ref_C, dim=-1).item())
    cos_BC = float(F.cosine_similarity(e_ref_B, e_ref_C, dim=-1).item())
    mean_same_id_cos = (cos_AB + cos_AC + cos_BC) / 3.0

    cos_mean_diff = float(F.cosine_similarity(e_ref_mean, e_diff, dim=-1).item())
    cos_target_mean = float(F.cosine_similarity(e_target, e_ref_mean, dim=-1).item())
    cos_target_diff = float(F.cosine_similarity(e_target, e_diff, dim=-1).item())
    identity_gap = mean_same_id_cos - cos_mean_diff

    print("Reference Consistency Metrics:")
    print(f"  cos(Ref_A, Ref_B):            {cos_AB:.6f}")
    print(f"  cos(Ref_A, Ref_C):            {cos_AC:.6f}")
    print(f"  cos(Ref_B, Ref_C):            {cos_BC:.6f}")
    print(f"  Mean Same-ID Reference Cos:   {mean_same_id_cos:.6f}")
    print(f"  cos(Ref_Mean, Different_ID):  {cos_mean_diff:.6f}")
    print(f"  Identity Gap (Same - Diff):   {identity_gap:.6f}")
    print(f"  cos(Target_7367, Ref_Mean):   {cos_target_mean:.6f}")
    print(f"  cos(Target_7367, Diff_ID):    {cos_target_diff:.6f}")

    # -------------------------------------------------------------------------
    # 4. Prepare Latents and Phase 4 Regional Conditioning (Strict Reference-Only)
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Precomputing Phase-4 Identity Conditioning & Input Latents...")
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        tok_A = arcface(prep_arcface_input(ref_a_rgb).resize_(1, 3, 256, 256) if False else torch.from_numpy(cv2.resize(ref_a_rgb, (256, 256))).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0)["spatial_tokens"]
        tok_B = arcface(torch.from_numpy(cv2.resize(ref_b_rgb, (256, 256))).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0)["spatial_tokens"]
        tok_C = arcface(torch.from_numpy(cv2.resize(ref_c_rgb, (256, 256))).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0)["spatial_tokens"]

        w = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32, device=device)
        t_fused = phase4(
            target_tokens=None, # STRICT TARGET ANTI-LEAKAGE
            target_mask=mask_tensor_256,
            reference_tokens_list=[tok_A, tok_B, tok_C],
            reference_weights=w
        )["fused_tokens"]

    # Prepare 9-channel latent input matching Phase 7
    padded_img, padded_mask, _ = AspectRatioPreserver.letterbox(
        image=target_rgb, mask=mask, target_size=(512, 512)
    )
    pil_original = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

    init_image_tensor = pipe.image_processor.preprocess(pil_original, height=512, width=512).to(device)
    mask_tensor_proc = pipe.mask_processor.preprocess(pil_mask, height=512, width=512).to(device)
    masked_image_tensor = init_image_tensor * (mask_tensor_proc < 0.5)

    generator = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        target_latents = pipe._encode_vae_image(init_image_tensor, generator=generator)
        masked_image_latents = pipe._encode_vae_image(masked_image_tensor, generator=generator)

    mask_latent = F.interpolate(mask_tensor_proc, size=(64, 64), mode="nearest")

    torch.manual_seed(42)
    fixed_noise = torch.randn((1, 4, 64, 64), device=device)
    fixed_timestep = torch.tensor([500], device=device, dtype=torch.long)
    alpha_prod_t = scheduler.alphas_cumprod[fixed_timestep.item()]
    beta_prod_t = 1.0 - alpha_prod_t

    noisy_latents = scheduler.add_noise(target_latents, fixed_noise, fixed_timestep)
    latent_9ch = torch.cat([noisy_latents, mask_latent, masked_image_latents], dim=1)

    # Text conditioning
    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    prompt_tok = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        clean_text_emb = text_encoder(prompt_tok)[0]

    # -------------------------------------------------------------------------
    # 5. Identity Loss Sanity Test (Vanilla vs Step 100 Checkpoints)
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Identity Loss Sanity Test: Comparing Vanilla vs Step 100 Predictions...")
    ckpt_100_path = PROJECT_ROOT / "checkpoints/decoupled_identity_attention_100steps/step_100.pt"
    manager.load_checkpoint(str(ckpt_100_path))
    manager.set_scale(0.1)
    manager.set_identity_tokens(t_fused)

    with torch.no_grad():
        # 1. Step 100 prediction
        pred_eps_100 = unet(latent_9ch, fixed_timestep, encoder_hidden_states=clean_text_emb).sample
        x0_pred_100 = (noisy_latents - (beta_prod_t ** 0.5) * pred_eps_100) / (alpha_prod_t ** 0.5)
        pred_img_100 = vae.decode(x0_pred_100 / vae.config.scaling_factor).sample
        pred_img_100_112 = F.interpolate(pred_img_100, size=(112, 112), mode="bilinear", align_corners=False)
        e_pred_100 = diff_arcface.extract_embedding(pred_img_100_112)

        # 2. Vanilla prediction (lambda = 0.0)
        manager.set_scale(0.0)
        pred_eps_vanilla = unet(latent_9ch, fixed_timestep, encoder_hidden_states=clean_text_emb).sample
        x0_pred_vanilla = (noisy_latents - (beta_prod_t ** 0.5) * pred_eps_vanilla) / (alpha_prod_t ** 0.5)
        pred_img_vanilla = vae.decode(x0_pred_vanilla / vae.config.scaling_factor).sample
        pred_img_vanilla_112 = F.interpolate(pred_img_vanilla, size=(112, 112), mode="bilinear", align_corners=False)
        e_pred_vanilla = diff_arcface.extract_embedding(pred_img_vanilla_112)

    manager.set_scale(0.1)

    cos_vanilla_ref = float(F.cosine_similarity(e_pred_vanilla, e_ref_mean, dim=-1).item())
    cos_vanilla_target = float(F.cosine_similarity(e_pred_vanilla, e_target, dim=-1).item())
    cos_vanilla_diff = float(F.cosine_similarity(e_pred_vanilla, e_diff, dim=-1).item())

    cos_100_ref = float(F.cosine_similarity(e_pred_100, e_ref_mean, dim=-1).item())
    cos_100_target = float(F.cosine_similarity(e_pred_100, e_target, dim=-1).item())
    cos_100_diff = float(F.cosine_similarity(e_pred_100, e_diff, dim=-1).item())

    print("Sanity Embedding Similarities:")
    print(f"  Vanilla Pred vs Reference Mean:     {cos_vanilla_ref:.6f}")
    print(f"  Vanilla Pred vs Target GroundTruth: {cos_vanilla_target:.6f}")
    print(f"  Vanilla Pred vs Different ID:       {cos_vanilla_diff:.6f}")
    print(f"  Step 100 Pred vs Reference Mean:    {cos_100_ref:.6f}")
    print(f"  Step 100 Pred vs Target GT:         {cos_100_target:.6f}")
    print(f"  Step 100 Pred vs Different ID:      {cos_100_diff:.6f}")

    # -------------------------------------------------------------------------
    # 6. Gradient Flow Proof & Lambda Sensitivity Calibration
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Proving End-to-End Gradient Flow and Calibrating Lambda_ID...")

    def execute_forward_and_loss(lam_id: float):
        t0 = time.time()

        # 1. UNet forward
        model_pred = unet(latent_9ch, fixed_timestep, encoder_hidden_states=clean_text_emb).sample
        t_unet = time.time() - t0

        # 2. Diffusion Noise Loss
        loss_diff = F.mse_loss(model_pred.float(), fixed_noise.float(), reduction="mean")

        # 3. x0 reconstruction
        t1 = time.time()
        x0_pred = (noisy_latents - (beta_prod_t ** 0.5) * model_pred) / (alpha_prod_t ** 0.5)
        t_x0 = time.time() - t1

        # 4. VAE decode (autograd active through frozen VAE)
        t2 = time.time()
        pred_img = vae.decode(x0_pred / vae.config.scaling_factor).sample
        t_vae = time.time() - t2

        # 5. Differentiable ArcFace forward
        t3 = time.time()
        pred_img_112 = F.interpolate(pred_img, size=(112, 112), mode="bilinear", align_corners=False)
        e_pred = diff_arcface.extract_embedding(pred_img_112)
        t_arcface = time.time() - t3

        # 6. Identity Loss
        loss_id = 1.0 - F.cosine_similarity(e_pred, e_ref_mean, dim=-1).mean()

        total_loss = loss_diff + lam_id * loss_id

        timing = {
            "unet_forward_sec": t_unet,
            "x0_recon_sec": t_x0,
            "vae_decode_sec": t_vae,
            "arcface_forward_sec": t_arcface
        }
        return total_loss, loss_diff, loss_id, timing

    # Test 6A: Diffusion Loss Only (lambda_id = 0.0)
    manager.set_scale(0.1)
    manager.set_identity_tokens(t_fused)
    for p in trainable_params:
        p.grad = None

    _, loss_diff_only, _, time_diff = execute_forward_and_loss(0.0)
    t_b0 = time.time()
    loss_diff_only.backward(retain_graph=False)
    t_backward_diff = time.time() - t_b0

    k_diff_grads = [p.grad.norm().item() for proc in manager.processors.values() for p in proc.to_k_id.parameters() if p.grad is not None]
    v_diff_grads = [p.grad.norm().item() for proc in manager.processors.values() for p in proc.to_v_id.parameters() if p.grad is not None]
    mean_k_diff = float(np.mean(k_diff_grads)) if k_diff_grads else 0.0
    mean_v_diff = float(np.mean(v_diff_grads)) if v_diff_grads else 0.0

    print("\n--- TEST 1: DIFFUSION LOSS ONLY ---")
    print(f"Diffusion Loss:      {loss_diff_only.item():.6f}")
    print(f"Mean K_id Grad Norm: {mean_k_diff:.8f}")
    print(f"Mean V_id Grad Norm: {mean_v_diff:.8f}")
    print(f"Backward Time:       {t_backward_diff:.2f}s")

    # Test 6B: Identity Loss Only
    for p in trainable_params:
        p.grad = None

    _, _, loss_id_only, time_id = execute_forward_and_loss(0.0)
    t_b1 = time.time()
    loss_id_only.backward(retain_graph=False)
    t_backward_id = time.time() - t_b1

    k_id_grads = [p.grad.norm().item() for proc in manager.processors.values() for p in proc.to_k_id.parameters() if p.grad is not None]
    v_id_grads = [p.grad.norm().item() for proc in manager.processors.values() for p in proc.to_v_id.parameters() if p.grad is not None]
    mean_k_id = float(np.mean(k_id_grads)) if k_id_grads else 0.0
    mean_v_id = float(np.mean(v_id_grads)) if v_id_grads else 0.0

    # Strict Frozen Check
    frozen_clean = True
    for mod_name, mod in [
        ("UNet_Original", [p for n, p in unet.named_parameters() if "to_k_id" not in n and "to_v_id" not in n]),
        ("VAE", list(vae.parameters())),
        ("CLIP", list(text_encoder.parameters())),
        ("Diff_ArcFace", list(diff_arcface.parameters())),
        ("Phase4", list(phase4.parameters()))
    ]:
        for p in mod:
            if p.grad is not None and torch.any(p.grad != 0.0):
                frozen_clean = False
                print(f"FATAL: Nonzero grad in frozen module {mod_name}!")

    print("\n--- TEST 2: IDENTITY LOSS ONLY ---")
    print(f"Identity Loss:       {loss_id_only.item():.6f}")
    print(f"Mean K_id Grad Norm: {mean_k_id:.8f}")
    print(f"Mean V_id Grad Norm: {mean_v_id:.8f}")
    print(f"K_id Identity / Diffusion Grad Ratio: {mean_k_id / (mean_k_diff + 1e-12):.2f}x")
    print(f"V_id Identity / Diffusion Grad Ratio: {mean_v_id / (mean_v_diff + 1e-12):.2f}x")
    print(f"Backward Time (through ArcFace+VAE+UNet): {t_backward_id:.2f}s")
    print(f"Frozen Parameter Integrity: {frozen_clean}")

    # Test 6C: Analytical Sensitivity Sweep over lambda_id
    print("\n--- TEST 3: ANALYTICAL LAMBDA_ID SENSITIVITY CALIBRATION ---")
    lambda_candidates = [0.01, 0.05, 0.10, 0.25]
    sensitivity_results = []

    for lam in lambda_candidates:
        for p in trainable_params:
            p.grad = None

        t_start = time.time()
        total_loss, l_diff, l_id, timing = execute_forward_and_loss(lam)
        total_loss.backward(retain_graph=False)
        step_cost = time.time() - t_start

        k_g = [p.grad.norm().item() for proc in manager.processors.values() for p in proc.to_k_id.parameters() if p.grad is not None]
        v_g = [p.grad.norm().item() for proc in manager.processors.values() for p in proc.to_v_id.parameters() if p.grad is not None]
        m_k = float(np.mean(k_g)) if k_g else 0.0
        m_v = float(np.mean(v_g)) if v_g else 0.0

        # Ratios
        id_k_component = lam * mean_k_id
        id_v_component = lam * mean_v_id
        ratio_k = id_k_component / (mean_k_diff + 1e-12)
        ratio_v = id_v_component / (mean_v_diff + 1e-12)

        res = {
            "lambda_id": lam,
            "diffusion_loss": float(l_diff.item()),
            "identity_loss": float(l_id.item()),
            "total_loss": float(total_loss.item()),
            "mean_k_grad_norm": m_k,
            "mean_v_grad_norm": m_v,
            "ratio_id_to_diff_k": ratio_k,
            "ratio_id_to_diff_v": ratio_v,
            "total_step_time_sec": step_cost
        }
        sensitivity_results.append(res)
        print(
            f"lambda={lam:<4.2f} | L_diff={l_diff.item():<8.6f} | L_id={l_id.item():<8.6f} | "
            f"K_grad={m_k:<10.8f} | V_grad={m_v:<10.8f} | "
            f"K_ratio={ratio_k:<6.2f}x | V_ratio={ratio_v:<6.2f}x | StepTime={step_cost:.1f}s"
        )

    # -------------------------------------------------------------------------
    # 7. Compile Final Report Data
    # -------------------------------------------------------------------------
    report_data = {
        "reference_consistency": {
            "cos_ref_A_ref_B": cos_AB,
            "cos_ref_A_ref_C": cos_AC,
            "cos_ref_B_ref_C": cos_BC,
            "mean_same_id_cos": mean_same_id_cos,
            "cos_ref_mean_diff_id": cos_mean_diff,
            "identity_gap": identity_gap,
            "cos_target_ref_mean": cos_target_mean,
            "cos_target_diff_id": cos_target_diff
        },
        "identity_loss_sanity": {
            "cos_vanilla_ref": cos_vanilla_ref,
            "cos_vanilla_target": cos_vanilla_target,
            "cos_vanilla_diff": cos_vanilla_diff,
            "cos_100_ref": cos_100_ref,
            "cos_100_target": cos_100_target,
            "cos_100_diff": cos_100_diff
        },
        "gradient_flow_proof": {
            "diffusion_only": {
                "loss": float(loss_diff_only.item()),
                "mean_k_grad": mean_k_diff,
                "mean_v_grad": mean_v_diff,
                "backward_time_sec": t_backward_diff
            },
            "identity_only": {
                "loss": float(loss_id_only.item()),
                "mean_k_grad": mean_k_id,
                "mean_v_grad": mean_v_id,
                "backward_time_sec": t_backward_id,
                "k_ratio_over_diff": mean_k_id / (mean_k_diff + 1e-12),
                "v_ratio_over_diff": mean_v_id / (mean_v_diff + 1e-12)
            },
            "frozen_parameter_integrity": frozen_clean
        },
        "lambda_sensitivity": sensitivity_results,
        "runtime_breakdown": {
            "unet_forward_sec": time_id["unet_forward_sec"],
            "x0_recon_sec": time_id["x0_recon_sec"],
            "vae_decode_sec": time_id["vae_decode_sec"],
            "arcface_forward_sec": time_id["arcface_forward_sec"],
            "backward_sec": t_backward_id,
            "total_estimated_step_cost_sec": float(np.mean([r["total_step_time_sec"] for r in sensitivity_results]))
        }
    }

    report_file = out_dir / "identity_loss_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)

    total_time = time.time() - start_total_time
    print(f"\nValidation suite completed in {total_time:.2f}s.")
    print(f"Report saved to {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
