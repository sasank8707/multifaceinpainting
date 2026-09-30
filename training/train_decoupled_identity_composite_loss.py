"""
Controlled 50-Step Composite Loss Training Pipeline for Decoupled Identity Cross-Attention in Stable Diffusion.

Objective:
L_total = L_diffusion + lambda_id * L_identity

where:
L_diffusion = MSE(epsilon_pred, epsilon_true)
L_identity = 1 - cosine(e_pred, e_ref_mean)
with:
x0_pred = (noisy_latents - sqrt(1 - alpha_bar_t) * model_pred) / sqrt(alpha_bar_t)
pred_img = VAE.decode(x0_pred / vae_scale)
e_pred = DifferentiableArcFace(pred_img_112)

Resumes directly from checkpoints/decoupled_identity_attention_100steps/step_100.pt.
Trains ONLY the 16 layers of to_k_id and to_v_id (~12.78M parameters).
All other components remain strictly frozen:
- Pretrained UNet original parameters (FROZEN)
- Pretrained VAE (FROZEN)
- Pretrained CLIP Text Encoder (FROZEN)
- Differentiable ArcFace backbone (FROZEN)
- Phase-4 Regional Fusion (FROZEN)

Anti-Leakage Guarantee:
- Target image provides WHERE mask context and ground truth reconstruction target only.
- Identity conditioning T_fused derives exclusively from K=3 verified same-identity references.
"""

import os
import sys
import time
import json
import argparse
import psutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset
from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import (
    DecoupledIdentityCrossAttentionManager,
    DecoupledIdentityAttnProcessor
)
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler


def get_parameter_tables(
    unet: nn.Module,
    manager: DecoupledIdentityCrossAttentionManager,
    vae: nn.Module,
    text_encoder: nn.Module,
    diff_arcface: nn.Module,
    arcface: nn.Module,
    phase4: nn.Module
):
    """Generates parameter audit tables and verifies frozen/trainable constraints."""
    modules = {
        "Decoupled_Identity_K_V": manager.get_trainable_parameters(),
        "UNet_Backbone_Original": [p for n, p in unet.named_parameters() if "to_k_id" not in n and "to_v_id" not in n],
        "VAE_Autoencoder": list(vae.parameters()),
        "CLIP_Text_Encoder": list(text_encoder.parameters()),
        "Differentiable_ArcFace": list(diff_arcface.parameters()),
        "Regional_ArcFace": list(arcface.parameters()),
        "Phase4_Regional_Fusion": list(phase4.parameters())
    }

    summary = {}
    total_trainable = 0
    total_frozen = 0

    table_lines = [
        "+" + "-"*35 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+",
        f"| {'Module Name':<33} | {'Trainable':<13} | {'Frozen':<13} | {'Total Params':<13} |",
        "+" + "-"*35 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+"
    ]

    for mod_name, param_list in modules.items():
        trainable_count = sum(p.numel() for p in param_list if p.requires_grad)
        frozen_count = sum(p.numel() for p in param_list if not p.requires_grad)
        total_count = trainable_count + frozen_count

        total_trainable += trainable_count
        total_frozen += frozen_count

        summary[mod_name] = {
            "trainable_params": trainable_count,
            "frozen_params": frozen_count,
            "total_params": total_count,
            "requires_grad_status": "ALL_TRAINABLE" if trainable_count == total_count and total_count > 0 else (
                "ALL_FROZEN" if frozen_count == total_count else "MIXED"
            )
        }

        table_lines.append(
            f"| {mod_name:<33} | {trainable_count:<13,d} | {frozen_count:<13,d} | {total_count:<13,d} |"
        )

    table_lines.append("+" + "-"*35 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
    table_lines.append(
        f"| {'TOTAL':<33} | {total_trainable:<13,d} | {total_frozen:<13,d} | {total_trainable + total_frozen:<13,d} |"
    )
    table_lines.append("+" + "-"*35 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")

    return summary, "\n".join(table_lines)


def build_fixed_diagnostic_data(pipe, diff_arcface, arcface, phase4, device):
    """Precomputes fixed evaluation tensors on CelebA-HQ subject 7367."""
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

    def prep_img(rgb_img, size=(256, 256)):
        resized = cv2.resize(rgb_img, size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    def prep_112(rgb_img):
        resized = cv2.resize(rgb_img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        tok_A = arcface(prep_img(ref_a_rgb))["spatial_tokens"]
        tok_B = arcface(prep_img(ref_b_rgb))["spatial_tokens"]
        tok_C = arcface(prep_img(ref_c_rgb))["spatial_tokens"]
        tok_diff = arcface(prep_img(diff_id_rgb))["spatial_tokens"]

        w_single = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        w_multi = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32, device=device)

        t_fused_A = phase4(target_tokens=None, target_mask=mask_tensor_256, reference_tokens_list=[tok_A], reference_weights=w_single)["fused_tokens"]
        t_fused_B = phase4(target_tokens=None, target_mask=mask_tensor_256, reference_tokens_list=[tok_B], reference_weights=w_single)["fused_tokens"]
        t_fused_C = phase4(target_tokens=None, target_mask=mask_tensor_256, reference_tokens_list=[tok_C], reference_weights=w_single)["fused_tokens"]
        t_fused_multi = phase4(target_tokens=None, target_mask=mask_tensor_256, reference_tokens_list=[tok_A, tok_B, tok_C], reference_weights=w_multi)["fused_tokens"]
        t_fused_diff = phase4(target_tokens=None, target_mask=mask_tensor_256, reference_tokens_list=[tok_diff], reference_weights=w_single)["fused_tokens"]

        # ArcFace Global Embeddings for biometric comparison
        e_ref_A = diff_arcface.extract_embedding(prep_112(ref_a_rgb))
        e_ref_B = diff_arcface.extract_embedding(prep_112(ref_b_rgb))
        e_ref_C = diff_arcface.extract_embedding(prep_112(ref_c_rgb))
        e_ref_mean = F.normalize((e_ref_A + e_ref_B + e_ref_C) / 3.0, p=2, dim=-1)
        e_diff = diff_arcface.extract_embedding(prep_112(diff_id_rgb))
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))

    # Latent preparation
    padded_img, padded_mask, _ = AspectRatioPreserver.letterbox(target_rgb, mask, target_size=(512, 512))
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
    scheduler = pipe.scheduler
    noisy_latents = scheduler.add_noise(target_latents, fixed_noise, fixed_timestep)
    latent_9ch = torch.cat([noisy_latents, mask_latent, masked_image_latents], dim=1)

    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    prompt_tok = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        clean_text_emb = pipe.text_encoder(prompt_tok)[0]

    return {
        "latent_9ch": latent_9ch,
        "noisy_latents": noisy_latents,
        "fixed_timestep": fixed_timestep,
        "clean_text_emb": clean_text_emb,
        "mask_64": (mask_latent > 0.5),
        "t_fused_A": t_fused_A,
        "t_fused_B": t_fused_B,
        "t_fused_C": t_fused_C,
        "t_fused_multi": t_fused_multi,
        "t_fused_diff": t_fused_diff,
        "e_ref_mean": e_ref_mean,
        "e_diff": e_diff,
        "e_target": e_target
    }


def evaluate_diagnostic(step, unet, vae, diff_arcface, manager, scheduler, diag_data, pristine_processors, device):
    """Evaluates comprehensive diagnostic metrics on the fixed subject 7367 benchmark."""
    latent = diag_data["latent_9ch"]
    noisy_latents = diag_data["noisy_latents"]
    timestep = diag_data["fixed_timestep"]
    text_emb = diag_data["clean_text_emb"]
    mask_64 = diag_data["mask_64"]

    t_fused_multi = diag_data["t_fused_multi"]
    t_fused_diff = diag_data["t_fused_diff"]
    e_ref_mean = diag_data["e_ref_mean"]
    e_diff = diag_data["e_diff"]
    e_target = diag_data["e_target"]

    alpha_prod_t = scheduler.alphas_cumprod[timestep.item()]
    beta_prod_t = 1.0 - alpha_prod_t

    # 1. Vanilla prediction (lambda = 0.0)
    manager.set_scale(0.0)
    with torch.no_grad():
        eps_vanilla = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()

    # 2. Same-ID Multi-Ref prediction (lambda = 0.1)
    manager.set_scale(0.1)
    manager.set_identity_tokens(t_fused_multi)
    manager.set_diagnostics_enabled(True)
    with torch.no_grad():
        eps_multi = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()
    manager.set_diagnostics_enabled(False)

    # 3. Different ID prediction (ID 3616)
    manager.set_identity_tokens(t_fused_diff)
    with torch.no_grad():
        eps_diff = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()

    # Reset identity tokens to same-ID
    manager.set_identity_tokens(t_fused_multi)

    # Decode x0 predictions to compute biometric alignment
    with torch.no_grad():
        x0_multi = (noisy_latents - (beta_prod_t ** 0.5) * eps_multi) / (alpha_prod_t ** 0.5)
        pred_img_multi = vae.decode(x0_multi / vae.config.scaling_factor).sample
        pred_img_multi_112 = F.interpolate(pred_img_multi.clamp(-1.0, 1.0), size=(112, 112), mode="bilinear", align_corners=False)
        e_pred_multi = diff_arcface.extract_embedding(pred_img_multi_112)

        cos_pred_ref = float(F.cosine_similarity(e_pred_multi, e_ref_mean, dim=-1).item())
        cos_pred_diff = float(F.cosine_similarity(e_pred_multi, e_diff, dim=-1).item())
        cos_pred_target = float(F.cosine_similarity(e_pred_multi, e_target, dim=-1).item())
        biometric_gap = cos_pred_ref - cos_pred_diff

    # Epsilon sensitivity metrics
    def compute_diff(e1, e2):
        l2 = float((e1 - e2).norm())
        rel_l2 = float(l2 / (e1.norm() + 1e-8))
        cos = float(F.cosine_similarity(e1.flatten(), e2.flatten(), dim=0))
        return {"l2_diff": l2, "rel_l2": rel_l2, "cosine_sim": cos}

    diff_id_vs_vanilla = compute_diff(eps_multi, eps_vanilla)
    diff_same_vs_diff = compute_diff(eps_multi, eps_diff)

    # Attention stats & KV norms
    k_norms, v_norms, entropies, max_probs = [], [], [], []
    for name, proc in manager.processors.items():
        with torch.no_grad():
            sample_k = proc.to_k_id(t_fused_multi)
            sample_v = proc.to_v_id(t_fused_multi)
            k_norms.append(float(sample_k.norm(dim=-1).mean()))
            v_norms.append(float(sample_v.norm(dim=-1).mean()))

            diag = proc.latest_diagnostics
            if diag:
                entropies.append(float(diag["entropy"]))
                max_probs.append(float(diag["max_prob"]))

    mean_k_norm = float(np.mean(k_norms)) if k_norms else 0.0
    mean_v_norm = float(np.mean(v_norms)) if v_norms else 0.0
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    mean_max_prob = float(np.mean(max_probs)) if max_probs else 0.0

    return {
        "step": step,
        "biometric_alignment": {
            "cos_pred_vs_ref_mean": cos_pred_ref,
            "cos_pred_vs_diff_id": cos_pred_diff,
            "cos_pred_vs_target_gt": cos_pred_target,
            "biometric_gap": biometric_gap
        },
        "epsilon_sensitivity": {
            "identity_vs_vanilla": diff_id_vs_vanilla,
            "same_id_vs_diff_id": diff_same_vs_diff
        },
        "identity_kv_norms": {
            "mean_key_norm": mean_k_norm,
            "mean_value_norm": mean_v_norm
        },
        "identity_attention_stats": {
            "mean_entropy": mean_entropy,
            "mean_max_prob": mean_max_prob,
            "uniform_entropy_theoretical": float(np.log(256)),
            "entropy_pct_of_uniform": float(mean_entropy / np.log(256)) * 100.0
        }
    }


def train_composite_loss(args):
    print("=" * 80)
    print("COMPOSITE IDENTITY-LOSS TRAINING: DECOUPLED CROSS-ATTENTION")
    print("=" * 80)
    start_total_time = time.time()

    # Optimal CPU threading
    torch.set_num_threads(4)
    device = torch.device("cpu")

    ckpt_dir = PROJECT_ROOT / "checkpoints/decoupled_identity_composite_loss"
    exp_dir = PROJECT_ROOT / "experiments/exp_decoupled_identity_composite_loss"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pretrained Pipeline & Differentiable ArcFace
    print("\n[Step 1/5] Loading Pretrained Stable Diffusion Inpainting & Differentiable ArcFace...")
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
    scheduler = pipe.scheduler

    # Freeze non-trainable components
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    diff_arcface = DifferentiableArcFaceBackbone("models/pretrained/w600k_r50.onnx").to(device)
    diff_arcface.eval()
    for p in diff_arcface.parameters():
        p.requires_grad = False

    arcface = RegionalIdentityEncoder(token_dim=512).to(device)
    arcface.eval()
    for p in arcface.parameters():
        p.requires_grad = False

    phase4 = RegionalIdentityTransformer(
        token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0
    ).to(device)
    phase4.eval()
    for p in phase4.parameters():
        p.requires_grad = False

    # 2. Attach Decoupled Identity Cross-Attention & Load step_100 Checkpoint
    print("\n[Step 2/5] Initializing Decoupled Cross-Attention & Loading step_100.pt...")
    pristine_vanilla_processors = dict(unet.attn_processors)
    manager = DecoupledIdentityCrossAttentionManager(unet, id_dim=512, default_scale=0.1)
    manager.original_processors = dict(pristine_vanilla_processors)
    manager.install(zero_init_v=False)

    # Load starting checkpoint (step_100.pt)
    init_ckpt_path = PROJECT_ROOT / args.resume_checkpoint
    assert init_ckpt_path.exists(), f"Starting checkpoint {init_ckpt_path} not found!"
    manager.load_checkpoint(str(init_ckpt_path))
    print(f"Successfully loaded checkpoint: {init_ckpt_path}")

    trainable_params = manager.get_trainable_parameters()
    for p in trainable_params:
        p.requires_grad = True

    param_summary, param_table_str = get_parameter_tables(
        unet, manager, vae, text_encoder, diff_arcface, arcface, phase4
    )
    print("\nPARAMETER STATUS AUDIT TABLE:")
    print(param_table_str)

    assert param_summary["Decoupled_Identity_K_V"]["trainable_params"] == 12779520, "Expected 12,779,520 trainable params!"
    assert param_summary["UNet_Backbone_Original"]["trainable_params"] == 0, "UNet original params must be frozen!"
    assert param_summary["VAE_Autoencoder"]["trainable_params"] == 0, "VAE must be frozen!"
    assert param_summary["CLIP_Text_Encoder"]["trainable_params"] == 0, "CLIP must be frozen!"
    assert param_summary["Differentiable_ArcFace"]["trainable_params"] == 0, "Differentiable ArcFace must be frozen!"
    assert param_summary["Regional_ArcFace"]["trainable_params"] == 0, "Regional ArcFace must be frozen!"
    assert param_summary["Phase4_Regional_Fusion"]["trainable_params"] == 0, "Phase 4 must be frozen!"

    # 3. Setup Dataset & Optimizer
    print("\n[Step 3/5] Setting up Dataset & Optimizer...")
    train_dataset = CelebAHQIdentityDataset(
        split="train",
        image_size=(512, 512),
        num_positive_references=3,
        seed=42
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    prompt_tok = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        clean_text_emb = text_encoder(prompt_tok)[0]

    # Precompute Diagnostic Evaluation Data
    print("\n[Step 4/5] Precomputing Diagnostic Benchmark (Subject 7367)...")
    diag_data = build_fixed_diagnostic_data(pipe, diff_arcface, arcface, phase4, device)

    base_step = args.start_step
    print(f"\nRunning Baseline Diagnostic at Step {base_step} (Before Composite Training)...")
    diag_initial = evaluate_diagnostic(base_step, unet, vae, diff_arcface, manager, scheduler, diag_data, pristine_vanilla_processors, device)
    print(f"  Step {base_step}: Biom Alignment: cos(Pred, Ref Mean) = {diag_initial['biometric_alignment']['cos_pred_vs_ref_mean']:.4f}")
    print(f"  Step {base_step}: Biom Gap:       cos(Ref) - cos(Diff) = {diag_initial['biometric_alignment']['biometric_gap']:+.4f}")
    print(f"  Step {base_step}: Sensitivity:    Diff-ID rel L2 diff = {diag_initial['epsilon_sensitivity']['same_id_vs_diff_id']['rel_l2']:.4f}")
    print(f"  Step {base_step}: Attention Ent:  {diag_initial['identity_attention_stats']['mean_entropy']:.4f} ({diag_initial['identity_attention_stats']['entropy_pct_of_uniform']:.2f}% of uniform)")
    sys.stdout.flush()

    total_steps = args.steps
    diag_interval = args.diag_interval

    training_metrics = []
    diagnostic_evaluations = [diag_initial]

    # 4. Training Loop
    print(f"\n[Step 5/5] Executing {total_steps} Composite Loss Optimization Steps...")
    sys.stdout.flush()
    data_iter = iter(train_loader)
    process = psutil.Process(os.getpid())

    for i in range(1, total_steps + 1):
        step_num = base_step + i
        step_start_time = time.time()

        # Dynamic lambda_id schedule: warmup over first 10 steps
        if i <= 10:
            lambda_id = 0.01 + (i - 1) / 9.0 * (args.lambda_id_target - 0.01)
        else:
            lambda_id = args.lambda_id_target

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        target = batch["target"].to(device)       # [1, 3, 512, 512]
        mask = batch["mask"].to(device)           # [1, 1, 512, 512]
        pos_refs = [r.to(device) for r in batch["pos_references"]] # 3 tensors of [1, 3, 512, 512]

        with torch.no_grad():
            latents = vae.encode(target).latent_dist.sample() * vae.config.scaling_factor
            masked_target = target * (1.0 - mask)
            masked_latents = vae.encode(masked_target).latent_dist.sample() * vae.config.scaling_factor
            mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

            # Extract ArcFace spatial tokens for all K=3 references
            ref_tokens_list = []
            ref_112_list = []
            for r in pos_refs:
                r_256 = F.interpolate(r, size=(256, 256), mode="bilinear", align_corners=False)
                ref_tokens_list.append(arcface(r_256)["spatial_tokens"])
                r_112 = F.interpolate(r, size=(112, 112), mode="bilinear", align_corners=False)
                ref_112_list.append(diff_arcface.extract_embedding(r_112))

            # Precompute normalized mean reference embedding for identity loss
            e_ref_mean_batch = F.normalize(torch.stack(ref_112_list).mean(dim=0), p=2, dim=-1)

            mask_256 = F.interpolate(mask, size=(256, 256), mode="nearest")
            ref_weights = torch.ones(1, len(ref_tokens_list), dtype=torch.float32, device=device) / float(len(ref_tokens_list))

            # Phase 4 Regional Fusion (Target tokens strictly None: Anti-Leakage)
            t_fused = phase4(
                target_tokens=None,
                target_mask=mask_256,
                reference_tokens_list=ref_tokens_list,
                reference_weights=ref_weights
            )["fused_tokens"]

            # Sample timestep and noise
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        latent_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)

        # Set identity conditioning
        manager.set_scale(0.1)
        manager.set_identity_tokens(t_fused)

        optimizer.zero_grad()

        # 1. UNet Forward
        model_pred = unet(latent_input, timesteps, encoder_hidden_states=clean_text_emb).sample

        # 2. Diffusion MSE Loss
        loss_diff = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        # 3. Clean x0 Reconstruction & Identity Loss
        alpha_prod_t = scheduler.alphas_cumprod[timesteps.item()]
        beta_prod_t = 1.0 - alpha_prod_t
        x0_pred = (noisy_latents - (beta_prod_t ** 0.5) * model_pred) / (alpha_prod_t ** 0.5)

        # Decode x0 via differentiable VAE
        pred_img = vae.decode(x0_pred / vae.config.scaling_factor).sample
        pred_img_112 = F.interpolate(pred_img.clamp(-1.0, 1.0), size=(112, 112), mode="bilinear", align_corners=False)
        e_pred = diff_arcface.extract_embedding(pred_img_112)

        # Biometric Identity Loss
        loss_id = 1.0 - F.cosine_similarity(e_pred, e_ref_mean_batch, dim=-1).mean()

        # Total Composite Loss
        loss_total = loss_diff + lambda_id * loss_id

        # Backward pass
        loss_total.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # Record gradient norms
        k_grads = [float(p.grad.norm().cpu()) for proc in manager.processors.values() for p in proc.to_k_id.parameters() if p.grad is not None]
        v_grads = [float(p.grad.norm().cpu()) for proc in manager.processors.values() for p in proc.to_v_id.parameters() if p.grad is not None]
        mean_k_grad = float(np.mean(k_grads)) if k_grads else 0.0
        mean_v_grad = float(np.mean(v_grads)) if v_grads else 0.0

        optimizer.step()
        step_elapsed = time.time() - step_start_time

        step_metric = {
            "step": step_num,
            "lambda_id": float(lambda_id),
            "loss_diff": float(loss_diff.item()),
            "loss_id": float(loss_id.item()),
            "loss_total": float(loss_total.item()),
            "mean_k_grad_norm": mean_k_grad,
            "mean_v_grad_norm": mean_v_grad,
            "step_time_sec": float(step_elapsed),
            "ram_gb": float(process.memory_info().rss / (1024**3))
        }
        training_metrics.append(step_metric)

        print(
            f"Step {step_num:3d} (rel {i:2d}/{total_steps}) | "
            f"L_diff: {loss_diff.item():.5f} | "
            f"L_id: {loss_id.item():.5f} | "
            f"lambda_id: {lambda_id:.3f} | "
            f"Total: {loss_total.item():.5f} | "
            f"Grad K/V: {mean_k_grad:.5f}/{mean_v_grad:.5f} | "
            f"Time: {step_elapsed:.1f}s"
        )
        sys.stdout.flush()

        # Diagnostic Evaluation and Checkpointing
        if i % diag_interval == 0 or i == total_steps:
            print(f"\n--- Diagnostic Evaluation at Step {step_num} ---")
            diag_res = evaluate_diagnostic(step_num, unet, vae, diff_arcface, manager, scheduler, diag_data, pristine_vanilla_processors, device)
            diagnostic_evaluations.append(diag_res)

            print(f"  Biometric Alignment: cos(Pred, Ref Mean) = {diag_res['biometric_alignment']['cos_pred_vs_ref_mean']:.4f}")
            print(f"  Biometric Gap:       cos(Ref) - cos(Diff) = {diag_res['biometric_alignment']['biometric_gap']:+.4f}")
            print(f"  Sensitivity:         Diff-ID rel L2 diff = {diag_res['epsilon_sensitivity']['same_id_vs_diff_id']['rel_l2']:.4f}")
            print(f"  Attention Entropy:   {diag_res['identity_attention_stats']['mean_entropy']:.4f} ({diag_res['identity_attention_stats']['entropy_pct_of_uniform']:.2f}% of uniform)")
            print(f"  Mean K/V Norms:      K={diag_res['identity_kv_norms']['mean_key_norm']:.4f}, V={diag_res['identity_kv_norms']['mean_value_norm']:.4f}")

            ckpt_path = ckpt_dir / f"step_{step_num}.pt"
            manager.save_checkpoint(
                str(ckpt_path),
                extra_data={"step": step_num, "optimizer_state_dict": optimizer.state_dict(), "diag": diag_res}
            )
            print(f"  Checkpoint saved: {ckpt_path}\n")
            sys.stdout.flush()

    # Save final logs and metrics
    with open(exp_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    with open(exp_dir / "diagnostic_evaluations.json", "w") as f:
        json.dump(diagnostic_evaluations, f, indent=2)

    total_training_time = time.time() - start_total_time
    summary_report = {
        "total_steps": total_steps,
        "start_step": base_step,
        "end_step": base_step + total_steps,
        "total_time_min": total_training_time / 60.0,
        "mean_step_time_sec": float(np.mean([m["step_time_sec"] for m in training_metrics])),
        "initial_diag": diag_initial,
        "final_diag": diagnostic_evaluations[-1]
    }
    with open(exp_dir / "summary_report.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    # Generate and save diagnostic curves plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps_train = [m["step"] for m in training_metrics]
        loss_diffs = [m["loss_diff"] for m in training_metrics]
        loss_ids = [m["loss_id"] for m in training_metrics]
        loss_totals = [m["loss_total"] for m in training_metrics]
        grad_ks = [m["mean_k_grad_norm"] for m in training_metrics]
        grad_vs = [m["mean_v_grad_norm"] for m in training_metrics]

        diag_steps = [d["step"] for d in diagnostic_evaluations]
        cos_refs = [d["biometric_alignment"]["cos_pred_vs_ref_mean"] for d in diagnostic_evaluations]
        cos_diffs = [d["biometric_alignment"]["cos_pred_vs_diff_id"] for d in diagnostic_evaluations]
        biom_gaps = [d["biometric_alignment"]["biometric_gap"] for d in diagnostic_evaluations]
        entropies = [d["identity_attention_stats"]["mean_entropy"] for d in diagnostic_evaluations]
        k_norms = [d["identity_kv_norms"]["mean_key_norm"] for d in diagnostic_evaluations]
        v_norms = [d["identity_kv_norms"]["mean_value_norm"] for d in diagnostic_evaluations]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Decoupled Identity Cross-Attention: Composite Loss Training", fontsize=16, fontweight="bold")

        # 1. Total Loss & Diffusion MSE
        axes[0, 0].plot(steps_train, loss_totals, label="Total Loss", color="#1f77b4", linewidth=2)
        axes[0, 0].plot(steps_train, loss_diffs, label="Diffusion MSE", color="#aec7e8", linestyle="--")
        axes[0, 0].set_title("Training Loss Progression")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Identity Loss
        axes[0, 1].plot(steps_train, loss_ids, label="Identity Loss (1 - cos)", color="#d62728", linewidth=2)
        axes[0, 1].set_title("Biometric Identity Loss (1 - cos)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Loss_id")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Gradient Norms (K vs V)
        axes[0, 2].plot(steps_train, grad_ks, label="Mean ||grad K_id||", color="#2ca02c")
        axes[0, 2].plot(steps_train, grad_vs, label="Mean ||grad V_id||", color="#9467bd")
        axes[0, 2].set_title("Gradient Norms (K_id vs V_id)")
        axes[0, 2].set_xlabel("Step")
        axes[0, 2].set_ylabel("L2 Norm")
        axes[0, 2].set_yscale("log")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Biometric Alignment (Pred vs Ref vs Diff-ID)
        axes[1, 0].plot(diag_steps, cos_refs, "o-", label="cos(Pred, Ref Mean)", color="#2ca02c", linewidth=2)
        axes[1, 0].plot(diag_steps, cos_diffs, "s--", label="cos(Pred, Diff ID)", color="#d62728", linewidth=2)
        axes[1, 0].plot(diag_steps, biom_gaps, "^-.", label="Biometric Gap", color="#ff7f0e", linewidth=2)
        axes[1, 0].set_title("Biometric Alignment & Identity Gap")
        axes[1, 0].set_xlabel("Diagnostic Step")
        axes[1, 0].set_ylabel("Cosine Similarity")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Cross-Attention Entropy
        uniform_h = np.log(256)
        axes[1, 1].plot(diag_steps, entropies, "o-", label="Mean Attention Entropy", color="#e377c2", linewidth=2)
        axes[1, 1].axhline(y=uniform_h, color="gray", linestyle=":", label=f"Uniform Entropy ({uniform_h:.2f})")
        axes[1, 1].set_title("Identity Cross-Attention Entropy")
        axes[1, 1].set_xlabel("Diagnostic Step")
        axes[1, 1].set_ylabel("Entropy (nats)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Weight Norms (K vs V)
        axes[1, 2].plot(diag_steps, k_norms, "o-", label="Mean Key Norm", color="#17becf", linewidth=2)
        axes[1, 2].plot(diag_steps, v_norms, "s-", label="Mean Value Norm", color="#bcbd22", linewidth=2)
        axes[1, 2].set_title("Weight Norm Evolution (K_id vs V_id)")
        axes[1, 2].set_xlabel("Diagnostic Step")
        axes[1, 2].set_ylabel("Mean Token Norm")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = exp_dir / "composite_loss_training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")

        # Also copy to artifacts directory if available
        brain_dir = Path("C:/Users/sasan/.gemini/antigravity/brain/fbe3e04a-609f-4ac9-8552-a9b57e8d7a2a/composite_training")
        brain_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(str(plot_path), str(brain_dir / "composite_loss_training_curves.png"))
        print(f"Artifact plot copied to: {brain_dir / 'composite_loss_training_curves.png'}")
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")

    print("\n" + "=" * 80)
    print("COMPOSITE IDENTITY-LOSS TRAINING COMPLETE")
    print(f"Total Time: {total_training_time/60.0:.2f} minutes")
    print(f"Metrics saved to: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite Identity-Loss Training for Decoupled Cross-Attention")
    parser.add_argument("--steps", type=int, default=50, help="Number of optimizer steps to execute")
    parser.add_argument("--start_step", type=int, default=100, help="Starting step number offset")
    parser.add_argument("--diag_interval", type=int, default=10, help="Diagnostic and checkpoint interval")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--lambda_id_target", type=float, default=0.05, help="Target identity loss weight")
    parser.add_argument("--resume_checkpoint", type=str, default="checkpoints/decoupled_identity_attention_100steps/step_100.pt")
    args = parser.parse_args()

    train_composite_loss(args)
