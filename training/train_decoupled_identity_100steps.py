"""
Controlled 100-Step Training Pipeline for Decoupled Identity Cross-Attention in Stable Diffusion.

Trains ONLY the 16 layers of to_k_id and to_v_id (~12.78M parameters).
All other components remain strictly frozen:
- Pretrained UNet original parameters (FROZEN)
- Pretrained VAE (FROZEN)
- Pretrained CLIP Text Encoder (FROZEN)
- ArcFace backbone (FROZEN)
- Phase-4 Regional Fusion (FROZEN)
- Old 77x768 identity adapter (FROZEN)

Anti-Leakage Guarantee:
- Target image provides WHERE mask context and ground truth reconstruction target only.
- Identity conditioning T_fused [B, 256, 512] derives exclusively from K=3 verified same-identity references.

Diagnostics at Steps 0, 25, 50, 75, 100:
- Fixed target CelebA-HQ 7367, exact 84,250-pixel mask, fixed latents and timestep.
- Measures: reference sensitivity, different-ID sensitivity, UNet epsilon response,
  masked vs unmasked response, target leakage, K/V norms, attention entropy, concentration,
  frozen parameter integrity, and failure mode detection.
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
from torch.utils.data import DataLoader
from PIL import Image

from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset
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
    arcface: nn.Module,
    phase4: nn.Module
):
    """Generates detailed parameter audit tables and verifies frozen/trainable constraints."""
    modules = {
        "Decoupled_Identity_K_V": manager.get_trainable_parameters(),
        "UNet_Backbone_Original": [p for n, p in unet.named_parameters() if "to_k_id" not in n and "to_v_id" not in n],
        "VAE_Autoencoder": list(vae.parameters()),
        "CLIP_Text_Encoder": list(text_encoder.parameters()),
        "ArcFace_Backbone": list(arcface.parameters()),
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
        f"| {'TOTAL SYSTEM PARAMETERS':<33} | {total_trainable:<13,d} | {total_frozen:<13,d} | {total_trainable + total_frozen:<13,d} |"
    )
    table_lines.append("+" + "-"*35 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")

    table_str = "\n".join(table_lines)
    return summary, table_str


def build_fixed_diagnostic_data(pipe, arcface, phase4, device):
    """Loads and precomputes the fixed diagnostic target, mask, references, latents, and text emb."""
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

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    def extract_fused(ref_imgs, weights=None):
        tokens_list = []
        with torch.no_grad():
            for r_rgb in ref_imgs:
                r_t = torch.from_numpy(cv2.resize(r_rgb, (256, 256))).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
                tokens_list.append(arcface(r_t)["spatial_tokens"])
            w = torch.tensor([weights if weights is not None else [1.0 / len(ref_imgs)] * len(ref_imgs)], dtype=torch.float32, device=device)
            f_out = phase4(
                target_tokens=None,
                target_mask=mask_tensor_256,
                reference_tokens_list=tokens_list,
                reference_weights=w
            )
            return f_out["fused_tokens"]

    t_fused_A = extract_fused([ref_a_rgb])
    t_fused_B = extract_fused([ref_b_rgb])
    t_fused_C = extract_fused([ref_c_rgb])
    t_fused_diff = extract_fused([diff_id_rgb])

    # Target anti-leakage test token
    with torch.no_grad():
        r_t_a = torch.from_numpy(cv2.resize(ref_a_rgb, (256, 256))).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
        tok_a = arcface(r_t_a)["spatial_tokens"]
        f_leak_test = phase4(
            target_tokens=torch.zeros(1, 256, 512, device=device), # zeroed target passed
            target_mask=mask_tensor_256,
            reference_tokens_list=[tok_a],
            reference_weights=torch.tensor([[1.0]], dtype=torch.float32, device=device)
        )["fused_tokens"]

    # Latent preparation
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
        masked_image_latents = pipe._encode_vae_image(masked_image_tensor, generator=generator)

    mask_latent = F.interpolate(mask_tensor_proc, size=(64, 64), mode="nearest")

    torch.manual_seed(42)
    fixed_noise = torch.randn((1, 4, 64, 64), device=device)
    fixed_timestep = torch.tensor([500], device=device, dtype=torch.long)
    latent_9ch = torch.cat([fixed_noise, mask_latent, masked_image_latents], dim=1)

    # Text conditioning
    prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    prompt_tok = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        clean_text_emb = pipe.text_encoder(prompt_tok)[0]

    mask_64 = (F.interpolate(mask_tensor_proc, size=(64, 64), mode="nearest") > 0.5).squeeze(0) # [1, 64, 64]

    return {
        "latent_9ch": latent_9ch,
        "fixed_timestep": fixed_timestep,
        "clean_text_emb": clean_text_emb,
        "t_fused_A": t_fused_A,
        "t_fused_B": t_fused_B,
        "t_fused_C": t_fused_C,
        "t_fused_diff": t_fused_diff,
        "f_leak_test": f_leak_test,
        "mask_64": mask_64
    }


def evaluate_diagnostic(
    step: int,
    unet: nn.Module,
    manager: DecoupledIdentityCrossAttentionManager,
    diag_data: dict,
    pristine_vanilla_procs: dict,
    device: torch.device
) -> dict:
    """Executes the fixed diagnostic suite measuring sensitivity, norms, entropy, and leakage."""
    latent = diag_data["latent_9ch"]
    timestep = diag_data["fixed_timestep"]
    text_emb = diag_data["clean_text_emb"]
    t_fused_A = diag_data["t_fused_A"]
    t_fused_B = diag_data["t_fused_B"]
    t_fused_C = diag_data["t_fused_C"]
    t_fused_diff = diag_data["t_fused_diff"]
    mask_64 = diag_data["mask_64"]

    # 1. Vanilla Epsilon (lambda = 0.0)
    manager.set_scale(0.0)
    with torch.no_grad():
        eps_vanilla = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()

    # 2. Ref A Epsilon (lambda = 0.1) with diagnostics recording enabled
    manager.set_scale(0.1)
    manager.set_identity_tokens(t_fused_A)
    manager.set_diagnostics_enabled(True)
    with torch.no_grad():
        eps_A = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()
    manager.set_diagnostics_enabled(False)

    # 3. Ref B Epsilon
    manager.set_identity_tokens(t_fused_B)
    with torch.no_grad():
        eps_B = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()

    # 4. Ref C Epsilon
    manager.set_identity_tokens(t_fused_C)
    with torch.no_grad():
        eps_C = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()

    # 5. Different ID Epsilon (3616)
    manager.set_identity_tokens(t_fused_diff)
    with torch.no_grad():
        eps_diff = unet(latent, timestep, encoder_hidden_states=text_emb).sample.clone()

    # Compute Sensitivities
    def compute_diff_metrics(e1, e2):
        l2 = float((e1 - e2).norm())
        rel_l2 = float(l2 / (e1.norm() + 1e-8))
        cos = float(F.cosine_similarity(e1.flatten(), e2.flatten(), dim=0))
        return {"l2_diff": l2, "rel_l2": rel_l2, "cosine_sim": cos}

    diff_AB = compute_diff_metrics(eps_A, eps_B)
    diff_AC = compute_diff_metrics(eps_A, eps_C)
    diff_A_diff = compute_diff_metrics(eps_A, eps_diff)
    diff_A_vanilla = compute_diff_metrics(eps_A, eps_vanilla)

    # Spatial Masked vs Unmasked Response (Ref A vs Vanilla)
    abs_diff = (eps_A - eps_vanilla).abs().squeeze(0) # [4, 64, 64]
    masked_mean = float(abs_diff[:, mask_64[0]].mean()) if mask_64[0].any() else 0.0
    unmasked_mean = float(abs_diff[:, ~mask_64[0]].mean()) if (~mask_64[0]).any() else 0.0
    masked_ratio = float(masked_mean / (unmasked_mean + 1e-8))

    # Target Anti-Leakage
    leak_l2 = float((t_fused_A - diag_data["f_leak_test"]).norm())
    leak_cos = float(F.cosine_similarity(t_fused_A.flatten(), diag_data["f_leak_test"].flatten(), dim=0))

    # Identity K/V Norm Statistics and Attention Entropy / Concentration
    k_norms = []
    v_norms = []
    entropies = []
    max_probs = []
    per_layer_stats = []

    for name, proc in manager.processors.items():
        with torch.no_grad():
            sample_k = proc.to_k_id(t_fused_A) # [1, 256, d]
            sample_v = proc.to_v_id(t_fused_A)
            k_norm = float(sample_k.norm(dim=-1).mean())
            v_norm = float(sample_v.norm(dim=-1).mean())
            k_norms.append(k_norm)
            v_norms.append(v_norm)

            diag = proc.latest_diagnostics
            if diag:
                entropies.append(float(diag["entropy"]))
                max_probs.append(float(diag["max_prob"]))
                per_layer_stats.append({
                    "layer_name": name,
                    "key_norm": k_norm,
                    "value_norm": v_norm,
                    "entropy": float(diag["entropy"]),
                    "max_prob": float(diag["max_prob"])
                })

    mean_k_norm = float(np.mean(k_norms)) if k_norms else 0.0
    mean_v_norm = float(np.mean(v_norms)) if v_norms else 0.0
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    mean_max_prob = float(np.mean(max_probs)) if max_probs else 0.0

    # NaN / Inf checks
    nan_inf = bool(torch.isnan(eps_A).any() or torch.isinf(eps_A).any())

    return {
        "step": step,
        "reference_sensitivity": {
            "ref_A_vs_ref_B": diff_AB,
            "ref_A_vs_ref_C": diff_AC,
            "ref_A_vs_diff_id": diff_A_diff
        },
        "unet_epsilon_response": {
            "identity_vs_vanilla": diff_A_vanilla
        },
        "masked_vs_unmasked": {
            "masked_mean_diff": masked_mean,
            "unmasked_mean_diff": unmasked_mean,
            "masked_ratio": masked_ratio
        },
        "target_leakage": {
            "l2_diff": leak_l2,
            "cosine_sim": leak_cos,
            "verified_zero_leakage": leak_l2 == 0.0
        },
        "identity_kv_norms": {
            "mean_key_norm": mean_k_norm,
            "mean_value_norm": mean_v_norm,
            "per_layer": per_layer_stats
        },
        "identity_attention_stats": {
            "mean_entropy": mean_entropy,
            "mean_max_prob": mean_max_prob
        },
        "nan_inf_detected": nan_inf
    }


def run_training_100steps(
    total_steps: int = 100,
    lambda_scale: float = 0.1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu"
):
    print("=" * 80)
    print("CONTROLLED 100-STEP TRAINING: DECOUPLED IDENTITY CROSS-ATTENTION")
    print("=" * 80)
    total_start_time = time.time()

    # Optimal CPU threading
    torch.set_num_threads(4)
    print(f"PyTorch using {torch.get_num_threads()} CPU threads.")

    device_obj = torch.device(device)

    exp_dir = PROJECT_ROOT / "experiments/exp_decoupled_identity_100steps"
    ckpt_dir = PROJECT_ROOT / "checkpoints/decoupled_identity_attention_100steps"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pretrained Models
    print("\n[Step 1/5] Loading Pretrained Stable Diffusion Inpainting Pipeline & Feature Extractors...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0] if snapshots else "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    ).to(device_obj)

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = DDPMScheduler.from_pretrained(load_path, subfolder="scheduler")

    arcface = RegionalIdentityEncoder(token_dim=512).to(device_obj)
    arcface.eval()

    phase4 = RegionalIdentityTransformer(
        token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0
    ).to(device_obj)
    phase4.eval()

    # 2. Freeze Non-Target Parameters and Attach Decoupled Identity Pathway
    print("\n[Step 2/5] Freezing pretrained backbones and attaching Decoupled Identity Cross-Attention...")
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    for p in arcface.parameters():
        p.requires_grad = False
    for p in phase4.parameters():
        p.requires_grad = False

    pristine_vanilla_processors = dict(unet.attn_processors)

    manager = DecoupledIdentityCrossAttentionManager(unet, id_dim=512, default_scale=lambda_scale)
    manager.original_processors = dict(pristine_vanilla_processors)
    manager.install(zero_init_v=False)

    trainable_params = manager.get_trainable_parameters()
    for p in trainable_params:
        p.requires_grad = True

    param_summary, param_table_str = get_parameter_tables(
        unet=unet,
        manager=manager,
        vae=vae,
        text_encoder=text_encoder,
        arcface=arcface,
        phase4=phase4
    )
    print("\nPARAMETER STATUS AUDIT TABLE:")
    print(param_table_str)

    assert param_summary["Decoupled_Identity_K_V"]["trainable_params"] == 12779520, "Expected exactly 12,779,520 trainable params!"
    assert param_summary["UNet_Backbone_Original"]["trainable_params"] == 0, "UNet original params must be frozen!"
    assert param_summary["VAE_Autoencoder"]["trainable_params"] == 0, "VAE must be frozen!"
    assert param_summary["CLIP_Text_Encoder"]["trainable_params"] == 0, "CLIP must be frozen!"
    assert param_summary["ArcFace_Backbone"]["trainable_params"] == 0, "ArcFace must be frozen!"
    assert param_summary["Phase4_Regional_Fusion"]["trainable_params"] == 0, "Phase 4 must be frozen!"

    # 3. Setup Dataset & Optimizer
    print("\n[Step 3/5] Setting up CelebA-HQ Identity-Group Dataset (K=3 references)...")
    train_dataset = CelebAHQIdentityDataset(
        split="train",
        image_size=(512, 512),
        num_positive_references=3,
        seed=42
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # Text prompt for training
    text_prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    text_tok = tokenizer(text_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device_obj)
    with torch.no_grad():
        clean_text_emb = text_encoder(text_tok)[0]

    # Pre-training snapshot of weights
    initial_weights = {}
    for name, proc in manager.processors.items():
        initial_weights[f"{name}.to_k_id"] = proc.to_k_id.weight.detach().clone()
        initial_weights[f"{name}.to_v_id"] = proc.to_v_id.weight.detach().clone()

    # Precompute diagnostic data
    print("\n[Step 4/5] Precomputing Fixed UNet Diagnostic Dataset (Target 7367, exact mask, Refs A/B/C/Diff)...")
    diag_data = build_fixed_diagnostic_data(pipe, arcface, phase4, device_obj)

    # Initial Diagnostic at Step 0
    print("\nRunning Baseline Diagnostic at Step 0...")
    diag_step0 = evaluate_diagnostic(0, unet, manager, diag_data, pristine_vanilla_processors, device_obj)
    print(f"Step 0: Identity vs Vanilla L2 = {diag_step0['unet_epsilon_response']['identity_vs_vanilla']['l2_diff']:.6f}")
    print(f"Step 0: Ref A vs Diff-ID L2 = {diag_step0['reference_sensitivity']['ref_A_vs_diff_id']['l2_diff']:.6f}")
    print(f"Step 0: Identity Attention Entropy = {diag_step0['identity_attention_stats']['mean_entropy']:.4f}")

    # Save Step 0 Checkpoint
    ckpt_step0_path = ckpt_dir / "step_0.pt"
    manager.save_checkpoint(
        str(ckpt_step0_path),
        extra_data={"step": 0, "optimizer_state_dict": optimizer.state_dict(), "diag": diag_step0}
    )
    print(f"Saved checkpoint: {ckpt_step0_path}")

    # Data structures for tracking
    training_metrics = []
    gradient_metrics = []
    diagnostic_evaluations = [diag_step0]
    frozen_param_integrity = {"step_0": True}

    step_times = []
    process = psutil.Process(os.getpid())

    # 4. Training Loop: 1 to 100
    print(f"\n[Step 5/5] Executing {total_steps} Optimization Steps...")
    data_iter = iter(train_loader)

    for step in range(1, total_steps + 1):
        step_start_time = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        target = batch["target"].to(device_obj)       # [1, 3, 512, 512]
        mask = batch["mask"].to(device_obj)           # [1, 1, 512, 512]
        pos_refs = [r.to(device_obj) for r in batch["pos_references"]] # 3 tensors of [1, 3, 512, 512]
        identity_id = batch["identity_id"][0]

        with torch.no_grad():
            latents = vae.encode(target).latent_dist.sample() * vae.config.scaling_factor
            masked_target = target * (1.0 - mask)
            masked_latents = vae.encode(masked_target).latent_dist.sample() * vae.config.scaling_factor
            mask_latent = F.interpolate(mask, size=(64, 64), mode="nearest")

            # Extract ArcFace spatial tokens for all K=3 references
            ref_tokens_list = []
            for r in pos_refs:
                r_256 = F.interpolate(r, size=(256, 256), mode="bilinear", align_corners=False)
                ref_tokens_list.append(arcface(r_256)["spatial_tokens"])

            mask_256 = F.interpolate(mask, size=(256, 256), mode="nearest")
            ref_weights = torch.ones(1, len(ref_tokens_list), dtype=torch.float32, device=device_obj) / float(len(ref_tokens_list))

            # Phase 4 Regional Fusion (Target tokens strictly None to enforce anti-leakage)
            t_fused = phase4(
                target_tokens=None,
                target_mask=mask_256,
                reference_tokens_list=ref_tokens_list,
                reference_weights=ref_weights
            )["fused_tokens"]

            # Sample timestep and noise
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device_obj).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        latent_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)

        # Set identity conditioning
        manager.set_scale(lambda_scale)
        manager.set_identity_tokens(t_fused)

        optimizer.zero_grad()
        model_pred = unet(latent_input, timesteps, encoder_hidden_states=clean_text_emb).sample
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss_val = float(loss.item())

        loss.backward()

        # Compute gradient norms for K and V
        k_grads = []
        v_grads = []
        for name, proc in manager.processors.items():
            if proc.to_k_id.weight.grad is not None:
                k_grads.append(float(proc.to_k_id.weight.grad.norm().cpu()))
            if proc.to_v_id.weight.grad is not None:
                v_grads.append(float(proc.to_v_id.weight.grad.norm().cpu()))

        mean_k_grad = float(np.mean(k_grads)) if k_grads else 0.0
        mean_v_grad = float(np.mean(v_grads)) if v_grads else 0.0

        optimizer.step()

        step_elapsed = time.time() - step_start_time
        step_times.append(step_elapsed)
        mem_mb = process.memory_info().rss / (1024 * 1024)

        # Calculate max parameter change
        param_deltas = []
        for name, proc in manager.processors.items():
            delta_k = (proc.to_k_id.weight.detach() - initial_weights[f"{name}.to_k_id"]).abs().max().item()
            delta_v = (proc.to_v_id.weight.detach() - initial_weights[f"{name}.to_v_id"]).abs().max().item()
            param_deltas.append(max(delta_k, delta_v))
        max_delta = max(param_deltas)

        # Record step metric
        step_record = {
            "step": step,
            "loss": loss_val,
            "mean_k_grad_norm": mean_k_grad,
            "mean_v_grad_norm": mean_v_grad,
            "max_param_change": max_delta,
            "lambda": lambda_scale,
            "step_time_seconds": step_elapsed,
            "cumulative_time_seconds": float(np.sum(step_times)),
            "memory_rss_mb": mem_mb,
            "identity_id": identity_id
        }
        training_metrics.append(step_record)

        grad_record = {
            "step": step,
            "mean_k_grad": mean_k_grad,
            "mean_v_grad": mean_v_grad,
            "per_layer_k_grads": k_grads,
            "per_layer_v_grads": v_grads
        }
        gradient_metrics.append(grad_record)

        # Log every 10 steps
        if step % 10 == 0 or step == 1:
            print(
                f"[Step {step:03d}/{total_steps}] Loss: {loss_val:.6f} | "
                f"K_grad: {mean_k_grad:.6f} | V_grad: {mean_v_grad:.6f} | "
                f"Max Delta: {max_delta:.6f} | Step Time: {step_elapsed:.1f}s | "
                f"Elapsed: {np.sum(step_times)/60:.1f}m | RAM: {mem_mb:.1f}MB"
            )
            sys.stdout.flush()

        # Checkpoints and full diagnostics at steps 25, 50, 75, 100
        if step in [25, 50, 75, 100]:
            print(f"\n>>> EXECUTING DIAGNOSTIC EVALUATION AT STEP {step} <<<")
            diag_res = evaluate_diagnostic(step, unet, manager, diag_data, pristine_vanilla_processors, device_obj)
            diagnostic_evaluations.append(diag_res)

            # Check frozen parameter integrity
            frozen_ok = True
            for mod_name, param_list in [
                ("UNet_Original", [p for n, p in unet.named_parameters() if "to_k_id" not in n and "to_v_id" not in n]),
                ("VAE", list(vae.parameters())),
                ("CLIP", list(text_encoder.parameters())),
                ("ArcFace", list(arcface.parameters())),
                ("Phase4", list(phase4.parameters()))
            ]:
                for p in param_list:
                    if p.grad is not None and torch.any(p.grad != 0.0):
                        frozen_ok = False
                        break
            frozen_param_integrity[f"step_{step}"] = frozen_ok

            print(f"Step {step} Diagnostic Summary:")
            print(f"  Identity vs Vanilla L2: {diag_res['unet_epsilon_response']['identity_vs_vanilla']['l2_diff']:.6f}")
            print(f"  Ref A vs Ref B (Same ID) L2: {diag_res['reference_sensitivity']['ref_A_vs_ref_B']['l2_diff']:.6f}")
            print(f"  Ref A vs Diff ID (3616) L2:  {diag_res['reference_sensitivity']['ref_A_vs_diff_id']['l2_diff']:.6f}")
            print(f"  Masked / Unmasked Ratio:      {diag_res['masked_vs_unmasked']['masked_ratio']:.4f}")
            print(f"  Attention Entropy:           {diag_res['identity_attention_stats']['mean_entropy']:.4f}")
            print(f"  Attention Concentration:     {diag_res['identity_attention_stats']['mean_max_prob']:.4f}")
            print(f"  Target Zero Leakage:         {diag_res['target_leakage']['verified_zero_leakage']}")
            print(f"  Frozen Params Integrity:     {frozen_ok}")

            # Save Checkpoint
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            manager.save_checkpoint(
                str(ckpt_path),
                extra_data={
                    "step": step,
                    "loss": loss_val,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "diag": diag_res
                }
            )
            print(f"Saved Checkpoint: {ckpt_path}")

            # Failure Mode Analysis
            step0_diff_id = diag_step0["reference_sensitivity"]["ref_A_vs_diff_id"]["l2_diff"]
            curr_diff_id = diag_res["reference_sensitivity"]["ref_A_vs_diff_id"]["l2_diff"]
            step0_id_vanilla = diag_step0["unet_epsilon_response"]["identity_vs_vanilla"]["l2_diff"]
            curr_id_vanilla = diag_res["unet_epsilon_response"]["identity_vs_vanilla"]["l2_diff"]
            uniform_entropy = np.log(256) # ~5.5451

            print("\nFailure Mode Audit at Step {step}:")
            print(f"  Diff-ID Sensitivity: Step 0 = {step0_diff_id:.6f} -> Step {step} = {curr_diff_id:.6f}")
            print(f"  Identity-vs-Vanilla: Step 0 = {step0_id_vanilla:.6f} -> Step {step} = {curr_id_vanilla:.6f}")
            print(f"  Entropy vs Uniform: {diag_res['identity_attention_stats']['mean_entropy']:.4f} (Uniform = {uniform_entropy:.4f})")

            # Check if epsilon changes grew without corresponding reference sensitivity
            id_vanilla_growth = curr_id_vanilla / (step0_id_vanilla + 1e-8)
            diff_id_growth = curr_diff_id / (step0_diff_id + 1e-8)
            print(f"  Growth Ratio: Identity-vs-Vanilla Growth = {id_vanilla_growth:.4f}x | Diff-ID Growth = {diff_id_growth:.4f}x")

            sys.stdout.flush()

            # Failure Mode Trigger Check
            failure_reasons = []
            if curr_diff_id <= step0_diff_id:
                failure_reasons.append(f"Ref A vs Different-ID response did NOT increase ({curr_diff_id:.6f} <= {step0_diff_id:.6f})")
            if abs(diag_res['identity_attention_stats']['mean_entropy'] - uniform_entropy) < 0.05:
                failure_reasons.append(f"Identity attention remains effectively uniform (entropy {diag_res['identity_attention_stats']['mean_entropy']:.4f} vs uniform {uniform_entropy:.4f})")
            if id_vanilla_growth > 1.3 and diff_id_growth < 1.02:
                failure_reasons.append(f"Identity-conditioned epsilon changes grew ({id_vanilla_growth:.2f}x) without corresponding reference sensitivity ({diff_id_growth:.2f}x)")

            # Check if loss decreased while identity sensitivity failed to improve
            if len(failure_reasons) >= 2 and loss_val < training_metrics[0]["loss"]:
                print(f"\n================================================================================")
                print(f"FAILURE MODE DETECTED AT CHECKPOINT STEP {step}:")
                for r in failure_reasons:
                    print(f"  * {r}")
                print("CONCLUSION: The model is learning denoising but NOT identity conditioning.")
                print(f"STOPPING TRAINING AT STEP {step} AS DIRECTED.")
                print(f"================================================================================")
                sys.stdout.flush()
                break

            # Intermediate save of metrics
            with open(exp_dir / "training_metrics.json", "w") as f:
                json.dump(training_metrics, f, indent=2)
            with open(exp_dir / "gradient_metrics.json", "w") as f:
                json.dump(gradient_metrics, f, indent=2)
            with open(exp_dir / "diagnostic_evaluations.json", "w") as f:
                json.dump(diagnostic_evaluations, f, indent=2)
            with open(exp_dir / "frozen_parameter_integrity.json", "w") as f:
                json.dump(frozen_param_integrity, f, indent=2)

    # 5. Finalize and Save Artifacts
    total_elapsed = time.time() - total_start_time
    print(f"\nTraining completed in {total_elapsed/60:.2f} minutes ({total_elapsed:.1f}s).")

    training_config = {
        "model": "runwayml/stable-diffusion-inpainting",
        "total_steps": total_steps,
        "batch_size": 1,
        "K_references": 3,
        "lambda_scale": lambda_scale,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "trainable_parameters": param_summary["Decoupled_Identity_K_V"]["trainable_params"],
        "frozen_parameters": sum(s["total_params"] for k, s in param_summary.items() if k != "Decoupled_Identity_K_V"),
        "resolution": "512x512",
        "total_runtime_seconds": total_elapsed,
        "device": device
    }

    with open(exp_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    with open(exp_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    with open(exp_dir / "gradient_metrics.json", "w") as f:
        json.dump(gradient_metrics, f, indent=2)
    with open(exp_dir / "diagnostic_evaluations.json", "w") as f:
        json.dump(diagnostic_evaluations, f, indent=2)
    with open(exp_dir / "frozen_parameter_integrity.json", "w") as f:
        json.dump(frozen_param_integrity, f, indent=2)

    print(f"\nAll artifacts successfully saved to {exp_dir}")
    print("=" * 80)
    print("FINISHED CONTROLLED 100-STEP RUN. STOPPING AS DIRECTED.")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="100-step controlled training of decoupled identity cross-attention")
    parser.add_argument("--steps", type=int, default=100, help="Total optimization steps (default: 100)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (default: cpu)")
    parser.add_argument("--lambda_scale", type=float, default=0.1, help="Lambda scale factor (default: 0.1)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = parser.parse_args()

    run_training_100steps(
        total_steps=args.steps,
        lambda_scale=args.lambda_scale,
        learning_rate=args.lr,
        weight_decay=1e-4,
        device=args.device
    )
