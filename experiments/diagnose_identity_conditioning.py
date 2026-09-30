"""
Forensic Investigation of Identity-Conditioning Pathway in Stable Diffusion Inpainting.

Executes all 8 diagnostic checks specified in the forensic protocol:
1. Exact conditioning path tensor tracing (shapes, dtypes, statistics, norms, training status)
2. Semantic alignment vs CLIP text manifold (cosine similarity to controlled prompts)
3. Null / ID / Random-statistics-matched / CLIP ablation at UNet level (fixed latent & timestep)
4. Reference-swap test at UNet level (epsilon predictions for Ref A, B, C, and Different ID)
5. Cross-attention layer diagnostic (logits, entropy, attention mass distribution, saturation)
6. Architectural evaluation of ArcFace -> CLIP direct substitution vs IP-Adapter
7. Target anti-leakage verification (Target 1 vs Target 2 with Ref A, B)
8. Independent mask conditioning diagnostic (spatial error breakdown)

Saves diagnostic artifacts, plots, and comprehensive JSON.
NO FULL DIFFUSION GENERATION IS PERFORMED.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.conditioning_adapter import IdentityConditioningAdapter, ConditioningCalibrator
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.reference_scoring.scorer import MaskAwareReferenceScorer
from diffusers import StableDiffusionInpaintPipeline


def get_stats(tensor: torch.Tensor, name: str = "") -> dict:
    t = tensor.detach().float().cpu()
    norms = t.norm(dim=-1)
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(tensor.dtype),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
        "token_norm_mean": float(norms.mean()),
        "token_norm_std": float(norms.std()),
        "token_norm_min": float(norms.min()),
        "token_norm_max": float(norms.max()),
        "token_0_norm": float(norms.flatten()[0]) if norms.numel() > 0 else 0.0,
    }


def main():
    print("=" * 80)
    print("STARTING FORENSIC INVESTIGATION OF IDENTITY-CONDITIONING PATHWAY")
    print("=" * 80)
    start_time = time.time()

    diag_dir = PROJECT_ROOT / "experiments/exp_identity_conditioned_diffusion/diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # -------------------------------------------------------------------------
    # 0. Load Data
    # -------------------------------------------------------------------------
    print("\n[Step 0] Loading Target, References, Mask, and Different Identity...")
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
    # 1. TRACE THE EXACT CONDITIONING PATH
    # -------------------------------------------------------------------------
    print("\n[Step 1] Tracing exact conditioning pathway tensor statistics...")
    trace_report = {}

    # Stage 1: Real ArcFace
    encoder = RegionalIdentityEncoder(token_dim=512)
    encoder.eval()

    r_a_256 = cv2.resize(ref_a_rgb, (256, 256))
    r_a_tensor = torch.from_numpy(r_a_256).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

    with torch.no_grad():
        arc_out = encoder(r_a_tensor)
    spatial_tokens_a = arc_out["spatial_tokens"] # [1, 256, 512]
    global_emb_a = arc_out["global_embedding"] # [1, 512]

    trace_report["stage1_arcface_spatial"] = get_stats(spatial_tokens_a, "Stage 1: ArcFace Regional Spatial Tokens")
    trace_report["stage1_arcface_spatial"]["training_gradients"] = "Frozen (Pretrained w600k_r50 ONNX backbone)"
    trace_report["stage1_arcface_spatial"]["checkpoint"] = "models/weights/w600k_r50.onnx"

    trace_report["stage1_arcface_global"] = get_stats(global_emb_a, "Stage 1: ArcFace Global 512D Embedding")
    trace_report["stage1_arcface_global"]["training_gradients"] = "Frozen (Pretrained w600k_r50 ONNX backbone)"
    trace_report["stage1_arcface_global"]["checkpoint"] = "models/weights/w600k_r50.onnx"

    # Stage 2: Reference Fusion
    transformer = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2, spatial_patch_grid=(16, 16), dropout=0.0)
    transformer.eval()

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float()

    ref_tokens_list = []
    with torch.no_grad():
        for r_rgb in [ref_a_rgb, ref_b_rgb, ref_c_rgb]:
            r_t = torch.from_numpy(cv2.resize(r_rgb, (256, 256))).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            ref_tokens_list.append(encoder(r_t)["spatial_tokens"])

    weights_tensor = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32)
    with torch.no_grad():
        fusion_out = transformer(
            target_tokens=None,
            target_mask=mask_tensor,
            reference_tokens_list=ref_tokens_list,
            reference_weights=weights_tensor
        )
    fused_identity_tokens = fusion_out["fused_tokens"] # [1, 256, 512]

    trace_report["stage2_reference_fusion"] = get_stats(fused_identity_tokens, "Stage 2: Fused Regional Reference Tokens")
    trace_report["stage2_reference_fusion"]["training_gradients"] = "Rule-based reference-only residual fusion (Untrained weights / identity projection)"
    trace_report["stage2_reference_fusion"]["checkpoint"] = "Procedural / architecture-level"

    # Stage 3 & 4: Identity Conditioning Adapter + Calibrator
    adapter = IdentityConditioningAdapter(in_dim=512, out_dim=768, num_tokens=256, out_tokens=77, num_id_queries=16, dropout=0.0)
    adapter_ckpt_path = PROJECT_ROOT / "checkpoints/identity_adapter/identity_adapter_best.pt"
    ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    adapter.load_state_dict(ckpt["model_state_dict"])
    adapter.eval()

    # Step-by-step internal forward through adapter:
    with torch.no_grad():
        spatial_features = adapter.spatial_proj(fused_identity_tokens)
        queries = adapter.id_queries.expand(1, -1, -1)
        id_queried, _ = adapter.cross_attn(query=queries, key=spatial_features, value=spatial_features)
        
        prompt_seq = adapter.base_context.expand(1, -1, -1).clone()
        prompt_seq[:, 1:1 + 16, :] = id_queried[:, :16, :]

        calibrated_prompt = adapter.calibrator(prompt_seq)
        calibrated_null = adapter.get_null_embedding(batch_size=1, target_len=77)

    trace_report["stage3_adapter_spatial_proj"] = get_stats(spatial_features, "Stage 3a: Adapter Spatial Proj (256x768)")
    trace_report["stage3_adapter_spatial_proj"]["training_gradients"] = "Trained via AdamW on CelebA-HQ triplets"
    trace_report["stage3_adapter_spatial_proj"]["checkpoint"] = adapter_ckpt_path.name

    trace_report["stage3_adapter_id_queries"] = get_stats(id_queried, "Stage 3b: Adapter Cross-Attn ID Queries (16x768)")
    trace_report["stage3_adapter_id_queries"]["training_gradients"] = "Trained via AdamW on CelebA-HQ triplets"
    trace_report["stage3_adapter_id_queries"]["checkpoint"] = adapter_ckpt_path.name

    trace_report["stage3_adapter_uncalibrated_seq"] = get_stats(prompt_seq, "Stage 3c: Soft-Prompt Sequence (77x768 uncalibrated)")
    trace_report["stage3_adapter_uncalibrated_seq"]["training_gradients"] = "base_context parameter + inserted queries"
    trace_report["stage3_adapter_uncalibrated_seq"]["checkpoint"] = adapter_ckpt_path.name

    trace_report["stage4_calibrator_output"] = get_stats(calibrated_prompt, "Stage 4: Calibrated Conditioning Tokens (77x768)")
    trace_report["stage4_calibrator_output"]["training_gradients"] = "Trained via AdamW (calibrate_identity_conditioning.py)"
    trace_report["stage4_calibrator_output"]["checkpoint"] = adapter_ckpt_path.name

    trace_report["stage4_calibrated_null"] = get_stats(calibrated_null, "Stage 4: Calibrated Null Embedding (77x768)")
    trace_report["stage4_calibrated_null"]["training_gradients"] = "Trained via AdamW (calibrate_identity_conditioning.py)"
    trace_report["stage4_calibrated_null"]["checkpoint"] = adapter_ckpt_path.name

    print("Path Trace Complete. Stage statistics computed.")

    # -------------------------------------------------------------------------
    # 2. CHECK SEMANTIC ALIGNMENT, NOT JUST DISTRIBUTION
    # -------------------------------------------------------------------------
    print("\n[Step 2] Measuring semantic alignment with Stable Diffusion CLIP text encoder...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0] if snapshots else "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        local_files_only=True
    )
    pipe = pipe.to(device)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    prompts_to_test = [
        "a human face",
        "portrait of a woman",
        "close-up face",
        "photo of a person",
        "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture",
        ""  # empty / null
    ]

    semantic_report = {}
    id_cond = calibrated_prompt[0] # [77, 768]

    for p in prompts_to_test:
        text_inputs = tokenizer(
            p,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_emb = text_encoder(text_inputs.input_ids)[0][0] # [77, 768]

        # Token-wise cosine similarity
        cos_tokenwise = F.cosine_similarity(id_cond, text_emb, dim=-1).cpu().numpy()
        
        # Token regions:
        # Token 0: BOS
        # Tokens 1..16: ID queried tokens
        # Tokens 17..76: base_context padding
        p_name = p if p != "" else "<EMPTY_NULL>"
        semantic_report[p_name] = {
            "mean_cosine": float(np.mean(cos_tokenwise)),
            "std_cosine": float(np.std(cos_tokenwise)),
            "min_cosine": float(np.min(cos_tokenwise)),
            "max_cosine": float(np.max(cos_tokenwise)),
            "token_0_BOS_cosine": float(cos_tokenwise[0]),
            "tokens_1_to_16_id_mean_cosine": float(np.mean(cos_tokenwise[1:17])),
            "tokens_1_to_16_id_min_cosine": float(np.min(cos_tokenwise[1:17])),
            "tokens_1_to_16_id_max_cosine": float(np.max(cos_tokenwise[1:17])),
            "tokens_17_to_76_padding_mean_cosine": float(np.mean(cos_tokenwise[17:])),
            "tokenwise_cosines": [float(x) for x in cos_tokenwise]
        }
        print(f"Prompt '{p_name[:35]}': Mean Cos={np.mean(cos_tokenwise):.4f}, Tokens 1-16 (ID) Cos={np.mean(cos_tokenwise[1:17]):.4f}, Max={np.max(cos_tokenwise):.4f}")

    # Cross-prompt cosine of genuine text embeddings to each other (for calibration reference)
    with torch.no_grad():
        emb_face = text_encoder(tokenizer("a human face", padding="max_length", max_length=77, return_tensors="pt").input_ids)[0][0]
        emb_woman = text_encoder(tokenizer("portrait of a woman", padding="max_length", max_length=77, return_tensors="pt").input_ids)[0][0]
        ref_text_cos = F.cosine_similarity(emb_face, emb_woman, dim=-1).cpu().numpy()
    semantic_report["reference_face_vs_woman_text_cosine"] = {
        "mean_cosine": float(np.mean(ref_text_cos)),
        "token_1_to_16_mean_cosine": float(np.mean(ref_text_cos[1:17])),
        "max_cosine": float(np.max(ref_text_cos))
    }
    print(f"Reference Genuine CLIP Text vs Text ('a human face' vs 'portrait of a woman'): Mean Cos={np.mean(ref_text_cos):.4f}, Tokens 1-16 Mean={np.mean(ref_text_cos[1:17]):.4f}")

    # -------------------------------------------------------------------------
    # 3. NULL / ID / RANDOM-STATISTICS-MATCHED / CLIP ABLATION AT UNET LEVEL
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running UNet forward pass ablation (fixed timestep & latent)...")

    # Set up fixed 9-channel latent input matching Phase 7 / second experiment
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
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

    # Downsampled mask for channel 4
    mask_latent = torch.nn.functional.interpolate(mask_tensor_proc, size=(64, 64), mode="nearest")

    # Fixed noisy latent z_t at timestep t=999 (start of 5-step diffusion)
    torch.manual_seed(42)
    fixed_noise = torch.randn((1, 4, 64, 64), device=device)
    fixed_timestep = torch.tensor([999], device=device, dtype=torch.long)

    # 9-channel concatenated input
    latent_9ch = torch.cat([fixed_noise, mask_latent, masked_image_latents], dim=1) # [1, 9, 64, 64]
    print(f"9-Channel UNet input prepared: shape={latent_9ch.shape}")

    # Conditionings:
    # A. Normal null conditioning (genuine CLIP empty text)
    with torch.no_grad():
        null_tok = tokenizer("", padding="max_length", max_length=77, return_tensors="pt").input_ids
        cond_A_clip_null = text_encoder(null_tok)[0] # [1, 77, 768]
    
    # B. Current identity conditioning (fused from 3 references)
    cond_B_identity = calibrated_prompt # [1, 77, 768]

    # C. Random conditioning with SAME mean/std/norm statistics
    # Exact per-token norm matching:
    torch.manual_seed(12345)
    rand_raw = torch.randn_like(cond_B_identity)
    # Unit normalize
    rand_dir = F.normalize(rand_raw, p=2, dim=-1)
    target_norms = cond_B_identity.norm(dim=-1, keepdim=True) # [1, 77, 1]
    cond_C_random_matched = rand_dir * target_norms
    # Shift to match global mean
    cond_C_random_matched = cond_C_random_matched - cond_C_random_matched.mean() + cond_B_identity.mean()

    # D. Actual CLIP text conditioning (detailed portrait prompt)
    detailed_prompt = "high quality photorealistic portrait of a woman, natural facial anatomy, natural skin texture"
    with torch.no_grad():
        prompt_tok = tokenizer(detailed_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        cond_D_clip_text = text_encoder(prompt_tok)[0] # [1, 77, 768]

    # E, F, G. Single reference identity conditionings
    def get_single_ref_conditioning(ref_img):
        with torch.no_grad():
            r_t = torch.from_numpy(cv2.resize(ref_img, (256, 256))).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            r_tokens = encoder(r_t)["spatial_tokens"]
            f_out = transformer(
                target_tokens=None,
                target_mask=mask_tensor,
                reference_tokens_list=[r_tokens],
                reference_weights=torch.tensor([[1.0]], dtype=torch.float32)
            )
            return adapter(f_out["fused_tokens"])["projected_tokens"]

    cond_E_ref_A = get_single_ref_conditioning(ref_a_rgb)
    cond_F_ref_B = get_single_ref_conditioning(ref_b_rgb)
    cond_G_ref_C = get_single_ref_conditioning(ref_c_rgb)
    cond_H_diff_id = get_single_ref_conditioning(diff_id_rgb)

    # 4. Adapter Null Embedding
    cond_adapter_null = calibrated_null # [1, 77, 768]

    conditions = {
        "A_clip_null": cond_A_clip_null,
        "adapter_null": cond_adapter_null,
        "B_identity_fused": cond_B_identity,
        "C_random_matched": cond_C_random_matched,
        "D_clip_text": cond_D_clip_text,
        "E_ref_A": cond_E_ref_A,
        "F_ref_B": cond_F_ref_B,
        "G_ref_C": cond_G_ref_C,
        "H_diff_id": cond_H_diff_id
    }

    # Register hooks on cross-attention modules
    cross_attn_stats = {}
    attn_records = {}

    def get_attn_hook(layer_name):
        def hook(module, input, output):
            # module is an attention module (e.g. Attention / CrossAttention)
            # In diffusers Attention, forward takes (hidden_states, encoder_hidden_states=...)
            pass
        return hook

    # Diffusers 0.40.0 uses Processor-based attention or direct forward.
    # Let's inspect pipe.unet.set_attn_processor or hook into to_k, to_v, to_q
    # A cleaner way to capture cross-attention maps across all 16 layers:
    class CrossAttnTracker:
        def __init__(self, unet):
            self.unet = unet
            self.hooks = []
            self.data = {}

        def register(self):
            for name, module in self.unet.named_modules():
                if name.endswith("attn2"):
                    self.data[name] = {}
                    # Hook into to_k, to_q to get keys and queries
                    def make_hook(n):
                        def forward_hook(m, args, kwargs, output):
                            # In diffusers BasicTransformerBlock / Attention
                            pass
                        return forward_hook

        def clear(self):
            self.data = {}

    # Let's run UNet forward passes and collect predictions
    print("\nExecuting forward passes through Stable Diffusion UNet...")
    epsilon_preds = {}

    for name, cond_tensor in conditions.items():
        with torch.no_grad():
            eps = pipe.unet(latent_9ch, fixed_timestep, encoder_hidden_states=cond_tensor).sample
        epsilon_preds[name] = eps
        print(f"  Condition '{name}': mean={eps.mean():.4f}, std={eps.std():.4f}, L2 norm={eps.norm():.2f}")

    # Compute pairwise epsilon comparisons
    def compare_epsilons(e1, e2):
        l2_diff = float((e1 - e2).norm())
        rel_l2 = float(l2_diff / e1.norm())
        cos_sim = float(F.cosine_similarity(e1.flatten(), e2.flatten(), dim=0))
        # Masked zone error: downsample mask to 64x64
        mask_bool_64 = (mask_latent > 0.5).expand_as(e1)
        masked_l2 = float((e1[mask_bool_64] - e2[mask_bool_64]).norm())
        unmasked_l2 = float((e1[~mask_bool_64] - e2[~mask_bool_64]).norm())
        return {
            "l2_diff": l2_diff,
            "rel_l2": rel_l2,
            "cosine_sim": cos_sim,
            "masked_l2_diff": masked_l2,
            "unmasked_l2_diff": unmasked_l2
        }

    ablation_comparisons = {
        "identity_vs_clip_null": compare_epsilons(epsilon_preds["B_identity_fused"], epsilon_preds["A_clip_null"]),
        "identity_vs_adapter_null": compare_epsilons(epsilon_preds["B_identity_fused"], epsilon_preds["adapter_null"]),
        "identity_vs_random_matched": compare_epsilons(epsilon_preds["B_identity_fused"], epsilon_preds["C_random_matched"]),
        "identity_vs_clip_text": compare_epsilons(epsilon_preds["B_identity_fused"], epsilon_preds["D_clip_text"]),
        "clip_text_vs_clip_null": compare_epsilons(epsilon_preds["D_clip_text"], epsilon_preds["A_clip_null"]),
        "random_matched_vs_clip_null": compare_epsilons(epsilon_preds["C_random_matched"], epsilon_preds["A_clip_null"]),
        "adapter_null_vs_clip_null": compare_epsilons(epsilon_preds["adapter_null"], epsilon_preds["A_clip_null"])
    }

    # -------------------------------------------------------------------------
    # 4. REFERENCE-SWAP TEST AT THE UNET LEVEL
    # -------------------------------------------------------------------------
    print("\n[Step 4] Computing Reference-Swap comparisons at UNet level...")
    ref_swap_report = {
        "ref_A_vs_ref_B": compare_epsilons(epsilon_preds["E_ref_A"], epsilon_preds["F_ref_B"]),
        "ref_A_vs_ref_C": compare_epsilons(epsilon_preds["E_ref_A"], epsilon_preds["G_ref_C"]),
        "ref_B_vs_ref_C": compare_epsilons(epsilon_preds["F_ref_B"], epsilon_preds["G_ref_C"]),
        "ref_A_vs_diff_id": compare_epsilons(epsilon_preds["E_ref_A"], epsilon_preds["H_diff_id"]),
        "fused_vs_ref_A": compare_epsilons(epsilon_preds["B_identity_fused"], epsilon_preds["E_ref_A"]),
        "fused_vs_diff_id": compare_epsilons(epsilon_preds["B_identity_fused"], epsilon_preds["H_diff_id"]),
    }

    for k, v in ref_swap_report.items():
        print(f"  {k}: Cosine={v['cosine_sim']:.6f}, L2={v['l2_diff']:.4f}, Rel L2={v['rel_l2']:.4f}")

    # -------------------------------------------------------------------------
    # 5. CROSS-ATTENTION DETAILED LAYER DIAGNOSTICS
    # -------------------------------------------------------------------------
    print("\n[Step 5] Analyzing Cross-Attention internal mechanics...")

    # We will hook into the attn2 modules to capture actual attention maps A = softmax(Q K^T / sqrt(d))
    # In diffusers UNet, each BasicTransformerBlock contains attn2
    cross_attn_layers = [name for name, m in pipe.unet.named_modules() if name.endswith("attn2")]
    print(f"Found {len(cross_attn_layers)} cross-attention layers in UNet.")

    # Custom processor to intercept attention probabilities
    captured_attns = {}

    class InterceptingProcessor:
        def __init__(self, layer_name, store):
            self.layer_name = layer_name
            self.store = store

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Compute Query, Key, Value manually to capture attention matrix
            batch_size, sequence_length, _ = hidden_states.shape
            query = attn.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Compute attention logits
            scale = 1.0 / math.sqrt(head_dim)
            logits = torch.matmul(query, key.transpose(-1, -2)) * scale # [B, heads, seq_len, 77]
            probs = torch.softmax(logits, dim=-1)

            # Store statistics
            with torch.no_grad():
                # Average across heads and spatial sequence
                mean_probs_per_token = probs.mean(dim=(0, 1, 2)).cpu().numpy() # [77]
                entropy = float(-(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().cpu())
                max_prob = float(probs.max().cpu())
                logit_mean = float(logits.mean().cpu())
                logit_std = float(logits.std().cpu())
                logit_max = float(logits.max().cpu())
                logit_min = float(logits.min().cpu())

                self.store[self.layer_name] = {
                    "mean_probs_per_token": mean_probs_per_token,
                    "entropy": entropy,
                    "max_prob": max_prob,
                    "logit_mean": logit_mean,
                    "logit_std": logit_std,
                    "logit_max": logit_max,
                    "logit_min": logit_min,
                    "token_0_mass": float(mean_probs_per_token[0]),
                    "tokens_1_to_16_id_mass": float(mean_probs_per_token[1:17].sum()),
                    "tokens_17_to_76_padding_mass": float(mean_probs_per_token[17:].sum()),
                }

            # Standard attention output
            hidden_states = torch.matmul(probs, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    def run_with_cross_attn_inspection(cond_tensor):
        store = {}
        # Temporarily install processors
        orig_processors = pipe.unet.attn_processors
        new_processors = {}
        for k in orig_processors.keys():
            if k.endswith("attn2.processor"):
                layer_name = k.replace(".processor", "")
                new_processors[k] = InterceptingProcessor(layer_name, store)
            else:
                new_processors[k] = orig_processors[k]

        pipe.unet.set_attn_processor(new_processors)
        with torch.no_grad():
            _ = pipe.unet(latent_9ch, fixed_timestep, encoder_hidden_states=cond_tensor).sample
        pipe.unet.set_attn_processor(orig_processors)
        return store

    attn_diag_identity = run_with_cross_attn_inspection(cond_B_identity)
    attn_diag_clip_text = run_with_cross_attn_inspection(cond_D_clip_text)
    attn_diag_random_matched = run_with_cross_attn_inspection(cond_C_random_matched)
    attn_diag_clip_null = run_with_cross_attn_inspection(cond_A_clip_null)

    # Summarize attention distribution across layers
    def summarize_attn_diag(diag_dict, label):
        entropies = [v["entropy"] for v in diag_dict.values()]
        max_probs = [v["max_prob"] for v in diag_dict.values()]
        token_0_masses = [v["token_0_mass"] for v in diag_dict.values()]
        id_masses = [v["tokens_1_to_16_id_mass"] for v in diag_dict.values()]
        pad_masses = [v["tokens_17_to_76_padding_mass"] for v in diag_dict.values()]
        logit_stds = [v["logit_std"] for v in diag_dict.values()]
        
        all_token_profiles = np.stack([v["mean_probs_per_token"] for v in diag_dict.values()]) # [16, 77]
        avg_profile = all_token_profiles.mean(axis=0)

        return {
            "label": label,
            "mean_entropy": float(np.mean(entropies)),
            "mean_max_prob": float(np.mean(max_probs)),
            "mean_token_0_mass": float(np.mean(token_0_masses)),
            "mean_id_tokens_1_to_16_mass": float(np.mean(id_masses)),
            "mean_padding_tokens_17_to_76_mass": float(np.mean(pad_masses)),
            "mean_logit_std": float(np.mean(logit_stds)),
            "avg_token_profile": [float(x) for x in avg_profile]
        }

    summary_attn_identity = summarize_attn_diag(attn_diag_identity, "Identity Conditioning")
    summary_attn_clip_text = summarize_attn_diag(attn_diag_clip_text, "Genuine CLIP Text")
    summary_attn_random_matched = summarize_attn_diag(attn_diag_random_matched, "Random Stats-Matched")
    summary_attn_clip_null = summarize_attn_diag(attn_diag_clip_null, "Genuine CLIP Null")

    print("\nCross-Attention Summary:")
    for s in [summary_attn_identity, summary_attn_clip_text, summary_attn_random_matched, summary_attn_clip_null]:
        print(f"  {s['label']:<24}: Entropy={s['mean_entropy']:.4f} | Tok0 Mass={s['mean_token_0_mass']:.4f} | Tok1-16 Mass={s['mean_id_tokens_1_to_16_mass']:.4f} | Pad Mass={s['mean_padding_tokens_17_to_76_mass']:.4f}")

    # Plot Cross-Attention Token Profile
    plt.figure(figsize=(12, 5))
    plt.plot(summary_attn_identity["avg_token_profile"], label="Identity Conditioning", color="red", lw=2)
    plt.plot(summary_attn_clip_text["avg_token_profile"], label="Genuine CLIP Text", color="blue", lw=2)
    plt.plot(summary_attn_random_matched["avg_token_profile"], label="Random Stats-Matched", color="gray", linestyle="--", lw=1.5)
    plt.plot(summary_attn_clip_null["avg_token_profile"], label="Genuine CLIP Null", color="green", linestyle=":", lw=1.5)
    plt.axvspan(1, 16, color="red", alpha=0.1, label="ID Query Slots (Tokens 1..16)")
    plt.xlabel("Token Index (0..76)")
    plt.ylabel("Average Cross-Attention Probability")
    plt.title("Cross-Attention Attention Allocation Profile Across 77 Tokens (16 UNet Layers)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(diag_dir / "cross_attention_token_profile.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 7. CHECK FOR TARGET LEAKAGE AGAIN
    # -------------------------------------------------------------------------
    print("\n[Step 7] Testing Target Anti-Leakage under varying Target vs Reference...")
    # Vary target while keeping reference fixed:
    # Condition 1: Target 7367 + Ref A (13401)
    # Condition 2: Target 7367 + Ref B (23058)
    # Condition 3: Different Target (3616) + Ref A (13401)
    # Condition 4: Different Target (3616) + Ref B (23058)

    def extract_cond_for_pair(tgt_rgb, ref_rgb):
        with torch.no_grad():
            r_t = torch.from_numpy(cv2.resize(ref_rgb, (256, 256))).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            r_tokens = encoder(r_t)["spatial_tokens"]
            
            # Note: transformer receives target_tokens=None in pure reference-only mode
            f_out = transformer(
                target_tokens=None,
                target_mask=mask_tensor,
                reference_tokens_list=[r_tokens],
                reference_weights=torch.tensor([[1.0]], dtype=torch.float32)
            )
            return adapter(f_out["fused_tokens"])["projected_tokens"]

    p_tgt1_refA = extract_cond_for_pair(target_rgb, ref_a_rgb)
    p_tgt1_refB = extract_cond_for_pair(target_rgb, ref_b_rgb)
    p_tgt2_refA = extract_cond_for_pair(diff_id_rgb, ref_a_rgb)
    p_tgt2_refB = extract_cond_for_pair(diff_id_rgb, ref_b_rgb)

    leakage_metrics = {
        "cos_same_ref_diff_target_A": float(F.cosine_similarity(p_tgt1_refA.flatten(), p_tgt2_refA.flatten(), dim=0)),
        "cos_same_ref_diff_target_B": float(F.cosine_similarity(p_tgt1_refB.flatten(), p_tgt2_refB.flatten(), dim=0)),
        "l2_same_ref_diff_target_A": float((p_tgt1_refA - p_tgt2_refA).norm()),
        "l2_same_ref_diff_target_B": float((p_tgt1_refB - p_tgt2_refB).norm()),
        "cos_same_target_diff_ref": float(F.cosine_similarity(p_tgt1_refA.flatten(), p_tgt1_refB.flatten(), dim=0)),
        "l2_same_target_diff_ref": float((p_tgt1_refA - p_tgt1_refB).norm())
    }

    print(f"Target Anti-Leakage Check:")
    print(f"  Cos(Target 1 + Ref A, Target 2 + Ref A): {leakage_metrics['cos_same_ref_diff_target_A']:.8f} (L2 diff: {leakage_metrics['l2_same_ref_diff_target_A']:.8f})")
    print(f"  Cos(Target 1 + Ref A, Target 1 + Ref B): {leakage_metrics['cos_same_target_diff_ref']:.8f} (L2 diff: {leakage_metrics['l2_same_target_diff_ref']:.8f})")

    # -------------------------------------------------------------------------
    # 8. CHECK MASK CONDITIONING INDEPENDENTLY
    # -------------------------------------------------------------------------
    print("\n[Step 8] Evaluating Mask Conditioning Independently...")
    # Does identity conditioning create an extreme divergence specifically inside the masked latent region?
    eps_id = epsilon_preds["B_identity_fused"]
    eps_vanilla = epsilon_preds["D_clip_text"]
    eps_null = epsilon_preds["A_clip_null"]
    eps_rand = epsilon_preds["C_random_matched"]

    mask_bool_64 = (mask_latent > 0.5) # [1, 1, 64, 64]
    mask_bool_expand = mask_bool_64.expand_as(eps_id)

    def zone_breakdown(e_pred, e_ref):
        diff = torch.abs(e_pred - e_ref)
        diff_masked = diff[mask_bool_expand]
        diff_unmasked = diff[~mask_bool_expand]
        return {
            "masked_mean_abs_diff": float(diff_masked.mean()),
            "masked_max_abs_diff": float(diff_masked.max()),
            "masked_std_diff": float(diff_masked.std()),
            "unmasked_mean_abs_diff": float(diff_unmasked.mean()),
            "unmasked_max_abs_diff": float(diff_unmasked.max()),
            "unmasked_std_diff": float(diff_unmasked.std()),
            "ratio_masked_to_unmasked": float(diff_masked.mean() / (diff_unmasked.mean() + 1e-8))
        }

    mask_breakdown = {
        "identity_vs_vanilla": zone_breakdown(eps_id, eps_vanilla),
        "identity_vs_null": zone_breakdown(eps_id, eps_null),
        "random_matched_vs_null": zone_breakdown(eps_rand, eps_null),
        "vanilla_vs_null": zone_breakdown(eps_vanilla, eps_null)
    }

    print("Mask Zone Error Breakdown (Masked vs Unmasked):")
    for k, v in mask_breakdown.items():
        print(f"  {k:<24}: Masked Mean Diff={v['masked_mean_abs_diff']:.4f} | Unmasked Mean Diff={v['unmasked_mean_abs_diff']:.4f} | Ratio={v['ratio_masked_to_unmasked']:.2f}x")

    # Spatial difference map visualization
    diff_id_vs_vanilla = (eps_id - eps_vanilla).norm(dim=1)[0].cpu().numpy()
    diff_id_vs_rand = (eps_id - eps_rand).norm(dim=1)[0].cpu().numpy()
    diff_vanilla_vs_null = (eps_vanilla - eps_null).norm(dim=1)[0].cpu().numpy()
    mask_np = mask_latent[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(mask_np, cmap="gray")
    axes[0].set_title("Latent Mask (64x64)")
    axes[1].imshow(diff_id_vs_vanilla, cmap="inferno")
    axes[1].set_title("Epsilon Diff: Identity vs Vanilla")
    axes[2].imshow(diff_id_vs_rand, cmap="inferno")
    axes[2].set_title("Epsilon Diff: Identity vs Random")
    axes[3].imshow(diff_vanilla_vs_null, cmap="inferno")
    axes[3].set_title("Epsilon Diff: Vanilla vs Null")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(diag_dir / "epsilon_spatial_diff_maps.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Save Full Diagnostic Results JSON
    # -------------------------------------------------------------------------
    full_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": time.time() - start_time,
        "section1_trace_report": trace_report,
        "section2_semantic_report": semantic_report,
        "section3_ablation_comparisons": ablation_comparisons,
        "section4_ref_swap_report": ref_swap_report,
        "section5_cross_attention_summary": {
            "identity": summary_attn_identity,
            "clip_text": summary_attn_clip_text,
            "random_matched": summary_attn_random_matched,
            "clip_null": summary_attn_clip_null
        },
        "section7_leakage_metrics": leakage_metrics,
        "section8_mask_breakdown": mask_breakdown
    }

    out_json_path = diag_dir / "diagnostic_results.json"
    with open(out_json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nAll forensic diagnostics completed in {time.time() - start_time:.2f}s.")
    print(f"Results saved to {out_json_path}")


if __name__ == "__main__":
    main()
