"""
Principled Calibration of Identity Conditioning for Diffusion Inpainting.
Aligns IdentityConditioningAdapter cross-attention tokens with the Stable Diffusion CLIP text manifold:
- Matches per-token norm distribution (Token 0 ~43.82, Tokens 1-76 ~28.38, Mean ~28.5)
- Matches global scalar mean (~ -0.112) and standard deviation (~ 1.025)
- Preserves directional angular geometry of biometric tokens
- Keeps ArcFace backbone and adapter feature extraction weights strictly frozen
- Updates checkpoints/identity_adapter/identity_adapter_best.pt
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion.conditioning_adapter import (
    IdentityConditioningAdapter,
    ConditioningCalibrator
)

CKPT_PATH = PROJECT_ROOT / "checkpoints" / "identity_adapter" / "identity_adapter_best.pt"


def run_calibration():
    print("=" * 70)
    print("STARTING PRINCIPLED IDENTITY CONDITIONING CALIBRATION")
    print("=" * 70)

    # 1. Load existing checkpoint
    assert CKPT_PATH.exists(), f"Checkpoint not found at {CKPT_PATH}"
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    print(f"Loaded checkpoint: Epoch {ckpt.get('epoch')}, Val Loss: {ckpt.get('val_loss'):.4f}")

    # 2. Instantiate adapter
    adapter = IdentityConditioningAdapter(
        in_dim=512,
        out_dim=768,
        num_tokens=256,
        out_tokens=77,
        num_id_queries=16,
        dropout=0.0
    )

    # Load existing weights (strict=False because calibrator keys are newly added)
    load_res = adapter.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("Loaded existing weights. Missing keys (new calibrator):", load_res.missing_keys)

    # 3. Freeze all parameters EXCEPT calibrator
    for name, param in adapter.named_parameters():
        if "calibrator" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = [p for p in adapter.parameters() if p.requires_grad]
    frozen_params = [p for p in adapter.parameters() if not p.requires_grad]
    print(f"Trainable parameters (calibrator only): {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen parameters: {sum(p.numel() for p in frozen_params):,}")

    # 4. Define Target CLIP Manifold Statistics (Empirically measured from SD inpainting text encoder)
    # Token 0: 43.8203, Tokens 1-76: 28.38
    target_token_norms = torch.full((1, 77), 28.38)
    target_token_norms[0, 0] = 43.8203
    target_global_mean = -0.1116
    target_global_std = 1.0253
    target_null_mean_norm = 28.9096

    # Measure BEFORE calibration
    adapter.eval()
    dummy_input = F.normalize(torch.randn(10, 256, 512), p=2, dim=-1)
    with torch.no_grad():
        # Temporarily bypass calibrator to get pre-calibration baseline
        b, m, d = dummy_input.shape
        spatial_features = adapter.spatial_proj(dummy_input)
        queries = adapter.id_queries.expand(b, -1, -1)
        id_queried, _ = adapter.cross_attn(query=queries, key=spatial_features, value=spatial_features)
        raw_prompt_seq = adapter.base_context.expand(b, -1, -1).clone()
        raw_prompt_seq[:, 1:1 + adapter.num_id_queries, :] = id_queried[:, :adapter.num_id_queries, :]
        raw_norms = torch.norm(raw_prompt_seq, p=2, dim=-1)

    before_stats = {
        "mean_token_norm": float(raw_norms.mean().item()),
        "std_token_norm": float(raw_norms.std().item()),
        "min_token_norm": float(raw_norms.min().item()),
        "max_token_norm": float(raw_norms.max().item()),
        "global_mean": float(raw_prompt_seq.mean().item()),
        "global_std": float(raw_prompt_seq.std().item()),
    }
    print("\n--- STATS BEFORE CALIBRATION ---")
    for k, v in before_stats.items():
        print(f"  {k}: {v:.4f}")

    # 5. Optimization Loop: Calibrate calibrator parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-5)
    torch.manual_seed(42)

    adapter.train()
    num_steps = 100
    print(f"\nRunning {num_steps} calibration steps...")

    for step in range(1, num_steps + 1):
        # Generate varied ArcFace synthetic representations (batch size 8)
        batch_tokens = F.normalize(torch.randn(8, 256, 512), p=2, dim=-1)

        out = adapter(identity_tokens=batch_tokens)
        proj_tokens = out["projected_tokens"] # [8, 77, 768]
        token_norms = torch.norm(proj_tokens, p=2, dim=-1) # [8, 77]

        # Target token norm loss
        loss_norm = F.mse_loss(token_norms, target_token_norms.expand(8, 77))

        # First-order scalar distribution losses
        curr_mean = proj_tokens.mean()
        curr_std = proj_tokens.std()
        loss_mean = (curr_mean - target_global_mean) ** 2
        loss_std = (curr_std - target_global_std) ** 2

        # Null embedding calibration loss
        null_out = adapter.get_null_embedding(batch_size=8, target_len=77)
        null_norms = torch.norm(null_out, p=2, dim=-1)
        loss_null_norm = F.mse_loss(null_norms, target_token_norms.expand(8, 77))
        loss_null_mean = (null_out.mean() - target_global_mean) ** 2
        loss_null_std = (null_out.std() - target_global_std) ** 2

        # Angular diversity loss between distinct identity samples
        sample_a = proj_tokens[0:4, 1:17]
        sample_b = proj_tokens[4:8, 1:17]
        cos_diversity = F.cosine_similarity(sample_a, sample_b, dim=-1).mean()
        loss_diversity = (cos_diversity - 0.0) ** 2

        total_loss = (
            loss_norm * 1.0 +
            loss_mean * 10.0 +
            loss_std * 5.0 +
            loss_null_norm * 0.5 +
            loss_null_mean * 5.0 +
            loss_null_std * 2.5 +
            loss_diversity * 2.0
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == num_steps:
            print(f"Step {step:3d} | Loss: {total_loss.item():.5f} | "
                  f"Norm: {token_norms.mean().item():.2f} (std {token_norms.std().item():.2f}) | "
                  f"Mean: {curr_mean.item():.4f} | Std: {curr_std.item():.4f}")

    # 6. Evaluation of Calibrated Adapter
    adapter.eval()
    with torch.no_grad():
        test_tokens = F.normalize(torch.randn(20, 256, 512), p=2, dim=-1)
        test_out = adapter(identity_tokens=test_tokens)
        cal_proj = test_out["projected_tokens"]
        cal_norms = torch.norm(cal_proj, p=2, dim=-1)

        null_test = adapter.get_null_embedding(batch_size=20, target_len=77)
        null_norms = torch.norm(null_test, p=2, dim=-1)

    after_stats = {
        "mean_token_norm": float(cal_norms.mean().item()),
        "std_token_norm": float(cal_norms.std().item()),
        "min_token_norm": float(cal_norms.min().item()),
        "max_token_norm": float(cal_norms.max().item()),
        "global_mean": float(cal_proj.mean().item()),
        "global_std": float(cal_proj.std().item()),
        "null_mean_norm": float(null_norms.mean().item()),
        "null_min_norm": float(null_norms.min().item()),
        "null_max_norm": float(null_norms.max().item()),
        "token_0_norm": float(cal_norms[:, 0].mean().item()),
        "tokens_1_76_mean_norm": float(cal_norms[:, 1:].mean().item()),
    }

    print("\n--- STATS AFTER CALIBRATION ---")
    for k, v in after_stats.items():
        print(f"  {k}: {v:.4f}")

    # 7. Quality Checks on Calibrated Conditioning Tokens
    assert 27.0 <= after_stats["mean_token_norm"] <= 30.0, f"Mean norm out of range: {after_stats['mean_token_norm']}"
    assert 40.0 <= after_stats["token_0_norm"] <= 46.0, f"Token 0 norm out of range: {after_stats['token_0_norm']}"
    assert abs(after_stats["global_mean"] - (-0.1116)) < 0.05, f"Global mean out of range: {after_stats['global_mean']}"
    assert abs(after_stats["global_std"] - 1.0253) < 0.10, f"Global std out of range: {after_stats['global_std']}"
    assert not torch.isnan(cal_proj).any(), "NaNs detected in calibrated tokens"
    assert not torch.isinf(cal_proj).any(), "Infs detected in calibrated tokens"
    print("\nCalibration Quality Checks: ALL PASSED.")

    # 8. Save Updated Checkpoint
    ckpt["model_state_dict"] = adapter.state_dict()
    ckpt["calibration_stats"] = {
        "before": before_stats,
        "after": after_stats,
        "target": {
            "mean_token_norm": 28.53,
            "token_0_norm": 43.82,
            "tokens_1_76_norm": 28.38,
            "global_mean": target_global_mean,
            "global_std": target_global_std
        }
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"\nSuccessfully saved calibrated model state to {CKPT_PATH}")

    # Save diagnostic summary json
    diag_path = PROJECT_ROOT / "experiments" / "calibration_diagnostic.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump(ckpt["calibration_stats"], f, indent=2)
    print(f"Saved diagnostic report to {diag_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_calibration()
