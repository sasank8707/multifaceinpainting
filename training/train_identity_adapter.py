"""
Training pipeline for IdentityConditioningAdapter.
Optimizes the adapter to map Phase 4 regional ArcFace tokens into diffusion conditioning representations:
- Same-identity consistency objective (representations from same subject must align)
- Contrastive separation objective (representations from different identities are separated)
- ArcFace biometric probe preservation
- CLIP manifold regularization (matching norm distribution of valid token embeddings)
Saves checkpoints and training history.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.diffusion.conditioning_adapter import IdentityConditioningAdapter


def train_identity_adapter(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 5e-4,
    device: str = "cpu",
    smoke_test: bool = True,
    smoke_subset_size: int = 8,
    margin: float = 0.30,
    checkpoint_dir: str = "checkpoints/identity_adapter"
) -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING IDENTITY CONDITIONING ADAPTER TRAINING PIPELINE")
    print("=" * 70)

    device_obj = torch.device(device)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & DataLoader
    train_ds = CelebAHQIdentityDataset(
        split="train",
        smoke_subset_size=smoke_subset_size if smoke_test else None,
        seed=42
    )
    val_ds = CelebAHQIdentityDataset(
        split="val",
        smoke_subset_size=4 if smoke_test else None,
        seed=123
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Frozen Feature Extractor + Trainable Adapter
    frozen_encoder = RegionalIdentityEncoder(token_dim=512).to(device_obj)
    frozen_encoder.eval()
    for param in frozen_encoder.parameters():
        param.requires_grad = False

    adapter = IdentityConditioningAdapter(
        in_dim=512,
        out_dim=768,
        num_tokens=256,
        out_tokens=77,
        num_id_queries=16,
        dropout=0.0
    ).to(device_obj)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    # 3. Training Loop
    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = ckpt_dir / "identity_adapter_best.pt"

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        adapter.train()
        train_losses = []

        for step, batch in enumerate(train_loader):
            targets = batch["target"].to(device_obj)
            pos_ref = batch["pos_references"][0].to(device_obj)
            neg_ref = batch["neg_reference"].to(device_obj)

            # Extract frozen ArcFace spatial tokens [B, 256, 512] and global embeddings [B, 512]
            with torch.no_grad():
                out_t = frozen_encoder(targets)
                out_p = frozen_encoder(pos_ref)
                out_n = frozen_encoder(neg_ref)

                t_tokens, t_global = out_t["spatial_tokens"], out_t["global_embedding"]
                p_tokens, p_global = out_p["spatial_tokens"], out_p["global_embedding"]
                n_tokens, n_global = out_n["spatial_tokens"], out_n["global_embedding"]

            optimizer.zero_grad()

            # Forward pass through trainable adapter
            cond_t = adapter(identity_tokens=t_tokens)
            cond_p = adapter(identity_tokens=p_tokens)
            cond_n = adapter(identity_tokens=n_tokens)

            # Flatten conditioning tokens for cosine distance
            proj_t = cond_t["projected_tokens"].flatten(1) # [B, 77*768]
            proj_p = cond_p["projected_tokens"].flatten(1)
            proj_n = cond_n["projected_tokens"].flatten(1)

            sim_pos = F.cosine_similarity(proj_t, proj_p, dim=-1) # [B]
            sim_neg = F.cosine_similarity(proj_t, proj_n, dim=-1) # [B]

            # 1. Same-identity consistency loss
            loss_cons = torch.mean(1.0 - sim_pos)

            # 2. Different-identity contrastive triplet loss
            loss_cont = torch.mean(F.relu(sim_neg - sim_pos + margin))

            # 3. ArcFace biometric preservation via linear probe
            probe_t = cond_t["probe_arcface"] # [B, 512]
            loss_probe = torch.mean(1.0 - F.cosine_similarity(probe_t, t_global, dim=-1))

            # 4. CLIP manifold norm regularization (target norm around 28.0)
            token_norms = cond_t["projected_tokens"].norm(dim=-1) # [B, 77]
            loss_clip = torch.mean((token_norms - 28.0) ** 2) * 1e-4

            loss_total = 1.0 * loss_cons + 1.0 * loss_cont + 1.0 * loss_probe + 1.0 * loss_clip
            loss_total.backward()
            optimizer.step()

            train_losses.append(loss_total.item())

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses))

        # Validation
        adapter.eval()
        val_losses = []
        val_pos_sims = []
        val_neg_sims = []

        with torch.no_grad():
            for batch in val_loader:
                targets = batch["target"].to(device_obj)
                pos_ref = batch["pos_references"][0].to(device_obj)
                neg_ref = batch["neg_reference"].to(device_obj)

                out_t = frozen_encoder(targets)
                out_p = frozen_encoder(pos_ref)
                out_n = frozen_encoder(neg_ref)

                cond_t = adapter(identity_tokens=out_t["spatial_tokens"])
                cond_p = adapter(identity_tokens=out_p["spatial_tokens"])
                cond_n = adapter(identity_tokens=out_n["spatial_tokens"])

                proj_t = cond_t["projected_tokens"].flatten(1)
                proj_p = cond_p["projected_tokens"].flatten(1)
                proj_n = cond_n["projected_tokens"].flatten(1)

                sim_pos = F.cosine_similarity(proj_t, proj_p, dim=-1)
                sim_neg = F.cosine_similarity(proj_t, proj_n, dim=-1)

                loss_v = torch.mean(1.0 - sim_pos) + torch.mean(F.relu(sim_neg - sim_pos + margin))
                val_losses.append(loss_v.item())
                val_pos_sims.extend(sim_pos.cpu().tolist())
                val_neg_sims.extend(sim_neg.cpu().tolist())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss
        avg_pos_sim = float(np.mean(val_pos_sims)) if val_pos_sims else 0.0
        avg_neg_sim = float(np.mean(val_neg_sims)) if val_neg_sims else 0.0

        print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Pos Sim: {avg_pos_sim:.4f} | Neg Sim: {avg_neg_sim:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_pos_sim": avg_pos_sim,
            "val_neg_sim": avg_neg_sim
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "history": history
            }, best_checkpoint_path)

    total_train_time = time.time() - start_train_time

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump({
            "total_time_seconds": total_train_time,
            "best_val_loss": best_val_loss,
            "epochs": epochs,
            "history": history
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("IDENTITY ADAPTER TRAINING COMPLETED!")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_train_time:.2f}s")
    print("=" * 70)

    return {
        "checkpoint_path": str(best_checkpoint_path),
        "best_val_loss": best_val_loss,
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke-test", action="store_true", default=True)
    parser.add_argument("--smoke-subset-size", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/identity_adapter")
    args = parser.parse_args()

    train_identity_adapter(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        smoke_test=args.smoke_test,
        smoke_subset_size=args.smoke_subset_size,
        checkpoint_dir=args.checkpoint_dir
    )
