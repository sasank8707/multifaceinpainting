"""
Training pipeline for Phase 5 Coarse Reconstructor.
Optimizes CoarseUNetGenerator using:
- L1 reconstruction loss
- Mask-weighted L1 loss
- 5x5 Laplacian loss (Wei et al. 2019)
- Region-weighted loss
- Boundary continuity loss
Saves model checkpoints and generates validation proofs of non-constant facial reconstruction.
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
from torch.utils.data import DataLoader

from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset
from modules.structural_guidance.coarse_reconstructor import CoarseReconstructor
from training.losses.laplace_loss import CoarseReconstructionLoss


def train_coarse_reconstructor(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 2e-4,
    device: str = "cpu",
    smoke_test: bool = True,
    smoke_subset_size: int = 8,
    checkpoint_dir: str = "checkpoints/coarse_reconstructor"
) -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING PHASE 5 COARSE RECONSTRUCTOR TRAINING PIPELINE")
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

    # 2. Model, Loss, Optimizer
    model = CoarseReconstructor(
        in_channels=4,
        out_channels=3,
        base_channels=32,
        identity_token_dim=512,
        num_heads=8,
        rw_multiplier_m=5.0
    ).to(device_obj)

    loss_fn = CoarseReconstructionLoss(
        weight_rec=1.0,
        weight_mask=5.0,
        weight_laplace=1.0,
        weight_rw=1.0,
        weight_boundary=2.0,
        rw_multiplier_m=5.0
    ).to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    # 3. Training Loop
    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = ckpt_dir / "coarse_reconstructor_best.pt"

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader):
            targets = batch["target"].to(device_obj)
            masks = batch["mask"].to(device_obj)
            masked_imgs = batch["masked_image"].to(device_obj)

            optimizer.zero_grad()
            out = model(masked_image=masked_imgs, mask=masks)
            pred = out["composited_output"]

            loss_dict = loss_fn(pred=pred, target=targets, mask=masks)
            loss = loss_dict["total_loss"]
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        val_stds = []

        with torch.no_grad():
            for batch in val_loader:
                targets = batch["target"].to(device_obj)
                masks = batch["mask"].to(device_obj)
                masked_imgs = batch["masked_image"].to(device_obj)

                out = model(masked_image=masked_imgs, mask=masks)
                pred = out["composited_output"]
                loss_dict = loss_fn(pred=pred, target=targets, mask=masks)
                val_losses.append(loss_dict["total_loss"].item())

                # Measure spatial variation in masked region
                for b_i in range(pred.shape[0]):
                    m_bool = (masks[b_i, 0] > 0.5)
                    if m_bool.sum() > 0:
                        pred_masked = pred[b_i, :, m_bool].cpu().numpy()
                        val_stds.append(float(np.std(pred_masked)))

        avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss
        avg_val_std = float(np.mean(val_stds)) if val_stds else 0.0

        print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Masked Std: {avg_val_std:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_masked_std": avg_val_std
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "history": history
            }, best_checkpoint_path)

    total_train_time = time.time() - start_train_time

    # Save summary history
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump({
            "total_time_seconds": total_train_time,
            "best_val_loss": best_val_loss,
            "epochs": epochs,
            "history": history
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("COARSE RECONSTRUCTOR TRAINING COMPLETED!")
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
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke-test", action="store_true", default=True)
    parser.add_argument("--smoke-subset-size", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/coarse_reconstructor")
    args = parser.parse_args()

    train_coarse_reconstructor(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        smoke_test=args.smoke_test,
        smoke_subset_size=args.smoke_subset_size,
        checkpoint_dir=args.checkpoint_dir
    )
