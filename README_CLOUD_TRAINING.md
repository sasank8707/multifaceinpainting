# Multi-Reference Facial Inpainting: Cloud-GPU Training Guide
### Resumable Decoupled Identity Cross-Attention Training on NVIDIA GPUs (T4 / L4 / A100)

This guide documents the complete cloud-training workflow for training the decoupled identity cross-attention pathway in Stable Diffusion, continuing directly from the verified **step-150 composite loss checkpoint**.

---

## 🔒 1. Protected Baseline Checkpoint

The step-150 checkpoint represents our immutable baseline and permanent rollback point. It is protected against any overwriting or modification.

- **Protected Location**: [`checkpoints/baselines/step_150_best/step_150.pt`](file:///c:/Users/sasan/Desktop/multifaceinpainting/checkpoints/baselines/step_150_best/step_150.pt)
- **Manifest**: [`checkpoints/baselines/step_150_best/manifest.json`](file:///c:/Users/sasan/Desktop/multifaceinpainting/checkpoints/baselines/step_150_best/manifest.json)
- **SHA-256 Checksum**: `e4ff5cff392ccca3522b1cc3c937c6a372bb9075ffa3f837e488c810a287be89`
- **File Size**: `153,394,949 bytes` (146.29 MB)
- **Training Step**: `150`
- **Scale ($\lambda$)**: `0.10`
- **Token Dimension ($d_{\text{id}}$)**: `512`
- **Trainable Parameters**: `12,779,520` (16 layers of `to_k_id` and `to_v_id`)

### Verification Command
To verify that the protected checkpoint is present, uncorrupted, and loads correctly:
```bash
python scripts/verify_best_checkpoint.py
```

---

## 🏗️ 2. Architectural Preservation & Parameter Audit

The model architecture, conditioning scale, and loss formulation are strictly preserved from the existing local experiments:

| Module | Status | Trainable Parameters | Frozen Parameters | Total Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **Decoupled Identity Cross-Attention (`to_k_id`, `to_v_id`)** | **TRAINABLE** | **12,779,520** | **0** | **12,779,520** |
| UNet Backbone Original Layers | FROZEN | 0 | 859,520,004 | 859,520,004 |
| VAE Autoencoder | FROZEN | 0 | 83,653,863 | 83,653,863 |
| CLIP Text Encoder | FROZEN | 0 | 123,060,480 | 123,060,480 |
| Differentiable ArcFace Backbone (`w600k_r50.onnx`) | FROZEN | 0 | 43,594,409 | 43,594,409 |
| Regional ArcFace Feature Extractor | FROZEN | 0 | 0 (OpenCV DNN) | 0 |
| Phase-4 Regional Transformer Fusion | FROZEN | 0 | 7,886,849 | 7,886,849 |
| **TOTAL** | | **12,779,520** | **1,074,136,556** | **1,086,916,076** |

- **Strict Target Anti-Leakage**: In Phase-4 Regional Transformer, `target_tokens=None`. All identity conditioning tokens derive 100% from the verified same-identity reference images.
- **Objective Function**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda_{\text{id}} \cdot \mathcal{L}_{\text{identity}}$$
  where $\mathcal{L}_{\text{diffusion}} = \text{MSE}(\hat{\epsilon}, \epsilon)$ and $\mathcal{L}_{\text{identity}} = 1 - \cos(\hat{e}, e_{\text{ref\_mean}})$.

---

## ⚡ 3. Cloud-GPU Execution (Google Colab & Kaggle)

### Recommended Cloud Setup
- **Google Colab**: Runtime $\rightarrow$ Change runtime type $\rightarrow$ Select **T4 GPU** (or A100 for fastest throughput).
- **Kaggle**: Settings $\rightarrow$ Accelerator $\rightarrow$ **GPU T4 x 2** $\rightarrow$ Internet ON.

### Pre-Flight Smoke Test (2–5 steps)
Before starting a longer training run, execute the automated smoke test on the GPU:
```bash
python scripts/train_identity_cloud.py \
  --resume checkpoints/baselines/step_150_best/step_150.pt \
  --device cuda \
  --mixed_precision fp16 \
  --smoke_test \
  --smoke_steps 3
```
This tests:
1. Loading baseline `step_150.pt` and restoring optimizer state.
2. Verification that the first step continues at **step 151**.
3. CUDA execution, forward pass, and mixed precision autocast.
4. Loss computation and finite gradient check.
5. Optimizer update and GradScaler step.
6. Checkpoint saving and reload integrity.

### Resumable Cloud Training Command
```bash
python scripts/train_identity_cloud.py \
  --resume checkpoints/baselines/step_150_best/step_150.pt \
  --max_steps 500 \
  --save_every 100 \
  --device cuda \
  --mixed_precision fp16 \
  --output_dir checkpoints/cloud_training \
  --learning_rate 1e-4 \
  --lambda_id 0.10 \
  --seed 42
```

To resume from an intermediate cloud checkpoint (e.g. `step_500.pt` to train to `1000`):
```bash
python scripts/train_identity_cloud.py \
  --resume checkpoints/cloud_training/step_500.pt \
  --max_steps 1000 \
  --save_every 250 \
  --device cuda
```

---

## 🎯 4. Training Milestones Strategy

Rather than unbounded training, checkpoints are evaluated at specific empirical milestones:
- **Milestone 0**: `150` (Baseline)
- **Milestone 1**: `500` (+350 steps)
- **Milestone 2**: `1000` (+500 steps)
- **Milestone 3**: `2500` (+1500 steps)
- **Milestone 4**: `5000` (+2500 steps)
- **Milestone 5**: `10000` (+5000 steps)

All checkpoints are saved in [`checkpoints/cloud_training/`](file:///c:/Users/sasan/Desktop/multifaceinpainting/checkpoints/cloud_training/) as:
- `step_500.pt`
- `step_1000.pt`
- `step_2500.pt`
- `step_5000.pt`
- `step_10000.pt`
- `latest.pt`

---

## 📊 5. Unified Checkpoint Evaluation Workflow

To measure whether further training improves, plateaus, or degrades identity alignment and reconstruction quality, evaluate all milestones using the identical Phase-7 protocol:

```bash
python scripts/evaluate_checkpoints.py \
  --checkpoints \
    checkpoints/baselines/step_150_best/step_150.pt \
    checkpoints/cloud_training/step_500.pt \
    checkpoints/cloud_training/step_1000.pt \
  --device cuda
```

### Metrics Tracked
1. **Biometric Similarity ($\text{Sim}_{\text{Ref}}$)**: Cosine similarity against mean reference ArcFace embedding.
2. **Target Similarity ($\text{Sim}_{\text{GT}}$)**: Cosine similarity against unmasked target ground truth.
3. **Impostor Similarity ($\text{Sim}_{\text{Diff}}$)**: Cosine similarity against different-identity subject (`3616`).
4. **Biometric Identity Gap**: $\Delta_{\text{ID}} = \text{Sim}_{\text{Ref}} - \text{Sim}_{\text{Diff}}$.
5. **Reconstruction Quality**: PSNR (dB), SSIM, MAE, RMSE inside the masked region.
6. **Edge & Gradient Errors**: Sobel first-order error, Laplacian 5x5 high-frequency transition error.
7. **Unmasked Invariance**: 100% bit-exact pixel preservation verification ($\Delta_{\text{max}} == 0.0$).
