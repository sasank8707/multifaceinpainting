# Training Plan & Objective Formulations

## 1. Overview & Pedagogical Objectives

The training regimen trains/adapts the novel research components while freezing heavy foundational backbones to ensure high parameter efficiency, numerical stability, and feasibility within standard 8–12 GB VRAM constraints (or fast CPU smoke testing).

### Trainable vs Frozen Component Separation
- **Frozen**:
  - Pretrained VAE Autoencoder (Latent Diffusion encoder/decoder).
  - Pretrained Identity Feature Extractor (ArcFace / MobileFaceNet for feature extraction).
  - Facial Landmark Detector (MediaPipe / Dlib 68-point).
  - Held-out Biometric Verifier (ArcFace-R100 MS1MV3) — **never receives gradients**.
- **Trainable**:
  - **Coarse Edge Generator** (U-Net predicting initial layout and organ contours).
  - **Mask-Aware Reference Scoring Head** (Retrieval scoring module).
  - **Regional Identity-Token Transformer** (Cross-attention fusion module).
  - **Diffusion Conditioning Adapters / Cross-Attention Projections**.

---

## 2. Loss Function Formulation

The total training objective is a composite loss designed to penalize pixel distortion, perceptual divergence, biometric identity drift, and edge discontinuity:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_1 \mathcal{L}_{\text{perc}} + \lambda_2 \mathcal{L}_{\text{id}} + \lambda_3 \mathcal{L}_{\text{landmark}} + \lambda_4 \mathcal{L}_{\text{edge}} + \lambda_5 \mathcal{L}_{\text{mask}} + \lambda_6 \mathcal{L}_{\text{boundary}}$$

### 2.1 Reconstruction Loss ($\mathcal{L}_{\text{rec}}$)
Smooth L1 / Charbonnier loss over the entire image:
$$\mathcal{L}_{\text{rec}} = \frac{1}{H \times W} \sum_{x, y} \sqrt{(\mathbf{I}_{\text{gt}}(x, y) - \hat{\mathbf{I}}(x, y))^2 + \epsilon^2}$$
where $\epsilon = 10^{-3}$.

### 2.2 Masked-Region Specific Loss ($\mathcal{L}_{\text{mask}}$)
Emphasizes error within the missing/occluded region $\Omega$:
$$\mathcal{L}_{\text{mask}} = \frac{1}{\sum \mathbf{M}} \|\mathbf{M} \odot (\mathbf{I}_{\text{gt}} - \hat{\mathbf{I}})\|_1$$

### 2.3 Laplace Edge Loss ($\mathcal{L}_{\text{edge}}$) — *Wei et al., IEEE Access 2019*
To prevent blurry reconstructions in high-frequency facial organs (eyelids, nostrils, lip contours), we incorporate the base paper's second-order differential Laplacian operator $\Delta$:
$$\mathcal{L}_{\text{edge}} = \|\Delta \mathbf{I}_{\text{gt}} - \Delta \hat{\mathbf{I}}\|_1$$
Implemented using a fixed 2D convolution kernel with the $5 \times 5$ Laplacian operator:
$$\mathbf{K}_{\text{Laplace}} = \begin{bmatrix}
0 & 0 & -1 & 0 & 0 \\
0 & -1 & -2 & -1 & 0 \\
-1 & -2 & 16 & -2 & -1 \\
0 & -1 & -2 & -1 & 0 \\
0 & 0 & -1 & 0 & 0
\end{bmatrix}$$

### 2.4 Perceptual Loss ($\mathcal{L}_{\text{perc}}$)
VGG-19 or LPIPS multi-scale feature distance:
$$\mathcal{L}_{\text{perc}} = \sum_{l=1}^L \frac{1}{C_l H_l W_l} \|\phi_l(\mathbf{I}_{\text{gt}}) - \phi_l(\hat{\mathbf{I}})\|_1$$

### 2.5 Identity Consistency Loss ($\mathcal{L}_{\text{id}}$)
Cosine distance in the embedding space of a training identity extractor:
$$\mathcal{L}_{\text{id}} = 1 - \frac{\mathbf{e}_{\text{train}}(\mathbf{I}_{\text{gt}}) \cdot \mathbf{e}_{\text{train}}(\hat{\mathbf{I}})}{\|\mathbf{e}_{\text{train}}(\mathbf{I}_{\text{gt}})\| \|\mathbf{e}_{\text{train}}(\hat{\mathbf{I}})\|}$$
*(Note: Evaluated with training feature extractor; the held-out ArcFace-R100 MS1MV3 verifier is strictly withheld from this loss).*

### 2.6 Structural Landmark Loss ($\mathcal{L}_{\text{landmark}}$)
L2 distance between predicted facial landmarks on the reconstructed output and ground-truth landmarks:
$$\mathcal{L}_{\text{landmark}} = \frac{1}{|\mathcal{L}|} \sum_{k=1}^{68} \|\hat{\mathbf{l}}_k - \mathbf{l}_{k, \text{gt}}\|_2$$

### 2.7 Boundary Continuity Loss ($\mathcal{L}_{\text{boundary}}$)
Penalizes color/edge disparity along the transition band $\partial \Omega$:
$$\mathcal{L}_{\text{boundary}} = \|\mathbf{B} \odot (\nabla \mathbf{I}_{\text{gt}} - \nabla \hat{\mathbf{I}})\|_1$$
where $\mathbf{B}$ is a dilated boundary band mask ($\text{radius}=3\text{px}$).

---

## 3. Staged Training Roadmap

Training is split into four decoupled stages to prevent gradient instability:

| Stage | Module Target | Input / Supervision | Active Losses | Epochs | Batch Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | **Coarse Edge U-Net** | $\mathbf{I}_{\text{masked}} \to \mathbf{X}_{\text{coarse}}$ | $\mathcal{L}_{\text{rec}} + \lambda_4 \mathcal{L}_{\text{edge}} + \lambda_5 \mathcal{L}_{\text{mask}}$ | 30 | 16 |
| **Stage 2** | **Reference Scorer** | $\{\mathbf{R}_i\}, \Omega \to w_i$ | Cross-entropy vs optimal reference overlap | 15 | 32 |
| **Stage 3** | **Regional Fusion Transformer** | $\mathbf{Q}_{\text{mask}}, \{\mathbf{K}_i, \mathbf{V}_i\} \to \mathbf{T}_{\text{fused}}$ | Feature alignment + $\mathcal{L}_{\text{id}}$ | 25 | 16 |
| **Stage 4** | **Diffusion Cross-Attention Adapters** | Denoising latents $\mathbf{z}_t + \mathbf{T}_{\text{fused}}$ | Diffusion noise MSE + $\mathcal{L}_{\text{id}}$ | 40 | 8 |

---

## 4. Hyperparameter Configuration

Default loss weighting schedule:
- $\lambda_{\text{rec}} = 1.0$
- $\lambda_{\text{mask}} = 5.0$
- $\lambda_{\text{edge}} = 1.0$ (Laplacian loss weight $\gamma$ from Wei et al. 2019)
- $\lambda_{\text{perc}} = 0.2$
- $\lambda_{\text{id}} = 0.5$
- $\lambda_{\text{landmark}} = 0.1$
- $\lambda_{\text{boundary}} = 2.0$

Optimizer & Learning Rate:
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-2}$)
- Learning Rate: $\eta = 1 \times 10^{-4}$ with Cosine Annealing warmup (first 5% steps)
- Gradient Clipping: $\max \|\mathbf{g}\|_2 = 1.0$

---

## 5. Hardware & Memory Optimization Strategies

For environments matching the team PPT specification (NVIDIA CUDA GPU with 8–12 GB VRAM):
1. **Mixed Precision**: Automatic Mixed Precision (AMP) with FP16 / BF16.
2. **Gradient Checkpointing**: Enabled across Transformer blocks and UNet mid-blocks.
3. **FlashAttention / Scaled Dot-Product Attention (SDPA)**: Native PyTorch 2.x SDPA with $O(N)$ memory scaling.
4. **Gradient Accumulation**: Step accumulation ($k=2$ or $4$) to achieve effective batch size 32 without VRAM overflow.

For local CPU execution:
- Lightweight deterministic mock backbones with dimension-matched tensor layers allow complete unit and integration tests to execute in $< 5$ seconds without CUDA dependency.
