# Evaluation Benchmark & Diagnostic Protocol

## 1. Overview & Verification Philosophy

Biometric identity-preserving inpainting requires evaluating two distinct, often conflicting dimensions:
1. **Biometric Identity Fidelity**: Ensuring the generated facial organs belong specifically to the intended subject.
2. **Photorealistic Perceptual Quality**: Ensuring edge continuity, natural skin texture, and absence of blurring or boundary artifacts.

> [!IMPORTANT]
> A simple pixel change in the masked region does **not** signify success. Evaluation must compare reconstructed facial organs against ground-truth targets both visually, structurally (landmarks, edges), and biometrically. Unmasked pixels must be preserved with 100% fidelity.

---

## 2. Quantitative Metric Formulations

### 2.1 Biometric Verification Metrics (Held-Out Matcher)
Identity verification is evaluated using a strictly held-out matcher: **ArcFace-R100 trained on MS1MV3**. This matcher is never accessed during training, feature loss calculation, or model parameter optimization.

1. **Cosine Identity Similarity**:
   $$\text{Sim}_{\text{ID}}(\hat{\mathbf{I}}, \mathbf{R}) = \frac{\mathbf{e}_{\text{heldout}}(\hat{\mathbf{I}}) \cdot \mathbf{e}_{\text{heldout}}(\mathbf{R})}{\|\mathbf{e}_{\text{heldout}}(\hat{\mathbf{I}})\| \|\mathbf{e}_{\text{heldout}}(\mathbf{R})\|}$$
   Evaluated against $\ge 3$ withheld unmasked reference images per identity.
2. **TAR @ FAR = 0.1%**:
   True Accept Rate at a False Accept Rate threshold of $10^{-3}$ on the matched pair distribution vs genuine/impostor impostor pairs.
3. **ROC AUC**:
   Area Under the Receiver Operating Characteristic curve distinguishing same-identity reconstructions from cross-identity impostors.

### 2.2 Perceptual & Reconstruction Quality Metrics
1. **PSNR (Peak Signal-to-Noise Ratio)**:
   $$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{MAX}^2}{\text{MSE}} \right)$$
   Evaluated both on the **whole image** and specifically on the **masked region $\Omega$** ($\text{PSNR}_{\text{mask}}$).
2. **SSIM (Structural Similarity Index)**: Evaluates luminance, contrast, and structural preservation.
3. **LPIPS (Learned Perceptual Image Patch Similarity)**: Evaluates deep feature perceptual distance using AlexNet/VGG backbones.
4. **FID (Fréchet Inception Distance)**: Computed over $\ge 5,000$ reconstructed crops vs ground-truth crops to evaluate distributional realism.
5. **No-Reference Perceptual Metrics (NIQE & BRISQUE)**:
   Evaluates naturalness and absence of generative artifacts without reference bias. Essential for catching unrealistic outputs that fool identity matchers.
6. **Laplacian Clarity Metric** (*Wei et al., IEEE Access 2019*):
   $$\text{Clarity}_{\text{Laplace}}(\mathbf{I}) = \frac{1}{H \times W} \sum_{x, y} |\Delta \mathbf{I}(x, y)|$$
   Measures high-frequency organ contour sharpness.

### 2.3 Structural Fidelity Metrics
1. **Landmark L2 Error**:
   Mean Euclidean pixel distance between detected 68-point landmarks on the reconstructed output and the ground truth:
   $$\text{Error}_{\text{landmark}} = \frac{1}{68} \sum_{k=1}^{68} \|\hat{\mathbf{l}}_k - \mathbf{l}_{k, \text{gt}}\|_2$$
2. **Boundary Continuity Error**:
   Mean gradient difference along the mask perimeter band $\partial \Omega$.

### 2.4 Integrity & Conservation Checks
1. **Mask Pixel Modification Rate**:
   $$\%_{\text{masked\_changed}} = \frac{\sum_{(x,y) \in \Omega} \mathbb{I}(|\hat{\mathbf{I}}(x,y) - \mathbf{I}_{\text{masked}}(x,y)| > \tau)}{|\Omega|} \times 100\%$$
2. **Unmasked Pixel Invariance**:
   $$\%_{\text{unmasked\_changed}} = \frac{\sum_{(x,y) \notin \Omega} \mathbb{I}(|\hat{\mathbf{I}}(x,y) - \mathbf{I}_{\text{orig}}(x,y)| > 0)}{H \times W - |\Omega|} \times 100\% \quad (\text{Target} = 0.00\%)$$

---

## 3. Benchmark Protocol: Identity Fidelity vs Mask Severity

The primary evaluation benchmark evaluates identity preservation across 4 progressive mask severity levels:

| Severity Tier | Target Facial Occlusion | Evaluated Properties | Research Significance |
| :--- | :--- | :--- | :--- |
| **Tier 1 (25%)** | 20%–30% of face box | Single organ (e.g. mouth or nose) | Localized context connection, color/skin tone continuity. |
| **Tier 2 (50%)** | 45%–55% of face box | Multiple organs (e.g. eyes + nose) | Intermediate occlusion; tests landmark consistency across organs. |
| **Tier 3 (75%)** | 70%–80% of face box | Major face area (eyes, nose, mouth) | Severe occlusion; remaining context provides minimal identity cues. |
| **Tier 4 (100%)** | $\ge 95\%$ (Full Face) | Entire facial region occluded | Identity evidence comes **100% from references**, isolating retrieval and fusion. |

---

## 4. Ablation Study Matrix (A1–A7)

To validate each novel component proposed in the research, the benchmark executes the 7 formal ablations defined in the Team PPT:

| Ablation Code | Reference Selection Strategy | Identity Fusion Strategy | Candidate Selection (O3) | Research Hypothesis Tested |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | Single random reference | Direct injection (single token) | None ($K=1$) | Baseline single-reference lower bound. |
| **A2** | Random multi-reference ($N=3$) | Mean token averaging | None ($K=1$) | Does unguided multi-reference help or add noise? |
| **A3** | Globally best reference (quality only) | Direct injection (single token) | None ($K=1$) | Global quality ranking vs mask-aware ranking. |
| **A4** | Multi-reference (no scoring) | Uniform equal-weight attention | None ($K=1$) | Equal multi-reference attention vs learned fusion. |
| **A5** | Multi-reference with global score | Global cross-attention (no mask condition) | None ($K=1$) | Mask-conditioned spatial attention vs global attention. |
| **A6** | **Mask-Aware Regional Fusion (O1+O2)** | Mask-conditioned regional cross-attention | None ($K=1$) | Evaluates full generative fusion prior to reranking. |
| **A7 (Full)** | **Mask-Aware Regional Fusion (O1+O2)** | **Mask-conditioned regional cross-attention** | **Verifier-Aware Sampling (O3)** | **Full proposed system with quality-constrained reranking.** |

---

## 5. Baseline Method Comparisons (B0–B4)

| Baseline ID | Description | Generative Type | Reference Conditioning |
| :--- | :--- | :--- | :--- |
| **B0** | Classical Inpainting (Telea / Navier-Stokes / PatchMatch) | Non-generative | None |
| **B1** | Single-Image Generative Inpainting (Vanilla U-Net / SD Inpainting) | Diffusion / GAN | None |
| **B2** | Single-Reference Identity Inpainting | Diffusion | Single global ArcFace embedding |
| **B3** | Multi-Reference Inpainting (Uniform Fusion) | Transformer-Diffusion | Multi-reference uniform average |
| **B4** | **Proposed Method (O1 Retrieve + O2 Fuse + O3 Verify)** | **Hybrid Transformer-Diffusion** | **Mask-Aware Regional Token Fusion** |

---

## 6. Diagnostic Evaluation Panel

Every experimental evaluation automatically generates and serializes an 8-panel diagnostic composite:
1. **Original Ground Truth** ($\mathbf{I}_{\text{orig}}$)
2. **Masked Target Input** ($\mathbf{I}_{\text{masked}}$)
3. **Applied Mask** ($\mathbf{M}$)
4. **Reconstructed Output** ($\hat{\mathbf{I}}$)
5. **Reconstruction-Only Region** ($\mathbf{M} \odot \hat{\mathbf{I}}$)
6. **Error / Difference Image** ($|\mathbf{I}_{\text{orig}} - \hat{\mathbf{I}}| \times 3$)
7. **Ranked Reference Images** with calculated retrieval scores ($w_1, w_2, w_3$)
8. **Zoomed Reconstructed Organ Detail** with overlaid facial landmarks
