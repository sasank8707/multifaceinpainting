# System Architecture Specification

## 1. Executive Summary & Research Problem

This document specifies the research architecture for **Multi-Reference Identity-Preserving Facial Image Inpainting Using Hybrid Transformer-Diffusion Networks**.

### Core Research Objective
Given a damaged/masked target facial image $\mathbf{I}_{\text{target}} \in \mathbb{R}^{H \times W \times 3}$, a binary/soft mask $\mathbf{M} \in [0, 1]^{H \times W}$, and a set of $N$ unconstrained reference images of the **same identity** $\{\mathbf{R}_1, \dots, \mathbf{R}_N\}$, reconstruct the missing/degraded regions $\Omega = \{(x,y) \mid \mathbf{M}(x,y) > 0\}$ such that:
1. **Biometric Identity** of the subject is strictly preserved ($\Delta \text{ID} \approx 0$).
2. **Facial Geometry & Landmarks** are structurally coherent and naturally connected to unmasked pixels.
3. **Photometric & Texture Fidelity** (skin tone, lighting, pores, expression) blend seamlessly across the boundary $\partial \Omega$.
4. **Surrounding Context & Non-Face Regions** are restored without injecting spurious facial features.

---

## 2. Theoretical Foundations & Synthesis

The architecture synthesizes core contributions from two authoritative references:

### A. Team Project PPT (*Mask-Aware Multi-Reference Fusion with Verifier-Aware Sampling*)
1. **O1 — RETRIEVE (Mask-Conditioned Reference Scoring)**:
   - Evaluates reference relevance conditioned on the missing region geometry $\Omega$ rather than using uniform global reference weights.
   - Computes:
     - 3D head pose estimation (yaw $\theta_y$, pitch $\theta_p$).
     - Image quality (Laplacian sharpness, exposure, noise).
     - Landmark visibility within the unmasked region of each reference corresponding to $\Omega$.
     - Geometric overlap with the occluded region.
   - Selects top-$K$ references per region of interest.
2. **O2 — FUSE (Regional Identity-Token Fusion)**:
   - Extracts patch-level identity features aligned to facial landmark coordinates.
   - Implements mask-conditioned cross-attention between spatial query tokens and reference key-value pairs.
   - Learns weighted multi-reference aggregation and projects tokens into the diffusion conditioning path.
3. **O3 — VERIFY (Verifier-Aware Sampling)**:
   - Generates $K$ stochastic diffusion candidates ($K=8\text{--}16$ in evaluation, $K=1$ in fast development).
   - Evaluates candidates using a **held-out ArcFace-R100** matcher and no-reference perceptual quality estimators (NIQE / BRISQUE) alongside structural landmark alignment.
   - Performs joint optimization:
     $$\mathbf{C}^* = \arg\max_{c \in \{1,\dots,K\}} \left[ \lambda \cdot \mathcal{S}_{\text{ID}}(c) + (1-\lambda) \cdot \mathcal{S}_{\text{Quality}}(c) \right]$$
   - The verifier is **strictly inference-time only** and receives **zero training gradients**.

### B. Base Research Paper (*Wei et al., IEEE Access, 2019*)
1. **Semantic Prior & Coarse Structural Generation**:
   - Semantic inpainting cannot rely on simple pixel copying or texture synthesis; high-level structural priors are mandatory.
   - A U-Net generator produces a coarse reconstruction $\mathbf{X}_{\text{coarse}}$ to anchor semantic geometry before fine-grained synthesis.
2. **Laplace Loss for Facial Edge Preservation**:
   - Standard L1/L2 pixel losses treat all pixels equally, leading to blurry edges in facial organs (eyes, lips, nostrils).
   - Second-order differential edge penalty (Laplacian operator $\Delta$):
     $$\mathcal{L}_{\text{Laplace}} = \mathbb{E} \left[ \|\Delta \mathbf{I}_{\text{real}} - \Delta \mathbf{I}_{\text{gen}}\|_1 \right]$$
     accelerates edge convergence and sharpens structural contours.
3. **Region Weighting (RW) & Boundary Continuity**:
   - Organ boundaries (high edge response) require stronger structural alignment than smooth skin areas.
   - Edge responses in blurry regions are scaled by factor $m > 1$ ($m=5$) to balance boundary matching.
4. **Laplacian Clarity & PSNR Trade-Off**:
   - High-frequency details and natural textures may slightly lower pixel-level PSNR compared to blurry averages, while dramatically improving perceptual realism and biometric accuracy.

---

## 3. Component Taxonomy

To ensure mathematical rigor, clean separation of concerns, and reproducible evaluation, all system components are categorized into four operational types:

| Component Name | Operational Nature | Weights Status | Gradient Flow | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Mask & Geometry Analyzer** | Deterministic Algorithmic | None (No weights) | None | Connected component labeling, convex hulls, face overlap computation, ROI coordinate extraction. |
| **Face & Landmark Detector** | Pretrained Backbone | Frozen | None | MediaPipe / OpenCV face detector + 68/5-point landmark extractor + pose estimator (solvePnP). |
| **Mask-Aware Reference Scorer** | Modular Trainable Head | Trainable / Adaptable | Yes (in retrieval training) | Projects pose, quality, and mask-region landmark visibility into scalar reference importance weights $w_i$. |
| **Regional Identity Tokenizer** | Pretrained Feature Extractor | Frozen | None | Extracts local patch-level identity tokens $\mathbf{F}_i \in \mathbb{R}^{P \times D}$ from candidate references. |
| **Regional Transformer Fusion** | Novel Trainable Module | Trainable | Yes (in end-to-end training) | Mask-conditioned cross-attention aggregating multi-reference tokens into spatial conditioning tokens $\mathbf{T}_{\text{fused}}$. |
| **Coarse Edge Generator** | Trainable U-Net | Trainable | Yes (with Laplace loss) | Predicts coarse semantic layout and sharp organ edges $\mathbf{X}_{\text{coarse}}$ directly from masked input. |
| **Diffusion Inpainting Backbone** | Pretrained + Trainable LoRA/CrossAttn | Frozen backbone, trainable adapters | Yes (through conditioning cross-attention) | Denoises latent representation conditioned on $\mathbf{T}_{\text{fused}}$ and structural edge priors. |
| **Boundary Refinement & Blending** | Deterministic Algorithmic | None | None | Laplacian pyramid / Poisson blending ensuring unmasked pixels are 100% bit-exact preserved. |
| **Held-Out ArcFace-R100 Matcher** | Pretrained Biometric Evaluator | Frozen | **STRICTLY NONE** | MS1MV3-trained ArcFace-R100 used exclusively for O3 reranking and final reporting. Never used in loss computation. |
| **Quality & Structural Verifier** | Deterministic Evaluator | Frozen | None | Computes NIQE, BRISQUE, and landmark L2 distance for candidate validation. |

---

## 4. End-to-End Pipeline & Unified Routing Layer

```
                        Input Image I + Mask M (+ References R_1..R_N)
                                         │
                                         ▼
                             [ Mask Analysis & Profiling ]
                             - Connected Component Analysis
                             - Bounding Box Context Expansion
                             - Face-Mask Intersection Ratio (IoU_face)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         [ Face Overlap > 0 ]                           [ Face Overlap == 0 ]
                 │                                               │
    ┌────────────┴────────────┐                                  │
    ▼                         ▼                                  │
[ Severe Hole / Inpainting ] [ Localized Degradation ]           │
(Wrinkle/Scar/Blemish/Burn)                                      │
    │                         │                                  │
    ▼                         ▼                                  ▼
[ MODE 1: Facial Inpainting ] [ MODE 2: Facial Restoration ]  [ MODE 4: Object Removal ]
- Multi-Reference Scoring     - Residual Degradation Mask     - Semantic Scene Completion
- Regional Identity Fusion    - Base Geometry Retention       - Texture Continuity
- Landmark Guidance           - Local Texture Synthesis       - No Face Token Injection
    │                         │                                  │
    └─────────────────────────┼──────────────────────────────────┘
                              ▼
                  [ MODE 3: Multi-Mask Routing ]
                  (Per-component classification & coordinated synthesis)
                              │
                              ▼
            [ Coarse Semantic & Edge Guidance (Wei et al. 2019) ]
            - U-Net coarse layout prediction
            - 5x5 Laplacian Edge Constraint
                              │
                              ▼
            [ Hybrid Transformer-Diffusion Conditioning ]
            - Cross-attention with fused identity tokens T_fused
            - Structural edge conditioning
                              │
                              ▼
             [ Diffusion Candidate Sampling (K >= 1) ]
                              │
                              ▼
            [ O3: Verifier-Aware Candidate Selection ]
            - S_joint = λ * S_ID + (1-λ) * S_Quality
            - Outlier / artifact candidate rejection
                              │
                              ▼
             [ Boundary-Preserving Reconstruction ]
             - ROI Inverse Composition
             - Strict Unmasked Pixel Preservation: I_out(1-M) == I_target(1-M)
                              │
                              ▼
                         Final Output
```

---

## 5. Mathematical Formulations

### 5.1 Mask-Aware Reference Scoring (O1 Retrieve)
For each candidate reference $\mathbf{R}_i$ ($i \in \{1,\dots,N\}$) and masked region $\Omega$:
1. **Pose Compatibility**:
   $$\mathcal{S}_{\text{pose}}(\mathbf{R}_i, \mathbf{I}_{\text{target}}) = \exp\left( - \frac{(\theta_{y,i} - \theta_{y,t})^2}{2 \sigma_{\text{yaw}}^2} - \frac{(\theta_{p,i} - \theta_{p,t})^2}{2 \sigma_{\text{pitch}}^2} \right)$$
2. **Quality Score**:
   $$\mathcal{S}_{\text{qual}}(\mathbf{R}_i) = \alpha_1 \cdot \text{Clarity}_{\text{Laplace}}(\mathbf{R}_i) + \alpha_2 \cdot \text{Illumination}(\mathbf{R}_i)$$
3. **Region Visibility & Mask Overlap**:
   Let $\mathcal{L}_{\Omega} = \{l_k \mid l_k \in \Omega\}$ be the set of facial landmarks occluded in the target. The visibility score measures the presence and unoccluded clarity of corresponding landmarks in reference $\mathbf{R}_i$:
   $$\mathcal{S}_{\text{vis}}(\mathbf{R}_i, \Omega) = \frac{1}{|\mathcal{L}_{\Omega}|} \sum_{l_k \in \mathcal{L}_{\Omega}} \mathbb{I}(l_{k,i} \text{ is visible and unoccluded in } \mathbf{R}_i)$$
4. **Total Region Weight**:
   $$w_i(\Omega) = \frac{\exp\left( \gamma_1 \mathcal{S}_{\text{pose}} + \gamma_2 \mathcal{S}_{\text{qual}} + \gamma_3 \mathcal{S}_{\text{vis}} \right)}{\sum_{j=1}^N \exp\left( \gamma_1 \mathcal{S}_{\text{pose},j} + \gamma_2 \mathcal{S}_{\text{qual},j} + \gamma_3 \mathcal{S}_{\text{vis},j} \right)}$$

### 5.2 Regional Identity-Token Fusion (O2 Fuse)
1. Reference features $\mathbf{F}_i \in \mathbb{R}^{P \times D}$ are extracted using a spatial patch encoder.
2. Spatial query tokens $\mathbf{Q} \in \mathbb{R}^{M \times D}$ are generated from the masked target ROI, augmented with spatial landmark positional encodings $\mathbf{P}_{\text{landmark}}$.
3. Multi-reference mask-conditioned cross-attention computes:
   $$\mathbf{A}_{i} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}_i^T}{\sqrt{D}} + \mathbf{M}_{\text{bias}} \right)$$
   $$\mathbf{T}_{\text{fused}} = \sum_{i=1}^N w_i(\Omega) \cdot \left( \mathbf{A}_i \mathbf{V}_i \right)$$
   where $\mathbf{K}_i, \mathbf{V}_i = \mathbf{F}_i \mathbf{W}_K, \mathbf{F}_i \mathbf{W}_V$.

### 5.3 Composite Training Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}} + \lambda_{\text{edge}} \mathcal{L}_{\text{Laplace}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask\_rec}}$$
where:
- $\mathcal{L}_{\text{rec}} = \|\mathbf{I}_{\text{gt}} - \hat{\mathbf{I}}\|_1$
- $\mathcal{L}_{\text{mask\_rec}} = \|\mathbf{M} \odot (\mathbf{I}_{\text{gt}} - \hat{\mathbf{I}})\|_1$
- $\mathcal{L}_{\text{Laplace}} = \|\Delta \mathbf{I}_{\text{gt}} - \Delta \hat{\mathbf{I}}\|_1$ (implemented via fixed 5x5 Laplacian kernel, per Wei et al.)
- $\mathcal{L}_{\text{id}} = 1 - \cos(\mathbf{e}_{\text{train}}(\mathbf{I}_{\text{gt}}), \mathbf{e}_{\text{train}}(\hat{\mathbf{I}}))$ (using training feature extractor, **not** the held-out verifier).

---

## 6. Strict Architectural Constraints

1. **No Reference Pixel Copying**: References contribute representations, identity embeddings, and spatial attention tokens only. Direct affine warping, patch copying, or alpha blending of reference pixels into the target is strictly forbidden.
2. **Resolution Abstraction**:
   - The model backbone operates at internal normalized resolutions (e.g. $256 \times 256$ or $512 \times 512$).
   - The system preserves **input dimensions $(H, W)$ bit-exact** on unmasked pixels:
     $$\hat{\mathbf{I}}_{\text{final}}(x, y) = \begin{cases} \hat{\mathbf{I}}_{\text{recon}}(x, y) & \text{if } (x, y) \in \Omega \\ \mathbf{I}_{\text{target}}(x, y) & \text{if } (x, y) \notin \Omega \end{cases}$$
   - Localized masks use an expanded context ROI to maximize effective resolution.
3. **Inspectability**: All reference scores $w_i$, regional weights, landmark detections, and candidate verification scores must be exported in structured metadata logs.
