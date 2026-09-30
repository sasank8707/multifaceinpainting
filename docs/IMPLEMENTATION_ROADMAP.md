# Research Implementation Roadmap & Quality Gates

## 1. Development Principles & Gates

This roadmap enforces a strict research methodology:
$$\text{Hypothesis} \longrightarrow \text{Implementation} \longrightarrow \text{Controlled Experiment} \longrightarrow \text{Quantitative Validation} \longrightarrow \text{Gate Approval} \longrightarrow \text{Next Phase}$$

No phase may proceed until the preceding gate satisfies its explicit pass conditions. Unchecked assumptions and unverified dependencies are strictly barred.

---

## 2. Phase-by-Phase Roadmap

### PHASE 0: Environment & Hardware Audit
- **Objective**: Audit CPU, GPU, RAM, VRAM, and Python runtimes; establish device dispatch layer (`cpu` vs `cuda`).
- **Deliverables**:
  - Python 3.10 virtual environment and dependency setup scripts.
  - `modules/utils/device.py` providing device-agnostic tensor management.
  - Hardware audit unit test `tests/test_phase0_environment.py`.
- **Pass Condition**: Automated tests pass; device profiles are logged; tensor allocation succeeds on CPU and GPU without warning.

### PHASE 1: Dataset Validation & Multi-Reference Identity Indexing
- **Objective**: Parse `identity_CelebA.txt` and `CelebA-HQ-to-CelebA-mapping.txt` to discover multi-image identity sets ($N \ge 2$) with strict ground-truth mapping.
- **Deliverables**:
  - `modules/data/identity_index.py` parsing identity annotations and mapping files.
  - Verification of dataset splits (Train: 80%, Val: 10%, Test: 10% partitioned by identity ID).
  - Validation test `tests/test_phase1_dataset.py`.
- **Pass Condition**: Successfully verifies identity mappings; finds $>1,000$ identities with $\ge 2$ images; guarantees zero identity overlap across splits.

### PHASE 2: Ground-Truth Synthetic Masking & Reproducible Pairs
- **Objective**: Implement 10 deterministic mask generation algorithms and synthetic data pairing pipeline.
- **Deliverables**:
  - `modules/masking/synthetic_masks.py` supporting rectangles, ellipses, irregular brush strokes, semantic masks, multi-masks, degradation, and text/object masks.
  - Severity level calibrator (25%, 50%, 75%, 100%).
  - Bounding box, coverage, and ROI context extraction utilities.
- **Pass Condition**: Generates masks across all 10 types; verifies coverage tolerances; tests aspect ratio and dimension invariance.

### PHASE 3: Simple Baseline Inpainting & Reconstruction Gate
- **Objective**: Implement and validate a real baseline inpainting model capable of semantic reconstruction on known CelebA-HQ images.
- **Deliverables**:
  - Baseline inpainting pipeline `modules/inpainting/baseline_inpainting.py` (classical baseline B0 + single-image generative baseline B1).
  - Diagnostic comparison script on known ground-truth CelebA-HQ targets.
- **Critical Gate (Phase 3 Pass Condition)**:
  - Input: Known CelebA-HQ image $\mathbf{A}$ + synthetic 25% facial mask $\mathbf{M}$.
  - Output: $\hat{\mathbf{A}}$.
  - Must replace the masked region with newly generated facial structure.
  - Must preserve unmasked pixels ($\Delta = 0$).
  - Must demonstrate measurable reconstruction improvement over masked input.
  - If this fails, STOP. Do not proceed to multi-reference fusion.

### PHASE 4: Coarse Edge Guidance & Laplace Loss (*Wei et al. 2019*)
- **Objective**: Implement U-Net coarse layout generator with base paper's $5 \times 5$ Laplacian edge loss and Region Weighting (RW).
- **Deliverables**:
  - `training/losses/laplace_loss.py` implementing 2nd-order differential edge penalty.
  - `models/backbone/coarse_unet.py` for coarse semantic prior generation.
  - Region Weighting calculation module with blur amplification $m=5$.
- **Pass Condition**: Edge loss accelerates convergence; facial organ contours are significantly sharper than L1-only baseline; Laplacian clarity improves.

### PHASE 5: Facial Landmark & Structural Conditioning
- **Objective**: Extract 68-point landmarks, estimate 3D head pose (yaw, pitch, roll), and construct structural spatial conditioning.
- **Deliverables**:
  - `modules/face_analysis/landmark_detector.py` and `pose_estimator.py`.
  - Structural feature map rasterizer connecting detected landmarks across masked regions.
- **Pass Condition**: Correctly detects landmarks and estimates pose on target and reference faces; landmark L2 error is quantified against ground truth.

### PHASE 6: Single-Reference Identity Conditioning (Baseline B2)
- **Objective**: Condition inpainting pipeline on a single reference identity embedding.
- **Deliverables**:
  - `models/identity/feature_extractor.py` extracting identity embeddings from reference $\mathbf{R}_1$.
  - Conditioning projection into generator latent space.
- **Pass Condition**: Reconstructed facial features shift towards reference identity compared to unconditioned baseline B1.

### PHASE 7: Mask-Aware Reference Scoring (O1 Retrieve)
- **Objective**: Implement mask-conditioned multi-reference evaluation module.
- **Deliverables**:
  - `modules/reference_scoring/mask_aware_scorer.py` computing pose compatibility, image quality, landmark visibility within $\Omega$, and geometric overlap.
  - Top-$K$ selection per region of interest.
- **Pass Condition**: References showing the specific occluded organ receive demonstrably higher regional scores than references with hidden organs.

### PHASE 8: Regional Identity-Token Transformer Fusion (O2 Fuse)
- **Objective**: Implement patch-level landmark-aligned identity tokens and mask-conditioned cross-attention.
- **Deliverables**:
  - `models/transformer/regional_cross_attention.py`.
  - `modules/identity_fusion/regional_token_fusion.py` outputting fused spatial identity tokens $\mathbf{T}_{\text{fused}}$.
- **Pass Condition**: Multi-reference tokens fuse spatially; attention maps inspectably attend to relevant reference patches; ablations A2, A4, A5 tested.

### PHASE 9: Diffusion Inpainting Backbone & Hybrid Integration
- **Objective**: Inject fused regional tokens and coarse edge priors into diffusion denoising loop.
- **Deliverables**:
  - `models/diffusion/hybrid_inpaint_pipeline.py`.
  - Spatial cross-attention layers conditioning diffusion latents on $\mathbf{T}_{\text{fused}}$.
- **Pass Condition**: Diffusion samples realistic faces aligned with fused identity tokens and structural boundaries.

### PHASE 10: Facial Degradation / Restoration Mode (Mode 2)
- **Objective**: Implement restoration for existing but damaged facial regions (wrinkles, scars, burn marks, blemishes).
- **Deliverables**:
  - `modules/restoration/degradation_restorer.py`.
  - Residual restoration mask processing preserving underlying bone structure and natural skin texture.
- **Pass Condition**: Removes degradation marks without blurring skin or altering identity; unmasked facial structure remains intact.

### PHASE 11: Multiple Mask Inpainting Mode (Mode 3)
- **Objective**: Support multiple disconnected masks across different facial organs in a single coherent pass.
- **Deliverables**:
  - `modules/masking/multi_mask_router.py` performing connected-component analysis and per-region feature alignment.
- **Pass Condition**: Reconstructs disjoint masked organs (e.g. cheek + nose + mouth) with global geometric and illumination consistency.

### PHASE 12: Non-Face Object Removal Mode (Mode 4)
- **Objective**: Reconstruct non-facial masked objects (wall text, logos, background marks) using environment context without injecting facial features.
- **Deliverables**:
  - `modules/object_removal/context_inpaint.py`.
  - Semantic mask classifier isolating non-face regions ($IoU_{\text{face}} = 0$).
- **Pass Condition**: Completely removes synthetic text/objects; restores background texture/lighting seamlessly; zero facial feature intrusion.

### PHASE 13: Verifier-Aware Candidate Selection (O3 Verify)
- **Objective**: Implement inference-time reranking using held-out ArcFace-R100 matcher and quality estimators (NIQE/BRISQUE).
- **Deliverables**:
  - `modules/verification/candidate_verifier.py`.
  - Joint scoring function $\mathcal{S}_{\text{joint}} = \lambda \mathcal{S}_{\text{ID}} + (1-\lambda) \mathcal{S}_{\text{Quality}}$ with candidate rejection filters.
- **Pass Condition**: Reranking selects the candidate with the highest joint quality-identity score; rejects distorted or artifacted samples.

### PHASE 14: Systematic Ablation Study (A1–A7 & B0–B4)
- **Objective**: Execute all 7 ablations and 5 baseline comparisons on standardized test subsets.
- **Deliverables**:
  - `evaluation/ablations/runner.py`.
  - Comparative metric tables and significance testing.
- **Pass Condition**: Confirms research hypothesis: A7 outperforms A1–A6 across identity and quality metrics.

### PHASE 15: Full Evaluation Across Mask Severities (25%, 50%, 75%, 100%)
- **Objective**: Run full evaluation suite across all 4 severity tiers with held-out matcher.
- **Deliverables**:
  - `evaluation/reports/benchmark_report.py`.
  - Full evaluation panels, metric CSVs, ROC curves, and TAR@FAR=0.1% reports.
- **Pass Condition**: Full benchmark executed; diagnostic panels generated for all test cases.

### PHASE 16: Research Inspection Interface / Frontend
- **Objective**: Interactive interface exposing multi-region analysis, reference scores, fusion maps, candidate comparisons, and diagnostic panels.
- **Deliverables**:
  - Lightweight web-based / interactive UI.
- **Pass Condition**: User can upload target, references, mask, inspect region analysis, and view verified outputs.
