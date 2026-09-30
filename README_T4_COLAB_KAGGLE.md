# Multi-Reference Facial Inpainting: NVIDIA T4 / CUDA Setup Guide
### High-Performance Inference on Google Colab & Kaggle

This package enables running the complete multi-reference facial inpainting pipeline on an **NVIDIA T4 GPU (16 GB VRAM)** in Google Colab, Kaggle Kernels, or any CUDA environment.

---

## 🚀 Key Specifications & Parameters
All hyperparameters strictly match the locally calibrated and trained pipeline:
- **Trained Checkpoint**: `step_150.pt` (Decoupled Identity Cross-Attention weights with composite loss)
- **Classifier-Free Guidance (CFG)**: `7.5`
- **Identity Coupling Scale ($\lambda$)**: `0.10`
- **Dynamic Reference Scoring**: `MaskAwareReferenceScorer` (evaluates pose, quality, visibility, mask overlap)
- **Regional Fusion**: 3-Reference `RegionalIdentityTransformer` with strict target identity anti-leakage
- **Diffusion Backbone**: `runwayml/stable-diffusion-inpainting` (5 DDIM steps)
- **Unmasked Invariance**: 100% bit-exact unmasked pixel preservation
- **T4 Execution Speed**: **~0.5 - 0.7 seconds** for 5 diffusion steps (~110 ms/step) at ~3.8 GB VRAM.

---

## 📦 Package Contents
```
├── checkpoints/
│   └── decoupled_identity_composite_loss/
│       └── step_150.pt                       # 146 MB trained cross-attention weights
├── models/
│   ├── pretrained/
│   │   ├── w600k_r50.onnx                    # 166 MB ArcFace ResNet-50 backbone
│   │   ├── face_landmarker.task              # MediaPipe landmark model
│   │   └── face_detection_yunet_2023mar.onnx # OpenCV YuNet detector
│   ├── identity/                             # ArcFace backbone & differentiable modules
│   ├── transformer/                          # Regional transformer fusion
│   └── diffusion/                            # Decoupled cross-attention manager & adapter
├── modules/
│   ├── face_analysis/                        # Landmark, pose, and quality analysis
│   ├── reference_scoring/                    # MaskAwareReferenceScorer
│   └── utils/                                # Bit-exact invariance & geometry
├── scripts/
│   ├── verify_cuda_t4_setup.py               # Pre-flight environment & weights validation
│   └── inference_t4_gpu.py                   # Single standalone GPU inference CLI & API
├── notebooks/
│   └── kaggle_colab_t4_inference.ipynb       # 1-Click Jupyter Notebook with rich visualizations
├── user_tests/test1/
│   ├── target.jpg                            # 484x369 ground truth portrait
│   ├── mask.png                              # 484x369 binary mask (21.5% facial occlusion)
│   ├── ref1.jpg                              # Reference photo 1 (3/4 pose)
│   ├── ref2.jpg                              # Reference photo 2 (upward pose)
│   └── ref3.jpg                              # Reference photo 3 (frontal pose)
├── requirements_t4.txt                       # Minimal dependency specification
└── README_T4_COLAB_KAGGLE.md                 # This guide
```

---

## ⚡ Option A: Running in Google Colab (Recommended)

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **Set Runtime to GPU**:
   - Navigate to `Runtime` $\rightarrow$ `Change runtime type`.
   - Select **T4 GPU** under Hardware Accelerator.
3. **Upload the Package**:
   - Upload `multifaceinpainting_t4_package.zip` into the Colab file browser (`/content`).
   - Unzip it in a Colab code cell:
     ```bash
     !unzip -q multifaceinpainting_t4_package.zip -d multifaceinpainting
     %cd multifaceinpainting
     ```
4. **Open the Notebook or Run CLI**:
   - You can open `notebooks/kaggle_colab_t4_inference.ipynb` directly in Colab and run all cells sequentially.
   - Or run the standalone script directly:
     ```bash
     !pip install -q -r requirements_t4.txt
     !python scripts/verify_cuda_t4_setup.py
     !python scripts/inference_t4_gpu.py --target user_tests/test1/target.jpg \
                                         --mask user_tests/test1/mask.png \
                                         --ref1 user_tests/test1/ref1.jpg \
                                         --ref2 user_tests/test1/ref2.jpg \
                                         --ref3 user_tests/test1/ref3.jpg \
                                         --output_dir output/t4_run \
                                         --device cuda
     ```

---

## ⚡ Option B: Running in Kaggle Notebooks

1. **Create a New Notebook** on Kaggle: [kaggle.com/code](https://kaggle.com/code)
2. **Turn On GPU Accelerator**:
   - In the Notebook settings on the right panel, set **Accelerator** to **GPU T4 x 2**.
   - Set **Internet** to **Internet on**.
3. **Upload Dataset / Zip**:
   - Upload `multifaceinpainting_t4_package.zip` as a Kaggle Dataset, or clone your GitHub repository.
4. **Run Inference**:
   ```bash
   !unzip -q /kaggle/input/multifaceinpainting-t4/multifaceinpainting_t4_package.zip -d /kaggle/working/multifaceinpainting
   %cd /kaggle/working/multifaceinpainting
   !pip install -q -r requirements_t4.txt
   !python scripts/inference_t4_gpu.py --device cuda
   ```

---

## 📊 Output Artifacts & Verification

Every inference run automatically generates:
1. `final_reconstruction.png`: Full-resolution output with unmasked region 100% bit-exact preserved and occluded facial region reconstructed.
2. `masked_crop.png`: Close-up bounding box crop of the reconstructed organ.
3. `side_by_side_comparison.png`: 5-panel figure displaying:
   - Panel 1: Target Ground Truth
   - Panel 2: Masked Input
   - Panel 3: Reconstruction (annotated with identity cosine similarity, PSNR, SSIM)
   - Panel 4: 3 Reference Images with computed dynamic weights
   - Panel 5: Ground Truth vs Reconstructed Crop comparison
4. `metrics.json`: Full structured JSON metrics recording:
   - GPU device name and precision (CUDA FP16)
   - Diffusion runtime (~0.5 - 0.7s on T4) and per-step latency
   - Computed dynamic reference weights ($w_1, w_2, w_3$)
   - Unmasked invariance test result (`passed: true`, `violated_pixels: 0`)
   - ArcFace biometric cosine similarities (vs Reference Mean and Target GT)
   - Reconstruction quality metrics (PSNR, SSIM, MAE, RMSE, Sobel error, Laplacian error)
