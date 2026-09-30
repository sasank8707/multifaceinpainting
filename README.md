# Multi-Face Inpainting & Multi-Mode Image Restoration System

An identity-preserving facial restoration and computer vision system built as a Computer Science & Engineering Final Project. Combines localized damage detection, multi-reference biometric identity conditioning, and latent diffusion inpainting with strict canvas preservation guarantees.

---

## Key Features

### Mode 1: Dual-Path Face Inpainting & Facial Restoration
- **Path A: User-Supplied Mask**: Reconstructs damaged, occluded, or painted facial regions from a user-supplied binary mask.
- **Path B: Automatic Damage Detection**: Detects burns, scars, scratches, and lacerations automatically using LAB color anomalies, multi-scale texture deviations, and landmark-guided facial zone shielding.
- **Biometric Identity Conditioning (O1 & O2)**:
  - **O1 Retrieve**: Mask-aware scoring of 1–3 reference photos across pose compatibility, image quality, and landmark visibility.
  - **O2 Regional Fusion**: Cross-attention transformer fusing reference identity tokens into target query positions without target identity leakage.
- **Decoupled Identity Cross-Attention**: Injects multi-reference facial identity at calibrated scale $\lambda = 0.10$ into Stable Diffusion Inpainting (`step_150.pt`).
- **Mathematical Canvas Invariance**: Inward boundary feathering ($r = 7\text{ px}$) strictly inside the damaged region, guaranteeing exact bit-level invariance ($\Delta_{\max} = 0.0$, 0 pixels altered outside mask).

### Mode 2: Background Removal
- High-speed foreground person and object extraction using LRASPP MobileNetV3-Large segmentation.
- Produces clean, transparent RGBA PNG cutouts with 100% bit-exact original RGB pixels.

### Mode 3: Edge-Preserving Deblurring
- Multi-scale bilateral detail decomposition with adaptive unsharp gradient boosting.
- Enhances facial contours and edge acuity while suppressing high-frequency noise amplification.

---

## System Architecture

```
                       [ Input Target Face ]
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
      [ Path A ]                                  [ Path B ]
  User-Supplied Mask                    Automatic Damage Detector
           │                             (LAB Color + Texture Anomaly)
           └─────────────────────┬─────────────────────┘
                                 │
                     [ Normalized Binary Mask ]
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
     [ 1–3 Reference Photos ]             [ Target Geometry ]
              │                                     │
      [ O1 Scorer (Weights) ]               [ Spatial Mask ]
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
                     [ O2 Regional Transformer ]
                                 │
                     [ Fused Identity Tokens ]
                                 │
                                 ▼
                 [ Decoupled Identity Cross-Attention ]
                    (Stable Diffusion Inpainting UNet)
                                 │
                                 ▼
                   [ Latent Diffusion Denoising ]
                                 │
                                 ▼
                    [ Boundary Harmonization ]
                     (Inward Feathering r = 7 px)
                                 │
                                 ▼
                     [ Restored Output Canvas ]
                     (Bit-Exact Outside Mask: Δ = 0)
```

---

## Directory Structure

```
multifaceinpainting/
├── app.py                      # Flask Web Application & REST API
├── configs/                    # Training and inference configurations
├── docs/                       # Architecture specifications & documentation
├── models/
│   ├── diffusion/              # Decoupled Cross-Attention & adapter managers
│   ├── identity/               # ArcFace encoder & differentiable backbones
│   └── transformer/            # Regional Identity Transformer (O2 Fuse)
├── modules/
│   ├── face_analysis/          # FaceLandmarker & pose/quality extractors
│   ├── masking/                # Synthetic mask generators & task routing
│   ├── reference_scoring/      # Mask-Aware Reference Scorer (O1 Retrieve)
│   ├── restoration/            # Unified restoration pipeline & detectors
│   └── utils/                  # Geometry, device configs, and metrics
├── scripts/                    # CLI demonstration tools and evaluation suites
├── static/                     # CSS stylesheets, UI assets, and client JS
├── templates/                  # Responsive web dashboard templates
├── tests/                      # Pytest unit and integration test suite
└── user_tests/                 # External test cases and verification images
```

---

## Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/sasank8707/multifaceinpainting.git
cd multifaceinpainting

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000` to access the interactive web interface.

### 3. Run the Automated Test Suite
```bash
pytest tests/ -q
```

---

## Project Verification

- **Checkpoint**: Step 150 Decoupled Cross-Attention Adapter (`step_150.pt`)
- **Biometric Identity Recovery**: ArcFace Cosine vs Target GT $\ge 0.914$
- **Unmasked Invariance**: $\Delta_{\max} = 0.0$ (0 pixels modified outside mask)
- **Validated On**: Windows CPU & NVIDIA CUDA environments
