# Data Pipeline & Synthetic Pair Construction Protocol

## 1. Overview & Dataset Sources

To guarantee biometric identity preservation, all identity-conditioned training and evaluation pairs are constructed using verified multi-image identity sets from official identity annotations:

1. **CelebA-HQ ($1024 \times 1024$)**:
   - Primary high-resolution dataset for multi-reference identity inpainting.
   - **Identity Rule**: Filenames do **not** imply identity. Identities are strictly derived by cross-referencing `CelebA-HQ-to-CelebA-mapping.txt` with `identity_CelebA.txt`.
   - Only identities with $N \ge 2$ unconstrained photos are included in the multi-reference pairing pool (1 ground truth target + 1 to 3 reference candidates).
2. **CelebAMask-HQ ($1024 \times 1024$)**:
   - Provides 19-class pixel-level semantic segmentation masks:
     - Facial organs: `l_eye`, `r_eye`, `l_brow`, `r_brow`, `nose`, `u_lip`, `l_lip`, `mouth`.
     - Face structure: `skin`, `neck`, `l_ear`, `r_ear`.
     - Accessories & hair: `cloth`, `hair`, `hat`, `eyeglass`, `ear_r`, `neck_l`.
   - Used to generate semantically accurate organ masks and evaluate region-specific identity preservation.
3. **Flickr-Faces-HQ (FFHQ) ($1024 \times 1024$)**:
   - 70,000 high-quality face images featuring diverse age, ethnicity, lighting, accessories, and unconstrained backgrounds.
   - Used for generalisation benchmarking, non-face object removal, and out-of-distribution evaluation.

---

## 2. Multi-Reference Identity Indexing Pipeline

```
identity_CelebA.txt (CelebA image -> identity_id)
                      │
CelebA-HQ-to-CelebA-mapping.txt (HQ index -> CelebA image)
                      │
                      ▼
        [ Identity Indexer Engine ]
                      │
                      ▼
   Filter Identities with Count >= 2
                      │
                      ▼
   Partition Identities into Fixed Splits:
   - Train Split (80% of identities)
   - Validation Split (10% of identities)
   - Test Split (10% of identities)
                      │
                      ▼
   For Each Sample Query:
   Target = Image_T (Ground Truth)
   References = [Ref_1, Ref_2, Ref_3] (Withheld / Paired)
```

---

## 3. Synthetic Mask Generators (10 Operational Types)

The data pipeline supports 10 distinct, deterministic, reproducible mask generation strategies:

| Mask Type ID | Generator Name | Description & Parameters | Applicable Task Mode |
| :--- | :--- | :--- | :--- |
| `M01` | **Rectangular Mask** | Random coordinates $(x, y, w, h)$ covering $20\%\text{--}80\%$ of face bounding box. | Mode 1 (Inpainting) |
| `M02` | **Elliptical Mask** | Center, axes, and rotation angle $\theta \in [0, 180^\circ]$ creating smooth curved boundaries. | Mode 1 (Inpainting) |
| `M03` | **Irregular Free-form Mask** | Random walk bezier curves, multi-stroke brush lines ($10\text{--}40\text{px}$ width) with random endpoints. | Mode 1 (Inpainting) |
| `M04` | **Semantic Facial Organ Mask** | Derived from CelebAMask-HQ or landmark convex hulls (e.g. nose-only, eye-pair, mouth+jaw). | Mode 1 (Inpainting) |
| `M05` | **Multiple Disconnected Masks** | 2 to 5 non-overlapping independent mask patches distributed across different facial regions. | Mode 3 (Multi-Mask) |
| `M06` | **Large Facial Occlusion** | Severe occlusion covering $\ge 50\%$ to $100\%$ (full face) of the facial bounding box. | Mode 1 (Severe Inpainting) |
| `M07` | **Localized Skin Degradation** | High-frequency noise, local contrast shifts, color bleaching simulating scars/burn marks. | Mode 2 (Restoration) |
| `M08` | **Wrinkle & Texture Blemish** | High-frequency line-based degradation simulating deep wrinkles or dermatological blemishes. | Mode 2 (Restoration) |
| `M09` | **Non-Face Object Removal** | Synthetic bounding box / polygon positioned strictly in background / clothing region ($IoU_{\text{face}} = 0$). | Mode 4 (Object Removal) |
| `M10` | **Text & Watermark Mask** | Alpha-composited typography (e.g. letter "L", copyright timestamps) placed on background/walls. | Mode 4 (Object Removal) |

---

## 4. Mask Severity & Coverage Calculation

Mask coverage is quantified across three spatial scopes to eliminate misleading global statistics:

1. **Total Image Coverage**:
   $$\text{Cov}_{\text{image}} = \frac{\sum_{x, y} \mathbf{M}(x, y)}{H \times W}$$
2. **Face-Region Coverage**:
   Given detected face bounding box $B_{\text{face}} = [x_1, y_1, x_2, y_2]$:
   $$\text{Cov}_{\text{face}} = \frac{\sum_{(x, y) \in B_{\text{face}}} \mathbf{M}(x, y)}{(x_2 - x_1)(y_2 - y_1)}$$
3. **Region-of-Interest (ROI) Coverage**:
   $$\text{Cov}_{\text{ROI}} = \frac{\sum_{(x, y) \in B_{\text{ROI}}} \mathbf{M}(x, y)}{\text{Area}(B_{\text{ROI}})}$$

The evaluation benchmark standardizes on 4 formal target face severity tiers:
- **Tier 1 (25% Severity)**: $\text{Cov}_{\text{face}} \in [0.20, 0.30]$
- **Tier 2 (50% Severity)**: $\text{Cov}_{\text{face}} \in [0.45, 0.55]$
- **Tier 3 (75% Severity)**: $\text{Cov}_{\text{face}} \in [0.70, 0.80]$
- **Tier 4 (100% Full-Face)**: $\text{Cov}_{\text{face}} \ge 0.95$ (Face completely occluded; reconstruction relies entirely on reference retrieval & fusion)

---

## 5. Metadata Schema (`metadata.json`)

Every synthetic training and evaluation sample is serialized with complete audit provenance:

```json
{
  "sample_id": "celeba_hq_004128_s50_m04",
  "identity_id": "5805",
  "split": "test",
  "target_image": "data/processed/celeba_hq/004128.jpg",
  "ground_truth_target": "data/processed/celeba_hq/004128.jpg",
  "masked_target": "data/experiments/samples/004128_masked.png",
  "mask_image": "data/experiments/samples/004128_mask.png",
  "reference_images": [
    "data/processed/celeba_hq/000004.jpg",
    "data/processed/celeba_hq/008219.jpg"
  ],
  "image_dimensions": {
    "width": 1024,
    "height": 1024,
    "channels": 3
  },
  "mask_metadata": {
    "mask_type": "M04_semantic_facial_organ",
    "target_organs": ["nose", "mouth"],
    "num_components": 1,
    "coverage_image": 0.124,
    "coverage_face": 0.512,
    "coverage_roi": 0.385,
    "bounding_box": [320, 410, 710, 805]
  },
  "face_metadata": {
    "face_detected": true,
    "landmarks_68": [[324, 450], "..."],
    "pose_yaw": -4.2,
    "pose_pitch": 2.1,
    "pose_roll": 0.8
  },
  "task_mode": "MODE_1_FACIAL_INPAINTING",
  "random_seed": 42
}
```

---

## 6. Resolution Abstraction & Geometric Integrity

1. **Resolution Independence**: The system ingests arbitrary image dimensions $(H, W)$ and aspect ratios (e.g. $256 \times 256$, $512 \times 512$, $1024 \times 1024$, $640 \times 480$).
2. **Internal Normalization**:
   - Target and references are normalized to model tensor resolution $(H_{\text{model}}, W_{\text{model}})$ using aspect-ratio-preserving symmetric padding.
   - For localized masks, an expanded context window $\text{ROI} = \text{Expand}(B_{\text{mask}}, \text{margin}=0.3)$ is extracted and processed at native model resolution.
3. **Bit-Exact Boundary Restoration**:
   - The generated output is mapped back to the original geometry $(H, W)$.
   - Unmasked pixels are verified via hash/L1:
     $$\max | \mathbf{I}_{\text{out}}(1 - \mathbf{M}) - \mathbf{I}_{\text{in}}(1 - \mathbf{M}) | = 0$$
