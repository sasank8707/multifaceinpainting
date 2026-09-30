# Real Natural Damage Validation Interface

This directory serves as the testbed for validating the automatic damage mask detector and face restoration pipeline on real-world, naturally damaged facial images (e.g., burns, scars, scratches, abrasions, dermatological lesions).

> [!IMPORTANT]
> **No Synthetic or Fabricated Images**:
> Do not fabricate, synthesize, or place artificially masked faces here. Only authentic, unmanipulated photographs of subjects with natural facial damage and matching uncorrupted reference photos should be placed in this folder.

---

## Expected Directory Structure

Each evaluation case must reside in its own subdirectory without any mask:

```
user_tests/
└── natural_damage/
    ├── case01/
    │   ├── target.jpg          # The naturally damaged face (unmasked, authentic photo)
    │   ├── ref1.jpg            # Reference portrait 1 (same subject, undamaged)
    │   ├── ref2.jpg            # Reference portrait 2 (optional but recommended)
    │   └── ref3.jpg            # Reference portrait 3 (optional)
    ├── case02/
    │   ├── target.jpg
    │   ├── ref1.jpg
    │   └── ref2.jpg
    └── README.md
```

### Critical Protocol Rules:
1. **NO MASKS PROVIDED**: The user or tester must **never** provide a mask (`mask.png` or `mask.jpg`) in these folders. The detector (`DamageMaskDetector`) must infer the damaged region automatically.
2. **Same Subject References**: Reference photos (`ref1.jpg`, etc.) must depict the **same identity** under clean, undamaged conditions (e.g. pre-injury photos or alternate clear angles).
3. **Single Face Priority**: Images should depict a single individual. If multiple people are present, the safety check will flag `MULTIPLE_FACES_DETECTED` and abort to prevent ambiguous identity reconstruction.
4. **Mandatory Human Verification**: All automatic masks require visual inspection (`AUTO_MASK_REVIEW_REQUIRED`) via `human_review_checkpoint.png` before any restoration claim can be certified.

---

## Execution Command

To run validation on a case:

```powershell
.venv\Scripts\python.exe scripts/validate_auto_damage.py `
  --target user_tests/natural_damage/case01/target.jpg `
  --references user_tests/natural_damage/case01/ref1.jpg user_tests/natural_damage/case01/ref2.jpg `
  --checkpoint checkpoints/baselines/step_150_best/step_150.pt `
  --output_dir evaluation/visualizations/auto_damage_validation/natural_case01 `
  --label "Natural burn/scar real-world validation"
```

The script will generate:
1. Original target copy (`01_original_target.png`)
2. Auto-detected binary mask (`02_auto_mask.png`)
3. Mask overlay with bounding box (`03_mask_overlay.png`)
4. Reference gallery panel (`04_reference_gallery.png`)
5. Bit-exact restored output (`05_restored.png`)
6. Human review checkpoint (`human_review_checkpoint.png`)
7. Comprehensive detector diagnostics (`diagnostics/`)
8. Execution metadata JSON (`metadata.json`)
