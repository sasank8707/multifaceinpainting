"""
Packaging Script for Kaggle / Google Colab T4 Inference.
Creates a clean, self-contained zip package containing:
- All required model code (models/, modules/, configs/)
- Pretrained weights (ArcFace ONNX, MediaPipe task, YuNet detector)
- Trained checkpoint (step_150.pt)
- Verification & inference scripts (verify_cuda_t4_setup.py, inference_t4_gpu.py)
- Colab / Kaggle Jupyter Notebook (kaggle_colab_t4_inference.ipynb)
- Sample test data (user_tests/test1/)
- Requirements & Quickstart README
"""

import os
import sys
import zipfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
DIST_DIR.mkdir(parents=True, exist_ok=True)
ZIP_FILE = DIST_DIR / "multifaceinpainting_t4_package.zip"

INCLUDED_ITEMS = [
    # Trained checkpoint
    "checkpoints/decoupled_identity_composite_loss/step_150.pt",
    # Architecture source code & pretrained models
    "models",
    "modules",
    "configs",
    # Scripts & Notebooks
    "scripts/inference_t4_gpu.py",
    "scripts/verify_cuda_t4_setup.py",
    "notebooks/kaggle_colab_t4_inference.ipynb",
    "requirements_t4.txt",
    "README_T4_COLAB_KAGGLE.md",
    # User test 1 inputs
    "user_tests/test1/target.jpg",
    "user_tests/test1/mask.png",
    "user_tests/test1/ref1.jpg",
    "user_tests/test1/ref2.jpg",
    "user_tests/test1/ref3.jpg",
]

EXCLUDED_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".git",
    "*.pyc",
    "*.pyo",
    "data/raw",
    "data/processed",
    "checkpoints/archive",
    "dist"
]


def should_exclude(rel_path: str) -> bool:
    for pat in EXCLUDED_PATTERNS:
        if pat.startswith("*") and rel_path.endswith(pat[1:]):
            return True
        if pat in rel_path.replace("\\", "/").split("/"):
            return True
    return False


def build_package():
    print("=" * 80)
    print("PACKAGING MULTI-REFERENCE INPAINTING FOR COLAB / KAGGLE T4 GPU")
    print("=" * 80)
    sys.stdout.flush()
    t0 = time.time()

    total_files = 0
    total_uncompressed_bytes = 0
    added_paths = set()

    if ZIP_FILE.exists():
        ZIP_FILE.unlink()

    with zipfile.ZipFile(str(ZIP_FILE), "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item_rel in INCLUDED_ITEMS:
            item_path = PROJECT_ROOT / item_rel
            if not item_path.exists():
                print(f"  [WARN] Item not found: {item_rel}")
                continue

            if item_path.is_file():
                arcname = Path(item_rel).as_posix()
                if arcname not in added_paths:
                    print(f"  Adding file: {arcname} ({item_path.stat().st_size / (1024*1024):.2f} MB)")
                    sys.stdout.flush()
                    zf.write(str(item_path), arcname)
                    added_paths.add(arcname)
                    total_files += 1
                    total_uncompressed_bytes += item_path.stat().st_size

            elif item_path.is_dir():
                print(f"  Adding directory tree: {item_rel}/")
                sys.stdout.flush()
                for root, dirs, files in os.walk(str(item_path)):
                    dirs[:] = [d for d in dirs if not should_exclude(d)]
                    for f in files:
                        full_f = Path(root) / f
                        rel_f = full_f.relative_to(PROJECT_ROOT).as_posix()
                        if should_exclude(rel_f) or rel_f in added_paths:
                            continue
                        zf.write(str(full_f), rel_f)
                        added_paths.add(rel_f)
                        total_files += 1
                        total_uncompressed_bytes += full_f.stat().st_size

    zip_size_mb = ZIP_FILE.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)
    elapsed = time.time() - t0

    print("=" * 80)
    print(f"PACKAGE GENERATION COMPLETED SUCCESSFULLY in {elapsed:.1f}s")
    print(f"  Archive Location:    {ZIP_FILE}")
    print(f"  Compressed Size:     {zip_size_mb:.2f} MB")
    print(f"  Uncompressed Size:   {raw_size_mb:.2f} MB")
    print(f"  Files Bundled:       {total_files}")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    build_package()
