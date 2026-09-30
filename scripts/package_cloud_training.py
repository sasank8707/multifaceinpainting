"""
Utility script to package the project for remote Cloud-GPU training (Colab / Kaggle).
Creates dist/multifaceinpainting_cloud_training.zip containing all essential code,
models, configurations, datasets cache, and the protected step_150_best baseline.
"""

import os
import zipfile
from pathlib import Path

def create_cloud_package():
    root = Path(__file__).resolve().parent.parent
    dist_dir = root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    out_zip = dist_dir / "multifaceinpainting_cloud_training.zip"

    print("=" * 70)
    print("PACKAGING CLOUD TRAINING ARCHIVE")
    print("=" * 70)
    print(f"Target Archive: {out_zip}")

    include_dirs = [
        "checkpoints/baselines/step_150_best",
        "configs",
        "data/masks",
        "data/metadata",
        "data/processed/dataset_cache",
        "models",
        "modules",
        "notebooks",
        "scripts",
        "training"
    ]

    include_files = [
        "requirements_training_cloud.txt",
        "README_CLOUD_TRAINING.md"
    ]

    file_count = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for d in include_dirs:
            dp = root / d
            if dp.exists():
                for f in dp.rglob("*"):
                    if f.is_file() and not any(p.startswith(".") for p in f.parts) and "__pycache__" not in f.parts:
                        rel_path = f.relative_to(root)
                        z.write(f, rel_path)
                        file_count += 1

        for f_name in include_files:
            fp = root / f_name
            if fp.exists():
                z.write(fp, f_name)
                file_count += 1

    size_mb = os.path.getsize(out_zip) / (1024 ** 2)
    print(f"Archive successfully generated:")
    print(f"  Files Packed: {file_count}")
    print(f"  Archive Size: {size_mb:.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    create_cloud_package()
