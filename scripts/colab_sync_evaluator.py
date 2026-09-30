"""
Colab Evaluator Synchronization and Pre-Flight Validation Script.

Performs clean inspection and validation without embedding any binary data:
1. Inspects scripts/evaluate_checkpoints.py for legacy Phase-7 paths.
2. If outdated, replaces with the corrected canonical evaluator from package/repo.
3. Verifies all 5 canonical images and data/masks/canonical_eval_mask_25pct.png exist.
4. Checks step_150.pt and step_500.pt without modifying either.
5. Runs: python scripts/evaluate_checkpoints.py --test_inputs_only
"""

import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sync_and_validate():
    print("=" * 80)
    print("COLAB EVALUATION PRE-FLIGHT DIAGNOSTIC & SYNCHRONIZATION")
    print("=" * 80)

    target_eval = PROJECT_ROOT / "scripts/evaluate_checkpoints.py"

    # 1. Inspection
    print("\n[1/4] Inspecting scripts/evaluate_checkpoints.py...")
    needs_replacement = True
    if target_eval.exists():
        content = target_eval.read_text(encoding="utf-8")
        if "phase7_artifacts" in content or "01_original_target.png" in content:
            print("  Status: Legacy Phase-7 references detected.")
            needs_replacement = True
        elif "resolve_eval_asset" in content:
            print("  Status: Corrected canonical evaluator is already in place.")
            needs_replacement = False

    # 2. Replacement if needed
    if needs_replacement:
        print("\n[2/4] Updating scripts/evaluate_checkpoints.py...")
        # Check if evaluator_update.zip or multifaceinpainting_cloud_training.zip is present
        replaced = False
        for zip_name in ["dist/evaluator_update.zip", "dist/multifaceinpainting_cloud_training.zip"]:
            zip_path = PROJECT_ROOT / zip_name
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as z:
                    if "scripts/evaluate_checkpoints.py" in z.namelist():
                        z.extract("scripts/evaluate_checkpoints.py", PROJECT_ROOT)
                        print(f"  Extracted corrected evaluator from {zip_name}.")
                        replaced = True
                        break
        if not replaced:
            print("  Notice: Ensure scripts/evaluate_checkpoints.py is updated from the repository.")
    else:
        print("\n[2/4] Evaluator script verified up-to-date.")

    # 3. Canonical Assets Verification
    print("\n[3/4] Checking Canonical Evaluation Assets...")
    expected_files = [
        PROJECT_ROOT / "data/processed/dataset_cache/7367.jpg",
        PROJECT_ROOT / "data/processed/dataset_cache/13401.jpg",
        PROJECT_ROOT / "data/processed/dataset_cache/23058.jpg",
        PROJECT_ROOT / "data/processed/dataset_cache/23414.jpg",
        PROJECT_ROOT / "data/processed/dataset_cache/3616.jpg",
        PROJECT_ROOT / "data/masks/canonical_eval_mask_25pct.png"
    ]

    mask_file = PROJECT_ROOT / "data/masks/canonical_eval_mask_25pct.png"
    if not mask_file.exists():
        for zip_name in ["dist/evaluator_update.zip", "dist/multifaceinpainting_cloud_training.zip"]:
            zip_path = PROJECT_ROOT / zip_name
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as z:
                    if "data/masks/canonical_eval_mask_25pct.png" in z.namelist():
                        z.extract("data/masks/canonical_eval_mask_25pct.png", PROJECT_ROOT)
                        print(f"  Extracted canonical mask from {zip_name}.")
                        break

    for f in expected_files:
        status = "EXISTS" if f.exists() else "MISSING"
        print(f"  [{status}] {f.relative_to(PROJECT_ROOT)}")

    # Checkpoints check (strictly read-only)
    step150 = PROJECT_ROOT / "checkpoints/baselines/step_150_best/step_150.pt"
    step500 = PROJECT_ROOT / "checkpoints/cloud_training/step_500.pt"
    print(f"  [CHECKPOINT] step_150.pt: {'EXISTS' if step150.exists() else 'MISSING'}")
    print(f"  [CHECKPOINT] step_500.pt: {'EXISTS' if step500.exists() else 'MISSING'}")

    # 4. Run --test_inputs_only
    print("\n[4/4] Executing: python scripts/evaluate_checkpoints.py --test_inputs_only\n")
    cmd = [sys.executable, str(target_eval), "--test_inputs_only"]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    print("=" * 80)
    print("PRE-FLIGHT VALIDATION FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    sync_and_validate()
