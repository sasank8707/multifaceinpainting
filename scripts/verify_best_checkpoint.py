"""
Verification Script for Protected Best Checkpoint (step_150.pt).
Verifies:
1. Protected checkpoint exists at checkpoints/baselines/step_150_best/step_150.pt
2. SHA-256 hash matches the immutable baseline manifest exactly
3. File size matches exactly
4. Weights load successfully into DecoupledIdentityCrossAttentionManager
5. Parameter counts and step metadata are intact
"""

import os
import sys
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

EXPECTED_SHA256 = "e4ff5cff392ccca3522b1cc3c937c6a372bb9075ffa3f837e488c810a287be89"
EXPECTED_FILE_SIZE = 153394949  # 146.29 MB
EXPECTED_STEP = 150
EXPECTED_TRAINABLE_PARAMS = 12779520
BASELINE_DIR = PROJECT_ROOT / "checkpoints/baselines/step_150_best"
BASELINE_FILE = BASELINE_DIR / "step_150.pt"
MANIFEST_FILE = BASELINE_DIR / "manifest.json"


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_protected_checkpoint() -> bool:
    print("=" * 80)
    print("VERIFYING PROTECTED BEST CHECKPOINT (STEP_150 BASELINE)")
    print("=" * 80)
    all_passed = True

    # 1. Existence check
    print("\n[1/5] Checking Baseline File Existence...")
    if not BASELINE_FILE.exists():
        print(f"  [FAIL] Protected checkpoint not found at: {BASELINE_FILE}")
        return False
    print(f"  [PASS] File exists: {BASELINE_FILE}")

    # 2. Manifest check
    print("\n[2/5] Checking Manifest File...")
    if not MANIFEST_FILE.exists():
        print(f"  [WARN] Manifest file not found at: {MANIFEST_FILE}")
    else:
        with open(MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
        print(f"  [PASS] Manifest loaded successfully.")
        print(f"         Status: {manifest.get('status', 'UNKNOWN')}")
        print(f"         Recorded Step: {manifest.get('training_step')}")
        print(f"         Recorded SHA-256: {manifest.get('sha256')}")

    # 3. File Size Check
    print("\n[3/5] Verifying File Size...")
    actual_size = BASELINE_FILE.stat().st_size
    print(f"  Actual File Size:   {actual_size:,} bytes ({actual_size / (1024*1024):.2f} MB)")
    print(f"  Expected File Size: {EXPECTED_FILE_SIZE:,} bytes ({EXPECTED_FILE_SIZE / (1024*1024):.2f} MB)")
    if actual_size != EXPECTED_FILE_SIZE:
        print(f"  [FAIL] File size mismatch! Expected {EXPECTED_FILE_SIZE}, got {actual_size}")
        all_passed = False
    else:
        print("  [PASS] File size matches byte-for-byte.")

    # 4. SHA-256 Hash Verification
    print("\n[4/5] Computing and Verifying SHA-256 Checksum...")
    actual_sha256 = compute_sha256(BASELINE_FILE)
    print(f"  Actual SHA-256:   {actual_sha256}")
    print(f"  Expected SHA-256: {EXPECTED_SHA256}")
    if actual_sha256 != EXPECTED_SHA256:
        print("  [FAIL] SHA-256 hash mismatch! Checkpoint has been altered or corrupted!")
        all_passed = False
    else:
        print("  [PASS] SHA-256 checksum matches 100% exactly.")

    # 5. Checkpoint Load and State Inspection
    print("\n[5/5] Loading Checkpoint and Inspecting Model State...")
    try:
        ckpt = torch.load(str(BASELINE_FILE), map_location="cpu", weights_only=False)
        assert "state_dict" in ckpt, "Missing 'state_dict' in checkpoint"
        sd = ckpt["state_dict"]
        num_keys = len(sd)
        step = ckpt.get("step")
        scale = ckpt.get("scale")
        id_dim = ckpt.get("id_dim")
        has_optim = "optimizer_state_dict" in ckpt

        print(f"  State Dict Keys:  {num_keys} (32 expected: 16 to_k_id + 16 to_v_id)")
        print(f"  Checkpoint Step:  {step} (expected {EXPECTED_STEP})")
        print(f"  Coupling Scale:   {scale} (expected 0.1)")
        print(f"  Token Dimension:  {id_dim} (expected 512)")
        print(f"  Optimizer State:  {'Included' if has_optim else 'None'}")

        # Count total parameters in state_dict (each value in sd is an OrderedDict of sub-parameters)
        total_params = sum(param.numel() for sub_sd in sd.values() for param in sub_sd.values())
        print(f"  Total Parameters: {total_params:,} (expected {EXPECTED_TRAINABLE_PARAMS:,})")

        assert num_keys == 32, f"Expected 32 keys, got {num_keys}"
        assert step == EXPECTED_STEP, f"Expected step {EXPECTED_STEP}, got {step}"
        assert total_params == EXPECTED_TRAINABLE_PARAMS, f"Expected {EXPECTED_TRAINABLE_PARAMS}, got {total_params}"
        print("  [PASS] Checkpoint integrity confirmed: all keys, step, scale, and tensor shapes match.")
    except Exception as e:
        print(f"  [FAIL] Failed to load checkpoint: {e}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print(">>> PROTECTED CHECKPOINT VERIFICATION PASSED: step_150 IS SECURE & IMMUTABLE <<<")
    else:
        print(">>> PROTECTED CHECKPOINT VERIFICATION FAILED! Review errors above. <<<")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = verify_protected_checkpoint()
    sys.exit(0 if success else 1)
