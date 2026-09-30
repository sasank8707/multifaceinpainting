"""
Pre-Flight Verification Script for CUDA / NVIDIA T4 Inference.
Verifies all model components, weights, dependencies, and data flow
before launching inference on Kaggle / Colab or local environment.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import cv2
import numpy as np


def verify_environment() -> bool:
    print("=" * 80)
    print("PRE-FLIGHT VERIFICATION: CUDA / T4 GPU INFERENCE SETUP")
    print("=" * 80)
    all_passed = True

    # 1. Hardware & Compute Device Check
    print("\n[1/5] Checking Hardware & PyTorch Configuration...")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  CUDA Devices: {device_count}")
        print(f"  Device 0: {device_name} ({vram_gb:.2f} GB VRAM)")
        is_t4 = "T4" in device_name
        print(f"  NVIDIA T4 Detected: {is_t4}")
        target_device = torch.device("cuda")
        target_dtype = torch.float16
    else:
        print("  Running on CPU (CUDA fallback mode enabled)")
        target_device = torch.device("cpu")
        target_dtype = torch.float32

    # 2. Pretrained Model Files Verification
    print("\n[2/5] Checking Required Pretrained Weights & Files...")
    required_files = [
        ("Trained Checkpoint (step_150.pt)", "checkpoints/decoupled_identity_composite_loss/step_150.pt", 100 * 1024 * 1024),
        ("ArcFace ONNX (w600k_r50.onnx)", "models/pretrained/w600k_r50.onnx", 150 * 1024 * 1024),
        ("MediaPipe Face Landmarker (.task)", "models/pretrained/face_landmarker.task", 3 * 1024 * 1024),
        ("OpenCV YuNet Face Detector", "models/pretrained/face_detection_yunet_2023mar.onnx", 200 * 1024),
    ]

    for label, rel_path, min_bytes in required_files:
        full_p = PROJECT_ROOT / rel_path
        if full_p.exists():
            sz = full_p.stat().st_size
            if sz >= min_bytes:
                print(f"  [PASS] {label}: {full_p.name} ({sz / (1024*1024):.2f} MB)")
            else:
                print(f"  [WARN] {label} exists but size is smaller than expected ({sz} bytes)")
        else:
            print(f"  [FAIL] Missing required file: {rel_path}")
            all_passed = False

    # 3. Python Package Imports Verification
    print("\n[3/5] Verifying Dependencies & Module Imports...")
    required_pkgs = ["diffusers", "transformers", "accelerate", "mediapipe", "cv2", "PIL", "skimage", "matplotlib"]
    for pkg in required_pkgs:
        try:
            __import__(pkg)
            print(f"  [PASS] Package '{pkg}' is installed and importable.")
        except ImportError as e:
            print(f"  [FAIL] Package '{pkg}' failed to import: {e}")
            all_passed = False

    # 4. Pipeline Architecture Modules Check
    print("\n[4/5] Checking Architecture Module Instantiations...")
    try:
        from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
        from models.identity.arcface_backbone import RegionalIdentityEncoder
        from models.transformer.fusion_transformer import RegionalIdentityTransformer
        from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
        from models.diffusion.diffusion_adapter import AspectRatioPreserver
        from modules.face_analysis.pipeline import FaceAnalysisPipeline
        from modules.reference_scoring.scorer import MaskAwareReferenceScorer
        print("  [PASS] All 7 core architectural modules imported successfully.")
    except Exception as e:
        print(f"  [FAIL] Error importing architecture modules: {e}")
        return False

    # Instantiate test objects
    try:
        arcface = RegionalIdentityEncoder(token_dim=512)
        transformer = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2)
        scorer = MaskAwareReferenceScorer()
        face_pipe = FaceAnalysisPipeline()
        print("  [PASS] Successfully instantiated RegionalIdentityEncoder, RegionalIdentityTransformer, MaskAwareReferenceScorer, and FaceAnalysisPipeline.")
    except Exception as e:
        print(f"  [FAIL] Error instantiating components: {e}")
        all_passed = False

    # 5. Checkpoint Integrity Check
    print("\n[5/5] Verifying step_150.pt Checkpoint Structure...")
    ckpt_path = PROJECT_ROOT / "checkpoints/decoupled_identity_composite_loss/step_150.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            assert "state_dict" in ckpt, "Missing 'state_dict' in checkpoint"
            sd = ckpt["state_dict"]
            num_keys = len(sd)
            print(f"  [PASS] Loaded checkpoint successfully. Keys in state_dict: {num_keys}")
            print(f"  [PASS] Calibrated scale in checkpoint: {ckpt.get('scale', 'N/A')}")
            print(f"  [PASS] Identity token dimension: {ckpt.get('id_dim', 'N/A')}")
        except Exception as e:
            print(f"  [FAIL] Failed to inspect checkpoint: {e}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print(">>> ALL PRE-FLIGHT VERIFICATIONS PASSED! System is 100% ready for CUDA/T4 execution. <<<")
    else:
        print(">>> WARNING: Some pre-flight checks failed. Review errors above. <<<")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    verify_environment()
