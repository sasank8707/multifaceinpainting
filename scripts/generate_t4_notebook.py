"""
Generates the complete, production-grade Jupyter Notebook for Kaggle and Google Colab T4 GPU inference.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Multi-Reference Facial Inpainting with Decoupled Identity Cross-Attention\n",
            "### NVIDIA T4 GPU High-Performance Inference Pipeline (Google Colab & Kaggle)\n",
            "\n",
            "This notebook provides a 1-click, end-to-end inference pipeline for multi-reference facial inpainting accelerated by an **NVIDIA T4 GPU (16 GB VRAM)**.\n",
            "\n",
            "**Key Capabilities:**\n",
            "- **Dynamic Reference Scoring**: Evaluates pose angle, image quality, facial visibility, and mask overlap via `MaskAwareReferenceScorer`.\n",
            "- **Regional Transformer Fusion**: Decouples WHERE identity is missing from WHO the person is, strictly preventing target identity leakage.\n",
            "- **Decoupled Identity Cross-Attention**: Injects multi-reference facial identity at calibrated scale $\\lambda = 0.10$ into Stable Diffusion Inpainting (`step_150.pt`).\n",
            "- **GPU Acceleration**: Generates photorealistic inpaintings in **~0.5 - 0.7 seconds** on NVIDIA T4 with FP16 precision.\n",
            "- **Bit-Exact Invariance**: Enforces strict mathematical invariance for all unmasked canvas pixels."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Hardware Diagnostic & GPU Verification\n",
            "Check that an NVIDIA GPU (preferably T4) is allocated to this runtime."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!nvidia-smi\n",
            "\n",
            "import torch\n",
            "print(f\"PyTorch Version: {torch.__version__}\")\n",
            "print(f\"CUDA Available:  {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"Device Name:     {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"VRAM Available:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
            "else:\n",
            "    print(\"WARNING: No GPU detected! Set Colab/Kaggle runtime accelerator to GPU (T4).\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Environment Setup & Dependency Installation\n",
            "Install required libraries: `diffusers`, `transformers`, `accelerate`, `mediapipe`, `opencv-python`, `scikit-image`, and `onnxruntime`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q diffusers transformers accelerate mediapipe opencv-python scikit-image onnxruntime safetensors matplotlib"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Workspace Setup & Repository Path Configuration\n",
            "Detects current environment (Google Colab, Kaggle, or Local Clone) and verifies all required pretrained model assets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Auto-detect repository root\n",
            "possible_roots = [\n",
            "    Path.cwd(),\n",
            "    Path(\"/content/multifaceinpainting\"),\n",
            "    Path(\"/kaggle/working/multifaceinpainting\"),\n",
            "    Path.cwd().parent\n",
            "]\n",
            "\n",
            "repo_root = None\n",
            "for p in possible_roots:\n",
            "    if (p / \"checkpoints\" / \"decoupled_identity_composite_loss\" / \"step_150.pt\").exists():\n",
            "        repo_root = p.resolve()\n",
            "        break\n",
            "\n",
            "if repo_root is None:\n",
            "    repo_root = Path.cwd().resolve()\n",
            "\n",
            "if str(repo_root) not in sys.path:\n",
            "    sys.path.insert(0, str(repo_root))\n",
            "\n",
            "print(f\"Workspace Root: {repo_root}\")\n",
            "os.chdir(repo_root)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Pre-Flight Verification Check\n",
            "Runs automated integrity checks verifying that the trained checkpoint (`step_150.pt`), ArcFace ONNX, MediaPipe face landmarker, and YuNet detector are present and valid."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from scripts.verify_cuda_t4_setup import verify_environment\n",
            "\n",
            "all_passed = verify_environment()\n",
            "assert all_passed, \"Pre-flight checks failed! Please check missing weights or dependencies above.\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Execute Multi-Reference Inference on T4 GPU\n",
            "Runs the complete inference pipeline on `user_tests/test1/`:\n",
            "- Target Image: `user_tests/test1/target.jpg` (484x369 px)\n",
            "- Binary Mask: `user_tests/test1/mask.png` (484x369 px, ~21.5% facial occlusion)\n",
            "- 3 Reference Images: `ref1.jpg`, `ref2.jpg`, `ref3.jpg` (differing poses and lighting)\n",
            "- Checkpoint: `step_150.pt` (FP16 on CUDA)\n",
            "- Hyperparameters: CFG = 7.5, Identity Scale $\\lambda = 0.10$, 5 DDIM steps, Seed = 42"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from scripts.inference_t4_gpu import run_t4_inference\n",
            "\n",
            "metrics = run_t4_inference(\n",
            "    target_path=\"user_tests/test1/target.jpg\",\n",
            "    mask_path=\"user_tests/test1/mask.png\",\n",
            "    ref_paths=[\n",
            "        \"user_tests/test1/ref1.jpg\",\n",
            "        \"user_tests/test1/ref2.jpg\",\n",
            "        \"user_tests/test1/ref3.jpg\"\n",
            "    ],\n",
            "    checkpoint_path=\"checkpoints/decoupled_identity_composite_loss/step_150.pt\",\n",
            "    output_dir=\"user_tests/test1/output_t4\",\n",
            "    steps=5,\n",
            "    guidance_scale=7.5,\n",
            "    lambda_scale=0.10,\n",
            "    seed=42,\n",
            "    device_str=\"auto\",\n",
            "    use_fp16=True\n",
            ")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Inline Visual Results Comparison\n",
            "Displays the 5-panel side-by-side comparison panel and close-up cropped reconstruction."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from PIL import Image\n",
            "from IPython.display import display\n",
            "\n",
            "output_dir = Path(\"user_tests/test1/output_t4\")\n",
            "side_by_side = Image.open(output_dir / \"side_by_side_comparison.png\")\n",
            "display(side_by_side)\n",
            "\n",
            "print(\"Close-up Reconstructed Masked Region:\")\n",
            "crop_img = Image.open(output_dir / \"masked_crop.png\")\n",
            "display(crop_img)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Performance & Biometric Metrics Report\n",
            "Prints a formatted breakdown of dynamic reference weights, biometric cosine similarity, reconstruction metrics, and T4 runtime latency."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "\n",
            "with open(output_dir / \"metrics.json\", \"r\") as f:\n",
            "    m = json.load(f)\n",
            "\n",
            "print(\"=\" * 75)\n",
            "print(\"T4 GPU INFERENCE PERFORMANCE REPORT\")\n",
            "print(\"=\" * 75)\n",
            "print(f\"Device:                     {m['hardware']['device_name']} ({m['hardware']['dtype']})\")\n",
            "print(f\"Diffusion Runtime:          {m['timing']['diffusion_runtime_seconds']:.3f} s ({m['timing']['diffusion_step_latency_ms']:.1f} ms/step)\")\n",
            "print(f\"Total Pipeline Runtime:     {m['timing']['total_pipeline_runtime_seconds']:.3f} s\")\n",
            "print(\"-\" * 75)\n",
            "print(\"DYNAMIC REFERENCE WEIGHTS:\")\n",
            "for name, w in zip(m['reference_scoring']['ref_names'], m['reference_scoring']['dynamic_weights']):\n",
            "    print(f\"  - {name:8s}: {w:.4f}\")\n",
            "print(\"-\" * 75)\n",
            "print(\"BIOMETRIC COSINE SIMILARITY:\")\n",
            "print(f\"  - Recon vs Reference Mean:  {m['biometric_identity']['cos_recon_vs_ref_mean']:.4f}\")\n",
            "print(f\"  - Recon vs Ref 1 (3/4):     {m['biometric_identity']['cos_recon_vs_ref1']:.4f}\")\n",
            "print(f\"  - Recon vs Ref 2 (Up):      {m['biometric_identity']['cos_recon_vs_ref2']:.4f}\")\n",
            "print(f\"  - Recon vs Ref 3 (Frontal): {m['biometric_identity']['cos_recon_vs_ref3']:.4f}\")\n",
            "print(f\"  - Recon vs Target GT:       {m['biometric_identity']['cos_recon_vs_target_gt']:.4f}\")\n",
            "print(\"-\" * 75)\n",
            "print(\"RECONSTRUCTION QUALITY:\")\n",
            "print(f\"  - PSNR:                     {m['reconstruction_quality']['psnr']:.2f} dB\")\n",
            "print(f\"  - SSIM:                     {m['reconstruction_quality']['ssim']:.4f}\")\n",
            "print(f\"  - MAE:                      {m['reconstruction_quality']['mae']:.2f}\")\n",
            "print(f\"  - Bit-Exact Invariance:     {'PASSED (0 pixel violations)' if m['unmasked_invariance']['passed'] else 'FAILED'}\")\n",
            "print(\"=\" * 75)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Download Output Artifacts\n",
            "Compress output images and metrics into a zip file for download."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "\n",
            "zip_path = Path(\"t4_inference_results.zip\")\n",
            "shutil.make_archive(\"t4_inference_results\", \"zip\", str(output_dir))\n",
            "print(f\"Results zipped to {zip_path.name} ({zip_path.stat().st_size / 1024:.1f} KB)\")\n",
            "\n",
            "# Attempt Google Colab direct download\n",
            "try:\n",
            "    from google.colab import files\n",
            "    files.download(str(zip_path))\n",
            "    print(\"Triggered Colab browser download.\")\n",
            "except Exception:\n",
            "    print(f\"Ready for manual download: {zip_path.resolve()}\")"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_nb_path = NOTEBOOKS_DIR / "kaggle_colab_t4_inference.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print(f"Successfully created notebook at: {out_nb_path}")
