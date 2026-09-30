"""
Generates notebooks/cloud_training.ipynb containing the complete 10-step
workflow for Cloud-GPU training on Google Colab and Kaggle.
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
            "# Multi-Reference Facial Inpainting: Cloud-GPU Training Workflow\n",
            "### Resumable Decoupled Identity Cross-Attention Training on NVIDIA GPUs (T4 / L4 / A100)\n",
            "\n",
            "This notebook implements the complete, production-grade cloud training pipeline for the Multi-Reference Facial Inpainting project.\n",
            "\n",
            "**Core Principles & Guarantees:**\n",
            "- **Immutable Baseline Protection**: Checkpoint `step_150.pt` (SHA-256: `e4ff5cff...`) is protected at `checkpoints/baselines/step_150_best/` and is never overwritten.\n",
            "- **Resumed Training**: Training continues from `step_150.pt`; the first new optimization step is **step 151**.\n",
            "- **Frozen Backbone / Trainable K-V**: Exactly 12,779,520 cross-attention projection parameters (`to_k_id`, `to_v_id`) are trained. The UNet backbone, VAE, CLIP, Differentiable ArcFace, and Regional Transformer remain 100% frozen.\n",
            "- **Composite Loss**: Joint objective $\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{diffusion}} + 0.10 \\cdot \\mathcal{L}_{\\text{identity}}$.\n",
            "- **Training Milestones**: Checkpoints saved at 500, 1000, 2500, 5000, and 10000 steps to evaluate the empirical effect of extended training."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. GPU Verification & Diagnostic\n",
            "Verify that a CUDA GPU (e.g. NVIDIA T4, L4, or A100) is allocated and check available VRAM."
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
            "    device_name = torch.cuda.get_device_name(0)\n",
            "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
            "    print(f\"GPU Device:      {device_name}\")\n",
            "    print(f\"Total VRAM:      {vram_gb:.2f} GB\")\n",
            "    print(f\"CUDA Version:    {torch.version.cuda}\")\n",
            "else:\n",
            "    raise RuntimeError(\"CUDA GPU not detected! In Colab, go to Runtime -> Change runtime type -> T4 GPU.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dependency Installation\n",
            "Install required packages for Stable Diffusion inpainting, ArcFace, MediaPipe, and metrics."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q diffusers transformers accelerate mediapipe opencv-python scikit-image onnxruntime safetensors psutil tqdm matplotlib"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Project Setup & Workspace Path Resolution\n",
            "Sets up Python path and locates the project directory whether on Google Colab (`/content/multifaceinpainting`) or Kaggle."
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
            "possible_roots = [\n",
            "    Path.cwd(),\n",
            "    Path(\"/content/multifaceinpainting\"),\n",
            "    Path(\"/kaggle/working/multifaceinpainting\"),\n",
            "    Path.cwd().parent\n",
            "]\n",
            "\n",
            "repo_root = None\n",
            "for p in possible_roots:\n",
            "    if (p / \"checkpoints\" / \"baselines\" / \"step_150_best\" / \"step_150.pt\").exists():\n",
            "        repo_root = p.resolve()\n",
            "        break\n",
            "\n",
            "if repo_root is None:\n",
            "    repo_root = Path.cwd().resolve()\n",
            "\n",
            "if str(repo_root) not in sys.path:\n",
            "    sys.path.insert(0, str(repo_root))\n",
            "\n",
            "os.chdir(repo_root)\n",
            "print(f\"Workspace Root: {repo_root}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Protected Baseline Checkpoint Verification\n",
            "Verify that the protected `step_150.pt` checkpoint exists, matches the exact SHA-256 checksum (`e4ff5cff...`), and loads without corruption."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python scripts/verify_best_checkpoint.py"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Dataset Verification\n",
            "Verify that `identity_index.json` is loaded and locate CelebA-HQ images or `CelebAMask-HQ.zip`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "index_path = Path(\"data/metadata/identity_index.json\")\n",
            "assert index_path.exists(), f\"Missing {index_path}!\"\n",
            "\n",
            "with open(index_path, \"r\") as f:\n",
            "    meta = json.load(f)\n",
            "\n",
            "train_splits = meta.get(\"splits\", {}).get(\"train\", [])\n",
            "val_splits = meta.get(\"splits\", {}).get(\"val\", [])\n",
            "print(f\"Verified Dataset Index: {len(train_splits)} train identities, {len(val_splits)} val identities.\")\n",
            "print(f\"Total Images Mapped:   {len(meta.get('hq_to_id', {}))}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. CUDA Smoke Test (3 Steps: 151, 152, 153)\n",
            "Runs a brief 3-step test on CUDA to verify:\n",
            "1. Baseline weights load properly\n",
            "2. Exactly 12,779,520 trainable parameters active (rest frozen)\n",
            "3. Forward pass & composite loss calculation\n",
            "4. Backward pass & finite gradient verification\n",
            "5. Optimizer & AMP GradScaler step\n",
            "6. Checkpoint saving and reload integrity"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python scripts/train_identity_cloud.py \\\n",
            "  --resume checkpoints/baselines/step_150_best/step_150.pt \\\n",
            "  --device cuda \\\n",
            "  --mixed_precision fp16 \\\n",
            "  --smoke_test \\\n",
            "  --smoke_steps 3"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Resumable Cloud Training Execution\n",
            "Resume training from step 150 up to milestone target (e.g. step 500 or 1000).\n",
            "All new checkpoints are saved to `checkpoints/cloud_training/`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train up to Step 500 (milestone 1)\n",
            "!python scripts/train_identity_cloud.py \\\n",
            "  --resume checkpoints/baselines/step_150_best/step_150.pt \\\n",
            "  --max_steps 500 \\\n",
            "  --save_every 100 \\\n",
            "  --device cuda \\\n",
            "  --mixed_precision fp16 \\\n",
            "  --output_dir checkpoints/cloud_training \\\n",
            "  --learning_rate 1e-4 \\\n",
            "  --lambda_id 0.10"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Checkpoint Listing & Inspection\n",
            "List all checkpoints generated in `checkpoints/cloud_training/`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "ckpt_dir = Path(\"checkpoints/cloud_training\")\n",
            "ckpts = sorted(list(ckpt_dir.glob(\"*.pt\")))\n",
            "\n",
            "print(f\"Checkpoints in {ckpt_dir} ({len(ckpts)} files):\")\n",
            "for c in ckpts:\n",
            "    sz_mb = c.stat().st_size / (1024*1024)\n",
            "    print(f\"  - {c.name:<18} ({sz_mb:.2f} MB)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Checkpoint Evaluation Benchmark\n",
            "Evaluates each checkpoint on the canonical benchmark against the protected `step_150.pt` baseline:\n",
            "- ArcFace Biometric Similarity & Biometric Identity Gap\n",
            "- Perceptual & Reconstruction Metrics: PSNR, SSIM, MAE\n",
            "- Bit-Exact Unmasked Invariance (must pass 100%)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare baseline step_150 against new checkpoints\n",
            "eval_ckpts = [\"checkpoints/baselines/step_150_best/step_150.pt\"]\n",
            "if (ckpt_dir / \"step_500.pt\").exists():\n",
            "    eval_ckpts.append(str(ckpt_dir / \"step_500.pt\"))\n",
            "if (ckpt_dir / \"latest.pt\").exists():\n",
            "    eval_ckpts.append(str(ckpt_dir / \"latest.pt\"))\n",
            "\n",
            "ckpt_args = \" \".join(eval_ckpts)\n",
            "!python scripts/evaluate_checkpoints.py --checkpoints {ckpt_args} --device cuda"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Checkpoint Export & Download\n",
            "Bundle trained checkpoints into a zip archive and download to local machine."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "zip_out = Path(\"cloud_trained_checkpoints.zip\")\n",
            "shutil.make_archive(\"cloud_trained_checkpoints\", \"zip\", \"checkpoints/cloud_training\")\n",
            "print(f\"Created archive: {zip_out.name} ({zip_out.stat().st_size / (1024*1024):.2f} MB)\")\n",
            "\n",
            "try:\n",
            "    from google.colab import files\n",
            "    files.download(str(zip_out))\n",
            "    print(\"Triggered automatic browser download.\")\n",
            "except Exception:\n",
            "    print(f\"File ready for download at: {zip_out.resolve()}\")"
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

out_path = NOTEBOOKS_DIR / "cloud_training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print(f"Successfully generated {out_path}")
