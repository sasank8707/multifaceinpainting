"""
Final Gate Verification Suite: Identity Pipeline Integrity & Isolation Gate.

Checks ALL 8 requirements specified in the project mandate:
1. Real pretrained ArcFace loaded (not random).
2. Same-ID vs diff-ID ArcFace test passes with a clear positive gap.
3. Reference swap test passes (different references produce different T_fused).
4. Mask conditioning test passes (different masks produce different regional tokens).
5. Identity adapter retrained on real ArcFace features.
6. No target identity leakage (residual is NOT Q + T_weighted).
7. Coarse reconstructor is isolated from diffusion.
8. Diffusion latent channels 5:9 are assembled explicitly with no diffusers zero-masking.

Strict Mandate:
- Zero diffusion generations.
- Zero image generation.
- Purely deterministic verification and unit/integration testing.
- Prints a clean summary table showing PASS / FAIL for each check.
"""

import os
import sys
import inspect
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.conditioning_adapter import IdentityConditioningAdapter
from models.diffusion.diffusion_adapter import DiffusionInpaintingAdapter
from training.datasets.celeba_hq_dataset import CelebAHQIdentityDataset


class IdentityPipelineFinalGate:
    """Consolidated verification suite for all 8 identity pipeline gate checks."""

    def __init__(self):
        self.device = torch.device("cpu")

    def check_1_real_arcface_loaded(self) -> dict:
        """Check 1: Real pretrained ArcFace loaded (not random)."""
        encoder = RegionalIdentityEncoder(token_dim=512)
        onnx_exists = os.path.isfile(encoder.model_path)
        file_size_mb = os.path.getsize(encoder.model_path) / (1024 * 1024) if onnx_exists else 0.0

        is_pretrained = getattr(encoder, "is_pretrained", False)
        has_net = encoder.net is not None
        has_size = file_size_mb > 150.0  # Official buffalo_l w600k_r50 is ~166 MB

        dummy_img = torch.randn(1, 3, 112, 112)
        out = encoder(dummy_img)
        spatial_shape = out["spatial_tokens"].shape
        global_shape = out["global_embedding"].shape

        passed = is_pretrained and has_net and has_size and (spatial_shape == (1, 256, 512)) and (global_shape == (1, 512))
        return {
            "passed": passed,
            "details": f"Model: {Path(encoder.model_path).name} ({file_size_mb:.1f} MB) | is_pretrained={is_pretrained} | spatial={spatial_shape} | global={global_shape}"
        }

    def check_2_same_vs_diff_arcface_gap(self) -> dict:
        """Check 2: Same-ID vs diff-ID ArcFace test passes with a clear positive gap."""
        encoder = RegionalIdentityEncoder(token_dim=512)
        ds = CelebAHQIdentityDataset(split="test", smoke_subset_size=1)

        t_target = ds._load_image_tensor(7367).unsqueeze(0)   # ID 3111
        t_same = ds._load_image_tensor(13401).unsqueeze(0)    # ID 3111
        t_diff = ds._load_image_tensor(3616).unsqueeze(0)     # ID 3330

        with torch.no_grad():
            emb_target = encoder(t_target)["global_embedding"]
            emb_same = encoder(t_same)["global_embedding"]
            emb_diff = encoder(t_diff)["global_embedding"]

        sim_same = F.cosine_similarity(emb_target, emb_same, dim=-1).item()
        sim_diff = F.cosine_similarity(emb_target, emb_diff, dim=-1).item()
        gap = sim_same - sim_diff

        passed = (sim_same > 0.40) and (sim_diff < 0.20) and (gap > 0.40)
        return {
            "passed": passed,
            "details": f"Same-ID Sim: {sim_same:+.4f} | Diff-ID Sim: {sim_diff:+.4f} | Gap: {gap:+.4f} (Required: > 0.40)"
        }

    def check_3_reference_swap_discrimination(self) -> dict:
        """Check 3: Reference swap test passes (different references produce different T_fused)."""
        encoder = RegionalIdentityEncoder(token_dim=512)
        transformer = RegionalIdentityTransformer(
            token_dim=512,
            num_heads=8,
            num_layers=2,
            spatial_patch_grid=(16, 16),
            dropout=0.0
        )
        encoder.eval()
        transformer.eval()
        ds = CelebAHQIdentityDataset(split="test", smoke_subset_size=1)

        # Subject 3111 target and mask
        mask_256 = ds._generate_severity_mask(idx=0, severity=0.20)
        mask_t = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float()

        # Subject 3111 references (same identity)
        ref_a_ids = [13401, 23058, 23414]
        # Subject 3330 references (different identity)
        ref_b_ids = [5447, 6008, 7695]

        with torch.no_grad():
            refs_a = [encoder(ds._load_image_tensor(i).unsqueeze(0))["spatial_tokens"] for i in ref_a_ids]
            refs_b = [encoder(ds._load_image_tensor(i).unsqueeze(0))["spatial_tokens"] for i in ref_b_ids]

            weights = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32)

            out_a = transformer(
                target_tokens=None,
                target_mask=mask_t,
                reference_tokens_list=refs_a,
                reference_weights=weights
            )
            out_b = transformer(
                target_tokens=None,
                target_mask=mask_t,
                reference_tokens_list=refs_b,
                reference_weights=weights
            )

        fused_a = out_a["fused_tokens"] # [1, 256, 512]
        fused_b = out_b["fused_tokens"]

        cos_sim = F.cosine_similarity(fused_a.flatten(1), fused_b.flatten(1), dim=-1).item()
        diff_per_token = torch.norm(fused_a - fused_b, p=2, dim=-1).squeeze().numpy()
        mean_token_diff = float(np.mean(diff_per_token))

        # Gate requires cos_sim < 0.85 and delta_l2 > 5.0
        passed = (cos_sim < 0.85) and (mean_token_diff > 5.0)
        return {
            "passed": passed,
            "details": f"Ref Swap Cosine: {cos_sim:.4f} (Required: < 0.85) | Mean Token Delta L2: {mean_token_diff:.2f} (Required: > 5.0)"
        }

    def check_4_mask_conditioning(self) -> dict:
        """Check 4: Mask conditioning test passes (different masks produce different regional tokens)."""
        encoder = RegionalIdentityEncoder(token_dim=512)
        transformer = RegionalIdentityTransformer(
            token_dim=512,
            num_heads=8,
            num_layers=2,
            spatial_patch_grid=(16, 16),
            dropout=0.0
        )
        encoder.eval()
        transformer.eval()
        ds = CelebAHQIdentityDataset(split="test", smoke_subset_size=1)

        ref_ids = [13401, 23058, 23414]
        with torch.no_grad():
            refs_tokens = [encoder(ds._load_image_tensor(i).unsqueeze(0))["spatial_tokens"] for i in ref_ids]

        # Upper face mask vs Lower face mask
        mask1 = np.zeros((256, 256), dtype=np.float32)
        mask1[50:130, 60:196] = 1.0
        mask1_t = torch.from_numpy(mask1).unsqueeze(0).unsqueeze(0)

        mask2 = np.zeros((256, 256), dtype=np.float32)
        mask2[150:230, 70:186] = 1.0
        mask2_t = torch.from_numpy(mask2).unsqueeze(0).unsqueeze(0)

        weights = torch.tensor([[1.0/3, 1.0/3, 1.0/3]], dtype=torch.float32)

        with torch.no_grad():
            out_m1 = transformer(
                target_tokens=None,
                target_mask=mask1_t,
                reference_tokens_list=refs_tokens,
                reference_weights=weights
            )
            out_m2 = transformer(
                target_tokens=None,
                target_mask=mask2_t,
                reference_tokens_list=refs_tokens,
                reference_weights=weights
            )

        fused_m1 = out_m1["fused_tokens"]
        fused_m2 = out_m2["fused_tokens"]

        diff_per_token = torch.norm(fused_m1 - fused_m2, p=2, dim=-1).squeeze().numpy()
        mean_token_diff = float(np.mean(diff_per_token))
        sim_masks = F.cosine_similarity(fused_m1.flatten(1), fused_m2.flatten(1), dim=-1).item()

        passed = (sim_masks < 0.999) and (mean_token_diff > 0.10)
        return {
            "passed": passed,
            "details": f"Mask Delta L2: {mean_token_diff:.4f} (Required: > 0.10) | Cosine Sim: {sim_masks:.4f}"
        }

    def check_5_identity_adapter_retrained(self) -> dict:
        """Check 5: Identity adapter retrained on real ArcFace features."""
        ckpt_path = PROJECT_ROOT / "checkpoints" / "identity_adapter" / "identity_adapter_best.pt"
        if not ckpt_path.is_file():
            return {"passed": False, "details": "Checkpoint file not found"}

        ckpt = torch.load(ckpt_path, map_location="cpu")
        has_state = "model_state_dict" in ckpt
        val_loss = ckpt.get("val_loss", float("inf"))
        epoch = ckpt.get("epoch", 0)

        adapter = IdentityConditioningAdapter(
            in_dim=512,
            out_dim=768,
            num_tokens=256,
            out_tokens=77,
            num_id_queries=16,
            dropout=0.0
        )
        adapter.load_state_dict(ckpt["model_state_dict"])
        adapter.eval()

        dummy_arcface = torch.randn(1, 256, 512)
        dummy_arcface = F.normalize(dummy_arcface, p=2, dim=-1)

        with torch.no_grad():
            out = adapter(identity_tokens=dummy_arcface)

        proj_tokens = out["projected_tokens"]
        probe_arcface = out["probe_arcface"]

        has_correct_shape = (proj_tokens.shape == (1, 77, 768)) and (probe_arcface.shape == (1, 512))
        no_nans = not torch.isnan(proj_tokens).any() and not torch.isnan(probe_arcface).any()
        probe_norm = torch.norm(probe_arcface, dim=-1).item()
        valid_probe_norm = abs(probe_norm - 1.0) < 1e-3

        passed = has_state and (val_loss < 0.50) and has_correct_shape and no_nans and valid_probe_norm
        return {
            "passed": passed,
            "details": f"Epoch: {epoch} | Val Loss: {val_loss:.4f} | Output Shape: {proj_tokens.shape} | Probe Norm: {probe_norm:.4f}"
        }

    def check_6_no_target_identity_leakage(self) -> dict:
        """Check 6: No target identity leakage (residual is NOT Q + T_weighted)."""
        fusion_file = PROJECT_ROOT / "models" / "transformer" / "fusion_transformer.py"
        with open(fusion_file, "r") as f:
            src = f.read()

        has_target_leakage_line = "Q + T_weighted" in src
        has_pure_ref_residual = "T_weighted + T_ref_base" in src or "weighted_tokens * (0.5 + gate) + ref_base" in src

        transformer = RegionalIdentityTransformer(
            token_dim=512,
            num_heads=8,
            num_layers=2,
            spatial_patch_grid=(16, 16),
            dropout=0.0
        )
        transformer.eval()

        mask_t = torch.zeros(1, 1, 256, 256)
        weights = torch.tensor([[1.0]], dtype=torch.float32)
        zeros_ref = [torch.zeros(1, 256, 512)]
        rand_ref = [torch.randn(1, 256, 512)]

        with torch.no_grad():
            out_zero = transformer(
                target_tokens=None,
                target_mask=mask_t,
                reference_tokens_list=zeros_ref,
                reference_weights=weights
            )["fused_tokens"]
            out_rand = transformer(
                target_tokens=None,
                target_mask=mask_t,
                reference_tokens_list=rand_ref,
                reference_weights=weights
            )["fused_tokens"]

        diff_norm = torch.norm(out_zero - out_rand).item()

        passed = (not has_target_leakage_line) and has_pure_ref_residual and (diff_norm > 1.0)
        return {
            "passed": passed,
            "details": f"Target residual 'Q + T_weighted' present: {has_target_leakage_line} | Pure ref residual present: {has_pure_ref_residual} | Ref sensitivity norm: {diff_norm:.2f}"
        }

    def check_7_coarse_reconstructor_isolated(self) -> dict:
        """Check 7: Coarse reconstructor is isolated from diffusion."""
        diff_file = PROJECT_ROOT / "models" / "diffusion" / "diffusion_adapter.py"
        with open(diff_file, "r") as f:
            src = f.read()

        has_coarse_import = "CoarseReconstructor" in src
        has_coarse_ckpt_load = "coarse_reconstructor_best.pt" in src

        adapter = DiffusionInpaintingAdapter(backend="mock")
        has_coarse_attr = hasattr(adapter, "coarse_reconstructor") or hasattr(adapter, "coarse_model")

        passed = (not has_coarse_import) and (not has_coarse_ckpt_load) and (not has_coarse_attr)
        return {
            "passed": passed,
            "details": f"Imports CoarseReconstructor: {has_coarse_import} | Loads coarse checkpoint: {has_coarse_ckpt_load} | Has coarse attr: {has_coarse_attr}"
        }

    def check_8_channels_5_to_9_explicitly_assembled(self) -> dict:
        """Check 8: Diffusion latent channels 5:9 are assembled explicitly with no diffusers zero-masking."""
        diff_file = PROJECT_ROOT / "models" / "diffusion" / "diffusion_adapter.py"
        with open(diff_file, "r") as f:
            src = f.read()

        has_vae_encode = "_encode_vae_image" in src
        has_masked_latents_arg = "masked_image_latents=guided_latents" in src

        from diffusers import StableDiffusionInpaintPipeline
        pipe_src = inspect.getsource(StableDiffusionInpaintPipeline.__call__)
        bypasses_zero_mask = "if masked_image_latents is None:\n            masked_image = init_image * (mask_condition < 0.5)\n        else:\n            masked_image = masked_image_latents" in pipe_src

        passed = has_vae_encode and has_masked_latents_arg and bypasses_zero_mask
        return {
            "passed": passed,
            "details": f"Explicit VAE encode: {has_vae_encode} | Passes masked_image_latents: {has_masked_latents_arg} | Diffusers zero-masking bypassed: {bypasses_zero_mask}"
        }

    def run_all(self) -> dict:
        """Runs all 8 checks and compiles a markdown summary table."""
        checks = [
            ("1. Real Pretrained ArcFace Loaded", self.check_1_real_arcface_loaded),
            ("2. Same-ID vs Diff-ID ArcFace Gap", self.check_2_same_vs_diff_arcface_gap),
            ("3. Reference Swap Discrimination", self.check_3_reference_swap_discrimination),
            ("4. Mask Conditioning Sensitivity", self.check_4_mask_conditioning),
            ("5. Identity Adapter Retrained", self.check_5_identity_adapter_retrained),
            ("6. No Target Identity Leakage", self.check_6_no_target_identity_leakage),
            ("7. Coarse Reconstructor Isolated", self.check_7_coarse_reconstructor_isolated),
            ("8. Channels 5:9 Explicitly Assembled", self.check_8_channels_5_to_9_explicitly_assembled),
        ]

        all_passed = True
        summary_rows = []

        print("\n" + "=" * 90)
        print("RUNNING FINAL GATE VERIFICATION SUITE: IDENTITY PIPELINE INTEGRITY & ISOLATION GATE")
        print("=" * 90)

        for name, func in checks:
            res = func()
            status = "PASS" if res["passed"] else "FAIL"
            if not res["passed"]:
                all_passed = False
            summary_rows.append({
                "name": name,
                "status": status,
                "details": res["details"]
            })
            print(f"[{status}] {name}")
            print(f"       Details: {res['details']}")

        print("=" * 90)
        print(f"OVERALL GATE STATUS: {'ALL CHECKS PASSED (PROCEED READY)' if all_passed else 'GATE FAILED'}")
        print("=" * 90 + "\n")

        return {
            "all_passed": all_passed,
            "rows": summary_rows
        }


# Pytest hooks for automated CI
@pytest.fixture(scope="module")
def gate():
    return IdentityPipelineFinalGate()

def test_gate_check_1_real_arcface_loaded(gate):
    res = gate.check_1_real_arcface_loaded()
    assert res["passed"], res["details"]

def test_gate_check_2_same_vs_diff_arcface_gap(gate):
    res = gate.check_2_same_vs_diff_arcface_gap()
    assert res["passed"], res["details"]

def test_gate_check_3_reference_swap_discrimination(gate):
    res = gate.check_3_reference_swap_discrimination()
    assert res["passed"], res["details"]

def test_gate_check_4_mask_conditioning(gate):
    res = gate.check_4_mask_conditioning()
    assert res["passed"], res["details"]

def test_gate_check_5_identity_adapter_retrained(gate):
    res = gate.check_5_identity_adapter_retrained()
    assert res["passed"], res["details"]

def test_gate_check_6_no_target_identity_leakage(gate):
    res = gate.check_6_no_target_identity_leakage()
    assert res["passed"], res["details"]

def test_gate_check_7_coarse_reconstructor_isolated(gate):
    res = gate.check_7_coarse_reconstructor_isolated()
    assert res["passed"], res["details"]

def test_gate_check_8_channels_5_to_9_explicitly_assembled(gate):
    res = gate.check_8_channels_5_to_9_explicitly_assembled()
    assert res["passed"], res["details"]


if __name__ == "__main__":
    runner = IdentityPipelineFinalGate()
    res = runner.run_all()
    sys.exit(0 if res["all_passed"] else 1)
