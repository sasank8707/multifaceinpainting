"""
Verification of Pipeline Parity Fix on user_tests/test1/
Runs 5 steps with the exact PIL image & normalized mask fix.
Verifies that Cos vs Target GT reaches ~0.915+ and Cos vs Ref Mean reaches ~0.555+.
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.reference_scoring.scorer import MaskAwareReferenceScorer
from diffusers import StableDiffusionInpaintPipeline


def main():
    print("=" * 80)
    print("TESTING PIPELINE PARITY FIX (5 STEPS)")
    print("=" * 80)

    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)

    base_path = PROJECT_ROOT / "user_tests/test1"
    target_rgb = cv2.cvtColor(cv2.imread(str(base_path / "target.jpg")), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(str(base_path / "mask.png"), cv2.IMREAD_GRAYSCALE)
    mask = (mask_gray > 127).astype(np.float32)

    ref_rgbs = [
        cv2.cvtColor(cv2.imread(str(base_path / "ref1.jpg")), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread(str(base_path / "ref2.jpg")), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread(str(base_path / "ref3.jpg")), cv2.COLOR_BGR2RGB)
    ]

    # Face Analysis & O1
    face_pipeline = FaceAnalysisPipeline()
    t_feat = face_pipeline.analyze(target_rgb, mask)
    r_feats = [face_pipeline.analyze(r) for r in ref_rgbs]
    scorer = MaskAwareReferenceScorer(
        pose_weight=0.30, quality_weight=0.25, visibility_weight=0.30, overlap_weight=0.15,
        top_k=3, strict_identity=False
    )
    score_rep = scorer.score_references(
        target_analysis=t_feat, reference_analyses=r_feats,
        target_identity_id="user_test1", reference_identity_ids=["user_test1"]*3,
        reference_names=["ref1", "ref2", "ref3"]
    )
    norm_w = score_rep["components"][0]["normalized_weights"]
    ref_weights = [norm_w[k] for k in ["ref1", "ref2", "ref3"]]
    sum_w = sum(ref_weights)
    ref_weights = [w / sum_w for w in ref_weights]

    # O2 Fusion (Fixed: mask strictly 0.0 or 1.0)
    arcface = RegionalIdentityEncoder(token_dim=512).to(device)
    arcface.eval()
    phase4 = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(device)
    phase4.eval()

    def prep_256(img):
        r = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor_256 = torch.from_numpy(mask_256).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        toks = [arcface(prep_256(r))["spatial_tokens"] for r in ref_rgbs]
        w_t = torch.tensor([ref_weights], dtype=torch.float32, device=device)
        t_fused = phase4(
            target_tokens=None, target_mask=mask_tensor_256,
            reference_tokens_list=toks, reference_weights=w_t
        )["fused_tokens"]

    # Load SD
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0]

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path, torch_dtype=torch.float32, safety_checker=None, local_files_only=True
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=0.10)
    manager.install(zero_init_v=False)
    ckpt_file = PROJECT_ROOT / "checkpoints/baselines/step_150_best/step_150.pt"
    manager.load_checkpoint(str(ckpt_file))
    manager.set_scale(0.10)
    manager.set_identity_tokens(t_fused)

    # Letterbox and convert to PIL Image!
    padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
        image=target_rgb, mask=mask, target_size=(512, 512)
    )
    pil_img = Image.fromarray(padded_img)
    pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))
    prompt = "high quality photorealistic portrait of a young man, natural facial anatomy, natural skin texture"

    gen = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.time()
    diff_out = pipe(
        prompt=prompt,
        image=pil_img,
        mask_image=pil_mask,
        num_inference_steps=5,
        guidance_scale=7.5,
        generator=gen
    ).images[0]
    dt = time.time() - t0

    padded_cand = np.array(diff_out, dtype=np.uint8)
    final_reconstruction = AspectRatioPreserver.unletterbox(
        padded_candidate=padded_cand,
        original_image=target_rgb,
        original_mask=mask,
        transform_info=transform_info
    )

    passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(target_rgb, final_reconstruction, mask)
    print(f"Diffusion generation completed in {dt:.2f}s")
    print(f"Unmasked Invariance: Passed={passed_inv}, Max Delta={max_diff}, Violated Pixels={violated_pixels}")

    # ArcFace Metrics
    diff_arcface = DifferentiableArcFaceBackbone("models/pretrained/w600k_r50.onnx").to(device)
    diff_arcface.eval()
    def prep_112(img):
        resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    with torch.no_grad():
        e_recon = diff_arcface.extract_embedding(prep_112(final_reconstruction))
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb))
        e_refs = [diff_arcface.extract_embedding(prep_112(r)) for r in ref_rgbs]
        e_ref_mean = F.normalize(sum(e_refs) / 3.0, p=2, dim=-1)

        cos_target = float(F.cosine_similarity(e_recon, e_target).item())
        cos_ref_mean = float(F.cosine_similarity(e_recon, e_ref_mean).item())

    mask_bool = (mask > 0.5)
    mean_rgb = np.mean(final_reconstruction[mask_bool], axis=0)

    print(f"Fixed Output Cos vs Target GT:  {cos_target:.4f} (Goal: ~0.915)")
    print(f"Fixed Output Cos vs Ref Mean:   {cos_ref_mean:.4f} (Goal: ~0.555)")
    print(f"Fixed Masked RGB Mean:          R={mean_rgb[0]:.1f}, G={mean_rgb[1]:.1f}, B={mean_rgb[2]:.1f}")

    out_p = PROJECT_ROOT / "evaluation/visualizations/parity_audit/03_fixed_verified.png"
    Image.fromarray(final_reconstruction).save(out_p)
    print(f"Saved: {out_p}")


if __name__ == "__main__":
    main()
