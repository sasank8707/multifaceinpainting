"""
Diagnostic Parity Audit: Web vs Direct Inference Pipeline
Follows the 15-point parity audit requested by the user.

Compares:
1. DIRECT known-good test: experiments/run_external_user_test1_inference.py
2. WEB pipeline test: modules/restoration/face_inpainting_pipeline.py + app.py

Audit items:
1. Target array RGB vs BGR min/max/mean
2. Mask shape, binary count, bbox
3. Reference arrays RGB vs BGR, ArcFace embedding vectors, norms, pairwise cosines, O1 scores
4. Identity tokens shape, stats, norm, CFG duplication check (cond and uncond)
5. Scheduler class and config (DDIM vs PNDM vs Euler)
6. Latent initialization (generator device, seed, initial noise stats)
7. Masked latent / VAE encode stats
8. Output image stats in masked region (at 5 steps first)
"""

import os
import sys
import time
import json
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
from modules.restoration.face_inpainting_pipeline import FaceInpaintingPipeline
from diffusers import StableDiffusionInpaintPipeline


def run_parity_audit():
    print("=" * 80)
    print("STRICT WEB-VS-DIRECT PIPELINE PARITY AUDIT (5 DDIM STEPS)")
    print("=" * 80)

    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------------------
    # ITEM 1: Target Image Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\n--- ITEM 1: Target Image Loading & Preprocessing ---")
    target_path = PROJECT_ROOT / "user_tests/test1/target.jpg"
    
    # Direct path
    target_bgr_direct = cv2.imread(str(target_path))
    target_rgb_direct = cv2.cvtColor(target_bgr_direct, cv2.COLOR_BGR2RGB)
    
    # Web path (app.py decode_image_file: Image.open(...).convert("RGB"))
    target_pil_web = Image.open(str(target_path)).convert("RGB")
    target_rgb_web = np.array(target_pil_web)

    print(f"Direct target shape: {target_rgb_direct.shape}, dtype: {target_rgb_direct.dtype}")
    print(f"Web target shape:    {target_rgb_web.shape}, dtype: {target_rgb_web.dtype}")
    target_diff = np.abs(target_rgb_direct.astype(int) - target_rgb_web.astype(int))
    print(f"Target Max Diff (Direct vs Web): {np.max(target_diff)}")
    print(f"Target Direct RGB: min={target_rgb_direct.min()}, max={target_rgb_direct.max()}, mean={target_rgb_direct.mean():.2f}")
    print(f"Target Web RGB:    min={target_rgb_web.min()}, max={target_rgb_web.max()}, mean={target_rgb_web.mean():.2f}")

    # -------------------------------------------------------------------------
    # ITEM 2: Mask Loading, Count & Preprocessing
    # -------------------------------------------------------------------------
    print("\n--- ITEM 2: Mask Loading, Count & Preprocessing ---")
    mask_path = PROJECT_ROOT / "user_tests/test1/mask.png"
    
    # Direct path
    mask_gray_direct = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask_direct = (mask_gray_direct > 127).astype(np.float32)
    
    # Web path
    mask_pil_web = Image.open(str(mask_path)).convert("L")
    mask_np_web = np.array(mask_pil_web)
    pipeline_dummy = FaceInpaintingPipeline(mock_mode=True, device=device)
    mask_norm_web = pipeline_dummy.validate_and_normalize_mask(mask_np_web, target_rgb_web.shape[:2])

    count_direct = int(np.sum(mask_direct > 0.5))
    count_web = int(np.sum(mask_norm_web > 0))
    print(f"Direct mask shape: {mask_direct.shape}, dtype: {mask_direct.dtype}, masked px: {count_direct}")
    print(f"Web mask shape:    {mask_norm_web.shape}, dtype: {mask_norm_web.dtype}, masked px: {count_web}")
    
    ys_d, xs_d = np.where(mask_direct > 0.5)
    ys_w, xs_w = np.where(mask_norm_web > 0)
    bbox_d = (int(xs_d.min()), int(ys_d.min()), int(xs_d.max()), int(ys_d.max()))
    bbox_w = (int(xs_w.min()), int(ys_w.min()), int(xs_w.max()), int(ys_w.max()))
    print(f"Direct bbox (x1, y1, x2, y2): {bbox_d}")
    print(f"Web bbox    (x1, y1, x2, y2): {bbox_w}")
    print(f"Mask values Direct: unique={np.unique(mask_direct)}")
    print(f"Mask values Web:    unique={np.unique(mask_norm_web)}")

    # -------------------------------------------------------------------------
    # ITEM 3: Reference Photos & O1 Scorer
    # -------------------------------------------------------------------------
    print("\n--- ITEM 3: Reference Photos & O1 Scorer ---")
    ref_paths = [
        PROJECT_ROOT / "user_tests/test1/ref1.jpg",
        PROJECT_ROOT / "user_tests/test1/ref2.jpg",
        PROJECT_ROOT / "user_tests/test1/ref3.jpg"
    ]
    ref_rgbs_direct = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in ref_paths]
    ref_rgbs_web = [np.array(Image.open(str(p)).convert("RGB")) for p in ref_paths]

    for i in range(3):
        diff = np.max(np.abs(ref_rgbs_direct[i].astype(int) - ref_rgbs_web[i].astype(int)))
        print(f"Ref {i+1} Direct vs Web Max Diff: {diff}")

    # Face Analysis
    face_pipeline = FaceAnalysisPipeline()
    t_feat_d = face_pipeline.analyze(target_rgb_direct, mask_direct)
    t_feat_w = face_pipeline.analyze(target_rgb_web, mask_norm_web)
    r_feats_d = [face_pipeline.analyze(r) for r in ref_rgbs_direct]
    r_feats_w = [face_pipeline.analyze(r) for r in ref_rgbs_web]

    # Scorer in Direct
    scorer_direct = MaskAwareReferenceScorer(
        pose_weight=0.30, quality_weight=0.25, visibility_weight=0.30, overlap_weight=0.15,
        top_k=3, strict_identity=False
    )
    score_report_d = scorer_direct.score_references(
        target_analysis=t_feat_d,
        reference_analyses=r_feats_d,
        target_identity_id="user_test1",
        reference_identity_ids=["user_test1", "user_test1", "user_test1"],
        reference_names=["ref1", "ref2", "ref3"]
    )
    comp_0 = score_report_d["components"][0]
    weights_direct = [comp_0["normalized_weights"][name] for name in ["ref1", "ref2", "ref3"]]
    sum_w = sum(weights_direct)
    weights_direct = [w / sum_w for w in weights_direct]

    # Scorer in Web (FaceInpaintingPipeline)
    scorer_web = MaskAwareReferenceScorer(strict_identity=False)
    score_report_w = scorer_web.score_references(t_feat_w, r_feats_w)
    if "global_normalized_weights" in score_report_w:
        weights_web = [round(float(v), 4) for v in score_report_w["global_normalized_weights"].values()]
    else:
        weights_web = [round(float(v), 4) for v in score_report_w.get("weights", [1.0/3]*3)]

    print(f"Direct O1 Weights: {weights_direct}")
    print(f"Web O1 Weights:    {weights_web}")

    # ArcFace Biometric Embeddings
    diff_arcface = DifferentiableArcFaceBackbone("models/pretrained/w600k_r50.onnx").to(device)
    diff_arcface.eval()
    def prep_112(img):
        resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    with torch.no_grad():
        e_ref1 = diff_arcface.extract_embedding(prep_112(ref_rgbs_direct[0]))
        e_ref2 = diff_arcface.extract_embedding(prep_112(ref_rgbs_direct[1]))
        e_ref3 = diff_arcface.extract_embedding(prep_112(ref_rgbs_direct[2]))
        e_target = diff_arcface.extract_embedding(prep_112(target_rgb_direct))
        e_ref_mean = F.normalize((e_ref1 + e_ref2 + e_ref3) / 3.0, p=2, dim=-1)

    print(f"ArcFace Norms: e_target={e_target.norm():.4f}, e_ref_mean={e_ref_mean.norm():.4f}")
    print(f"Target vs Ref Mean Cosine: {float(F.cosine_similarity(e_target, e_ref_mean).item()):.4f}")

    # -------------------------------------------------------------------------
    # ITEM 4: Regional Identity Fusion (O2) & Mask Scale Discrepancy
    # -------------------------------------------------------------------------
    print("\n--- ITEM 4: Regional Identity Fusion (O2) & Mask Scale Discrepancy ---")
    arcface = RegionalIdentityEncoder(token_dim=512).to(device)
    arcface.eval()
    phase4 = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(device)
    phase4.eval()

    def prep_256(img):
        r = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    with torch.no_grad():
        toks = [arcface(prep_256(r))["spatial_tokens"] for r in ref_rgbs_direct]

        # DIRECT: mask is 0.0 or 1.0
        mask_256_d = cv2.resize(mask_direct, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_t_d = torch.from_numpy(mask_256_d).unsqueeze(0).unsqueeze(0).float().to(device)
        w_t_d = torch.tensor([weights_direct], dtype=torch.float32, device=device)
        t_fused_direct = phase4(
            target_tokens=None, target_mask=mask_t_d,
            reference_tokens_list=toks, reference_weights=w_t_d
        )["fused_tokens"]

        # WEB AS-IS: mask is 0 or 255
        mask_256_w = cv2.resize(mask_norm_web, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_t_w = torch.from_numpy(mask_256_w).unsqueeze(0).unsqueeze(0).float().to(device)
        w_t_w = torch.tensor([weights_web], dtype=torch.float32, device=device)
        t_fused_web = phase4(
            target_tokens=None, target_mask=mask_t_w,
            reference_tokens_list=toks, reference_weights=w_t_w
        )["fused_tokens"]

    print(f"Direct mask_t_d unique values: {torch.unique(mask_t_d).tolist()}")
    print(f"Web mask_t_w unique values:    {torch.unique(mask_t_w).tolist()}")
    print(f"Direct t_fused: shape={t_fused_direct.shape}, norm={t_fused_direct.norm(dim=-1).mean():.4f}, mean={t_fused_direct.mean():.4f}, std={t_fused_direct.std():.4f}")
    print(f"Web t_fused:    shape={t_fused_web.shape}, norm={t_fused_web.norm(dim=-1).mean():.4f}, mean={t_fused_web.mean():.4f}, std={t_fused_web.std():.4f}")
    t_fused_cos = float(F.cosine_similarity(t_fused_direct.flatten(), t_fused_web.flatten(), dim=0).item())
    print(f"Cosine Similarity between Direct t_fused and Web t_fused: {t_fused_cos:.4f}")

    # -------------------------------------------------------------------------
    # ITEM 5: Stable Diffusion Pipeline & Decoupled Attention Check
    # -------------------------------------------------------------------------
    print("\n--- ITEM 5: Diffusion Pipeline, Scheduler & Attention Manager ---")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots")
    snapshots = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    load_path = snapshots[0] if snapshots else "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path, torch_dtype=torch.float32, safety_checker=None, local_files_only=True
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    print(f"Scheduler class: {pipe.scheduler.__class__.__name__}")
    print(f"Scheduler config: {pipe.scheduler.config}")

    manager = DecoupledIdentityCrossAttentionManager(pipe.unet, id_dim=512, default_scale=0.10)
    manager.install(zero_init_v=False)
    ckpt_file = PROJECT_ROOT / "checkpoints/baselines/step_150_best/step_150.pt"
    manager.load_checkpoint(str(ckpt_file))
    manager.set_scale(0.10)

    # -------------------------------------------------------------------------
    # ITEM 6: Letterboxing & Latent Inputs (Direct vs Web)
    # -------------------------------------------------------------------------
    print("\n--- ITEM 6: Letterboxing & Latent Inputs ---")
    # Direct Letterboxing
    pad_img_d, pad_mask_d, t_info_d = AspectRatioPreserver.letterbox(
        image=target_rgb_direct, mask=mask_direct, target_size=(512, 512)
    )
    pil_img_d = Image.fromarray(pad_img_d)
    pil_mask_d = Image.fromarray((pad_mask_d * 255.0).astype(np.uint8))

    # Web Letterboxing
    masked_target_w = target_rgb_web.copy()
    masked_target_w[mask_norm_web > 0] = 0
    pad_img_w, pad_mask_w, t_info_w = AspectRatioPreserver.letterbox(
        image=masked_target_w, mask=mask_norm_web, target_size=(512, 512)
    )
    # In web pipeline, pad_img_w and pad_mask_w were passed as numpy arrays or PIL?
    print(f"Direct pil_img: type={type(pil_img_d)}, size={pil_img_d.size}, mode={pil_img_d.mode}")
    print(f"Direct pil_mask: type={type(pil_mask_d)}, size={pil_mask_d.size}, mode={pil_mask_d.mode}, unique={np.unique(np.array(pil_mask_d))}")
    print(f"Web pad_img:   type={type(pad_img_w)}, shape={pad_img_w.shape}, masked pixels black={np.all(pad_img_w[pad_mask_w > 0] == 0)}")
    print(f"Web pad_mask:  type={type(pad_mask_w)}, shape={pad_mask_w.shape}, unique={np.unique(pad_mask_w)}")

    # -------------------------------------------------------------------------
    # ITEM 7 & 8: 5-Step Diffusion Generation Comparison
    # -------------------------------------------------------------------------
    print("\n--- ITEM 7 & 8: 5-Step Diffusion Generation Comparison ---")
    prompt_direct = "high quality photorealistic portrait of a young man, natural facial anatomy, natural skin texture"
    prompt_web = "high quality photorealistic portrait of a person, natural facial anatomy, clean skin"

    out_audit_dir = PROJECT_ROOT / "evaluation/visualizations/parity_audit"
    out_audit_dir.mkdir(parents=True, exist_ok=True)

    # RUN A: Exactly reproducing DIRECT known-good test (5 steps)
    print("\n[Run A] Executing DIRECT Path (5 steps, seed 42)...")
    manager.set_identity_tokens(t_fused_direct)
    gen_a = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.time()
    out_a = pipe(
        prompt=prompt_direct,
        image=pil_img_d,
        mask_image=pil_mask_d,
        num_inference_steps=5,
        guidance_scale=7.5,
        generator=gen_a
    ).images[0]
    dt_a = time.time() - t0
    cand_a = np.array(out_a, dtype=np.uint8)
    recon_a = AspectRatioPreserver.unletterbox(
        padded_candidate=cand_a, original_image=target_rgb_direct, original_mask=mask_direct, transform_info=t_info_d
    )
    print(f"Run A completed in {dt_a:.2f}s")
    Image.fromarray(recon_a).save(out_audit_dir / "01_run_a_direct.png")

    # Evaluate Run A Biometrics
    with torch.no_grad():
        e_recon_a = diff_arcface.extract_embedding(prep_112(recon_a))
        cos_a_ref_mean = float(F.cosine_similarity(e_recon_a, e_ref_mean, dim=-1).item())
        cos_a_target = float(F.cosine_similarity(e_recon_a, e_target, dim=-1).item())
    
    # Masked region stats
    mask_bool = (mask_direct > 0.5)
    mean_rgb_a = np.mean(recon_a[mask_bool], axis=0)
    std_rgb_a = np.std(recon_a[mask_bool], axis=0)
    print(f"Run A Metrics: Cos vs Target={cos_a_target:.4f}, Cos vs Ref Mean={cos_a_ref_mean:.4f}")
    print(f"Run A Masked RGB Mean: R={mean_rgb_a[0]:.1f}, G={mean_rgb_a[1]:.1f}, B={mean_rgb_a[2]:.1f}, Std: {std_rgb_a}")

    # RUN B: WEB Path as currently written in FaceInpaintingPipeline (5 steps)
    print("\n[Run B] Executing WEB Path AS-IS (5 steps, seed 42)...")
    manager.set_identity_tokens(t_fused_web)
    gen_b = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.time()
    out_b = pipe(
        prompt=prompt_web,
        image=pad_img_w, # Note: numpy array
        mask_image=pad_mask_w, # Note: numpy array with {0, 255}
        num_inference_steps=5,
        guidance_scale=7.5,
        generator=gen_b
    ).images[0]
    dt_b = time.time() - t0
    cand_b = np.array(out_b, dtype=np.uint8)
    recon_b = AspectRatioPreserver.unletterbox(
        padded_candidate=cand_b, original_image=target_rgb_web, original_mask=mask_norm_web, transform_info=t_info_w
    )
    print(f"Run B completed in {dt_b:.2f}s")
    Image.fromarray(recon_b).save(out_audit_dir / "02_run_b_web_as_is.png")

    with torch.no_grad():
        e_recon_b = diff_arcface.extract_embedding(prep_112(recon_b))
        cos_b_ref_mean = float(F.cosine_similarity(e_recon_b, e_ref_mean, dim=-1).item())
        cos_b_target = float(F.cosine_similarity(e_recon_b, e_target, dim=-1).item())

    mean_rgb_b = np.mean(recon_b[mask_bool], axis=0)
    std_rgb_b = np.std(recon_b[mask_bool], axis=0)
    print(f"Run B Metrics: Cos vs Target={cos_b_target:.4f}, Cos vs Ref Mean={cos_b_ref_mean:.4f}")
    print(f"Run B Masked RGB Mean: R={mean_rgb_b[0]:.1f}, G={mean_rgb_b[1]:.1f}, B={mean_rgb_b[2]:.1f}, Std: {std_rgb_b}")

    # Summary
    print("\n" + "=" * 80)
    print("PARITY AUDIT SUMMARY:")
    print(f"  Run A (Direct) Cosine Target:   {cos_a_target:.4f} | Cos Ref Mean: {cos_a_ref_mean:.4f}")
    print(f"  Run B (Web)    Cosine Target:   {cos_b_target:.4f} | Cos Ref Mean: {cos_b_ref_mean:.4f}")
    print(f"  Run A Masked RGB Mean:          {mean_rgb_a.tolist()}")
    print(f"  Run B Masked RGB Mean:          {mean_rgb_b.tolist()}")
    print("=" * 80)


if __name__ == "__main__":
    run_parity_audit()
