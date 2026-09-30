"""
Unified Face Inpainting / Face Restoration Pipeline.
Supports Dual Input Modes:
- Path A: User-Supplied Mask (Inpainting)
- Path B: Automatic Damage Detection Mask (Restoration)

Routes both paths into the exact same verified research inpainting backend:
- O1 Mask-Aware Reference Scoring
- O2 Regional Identity Fusion (RegionalIdentityTransformer)
- Decoupled Identity Cross-Attention UNet (step_150.pt)
- AspectRatioPreserver Unletterbox with Strict Bit-Exact Canvas Invariance (Delta = 0.0)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.identity.differentiable_arcface import DifferentiableArcFaceBackbone
from models.identity.arcface_backbone import RegionalIdentityEncoder
from models.transformer.fusion_transformer import RegionalIdentityTransformer
from models.diffusion.decoupled_cross_attention import DecoupledIdentityCrossAttentionManager
from models.diffusion.diffusion_adapter import AspectRatioPreserver
from modules.utils.geometry import verify_unmasked_invariance
from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.reference_scoring.scorer import MaskAwareReferenceScorer
from modules.restoration.damage_detector import BaseDamageDetector, DamageMaskDetector, DamageDetectionResult

logger = logging.getLogger("FaceInpaintingPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FaceInpaintingPipeline:
    """
    Operational Face Inpainting & Face Restoration Pipeline.
    Unifies Path A (Supplied Mask) and Path B (Auto Damage Mask) while preserving
    the exact validated Step 150 research conditioning backend.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/baselines/step_150_best/step_150.pt",
        device: Optional[torch.device] = None,
        identity_scale: float = 0.10,
        guidance_scale: float = 7.5,
        default_steps: int = 50,
        damage_detector: Optional[BaseDamageDetector] = None,
        mock_mode: bool = False
    ):
        """
        Args:
            checkpoint_path: Relative or absolute path to adapter weights.
            device: Compute device (CPU or CUDA).
            identity_scale: Cross-attention steering weight lambda (default 0.10).
            guidance_scale: Diffusion classifier-free guidance scale.
            default_steps: Default inference denoising steps.
            damage_detector: Optional custom damage mask detector instance.
            mock_mode: If True, uses fast synthetic inpainting for sub-second testing.
        """
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.identity_scale = identity_scale
        self.guidance_scale = guidance_scale
        self.default_steps = default_steps
        self.mock_mode = mock_mode

        ckpt_file = (PROJECT_ROOT / checkpoint_path) if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
        assert ckpt_file.exists(), f"Checkpoint file not found: {ckpt_file}"
        self.checkpoint_path = ckpt_file

        # Initialize Analysis and Feature Modules
        self.face_pipeline = FaceAnalysisPipeline()
        self.reference_scorer = MaskAwareReferenceScorer(strict_identity=False)
        self.damage_detector = damage_detector or DamageMaskDetector()

        # Phase 4 Identity Encoder & Regional Transformer
        self.arcface_encoder = RegionalIdentityEncoder(token_dim=512).to(self.device)
        self.arcface_encoder.eval()
        self.fusion_transformer = RegionalIdentityTransformer(token_dim=512, num_heads=8, num_layers=2).to(self.device)
        self.fusion_transformer.eval()

        # Real Diffusion Pipeline Setup (lazy-loaded or mock)
        self.pipe = None
        self.manager = None
        if not self.mock_mode:
            self._init_diffusion_backend()
        else:
            logger.info("FaceInpaintingPipeline initialized in MOCK mode (fast testing).")

    def _init_diffusion_backend(self):
        """Initializes real Stable Diffusion Inpainting pipeline and Decoupled Cross-Attention."""
        from diffusers import StableDiffusionInpaintPipeline

        def find_sd_model():
            cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting/snapshots"))
            if cache_base.exists():
                dirs = [d for d in cache_base.iterdir() if d.is_dir()]
                if dirs:
                    return str(dirs[0])
            return "runwayml/stable-diffusion-inpainting"

        sd_path = find_sd_model()
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        logger.info(f"Loading Stable Diffusion Inpainting from {sd_path} on {self.device}...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_path,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.manager = DecoupledIdentityCrossAttentionManager(
            self.pipe.unet,
            id_dim=512,
            default_scale=self.identity_scale
        )
        self.manager.install(zero_init_v=False)
        self.manager.load_checkpoint(str(self.checkpoint_path))
        self.manager.to(device=self.device, dtype=self.pipe.unet.dtype)
        logger.info(f"Loaded Step 150 adapter weights into Decoupled Attention Manager from {self.checkpoint_path.name}")

    def validate_and_normalize_mask(
        self,
        mask: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Validates and normalizes any user-supplied or auto-generated mask.

        Requirements:
        1. Dimensions must match target image (H, W).
        2. Must be binary uint8 {0, 255}.
        3. Must not be empty (total masked pixels > 0).
        4. Must not cover 100% of image (unless confirmed).
        """
        if mask is None:
            raise ValueError("Mask cannot be None for validation.")

        # Convert to 2D single channel
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        target_h, target_w = target_shape[:2]
        if mask.shape[:2] != (target_h, target_w):
            raise ValueError(
                f"Mask shape {mask.shape[:2]} does not match target image dimensions {(target_h, target_w)}."
            )

        # Normalize to binary {0, 255}
        if mask.dtype == bool:
            binary = mask.astype(np.uint8) * 255
        elif np.issubdtype(mask.dtype, np.floating):
            binary = (mask > 0.5).astype(np.uint8) * 255
        else:
            binary = (mask > 127).astype(np.uint8) * 255

        num_masked = int(np.sum(binary > 0))
        total_pixels = target_h * target_w

        if num_masked == 0:
            raise ValueError("Supplied mask is completely empty (0 pixels masked). Inpainting requires a non-empty mask.")

        if num_masked >= (total_pixels * 0.98):
            raise ValueError(
                f"Supplied mask covers {num_masked}/{total_pixels} pixels (>{98}% of total canvas). "
                "Facial inpainting requires unmasked context for canvas preservation."
            )

        return binary

    def restore(
        self,
        target_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        reference_images: Optional[List[np.ndarray]] = None,
        seed: int = 42,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        identity_scale: Optional[float] = None,
        detector: Optional[BaseDamageDetector] = None,
        save_debug_vis: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        prompt: str = "high quality photorealistic portrait of a person, natural facial anatomy, clean skin",
        step_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Unified Entry Point for Face Inpainting and Face Restoration.

        Args:
            target_image: [H, W, 3] uint8 RGB image.
            mask: Optional [H, W] binary mask. If None, triggers Path B (Auto-Damage Detection).
            reference_images: Optional list of 1-3 [H, W, 3] reference portraits of same subject.
            seed: Random seed for diffusion reproducibility.
            steps: Number of denoising steps (default self.default_steps).
            guidance_scale: Classifier-free guidance scale.
            identity_scale: Decoupled attention weight lambda.
            detector: Optional custom damage detector override.
            save_debug_vis: Whether to render and save 5-panel debug visualization.
            output_dir: Output directory to save results and visualizations.
            prompt: Text prompt guiding diffusion context.

        Returns:
            Dictionary containing final restored image, normalized mask, execution metrics, and logs.
        """
        t_start = time.time()
        assert target_image is not None and target_image.ndim == 3, "target_image must be a 3-channel RGB image."
        h, w = target_image.shape[:2]
        total_image_pixels = h * w
        inference_steps = steps or self.default_steps
        cfg_scale = guidance_scale or self.guidance_scale
        lambda_val = identity_scale or self.identity_scale

        refs = reference_images if reference_images else []
        active_detector = detector or self.damage_detector

        # -------------------------------------------------------------
        # STEP 1: Route Input Path (Path A vs Path B)
        # -------------------------------------------------------------
        face_info = None
        detection_result = None

        if mask is not None:
            input_mode = "MASK_PROVIDED"
            logger.info("Routing to PATH A: User-supplied mask provided.")
            normalized_mask = self.validate_and_normalize_mask(mask, target_image.shape[:2])
            confidence = 1.0
            bbox_damage = None
        else:
            input_mode = "AUTO_MASK"
            logger.info("Routing to PATH B: No mask provided. Triggering automatic damage detection...")
            # Detect face first to pass shared face_info to detector
            try:
                face_info = self.face_pipeline.analyze(target_image)
            except Exception as e:
                logger.warning(f"Face analysis warning: {e}")

            detection_result = active_detector.detect(
                image=target_image,
                reference_images=refs,
                face_info=face_info.get("face_detection") if face_info else None
            )

            if detection_result.num_pixels == 0:
                raise ValueError(
                    "Automatic damage detector found zero damaged pixels on the face. "
                    "The face may be unoccluded, or damage contrast is below detection threshold. "
                    "Please provide a manual mask or adjust sensitivity."
                )

            normalized_mask = self.validate_and_normalize_mask(detection_result.mask, target_image.shape[:2])
            confidence = detection_result.confidence
            bbox_damage = detection_result.bbox
            logger.info(
                f"Auto-Damage detected: {detection_result.num_pixels:,} pixels "
                f"({detection_result.face_coverage_pct:.1f}% face, {detection_result.image_coverage_pct:.1f}% image), "
                f"Confidence: {confidence:.2f}"
            )

        num_masked_pixels = int(np.sum(normalized_mask > 0))
        img_coverage_pct = round(float(num_masked_pixels / total_image_pixels) * 100.0, 2)

        # -------------------------------------------------------------
        # STEP 2: Multi-Reference Identity Analysis & Scoring (O1)
        # -------------------------------------------------------------
        ref_weights = []
        if refs:
            logger.info(f"Processing {len(refs)} reference image(s) via MaskAwareReferenceScorer (O1)...")
            try:
                target_features = self.face_pipeline.analyze(target_image, normalized_mask)
                ref_features_list = [self.face_pipeline.analyze(r) for r in refs]
                scoring_result = self.reference_scorer.score_references(target_features, ref_features_list)
                if "global_normalized_weights" in scoring_result:
                    ref_weights = [round(float(w), 4) for w in scoring_result["global_normalized_weights"].values()]
                else:
                    ref_weights = [round(float(w), 4) for w in scoring_result.get("weights", [1.0 / len(refs)] * len(refs))]
            except Exception as e:
                logger.warning(f"Reference scoring fallback to equal weights: {e}")
                ref_weights = [round(1.0 / len(refs), 4)] * len(refs)
        else:
            logger.info("No references provided. Operating in unconditioned text/context mode.")

        # -------------------------------------------------------------
        # STEP 3: Multi-Reference Regional Token Fusion (O2)
        # -------------------------------------------------------------
        t_fused = None
        if refs and not self.mock_mode:
            def prep_256(img):
                r = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                return torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1.0

            mask_256 = cv2.resize(normalized_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_256_float = (mask_256 > 127).astype(np.float32)
            mask_tensor_256 = torch.from_numpy(mask_256_float).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                ref_tokens = []
                for r_img in refs:
                    t_tok = self.arcface_encoder(prep_256(r_img))["spatial_tokens"].to(self.device)
                    ref_tokens.append(t_tok)

                w_tensor = torch.tensor([ref_weights], dtype=torch.float32, device=self.device)
                t_fused = self.fusion_transformer(
                    target_tokens=None,
                    target_mask=mask_tensor_256,
                    reference_tokens_list=ref_tokens,
                    reference_weights=w_tensor
                )["fused_tokens"]

        # -------------------------------------------------------------
        # STEP 4: Inpainting Execution & Strict Canvas Preservation
        # -------------------------------------------------------------
        mask_binary = (normalized_mask > 127).astype(np.float32)

        padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
            image=target_image,
            mask=mask_binary,
            target_size=(512, 512)
        )
        pil_img = Image.fromarray(padded_img)
        pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

        t_diff_start = time.time()
        if not self.mock_mode:
            assert self.manager is not None and self.pipe is not None
            self.manager.set_scale(lambda_val)
            if t_fused is not None:
                self.manager.set_identity_tokens(t_fused.to(device=self.device, dtype=self.pipe.unet.dtype))

            gen = (
                torch.Generator(device="cpu").manual_seed(seed)
                if self.device.type == "cpu"
                else torch.Generator(device=self.device).manual_seed(seed)
            )

            def on_step_end(pipeline_obj, step_idx, timestep, callback_kwargs):
                if step_callback:
                    step_callback(step_idx + 1, inference_steps)
                return callback_kwargs

            diff_out = self.pipe(
                prompt=prompt,
                image=pil_img,
                mask_image=pil_mask,
                num_inference_steps=inference_steps,
                guidance_scale=cfg_scale,
                generator=gen,
                callback_on_step_end=on_step_end if step_callback else None
            ).images[0]

            padded_cand = np.array(diff_out, dtype=np.uint8)
            final_reconstructed = AspectRatioPreserver.unletterbox(
                padded_candidate=padded_cand,
                original_image=target_image,
                original_mask=normalized_mask,
                transform_info=transform_info
            )
        else:
            if step_callback:
                step_callback(inference_steps, inference_steps)
            # Fast deterministic mock synthesis for sub-second testing
            mock_canvas = target_image.copy()
            mask_bool = (normalized_mask > 0)
            rng = np.random.RandomState(seed)
            if refs:
                base_ref = cv2.resize(refs[0], (w, h), interpolation=cv2.INTER_AREA)
                noise = rng.normal(0, 4, mock_canvas.shape).astype(np.float32)
                blended = np.clip(0.85 * base_ref.astype(np.float32) + 0.15 * mock_canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                mock_canvas[mask_bool] = blended[mask_bool]
            else:
                blurred = cv2.GaussianBlur(target_image, (31, 31), 10.0)
                mock_canvas[mask_bool] = blurred[mask_bool]

            # Enforce mathematical unmasked invariance
            final_reconstructed = AspectRatioPreserver.unletterbox(
                padded_candidate=cv2.resize(mock_canvas, (512, 512)),
                original_image=target_image,
                original_mask=normalized_mask,
                transform_info=transform_info
            )

        inpainting_time = time.time() - t_diff_start

        # -------------------------------------------------------------
        # STEP 5: Verify Bit-Exact Canvas Invariance (Delta = 0.0)
        # -------------------------------------------------------------
        passed_inv, max_diff, violated_pixels = verify_unmasked_invariance(
            target_image, final_reconstructed, normalized_mask
        )
        assert passed_inv, f"Invariance violation: max unmasked diff = {max_diff}"

        total_runtime = time.time() - t_start

        # -------------------------------------------------------------
        # STEP 6: Debug Visualization (5-Panel Figure)
        # -------------------------------------------------------------
        vis_path = None
        if save_debug_vis:
            out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "evaluation/visualizations/restoration_runs")
            out_dir.mkdir(parents=True, exist_ok=True)
            vis_filename = f"{input_mode.lower()}_run_seed{seed}_{int(time.time())}.png"
            vis_path = out_dir / vis_filename

            # 5-Panel Display: [Target | Mask | Overlay | Reference(s) | Final Result]
            fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

            axes[0].imshow(target_image)
            axes[0].set_title("1. Original Input")
            axes[0].axis("off")

            axes[1].imshow(normalized_mask, cmap="gray")
            axes[1].set_title(f"2. {input_mode}\n({num_masked_pixels:,} px, {img_coverage_pct}%)")
            axes[1].axis("off")

            overlay = target_image.copy()
            overlay[normalized_mask > 0] = [230, 40, 40]
            blended_overlay = cv2.addWeighted(overlay, 0.5, target_image, 0.5, 0)
            axes[2].imshow(blended_overlay)
            axes[2].set_title(f"3. Mask Overlay\n(Conf: {confidence:.2f})")
            axes[2].axis("off")

            if refs:
                # Tile up to 3 references
                tile_h, tile_w = refs[0].shape[:2]
                ref_thumbs = [cv2.resize(r, (160, 160)) for r in refs[:3]]
                tiled_refs = np.hstack(ref_thumbs)
                axes[3].imshow(tiled_refs)
                axes[3].set_title(f"4. Ref Gallery (M={len(refs)})\nw={ref_weights}")
            else:
                axes[3].text(0.5, 0.5, "No References\n(Single Image)", ha="center", va="center")
                axes[3].set_title("4. Reference Gallery")
            axes[3].axis("off")

            axes[4].imshow(final_reconstructed)
            axes[4].set_title(f"5. Inpainted Result\n(Δ_max = {max_diff:.5f})", color="green", fontweight="bold")
            axes[4].axis("off")

            fig.tight_layout()
            fig.savefig(str(vis_path), dpi=150)
            plt.close(fig)
            logger.info(f"Saved 5-panel debug visualization to {vis_path}")

        # Save detector diagnostics if in AUTO_MASK mode
        diagnostics_paths = {}
        if detection_result is not None and save_debug_vis and output_dir:
            diag_dir = Path(output_dir) / "diagnostics"
            diagnostics_paths = detection_result.save_diagnostics(output_dir=diag_dir)

        # Structured Response
        status = detection_result.status if detection_result else "MASK_PROVIDED_VERIFIED"
        safety_passed = detection_result.safety_passed if detection_result else True
        safety_flags = detection_result.safety_flags if detection_result else []

        result = {
            "input_mode": input_mode,
            "status": status,
            "safety_passed": safety_passed,
            "safety_flags": safety_flags,
            "final_image": final_reconstructed,
            "normalized_mask": normalized_mask,
            "detection_result": detection_result,
            "mask_metrics": {
                "num_masked_pixels": num_masked_pixels,
                "image_coverage_pct": img_coverage_pct,
                "face_coverage_pct": detection_result.face_coverage_pct if detection_result else None,
                "detection_confidence": confidence,
                "damage_bbox": bbox_damage,
                "num_faces_detected": detection_result.num_faces_detected if detection_result else 1,
                "landmarks_reliable": detection_result.landmarks_reliable if detection_result else True
            },
            "reference_scoring": {
                "num_references": len(refs),
                "weights": ref_weights
            },
            "invariance": {
                "passed": passed_inv,
                "max_difference": max_diff,
                "violated_pixels": violated_pixels
            },
            "performance": {
                "inpainting_seconds": round(inpainting_time, 3),
                "total_pipeline_seconds": round(total_runtime, 3)
            },
            "debug_visualization_path": str(vis_path) if vis_path else None,
            "diagnostics_paths": diagnostics_paths
        }

        logger.info(
            f"Restoration Complete [{input_mode}] - Mask: {num_masked_pixels:,} px ({img_coverage_pct}%), "
            f"Inpainting: {inpainting_time:.2f}s, Invariance: PASS (Delta=0.0)"
        )
        return result
