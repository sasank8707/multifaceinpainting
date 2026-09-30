"""
High-Level End-to-End Pipeline Coordinator.
Chains Phase 2 Face Analysis, Phase 3 O1 Reference Scoring, Phase 4 O2 Regional Identity Fusion,
Phase 5 Coarse Layout Reconstruction, and Phase 6 Latent-Diffusion Backbone Integration.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from modules.face_analysis.pipeline import FaceAnalysisPipeline
from modules.reference_scoring.scorer import MaskAwareReferenceScorer
from modules.identity_fusion.regional_fusion import RegionalIdentityFusionModule
from modules.structural_guidance.coarse_reconstructor import CoarseReconstructor
from models.diffusion.diffusion_adapter import DiffusionInpaintingAdapter
from modules.masking.task_router import TaskMode
from modules.utils.device import get_device_config, DeviceConfig


class DiffusionInpaintingManager:
    """
    End-to-End Research Pipeline Manager synthesizing all project phases (1 through 6).
    """

    def __init__(
        self,
        device_config: Optional[DeviceConfig] = None,
        working_resolution: Tuple[int, int] = (256, 256),
        backend: str = "mock"
    ):
        self.device_config = device_config or get_device_config()
        self.working_resolution = working_resolution

        # Phase 2: Face and Mask Analysis
        self.face_pipeline = FaceAnalysisPipeline()

        # Phase 3: O1 Reference Scorer
        self.reference_scorer = MaskAwareReferenceScorer(strict_identity=False)

        # Phase 4: O2 Regional Identity Tokenizer and Fusion Transformer
        self.identity_fusion = RegionalIdentityFusionModule(
            token_dim=512,
            num_heads=8,
            num_layers=2,
            strict_identity=False
        ).to(self.device_config.device)

        # Phase 5: Coarse Semantic U-Net Reconstructor
        self.coarse_reconstructor = CoarseReconstructor(
            in_channels=4,
            out_channels=3,
            base_channels=32, # Responsive on CPU
            identity_token_dim=512,
            num_heads=8
        ).to(self.device_config.device)

        # Phase 6: Latent-Diffusion Backbone Adapter
        self.diffusion_adapter = DiffusionInpaintingAdapter(
            backend=backend,
            working_resolution=working_resolution,
            token_dim=512,
            cross_attention_dim=768,
            device_config=self.device_config
        ).to(self.device_config.device)

    def load_real_pipeline(self, model_id_or_path: str = "runwayml/stable-diffusion-inpainting") -> Any:
        """Loads real Diffusers inpainting pipeline on the diffusion adapter."""
        return self.diffusion_adapter.load_real_pipeline(model_id_or_path)

    def inpaint(
        self,
        target_image: np.ndarray,
        mask: np.ndarray,
        reference_images: Optional[List[np.ndarray]] = None,
        target_identity_id: Optional[str] = None,
        reference_identity_ids: Optional[List[str]] = None,
        num_candidates: int = 1,
        seed: Optional[int] = 42,
        num_inference_steps: int = 5,
        guidance_scale: float = 7.5
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Executes end-to-end multi-reference inpainting pipeline.
        
        Args:
            target_image: [H, W, 3] uint8 in [0, 255]
            mask: [H, W] float32 in [0, 1] or uint8 in [0, 255]
            reference_images: optional list of [H_ref, W_ref, 3] verified reference images
            target_identity_id: optional ground-truth identity ID
            reference_identity_ids: optional list of reference identity IDs
            num_candidates: K >= 1 candidates to generate
            seed: random seed for reproducibility
            num_inference_steps: diffusion steps (default 5 for CPU)
            guidance_scale: classifier-free guidance scale
            
        Returns:
            candidates: list of K [H, W, 3] reconstructed images with bit-exact unmasked pixels
            trace: execution trace across all pipeline phases
        """
        mask_f32 = (mask > 0.5).astype(np.float32) if mask.dtype != np.float32 else mask
        h, w = target_image.shape[:2]

        # 1. Phase 2: Face Analysis and Mask Profiling
        target_analysis = self.face_pipeline.analyze(target_image, mask_f32)
        routing = target_analysis.get("routing", {})
        task_mode_str = routing.get("task_mode", TaskMode.MODE_1_FACIAL_INPAINTING.value)
        face_info = target_analysis.get("face", {})
        has_face = face_info.get("detected", False)

        trace: Dict[str, Any] = {
            "task_mode": task_mode_str,
            "face_detected": has_face,
            "mask_coverage": target_analysis.get("mask_profile", {}).get("image_coverage", 0.0),
            "face_coverage": target_analysis.get("mask_profile", {}).get("face_region_coverage", 0.0)
        }

        # 2. Phase 3 & 4: Reference Scoring and Regional Identity Fusion
        identity_tokens = None
        scoring_report = None

        if (
            reference_images is not None
            and len(reference_images) > 0
            and task_mode_str != TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value
        ):
            # Analyze references
            ref_analyses = [self.face_pipeline.analyze(ref) for ref in reference_images]

            # Score references (Phase 3)
            scoring_report = self.reference_scorer.score_references(
                target_analysis=target_analysis,
                reference_analyses=ref_analyses,
                target_identity_id=target_identity_id,
                reference_identity_ids=reference_identity_ids
            )
            trace["reference_scoring"] = scoring_report

            # Convert numpy images to tensors for identity fusion (Phase 4)
            t_img_256 = cv2.resize(target_image, (256, 256))
            t_mask_256 = cv2.resize(mask_f32, (256, 256), interpolation=cv2.INTER_NEAREST)
            t_img_t = torch.from_numpy(t_img_256).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t_mask_t = torch.from_numpy(t_mask_256).unsqueeze(0).unsqueeze(0).float()

            ref_tensors = []
            for ref_np in reference_images:
                ref_256 = cv2.resize(ref_np, (256, 256))
                ref_t = torch.from_numpy(ref_256).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                ref_tensors.append(ref_t.to(self.device_config.device))

            t_img_t = t_img_t.to(self.device_config.device)
            t_mask_t = t_mask_t.to(self.device_config.device)

            fusion_out = self.identity_fusion(
                target_tensor=t_img_t,
                target_mask=t_mask_t,
                reference_tensors=ref_tensors,
                scoring_report=scoring_report,
                target_identity_id=target_identity_id,
                reference_identity_ids=reference_identity_ids
            )
            identity_tokens = fusion_out["fused_tokens"]
            trace["identity_conditioning_active"] = fusion_out["identity_conditioning_enabled"]
        else:
            trace["identity_conditioning_active"] = False

        # 3. Phase 5: Coarse Layout and Edge Reconstruction
        coarse_restored, coarse_meta = self.coarse_reconstructor.reconstruct_numpy(
            image_np=target_image,
            mask_np=mask_f32,
            identity_tokens=identity_tokens,
            task_mode=task_mode_str,
            device=self.device_config.device
        )
        trace["coarse_reconstruction"] = coarse_meta
        trace["coarse_image"] = coarse_restored

        # 4. Phase 6: Latent-Diffusion Candidate Generation
        candidates, diff_meta = self.diffusion_adapter.generate_candidates_numpy(
            image_np=target_image,
            mask_np=mask_f32,
            coarse_prediction_np=coarse_restored,
            identity_tokens=identity_tokens,
            task_mode=task_mode_str,
            num_candidates=num_candidates,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        trace["diffusion_generation"] = diff_meta

        return candidates, trace
