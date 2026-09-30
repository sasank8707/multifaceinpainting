"""
Unified Latent-Diffusion Inpainting Adapter.
Provides device-agnostic inpainting, aspect-ratio and resolution preservation,
multi-candidate generation (K >= 1), and seamless switching between
a high-speed deterministic mock backend and real Diffusers pipelines.
"""

from PIL import Image
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from modules.masking.task_router import TaskMode
from modules.utils.device import get_device_config, DeviceConfig
from modules.utils.geometry import verify_unmasked_invariance
from models.diffusion.mock_diffusion import MockDiffusionInpainter
from models.diffusion.conditioning_adapter import (
    IdentityConditioningAdapter,
    StructuralGuidanceAdapter,
    DiffusionConditioningBundle
)


class AspectRatioPreserver:
    """
    Utility for letterboxing and un-letterboxing images and masks
    to internal working resolutions without unintended aspect ratio distortion.
    """

    @staticmethod
    def letterbox(
        image: np.ndarray,
        mask: np.ndarray,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Pads and resizes image and mask to target_size while preserving aspect ratio.
        
        Args:
            image: [H, W, 3] uint8
            mask: [H, W] float32 in [0, 1] or uint8 in [0, 255]
            target_size: (H_target, W_target)
            
        Returns:
            padded_image: [H_target, W_target, 3] uint8
            padded_mask: [H_target, W_target] float32 in [0, 1]
            transform_info: metadata for inverse mapping
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_size

        # Compute scaling factor to fit inside target_size
        scale = min(target_h / orig_h, target_w / orig_w)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Resize
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        resized_mask = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Compute symmetric padding
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        # Pad image with edge reflection and mask with zeros
        padded_img = cv2.copyMakeBorder(
            resized_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        padded_mask = cv2.copyMakeBorder(
            resized_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0.0
        )

        info = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "new_h": new_h,
            "new_w": new_w,
            "scale": scale,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "target_size": target_size
        }
        return padded_img, padded_mask, info

    @staticmethod
    def unletterbox(
        padded_candidate: np.ndarray,
        original_image: np.ndarray,
        original_mask: np.ndarray,
        transform_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Inverse maps padded candidate back to exact original dimensions
        and enforces bit-exact preservation on unmasked pixels.
        
        Args:
            padded_candidate: [H_target, W_target, 3] uint8
            original_image: [H_orig, W_orig, 3] uint8
            original_mask: [H_orig, W_orig] float32 in [0, 1]
            transform_info: metadata dictionary from letterbox
            
        Returns:
            restored_image: [H_orig, W_orig, 3] uint8
        """
        pad_top = transform_info["pad_top"]
        pad_left = transform_info["pad_left"]
        new_h = transform_info["new_h"]
        new_w = transform_info["new_w"]
        orig_h = transform_info["orig_h"]
        orig_w = transform_info["orig_w"]

        # 1. Crop out padding
        cropped = padded_candidate[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

        # 2. Resize back to original dimensions
        restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # 3. Bit-exact unmasked pixel preservation
        final_output = original_image.copy()
        mask_bool = (original_mask > 0.5)
        final_output[mask_bool] = restored[mask_bool]

        return final_output


class DiffusionInpaintingAdapter(nn.Module):
    """
    Modular Inpainting Adapter coordinating latent-diffusion backbones.
    Supports both fast CPU mock execution and full Diffusers pipelines.
    """

    def __init__(
        self,
        backend: str = "mock",
        working_resolution: Tuple[int, int] = (256, 256),
        token_dim: int = 512,
        cross_attention_dim: int = 768,
        latent_channels: int = 4,
        device_config: Optional[DeviceConfig] = None
    ):
        super().__init__()
        self.backend = backend
        self.working_resolution = working_resolution
        self.token_dim = token_dim
        self.cross_attention_dim = cross_attention_dim
        self.latent_channels = latent_channels
        self.device_config = device_config or get_device_config()

        # Deterministic mock backend
        self.mock_inpainter = MockDiffusionInpainter(
            token_dim=token_dim,
            cross_attention_dim=cross_attention_dim,
            latent_channels=latent_channels
        )

        # Identity cross-attention conditioning adapter
        self.identity_adapter = IdentityConditioningAdapter(
            in_dim=token_dim,
            out_dim=cross_attention_dim,
            out_tokens=77,
            dropout=0.0
        )
        self._load_default_identity_adapter()

        # Real diffusers pipeline holder (lazy-loaded on demand)
        self.real_pipeline = None

    def _load_default_identity_adapter(self):
        """Loads retrained identity adapter weights if available."""
        from pathlib import Path
        ckpt_path = Path(__file__).resolve().parent.parent.parent / "checkpoints/identity_adapter/identity_adapter_best.pt"
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.identity_adapter.load_state_dict(state_dict)
            self.identity_adapter.eval()
            print(f"[DiffusionInpaintingAdapter] Loaded retrained identity adapter checkpoint from {ckpt_path.name}")

    def load_real_pipeline(
        self,
        model_id_or_path: str = "runwayml/stable-diffusion-inpainting",
        torch_dtype: Optional[torch.dtype] = None
    ) -> Any:
        """
        Loads the real Diffusers StableDiffusionInpaintPipeline.
        Prefers local cached snapshots if available.
        """
        from diffusers import StableDiffusionInpaintPipeline

        dtype = torch_dtype or self.device_config.dtype
        device = self.device_config.device

        # Check local HF cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        local_model_dir = os.path.join(
            cache_dir, "models--runwayml--stable-diffusion-inpainting", "snapshots"
        )
        load_path = model_id_or_path
        if os.path.isdir(local_model_dir):
            snapshots = [os.path.join(local_model_dir, d) for d in os.listdir(local_model_dir)]
            if snapshots:
                load_path = snapshots[0]

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            load_path,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True if os.path.isdir(load_path) else False
        )
        pipe = pipe.to(device)

        # Apply GPU memory optimizations if CUDA is active
        if self.device_config.is_cuda:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

        self.real_pipeline = pipe
        self.backend = "diffusers"
        return pipe

    def forward(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        coarse_prediction: torch.Tensor,
        bottleneck_features: Optional[torch.Tensor] = None,
        identity_tokens: Optional[torch.Tensor] = None,
        task_mode: str = TaskMode.MODE_1_FACIAL_INPAINTING.value,
        num_candidates: int = 1,
        seed: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Tensor forward pass executing inpainting generation."""
        if self.backend == "mock" or self.real_pipeline is None:
            return self.mock_inpainter(
                masked_image=masked_image,
                mask=mask,
                coarse_prediction=coarse_prediction,
                bottleneck_features=bottleneck_features,
                identity_tokens=identity_tokens,
                task_mode=task_mode,
                num_candidates=num_candidates,
                seed=seed
            )
        else:
            raise NotImplementedError(
                "Direct tensor forward on real pipeline is handled via generate_candidates_numpy."
            )

    def generate_candidates_numpy(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        coarse_prediction_np: np.ndarray,
        identity_tokens: Optional[torch.Tensor] = None,
        task_mode: str = TaskMode.MODE_1_FACIAL_INPAINTING.value,
        num_candidates: int = 1,
        seed: Optional[int] = 42,
        num_inference_steps: int = 5,
        guidance_scale: float = 7.5
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Generates K candidate reconstructions for arbitrary input dimensions
        with full aspect ratio preservation and bit-exact unmasked pixel preservation.
        
        Args:
            image_np: [H, W, 3] uint8 in [0, 255]
            mask_np: [H, W] float32 in [0, 1]
            coarse_prediction_np: [H, W, 3] uint8 in [0, 255]
            identity_tokens: [1, M, D] or None
            task_mode: operational task mode string
            num_candidates: K >= 1
            seed: random seed
            num_inference_steps: diffusion inference steps
            guidance_scale: classifier-free guidance scale
            
        Returns:
            candidates: list of K [H, W, 3] uint8 restored images
            metadata: execution and invariance verification metadata
        """
        orig_h, orig_w = image_np.shape[:2]

        # 1. Letterbox to working resolution
        padded_img, padded_mask, transform_info = AspectRatioPreserver.letterbox(
            image=image_np,
            mask=mask_np,
            target_size=self.working_resolution
        )
        padded_coarse, _, _ = AspectRatioPreserver.letterbox(
            image=coarse_prediction_np,
            mask=mask_np,
            target_size=self.working_resolution
        )

        # 2. Run inpainter at working resolution
        if self.backend == "diffusers" and self.real_pipeline is not None:
            # =========================================================================
            # STABLE DIFFUSION INPAINTING UNET 9-CHANNEL INPUT SPECIFICATION:
            # In StableDiffusionInpaintPipeline, the inpainting UNet receives a 9-channel
            # concatenated latent tensor along dimension 1:
            #   - Channels 0:4 (latents): The noisy latent state z_t undergoing iterative denoising.
            #   - Channels 4:5 (mask): The downsampled binary inpainting mask (1 = inpaint, 0 = keep).
            #   - Channels 5:9 (masked_image_latents): The VAE latent encoding of the input image.
            #
            # PHASE-5 COARSE STRUCTURAL GUIDANCE INJECTION:
            # Rather than supplying an empty black canvas inside the mask, Phase-5 coarse
            # reconstruction (padded_coarse) is composited into the masked area of guided_img_np.
            # When encoded by the VAE into channels 5:9, this serves as coarse structural guidance
            # through the standard masked-image latent pathway (not a separate ControlNet module),
            # providing spatial, edge, and anatomical priors directly to the UNet feature maps.
            # =========================================================================
            id_out = self.identity_adapter(
                identity_tokens=identity_tokens,
                task_mode=task_mode,
                batch_size=1
            )
            # Projected identity tokens [1, 77, 768] for cross-attention
            prompt_embeds = id_out["projected_tokens"].to(self.device_config.device, dtype=self.device_config.dtype)
            null_embeds = self.identity_adapter.get_null_embedding(
                batch_size=1, target_len=77
            ).to(self.device_config.device, dtype=self.device_config.dtype)

            # Phase-5 coarse structural guidance composited image for VAE latent channels 5:9
            mask_bool = (padded_mask > 0.5)
            pil_original = Image.fromarray(padded_img)
            pil_mask = Image.fromarray((padded_mask * 255.0).astype(np.uint8))

            # Explicitly construct normalized masked-image tensor in the exact semantic space
            # expected by pretrained Stable Diffusion inpainting (neutral 0.0 in masked region)
            init_image_tensor = self.real_pipeline.image_processor.preprocess(
                pil_original, height=self.working_resolution[0], width=self.working_resolution[1]
            ).to(device=self.device_config.device, dtype=self.device_config.dtype)

            mask_tensor = self.real_pipeline.mask_processor.preprocess(
                pil_mask, height=self.working_resolution[0], width=self.working_resolution[1]
            ).to(device=self.device_config.device, dtype=self.device_config.dtype)

            if coarse_prediction_np is not None:
                pil_coarse = Image.fromarray(padded_coarse)
                coarse_tensor = self.real_pipeline.image_processor.preprocess(
                    pil_coarse, height=self.working_resolution[0], width=self.working_resolution[1]
                ).to(device=self.device_config.device, dtype=self.device_config.dtype)
                masked_image_tensor = init_image_tensor * (mask_tensor < 0.5) + coarse_tensor * (mask_tensor >= 0.5)
            else:
                masked_image_tensor = init_image_tensor * (mask_tensor < 0.5)

            padded_candidates = []
            seeds_used = []
            base_seed = seed if seed is not None else 42

            for k in range(num_candidates):
                cand_seed = base_seed + k * 10007
                seeds_used.append(cand_seed)
                gen = torch.Generator(device=self.device_config.device_str).manual_seed(cand_seed)

                # Assemble channels 5:9 via VAE encoding from the properly normalized masked_image_tensor
                with torch.no_grad():
                    guided_latents = self.real_pipeline._encode_vae_image(masked_image_tensor, generator=gen)

                gen_pil = self.real_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=null_embeds,
                    image=pil_original,
                    mask_image=pil_mask,
                    masked_image_latents=guided_latents,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=gen
                ).images[0]

                cand_np = np.array(gen_pil, dtype=np.uint8)
                # Strict unmasked pixel preservation
                final_cand = padded_img.copy()
                final_cand[mask_bool] = cand_np[mask_bool]
                padded_candidates.append(final_cand)

            identity_active = id_out["identity_active"]
            seeds = seeds_used
        else:
            # Deterministic mock backend for ultra-fast CPU testing
            padded_candidates, mock_meta = self.mock_inpainter.generate_candidates_numpy(
                image_np=padded_img,
                mask_np=padded_mask,
                coarse_prediction_np=padded_coarse,
                identity_tokens=identity_tokens,
                task_mode=task_mode,
                num_candidates=num_candidates,
                seed=seed,
                device=self.device_config.device
            )
            identity_active = mock_meta["identity_active"]
            seeds = mock_meta["seeds"]

        # 3. Unletterbox each candidate back to original dimensions
        restored_candidates = []
        all_invariance_passed = True
        max_unmasked_diff = 0

        for cand_pad in padded_candidates:
            restored = AspectRatioPreserver.unletterbox(
                padded_candidate=cand_pad,
                original_image=image_np,
                original_mask=mask_np,
                transform_info=transform_info
            )
            # Verify invariance on original dimensions
            passed, diff, _ = verify_unmasked_invariance(image_np, restored, mask_np)
            if not passed:
                all_invariance_passed = False
            if diff > max_unmasked_diff:
                max_unmasked_diff = diff

            restored_candidates.append(restored)

        metadata = {
            "backend": self.backend,
            "num_candidates": num_candidates,
            "seeds": seeds,
            "task_mode": task_mode,
            "identity_active": identity_active,
            "original_dimensions": (orig_h, orig_w),
            "working_resolution": self.working_resolution,
            "unmasked_invariance_passed": all_invariance_passed,
            "max_unmasked_diff": max_unmasked_diff
        }

        return restored_candidates, metadata
