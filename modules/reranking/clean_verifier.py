"""
Clean O3 Verifier-Aware Sampling & Reranker Module.
Strict Protocol Implementation:
1. Zero Ground-Truth Access: Candidate selection has ZERO access to target_rgb,
   ground-truth target crops, ground-truth masks, or pixel error metrics (PSNR, SSIM, MAE, RMSE).
2. Matcher Independence: Candidate identity scoring uses an independent face recognition
   matcher (InceptionResnetV1 / FaceNet pretrained on VGGFace2), structurally and parametrically
   disjoint from the held-out evaluation ArcFace model (w600k_r50.onnx).
3. Blind Visual Quality: Candidate visual quality uses a blind, no-reference image quality
   metric (BRISQUE via piq) operating strictly on natural scene statistics.
4. Joint Selection: Explicitly combines normalized identity and visual quality scores:
   S_joint = w_id * S_id + w_qual * S_qual
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    InceptionResnetV1 = None

try:
    import piq
except ImportError:
    piq = None


class CleanO3Verifier:
    """
    Leak-free Verifier for Candidate Selection in Multi-Reference Facial Inpainting.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        weight_identity: float = 0.60,
        weight_quality: float = 0.40,
        facenet_weights: str = "vggface2"
    ):
        """
        Initializes the clean verifier with an independent face matcher and blind quality metric.
        
        Args:
            device: Compute device (CPU or CUDA).
            weight_identity: Weight for biometric identity score (default 0.60).
            weight_quality: Weight for blind visual quality score (default 0.40).
            facenet_weights: Pretrained weights for InceptionResnetV1 ('vggface2' or 'casia-webface').
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        assert abs(weight_identity + weight_quality - 1.0) < 1e-5, (
            f"Weights must sum to 1.0, got w_id={weight_identity}, w_qual={weight_quality}"
        )
        self.w_id = float(weight_identity)
        self.w_qual = float(weight_quality)

        # Initialize Independent Matcher: InceptionResnetV1 (FaceNet)
        if InceptionResnetV1 is None:
            raise ImportError(
                "facenet_pytorch is required for CleanO3Verifier independent matcher. "
                "Install with `pip install facenet-pytorch`."
            )
        self.matcher_name = f"InceptionResnetV1_{facenet_weights}"
        self.matcher = InceptionResnetV1(pretrained=facenet_weights).eval().to(self.device)
        for param in self.matcher.parameters():
            param.requires_grad = False

        # Verify blind quality availability
        if piq is None:
            raise ImportError("piq is required for CleanO3Verifier BRISQUE quality metric. Install with `pip install piq`.")
        self.quality_metric_name = "BRISQUE_piq"

    def _prep_face_for_facenet(self, rgb_img: np.ndarray) -> torch.Tensor:
        """
        Prepares an RGB image (H, W, 3) uint8 for FaceNet (1, 3, 160, 160).
        Standard fixed standardization: (img - 127.5) / 128.0
        """
        resized = cv2.resize(rgb_img, (160, 160), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        tensor = (tensor - 127.5) / 128.0
        return tensor

    def extract_identity_embedding(self, rgb_img: np.ndarray) -> torch.Tensor:
        """
        Extracts L2-normalized 512-d FaceNet identity embedding.
        """
        x = self._prep_face_for_facenet(rgb_img)
        with torch.no_grad():
            emb = self.matcher(x)
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def compute_reference_gallery_embedding(self, ref_rgbs: List[np.ndarray]) -> torch.Tensor:
        """
        Computes the unified reference gallery embedding from available reference images.
        """
        assert len(ref_rgbs) > 0, "Reference gallery must contain at least one image."
        embs = []
        for ref_img in ref_rgbs:
            e = self.extract_identity_embedding(ref_img)
            embs.append(e)
        all_embs = torch.cat(embs, dim=0) # [M, 512]
        mean_emb = all_embs.mean(dim=0, keepdim=True)
        return F.normalize(mean_emb, p=2, dim=-1)

    def compute_identity_score(
        self,
        candidate_rgb: np.ndarray,
        e_ref_gallery: torch.Tensor
    ) -> float:
        """
        Computes cosine identity similarity between candidate and reference gallery embedding.
        Returns a float score in [-1.0, 1.0].
        """
        e_cand = self.extract_identity_embedding(candidate_rgb)
        cos_sim = float(F.cosine_similarity(e_cand, e_ref_gallery, dim=-1).item())
        return cos_sim

    def compute_blind_quality_score(self, candidate_rgb: np.ndarray) -> Tuple[float, float]:
        """
        Computes blind, no-reference image quality score using BRISQUE on natural scene statistics.
        Returns:
            normalized_quality: float in [0.0, 1.0], where 1.0 = best quality.
            raw_brisque: raw BRISQUE score (lower is better, typically 0 to 100).
        """
        # Resize to standard evaluation scale (e.g., 256x256) for BRISQUE
        resized = cv2.resize(candidate_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            raw_val = piq.brisque(tensor, data_range=1.0)
            raw_score = float(raw_val.item())

        # Normalize BRISQUE:
        # Standard range is [0, 100]. Lower raw is better.
        # Slight negative values down to -10 can occur on pristine photos.
        clipped = max(0.0, min(100.0, raw_score))
        normalized = 1.0 - (clipped / 100.0) # 0 -> 1.0 (best), 100 -> 0.0 (worst)
        return float(normalized), float(raw_score)

    def score_candidate(
        self,
        candidate_rgb: np.ndarray,
        e_ref_gallery: torch.Tensor,
        seed: int
    ) -> Dict[str, Any]:
        """
        Scores a single candidate without any ground-truth target information.
        """
        s_id = self.compute_identity_score(candidate_rgb, e_ref_gallery)
        s_qual_norm, raw_brisque = self.compute_blind_quality_score(candidate_rgb)

        # Normalize identity score to [0, 1] range for harmonious combination with quality:
        # Cosine similarity is usually in [0, 1] for same identity; clamp negative values to 0.
        s_id_norm = max(0.0, min(1.0, s_id))

        s_joint = self.w_id * s_id_norm + self.w_qual * s_qual_norm

        return {
            "seed": seed,
            "s_identity_raw": round(s_id, 4),
            "s_identity_norm": round(s_id_norm, 4),
            "s_quality_norm": round(s_qual_norm, 4),
            "brisque_raw": round(raw_brisque, 4),
            "s_joint": round(s_joint, 4)
        }

    def rerank(
        self,
        candidates: List[np.ndarray],
        candidate_seeds: List[int],
        reference_rgbs: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Reranks K candidate images using ONLY the candidates and reference gallery.
        
        CRITICAL INFORMATION BARRIER:
        This function does NOT accept target_rgb or ground-truth masks.
        Zero target pixels are used in ranking.
        """
        assert len(candidates) > 0, "Candidates list cannot be empty."
        assert len(candidates) == len(candidate_seeds), "Mismatch between candidates and seeds."
        assert len(reference_rgbs) > 0, "Reference gallery cannot be empty."

        # Compute reference gallery embedding once
        e_ref_gallery = self.compute_reference_gallery_embedding(reference_rgbs)

        scored_candidates = []
        joint_scores = []

        for idx, (cand_img, s) in enumerate(zip(candidates, candidate_seeds)):
            info = self.score_candidate(cand_img, e_ref_gallery, seed=s)
            info["candidate_index"] = idx + 1
            scored_candidates.append(info)
            joint_scores.append(info["s_joint"])

        winning_idx = int(np.argmax(joint_scores))
        winner_info = scored_candidates[winning_idx]

        return {
            "winning_index": winning_idx + 1,
            "winning_seed": candidate_seeds[winning_idx],
            "winning_score": joint_scores[winning_idx],
            "winner_details": winner_info,
            "all_candidates": scored_candidates,
            "protocol": {
                "matcher": self.matcher_name,
                "quality_metric": self.quality_metric_name,
                "w_identity": self.w_id,
                "w_quality": self.w_qual,
                "k_candidates": len(candidates),
                "ground_truth_isolated": True
            }
        }
