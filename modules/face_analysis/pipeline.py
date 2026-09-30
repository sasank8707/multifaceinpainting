"""
Unified Face, Geometry, and Mask-Aware Analysis Pipeline.
Integrates face detection, 3D pose estimation, quality analysis,
region profiling, and task routing into a clean, reproducible research API.
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from modules.face_analysis.face_detector import FaceDetector
from modules.face_analysis.pose_estimator import PoseEstimator
from modules.face_analysis.quality_analyzer import QualityAnalyzer
from modules.masking.region_profiler import MaskRegionProfiler
from modules.masking.task_router import TaskRouter


class FaceAnalysisPipeline:
    """
    Modular, frozen analysis pipeline for arbitrary images and masks.
    """

    def __init__(
        self,
        model_path: str = "models/pretrained/face_landmarker.task",
        roi_margin_ratio: float = 0.35
    ):
        self.detector = FaceDetector(model_path=model_path)
        self.pose_estimator = PoseEstimator()
        self.quality_analyzer = QualityAnalyzer()
        self.region_profiler = MaskRegionProfiler(roi_margin_ratio=roi_margin_ratio)
        self.router = TaskRouter()

    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        is_degradation: bool = False,
        explicit_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete face, pose, quality, and mask geometry analysis.
        
        Args:
            image: Input image (H, W, 3) or (H, W) numpy array.
            mask: Optional binary mask (H, W) in [0, 1] or [0, 255].
            is_degradation: True if user specifies localized skin restoration.
            explicit_mode: User or experiment task override.
            
        Returns:
            Comprehensive analysis dictionary ready for JSON serialization.
        """
        h, w = image.shape[:2]
        
        # 1. Detect Face & Landmarks
        face_info = self.detector.detect(image)
        has_face = face_info is not None
        
        face_bbox = face_info["bbox"] if has_face else None
        dense_landmarks = face_info["dense_landmarks_pixel"] if has_face else None
        
        # 2. 3D Pose Estimation
        if has_face and dense_landmarks is not None:
            pose = self.pose_estimator.estimate_pose(np.array(dense_landmarks), (h, w))
        else:
            pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}
            
        # 3. Quality & Sharpness Analysis
        sharpness_info = self.quality_analyzer.compute_sharpness(image, face_bbox)
        illum_info = self.quality_analyzer.compute_illumination(image, face_bbox)
        composite_quality = self.quality_analyzer.compute_composite_quality_score(image, face_bbox)
        
        # 4. Mask-Aware Region Profiling
        if mask is not None:
            mask_profile = self.region_profiler.profile_mask(
                mask,
                face_bbox=face_bbox,
                landmarks_pixel=dense_landmarks
            )
            # Compute landmark visibility under mask
            if has_face and dense_landmarks is not None:
                vis_info = self.quality_analyzer.evaluate_landmark_visibility(
                    dense_landmarks,
                    mask
                )
            else:
                vis_info = {"visibility_ratio": 1.0, "occlusion_ratio": 0.0, "occluded_count": 0}
        else:
            mask_profile = {
                "num_components": 0,
                "total_masked_pixels": 0,
                "image_coverage": 0.0,
                "face_coverage": 0.0,
                "face_overlap_pixels": 0,
                "primary_region_type": "NO_MASK",
                "components": []
            }
            vis_info = {"visibility_ratio": 1.0, "occlusion_ratio": 0.0, "occluded_count": 0}

        # 5. Task Routing
        routing = self.router.route(
            mask_profile=mask_profile,
            face_info=face_info,
            explicit_mode=explicit_mode,
            is_degradation_flag=is_degradation
        )

        return {
            "image_dimensions": {"height": h, "width": w, "channels": image.shape[2] if image.ndim == 3 else 1},
            "face": {
                "detected": has_face,
                "bbox": face_bbox,
                "face_area": face_info["face_area"] if has_face else 0,
                "num_landmarks": face_info["num_landmarks"] if has_face else 0,
                "canonical_68": face_info["canonical_68_pixel"] if has_face else [],
            },
            "pose": pose,
            "quality": {
                "composite_score": composite_quality,
                "sharpness": sharpness_info,
                "illumination": illum_info,
                "landmark_visibility": vis_info
            },
            "mask_profile": mask_profile,
            "routing": routing
        }

    def save_analysis(self, analysis_dict: Dict[str, Any], output_path: str):
        """Serializes analysis results to disk in clean JSON format."""
        # Convert any remaining non-serializable items
        def _convert(val):
            if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
                return int(val)
            if isinstance(val, (np.float32, np.float64)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        with open(output_path, "w") as f:
            json.dump(analysis_dict, f, indent=2, default=_convert)
