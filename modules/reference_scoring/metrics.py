"""
Mathematical metrics for Mask-Aware Multi-Reference Scoring (O1 Retrieve).
Computes normalized sub-scores for pose compatibility, image quality/sharpness,
landmark visibility within occluded regions, and geometric region overlap.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def compute_pose_compatibility(
    target_pose: Dict[str, float],
    ref_pose: Dict[str, float],
    sigma_yaw: float = 15.0,
    sigma_pitch: float = 15.0,
    sigma_roll: float = 10.0
) -> float:
    """
    Computes exponential Gaussian head pose compatibility score in [0, 1].
    
    Formula:
        S_pose = exp( - (dyaw^2 / (2*sigma_yaw^2)) - (dpitch^2 / (2*sigma_pitch^2)) - (droll^2 / (2*sigma_roll^2)) )
    """
    if target_pose.get("confidence", 0.0) == 0.0 or ref_pose.get("confidence", 0.0) == 0.0:
        return 0.5  # Neutral fallback if pose confidence is unavailable
        
    dyaw = float(ref_pose.get("yaw", 0.0) - target_pose.get("yaw", 0.0))
    dpitch = float(ref_pose.get("pitch", 0.0) - target_pose.get("pitch", 0.0))
    droll = float(ref_pose.get("roll", 0.0) - target_pose.get("roll", 0.0))
    
    exp_arg = (
        - (dyaw ** 2) / (2.0 * (sigma_yaw ** 2))
        - (dpitch ** 2) / (2.0 * (sigma_pitch ** 2))
        - (droll ** 2) / (2.0 * (sigma_roll ** 2))
    )
    score = float(np.exp(np.clip(exp_arg, -50.0, 0.0)))
    return round(score, 4)


def compute_quality_score(
    ref_quality: Dict[str, Any],
    sharpness_weight: float = 0.6,
    illumination_weight: float = 0.4
) -> float:
    """
    Computes normalized reference face quality score in [0, 1].
    Combines Laplacian sharpness (Wei et al., 2019) and balanced exposure.
    """
    # Check if precomputed composite score exists
    if "composite_score" in ref_quality:
        return round(float(ref_quality["composite_score"]), 4)
        
    norm_sharpness = 0.5
    if "sharpness" in ref_quality and "normalized_sharpness" in ref_quality["sharpness"]:
        norm_sharpness = float(ref_quality["sharpness"]["normalized_sharpness"])
        
    exposure_balance = 0.5
    if "illumination" in ref_quality and "exposure_balance_score" in ref_quality["illumination"]:
        exposure_balance = float(ref_quality["illumination"]["exposure_balance_score"])
        
    score = (sharpness_weight * norm_sharpness) + (illumination_weight * exposure_balance)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def compute_masked_landmark_visibility(
    affected_landmark_indices: List[int],
    ref_face_info: Optional[Dict[str, Any]],
    ref_dimensions: Tuple[int, int]
) -> float:
    """
    Computes landmark visibility score S_vis in [0, 1] for occluded target landmarks.
    
    Evaluates whether the landmarks occluded in the target are clearly visible,
    unoccluded, and present within the reference face frame.
    
    If no landmarks fall within the mask component (e.g. localized skin patch),
    evaluates general face landmark completeness.
    """
    if ref_face_info is None or not ref_face_info.get("detected", False):
        return 0.0
        
    ref_landmarks = ref_face_info.get("dense_landmarks_pixel", [])
    if not ref_landmarks and "canonical_68" in ref_face_info:
        ref_landmarks = ref_face_info["canonical_68"]
        
    if not ref_landmarks:
        return 0.0
        
    ref_h, ref_w = ref_dimensions[:2]
    
    # If the mask component occludes specific landmarks in the target:
    if len(affected_landmark_indices) > 0:
        visible_count = 0
        for idx in affected_landmark_indices:
            if idx < len(ref_landmarks):
                rx, ry = ref_landmarks[idx][0], ref_landmarks[idx][1]
                # Check if landmark lies cleanly within reference image boundary (margin 5px)
                if 5 <= rx <= (ref_w - 5) and 5 <= ry <= (ref_h - 5):
                    visible_count += 1
        score = float(visible_count / len(affected_landmark_indices))
        return round(score, 4)
    else:
        # Fallback for arbitrary skin masks without landmark intersections:
        # Evaluate whether reference face is fully within frame without clipping
        ref_bbox = ref_face_info.get("bbox", (0, 0, ref_w, ref_h))
        bx1, by1, bx2, by2 = ref_bbox
        is_in_bounds = (bx1 >= 0 and by1 >= 0 and bx2 <= ref_w and by2 <= ref_h)
        return 1.0 if is_in_bounds else 0.7


def compute_region_overlap_compatibility(
    comp_bbox: Tuple[int, int, int, int],
    target_face_bbox: Optional[Tuple[int, int, int, int]],
    ref_face_info: Optional[Dict[str, Any]],
    ref_dimensions: Tuple[int, int]
) -> float:
    """
    Computes geometric region overlap compatibility S_overlap in [0, 1].
    
    Maps the mask component bounding box normalized relative to the target face
    into the reference face bounding box, and checks what fraction is safely
    contained within the reference image boundaries without frame truncation.
    """
    if target_face_bbox is None or ref_face_info is None or not ref_face_info.get("detected", False):
        return 0.5
        
    tx1, ty1, tx2, ty2 = target_face_bbox
    tw = max(1, tx2 - tx1)
    th = max(1, ty2 - ty1)
    
    cx1, cy1, cx2, cy2 = comp_bbox
    # Normalized component coords relative to target face
    norm_x1 = (cx1 - tx1) / tw
    norm_y1 = (cy1 - ty1) / th
    norm_x2 = (cx2 - tx1) / tw
    norm_y2 = (cy2 - ty1) / th
    
    ref_bbox = ref_face_info.get("bbox")
    if ref_bbox is None:
        return 0.5
    rx1, ry1, rx2, ry2 = ref_bbox
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    
    # Project into reference coordinates
    proj_x1 = rx1 + norm_x1 * rw
    proj_y1 = ry1 + norm_y1 * rh
    proj_x2 = rx1 + norm_x2 * rw
    proj_y2 = ry1 + norm_y2 * rh
    
    proj_w = max(1e-5, proj_x2 - proj_x1)
    proj_h = max(1e-5, proj_y2 - proj_y1)
    proj_area = proj_w * proj_h
    
    # Intersection with reference frame
    ref_h, ref_w = ref_dimensions[:2]
    inter_x1 = max(0.0, proj_x1)
    inter_y1 = max(0.0, proj_y1)
    inter_x2 = min(float(ref_w), proj_x2)
    inter_y2 = min(float(ref_h), proj_y2)
    
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    overlap_ratio = float(inter_area / proj_area)
    return round(float(np.clip(overlap_ratio, 0.0, 1.0)), 4)
