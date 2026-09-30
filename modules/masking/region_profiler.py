"""
Mask-Aware Region Profiling & Geometry Analysis Module.
Extracts connected mask components, computes bounding boxes, areas, coverage ratios
(image, face, ROI), calculates face-mask overlap and IoU, identifies occluded landmarks,
and semantically classifies regions without hardcoded organ assumptions.
"""

import math
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional


class MaskRegionProfiler:
    """
    Analyzes arbitrary single or multi-component masks relative to face geometry.
    Operates completely region-agnostically across any mask shape or coverage level.
    """

    def __init__(self, roi_margin_ratio: float = 0.35):
        self.roi_margin_ratio = roi_margin_ratio

    def profile_mask(
        self,
        mask: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks_pixel: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts comprehensive geometric and semantic profile of the mask.
        
        Args:
            mask: 2D array [0, 1] or [0, 255]
            face_bbox: Optional detected face bounding box (x1, y1, x2, y2)
            landmarks_pixel: Optional list of detected landmark coordinates
            
        Returns:
            Dict containing global mask metrics, per-component analysis, and classification.
        """
        h, w = mask.shape[:2]
        binary = (mask > 0.5).astype(np.uint8)
        total_image_pixels = h * w
        total_masked_pixels = int(np.sum(binary))
        
        image_coverage = float(total_masked_pixels / total_image_pixels) if total_image_pixels > 0 else 0.0
        
        # Face-region coverage and overlap
        face_area = 0
        face_overlap_pixels = 0
        face_coverage = 0.0
        face_iou = 0.0
        
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
            
            # Intersection of mask with face box
            face_mask_box = np.zeros((h, w), dtype=np.uint8)
            face_mask_box[fy1:fy2, fx1:fx2] = 1
            
            intersection = np.logical_and(binary, face_mask_box)
            face_overlap_pixels = int(np.sum(intersection))
            face_coverage = float(face_overlap_pixels / face_area)
            
            union = np.logical_or(binary, face_mask_box)
            face_iou = float(face_overlap_pixels / np.sum(union)) if np.sum(union) > 0 else 0.0

        # Connected component extraction
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        components = []
        for i in range(1, num_labels): # Skip label 0 (background)
            cx, cy, cw, ch, area = stats[i]
            comp_mask = (labels == i).astype(np.uint8)
            comp_bbox = (int(cx), int(cy), int(cx + cw), int(cy + ch))
            comp_area = int(area)
            comp_img_cov = float(comp_area / total_image_pixels)
            
            # Contextual ROI for component
            mx = int(cw * self.roi_margin_ratio)
            my = int(ch * self.roi_margin_ratio)
            roi_bbox = (
                max(0, cx - mx),
                max(0, cy - my),
                min(w, cx + cw + mx),
                min(h, cy + ch + my)
            )
            roi_area = max(1, (roi_bbox[2] - roi_bbox[0]) * (roi_bbox[3] - roi_bbox[1]))
            roi_coverage = float(comp_area / roi_area)
            
            # Component face overlap
            comp_face_overlap = 0
            comp_face_cov = 0.0
            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                comp_inter = comp_mask[fy1:fy2, fx1:fx2]
                comp_face_overlap = int(np.sum(comp_inter))
                comp_face_cov = float(comp_face_overlap / face_area)
                
            # Check affected landmarks for this component
            comp_affected_landmarks = []
            if landmarks_pixel is not None:
                for lm_idx, pt in enumerate(landmarks_pixel):
                    lx, ly = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= lx < w and 0 <= ly < h:
                        if comp_mask[ly, lx] == 1:
                            comp_affected_landmarks.append(lm_idx)

            # Classify component
            if face_bbox is None or comp_face_overlap == 0:
                comp_category = "NON_FACE"
            elif comp_face_overlap == comp_area:
                comp_category = "PURE_FACIAL"
            elif comp_face_overlap > 0 and comp_face_overlap < comp_area:
                comp_category = "MIXED_BOUNDARY"
            else:
                comp_category = "FACIAL"
                
            components.append({
                "component_id": i - 1,
                "bbox": comp_bbox,
                "roi_bbox": roi_bbox,
                "area_pixels": comp_area,
                "image_coverage": round(comp_img_cov, 5),
                "face_coverage": round(comp_face_cov, 5),
                "roi_coverage": round(roi_coverage, 5),
                "face_overlap_pixels": comp_face_overlap,
                "affected_landmarks": comp_affected_landmarks,
                "num_affected_landmarks": len(comp_affected_landmarks),
                "centroid": (round(float(centroids[i][0]), 2), round(float(centroids[i][1]), 2)),
                "category": comp_category
            })

        # Overall global category determination
        if face_bbox is None or face_overlap_pixels == 0:
            primary_region_type = "NON_FACE"
        elif face_overlap_pixels == total_masked_pixels:
            primary_region_type = "FACIAL"
        elif face_overlap_pixels > 0 and (total_masked_pixels - face_overlap_pixels) > (0.1 * total_masked_pixels):
            primary_region_type = "MIXED"
        else:
            primary_region_type = "FACIAL"

        # Global affected landmarks
        global_affected_landmarks = []
        if landmarks_pixel is not None:
            for lm_idx, pt in enumerate(landmarks_pixel):
                lx, ly = int(round(pt[0])), int(round(pt[1]))
                if 0 <= lx < w and 0 <= ly < h:
                    if binary[ly, lx] == 1:
                        global_affected_landmarks.append(lm_idx)

        # Global bounding box covering all mask components
        if components:
            min_x = min(c["bbox"][0] for c in components)
            min_y = min(c["bbox"][1] for c in components)
            max_x = max(c["bbox"][2] for c in components)
            max_y = max(c["bbox"][3] for c in components)
            global_mask_bbox = (min_x, min_y, max_x, max_y)
        else:
            global_mask_bbox = (0, 0, 0, 0)

        return {
            "image_dimensions": {"height": h, "width": w},
            "total_masked_pixels": total_masked_pixels,
            "image_coverage": round(image_coverage, 5),
            "face_coverage": round(face_coverage, 5),
            "face_overlap_pixels": face_overlap_pixels,
            "face_iou": round(face_iou, 5),
            "num_components": len(components),
            "components": components,
            "global_mask_bbox": global_mask_bbox,
            "global_affected_landmarks": global_affected_landmarks,
            "num_affected_landmarks": len(global_affected_landmarks),
            "primary_region_type": primary_region_type,
            "is_multi_mask": len(components) > 1,
            "is_facial": primary_region_type in ["FACIAL", "MIXED"],
            "is_pure_background": primary_region_type == "NON_FACE"
        }
