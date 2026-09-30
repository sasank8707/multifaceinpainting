"""
Comprehensive Synthetic Mask Generator.
Supports 10 deterministic, reproducible mask types across 4 severity tiers
(25%, 50%, 75%, 100%) and multi-component disconnected configurations.
"""

import math
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional


class MaskGenerator:
    """
    Generates diverse masks for facial inpainting, restoration, multi-mask, and object removal.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

    def set_seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def generate_rectangular_mask(
        self,
        shape: Tuple[int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_coverage: float = 0.25
    ) -> np.ndarray:
        """Type M01: Single rectangular mask with specified coverage."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            fw, fh = fx2 - fx1, fy2 - fy1
            target_area = fw * fh * target_coverage
            side = int(math.sqrt(target_area))
            rw = min(fw, max(10, side))
            rh = min(fh, max(10, int(target_area / rw)))
            rx = self.rng.randint(fx1, max(fx1 + 1, fx2 - rw + 1))
            ry = self.rng.randint(fy1, max(fy1 + 1, fy2 - rh + 1))
        else:
            target_area = h * w * target_coverage
            side = int(math.sqrt(target_area))
            rw = min(w, max(10, side))
            rh = min(h, max(10, int(target_area / rw)))
            rx = self.rng.randint(0, max(1, w - rw))
            ry = self.rng.randint(0, max(1, h - rh))
            
        mask[ry : ry + rh, rx : rx + rw] = 1.0
        return mask

    def generate_elliptical_mask(
        self,
        shape: Tuple[int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_coverage: float = 0.25
    ) -> np.ndarray:
        """Type M02: Rotated elliptical mask."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            cx = self.rng.randint(fx1 + (fx2 - fx1)//4, fx2 - (fx2 - fx1)//4)
            cy = self.rng.randint(fy1 + (fy2 - fy1)//4, fy2 - (fy2 - fy1)//4)
            target_area = (fx2 - fx1) * (fy2 - fy1) * target_coverage
        else:
            cx = self.rng.randint(w // 4, 3 * w // 4)
            cy = self.rng.randint(h // 4, 3 * h // 4)
            target_area = h * w * target_coverage
            
        r_mean = math.sqrt(target_area / math.pi)
        r_major = int(max(8, r_mean * self.rng.uniform(1.1, 1.5)))
        r_minor = int(max(8, target_area / (math.pi * r_major)))
        angle = self.rng.uniform(0, 180)
        
        mask_uint8 = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask_uint8, (cx, cy), (r_major, r_minor), angle, 0, 360, 255, -1)
        return (mask_uint8 > 127).astype(np.float32)

    def generate_irregular_mask(
        self,
        shape: Tuple[int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        num_strokes: int = 6,
        max_width: int = 30
    ) -> np.ndarray:
        """Type M03: Free-form irregular brush strokes / random walks."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if face_bbox is not None:
            bx1, by1, bx2, by2 = face_bbox
        else:
            bx1, by1, bx2, by2 = 0, 0, w, h
            
        for _ in range(num_strokes):
            x1 = self.rng.randint(bx1, bx2)
            y1 = self.rng.randint(by1, by2)
            for _ in range(self.rng.randint(2, 5)):
                x2 = int(np.clip(x1 + self.rng.randint(-w//4, w//4), bx1, bx2))
                y2 = int(np.clip(y1 + self.rng.randint(-h//4, h//4), by1, by2))
                width = self.rng.randint(8, max_width)
                cv2.line(mask, (x1, y1), (x2, y2), 255, width)
                cv2.circle(mask, (x2, y2), width // 2, 255, -1)
                x1, y1 = x2, y2
                
        return (mask > 127).astype(np.float32)

    def generate_multi_component_mask(
        self,
        shape: Tuple[int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        num_components: int = 3,
        component_types: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Type M05: Multiple disconnected mask regions across the face."""
        h, w = shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        components_info = []
        
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
        else:
            fx1, fy1, fx2, fy2 = 0, 0, w, h

        for comp_idx in range(num_components):
            # Pick a quadrant or region within face bbox
            cw = (fx2 - fx1) // 3
            ch = (fy2 - fy1) // 3
            qx = fx1 + (comp_idx % 3) * cw
            qy = fy1 + (comp_idx // 3) * ch
            sub_bbox = (qx, qy, min(fx2, qx + cw), min(fy2, qy + ch))
            
            sub_mask = self.generate_elliptical_mask(shape, face_bbox=sub_bbox, target_coverage=0.45)
            combined_mask = np.maximum(combined_mask, sub_mask)
            
            components_info.append({
                "component_id": comp_idx,
                "bbox": sub_bbox,
                "type": "facial_region"
            })
            
        return combined_mask, components_info

    def generate_severe_occlusion_mask(
        self,
        shape: Tuple[int, int],
        face_bbox: Tuple[int, int, int, int],
        severity_ratio: float = 0.75
    ) -> np.ndarray:
        """Type M06: Severe occlusion (50%, 75%, or 100% of the entire face)."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        fx1, fy1, fx2, fy2 = face_bbox
        fw, fh = fx2 - fx1, fy2 - fy1
        
        if severity_ratio >= 0.95: # Full face
            mask[fy1:fy2, fx1:fx2] = 1.0
        else:
            # Slice from bottom (covering mouth/nose) or top (eyes/nose)
            cover_h = int(fh * severity_ratio)
            from_top = bool(self.rng.choice([True, False]))
            if from_top:
                mask[fy1 : fy1 + cover_h, fx1:fx2] = 1.0
            else:
                mask[fy2 - cover_h : fy2, fx1:fx2] = 1.0
                
        return mask

    def generate_degradation_mask(
        self,
        shape: Tuple[int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        degradation_type: str = "scar"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Type M07 / M08: Localized skin degradation mask (scar, burn, wrinkle, blemish).
        Returns:
            restoration_mask: Binary region indicating where degradation is situated.
            synthetic_degraded_patch: Degraded noise/pattern to superimpose on target.
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            cx = self.rng.randint(fx1 + (fx2 - fx1)//3, fx2 - (fx2 - fx1)//3)
            cy = self.rng.randint(fy1 + (fy2 - fy1)//3, fy2 - (fy2 - fy1)//3)
        else:
            cx, cy = w // 2, h // 2
            
        if degradation_type in ["scar", "wrinkle"]:
            # Line with small branches
            x1, y1 = cx - self.rng.randint(15, 30), cy - self.rng.randint(10, 20)
            x2, y2 = cx + self.rng.randint(15, 30), cy + self.rng.randint(10, 20)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=self.rng.randint(3, 7))
        elif degradation_type in ["burn", "blemish"]:
            # Irregular cluster
            r = self.rng.randint(12, 28)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            # Rough edges
            noise = (self.rng.rand(h, w) > 0.3).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, noise)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = (mask > 60).astype(np.uint8) * 255
            
        binary_mask = (mask > 127).astype(np.float32)
        return binary_mask

    def generate_object_removal_mask(
        self,
        shape: Tuple[int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        object_type: str = "letter_L"
    ) -> np.ndarray:
        """
        Type M09 / M10: Non-face object mask strictly outside face bounding box (IoU_face == 0).
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Place in background (e.g. top-left, top-right, or bottom border)
        margin = 20
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            # Place either to the left of face, right of face, or above face
            choices = []
            if fx1 > 60: choices.append("left")
            if (w - fx2) > 60: choices.append("right")
            if fy1 > 60: choices.append("top")
            loc = self.rng.choice(choices) if choices else "top"
            
            if loc == "left":
                ox1, oy1 = margin, self.rng.randint(margin, h - 80)
            elif loc == "right":
                ox1, oy1 = fx2 + margin, self.rng.randint(margin, h - 80)
            else: # top
                ox1, oy1 = self.rng.randint(margin, w - 80), margin
        else:
            ox1, oy1 = margin, margin

        if object_type == "letter_L":
            # Draw letter L
            cv2.line(mask, (ox1, oy1), (ox1, oy1 + 50), 255, thickness=12)
            cv2.line(mask, (ox1, oy1 + 50), (ox1 + 35, oy1 + 50), 255, thickness=12)
        else: # Generic object block/box
            cv2.rectangle(mask, (ox1, oy1), (ox1 + 45, oy1 + 45), 255, -1)
            
        return (mask > 127).astype(np.float32)

    def analyze_mask(
        self,
        mask: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Computes connected components, coverage metrics, and face-overlap statistics.
        """
        h, w = mask.shape[:2]
        binary = (mask > 0.5).astype(np.uint8)
        
        num_masked_pixels = int(np.sum(binary))
        image_coverage = float(num_masked_pixels / (h * w))
        
        face_coverage = 0.0
        face_overlap_pixels = 0
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
            face_crop = binary[fy1:fy2, fx1:fx2]
            face_overlap_pixels = int(np.sum(face_crop))
            face_coverage = float(face_overlap_pixels / face_area)
            
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        components = []
        for i in range(1, num_labels): # Skip background 0
            cx, cy, cw, ch, area = stats[i]
            components.append({
                "label": i,
                "bbox": (int(cx), int(cy), int(cx + cw), int(cy + ch)),
                "area_pixels": int(area),
                "centroid": (float(centroids[i][0]), float(centroids[i][1]))
            })
            
        return {
            "image_coverage": image_coverage,
            "face_coverage": face_coverage,
            "face_overlap_pixels": face_overlap_pixels,
            "num_components": len(components),
            "components": components,
            "is_face_region": face_overlap_pixels > 0,
            "is_pure_background": face_overlap_pixels == 0
        }
