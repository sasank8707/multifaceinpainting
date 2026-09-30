"""
Pretrained Frozen Face and Landmark Detector.
Extracts 3D facial mesh landmarks and canonical 68-point landmarks using MediaPipe.
Operates completely frozen without gradients or trainable weights.
"""

import os
import numpy as np
import cv2
from typing import Optional, Dict, Any, List, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canonical 68-point landmark indices mapped from MediaPipe 468/478 mesh
# Reference: Canonical Dlib 68 mapping on MediaPipe mesh
CANONICAL_68_INDICES = [
    # Jawline (0-16)
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # Right eyebrow (17-21)
    70, 63, 105, 66, 107,
    # Left eyebrow (22-26)
    336, 296, 334, 293, 300,
    # Nose bridge & tip (27-35)
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    # Right eye (36-41)
    33, 160, 158, 133, 153, 144,
    # Left eye (42-47)
    362, 385, 387, 263, 373, 380,
    # Outer lip (48-59)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner lip (60-67)
    78, 81, 13, 311, 308, 402, 14, 178
]


class FaceDetector:
    """
    Frozen face and landmark detector.
    Provides robust bounding boxes, 478 dense landmarks, and 68 canonical landmarks.
    """
    
    def __init__(self, model_path: str = "models/pretrained/face_landmarker.task", max_faces: int = 4):
        self.model_path = model_path
        self.max_faces = max_faces
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker model not found at {self.model_path}. "
                "Ensure Phase 0 model assets are downloaded."
            )
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects face and landmarks in an image (RGB or BGR numpy array).
        Returns None if no face is detected.
        """
        h, w = image.shape[:2]
        # MediaPipe expects RGB format
        if image.ndim == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self.detector.detect(mp_image)
        
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None
            
        landmarks_raw = results.face_landmarks[0]
        
        # Extract dense landmarks (x, y, z)
        dense_normalized = []
        dense_pixel = []
        for lm in landmarks_raw:
            dense_normalized.append([float(lm.x), float(lm.y), float(lm.z)])
            dense_pixel.append([float(lm.x * w), float(lm.y * h), float(lm.z * w)])
            
        dense_pixel_arr = np.array(dense_pixel, dtype=np.float32)
        
        # Face bounding box with 5% contextual padding
        min_x = float(np.min(dense_pixel_arr[:, 0]))
        min_y = float(np.min(dense_pixel_arr[:, 1]))
        max_x = float(np.max(dense_pixel_arr[:, 0]))
        max_y = float(np.max(dense_pixel_arr[:, 1]))
        
        fw = max_x - min_x
        fh = max_y - min_y
        pad_x = fw * 0.05
        pad_y = fh * 0.05
        
        bbox = (
            int(max(0, min_x - pad_x)),
            int(max(0, min_y - pad_y)),
            int(min(w, max_x + pad_x)),
            int(min(h, max_y + pad_y))
        )
        
        # Canonical 68-point landmarks
        canonical_68 = []
        for idx in CANONICAL_68_INDICES:
            if idx < len(dense_pixel):
                canonical_68.append(dense_pixel[idx][:2])
            else:
                canonical_68.append([0.0, 0.0])
                
        return {
            "face_detected": True,
            "bbox": bbox,
            "face_area": int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            "num_landmarks": len(dense_pixel),
            "dense_landmarks_normalized": dense_normalized,
            "dense_landmarks_pixel": dense_pixel,
            "canonical_68_pixel": canonical_68,
            "image_dimensions": {"height": h, "width": w},
            "num_faces_detected": len(results.face_landmarks),
        }
