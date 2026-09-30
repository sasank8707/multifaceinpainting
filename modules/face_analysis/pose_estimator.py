"""
Head Pose Estimation Module.
Estimates 3D head pose (yaw, pitch, roll in degrees) using Perspective-n-Point (solvePnP)
with a canonical 3D facial landmark model. Computes pose compatibility metrics.
"""

import math
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple


class PoseEstimator:
    """
    Estimates 3D head orientation angles from detected facial landmarks.
    """

    # Canonical 3D facial model points (in mm, centered at nose bridge / tip)
    CANONICAL_3D_POINTS = np.array([
        [0.0, 0.0, 0.0],          # Nose tip (index 1)
        [0.0, -330.0, -65.0],     # Chin (index 152)
        [-225.0, 170.0, -135.0],  # Left eye outer corner (index 33)
        [225.0, 170.0, -135.0],   # Right eye outer corner (index 263)
        [-150.0, -150.0, -125.0], # Left mouth corner (index 61)
        [150.0, -150.0, -125.0]   # Right mouth corner (index 291)
    ], dtype=np.float64)

    # Corresponding MediaPipe landmark indices
    KEYPOINT_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self, sigma_yaw: float = 15.0, sigma_pitch: float = 15.0, sigma_roll: float = 10.0):
        self.sigma_yaw = sigma_yaw
        self.sigma_pitch = sigma_pitch
        self.sigma_roll = sigma_roll

    def estimate_pose(
        self,
        dense_landmarks_pixel: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Estimates yaw, pitch, and roll in degrees from 2D facial landmark coordinates.
        
        Args:
            dense_landmarks_pixel: array of shape (N, 2) or (N, 3) with (x, y) coordinates.
            image_shape: (height, width) of the image.
            
        Returns:
            Dict containing yaw, pitch, roll, and rotation vector.
        """
        h, w = image_shape[:2]
        
        # Extract corresponding 2D image points
        image_points = []
        for idx in self.KEYPOINT_INDICES:
            if idx < len(dense_landmarks_pixel):
                pt = dense_landmarks_pixel[idx]
                image_points.append([float(pt[0]), float(pt[1])])
            else:
                image_points.append([w / 2.0, h / 2.0])
                
        image_points = np.array(image_points, dtype=np.float64)

        # Approximate camera matrix assuming principal point at center
        focal_length = float(w)
        cam_matrix = np.array([
            [focal_length, 0.0, w / 2.0],
            [0.0, focal_length, h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.CANONICAL_3D_POINTS,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 0.0}

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Unpack angles (pitch, yaw, roll)
        raw_pitch = float(angles[0])
        raw_yaw = float(angles[1])
        raw_roll = float(angles[2])

        # Normalize pitch to [-90, 90] range relative to facing forward
        if raw_pitch > 90:
            pitch = raw_pitch - 180.0
        elif raw_pitch < -90:
            pitch = raw_pitch + 180.0
        else:
            pitch = raw_pitch
            
        yaw = raw_yaw
        roll = raw_roll

        return {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "confidence": 1.0
        }

    def compute_pose_similarity(
        self,
        pose1: Dict[str, float],
        pose2: Dict[str, float]
    ) -> float:
        """
        Computes exponential gaussian pose compatibility score S_pose in [0, 1].
        High score indicates similar head orientation between target and reference.
        """
        dyaw = pose1.get("yaw", 0.0) - pose2.get("yaw", 0.0)
        dpitch = pose1.get("pitch", 0.0) - pose2.get("pitch", 0.0)
        droll = pose1.get("roll", 0.0) - pose2.get("roll", 0.0)
        
        exp_arg = (
            - (dyaw ** 2) / (2.0 * (self.sigma_yaw ** 2))
            - (dpitch ** 2) / (2.0 * (self.sigma_pitch ** 2))
            - (droll ** 2) / (2.0 * (self.sigma_roll ** 2))
        )
        sim = float(np.exp(np.clip(exp_arg, -50.0, 0.0)))
        return round(sim, 4)
