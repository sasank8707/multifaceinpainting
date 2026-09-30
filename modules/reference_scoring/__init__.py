"""
Reference scoring package (O1 Retrieve).
"""

from modules.reference_scoring.scorer import MaskAwareReferenceScorer, IdentityMismatchError
from modules.reference_scoring.metrics import (
    compute_pose_compatibility,
    compute_quality_score,
    compute_masked_landmark_visibility,
    compute_region_overlap_compatibility
)

__all__ = [
    "MaskAwareReferenceScorer",
    "IdentityMismatchError",
    "compute_pose_compatibility",
    "compute_quality_score",
    "compute_masked_landmark_visibility",
    "compute_region_overlap_compatibility",
]
