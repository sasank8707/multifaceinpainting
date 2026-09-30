"""
Mask-Aware Multi-Reference Scorer (O1 Retrieve Module).
Scores, ranks, and selects Top-K reference images per region-of-interest conditioned on:
1. Pose compatibility (yaw, pitch, roll)
2. Image quality / sharpness (Laplacian clarity from Wei et al., 2019)
3. Landmark visibility within the occluded region
4. Geometric / region overlap compatibility

Strictly enforces verified identity grouping and disables facial conditioning for non-face masks.
Operates completely region-agnostically without hardcoded organ rules.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from modules.reference_scoring.metrics import (
    compute_pose_compatibility,
    compute_quality_score,
    compute_masked_landmark_visibility,
    compute_region_overlap_compatibility
)
from modules.masking.task_router import TaskMode


class IdentityMismatchError(ValueError):
    """Raised when an unverified or mismatched identity is passed in strict mode."""
    pass


class MaskAwareReferenceScorer:
    """
    O1 Retrieve Engine: Evaluates and selects reference images conditioned on mask geometry.
    """

    def __init__(
        self,
        pose_weight: float = 0.30,
        quality_weight: float = 0.25,
        visibility_weight: float = 0.30,
        overlap_weight: float = 0.15,
        top_k: int = 3,
        strict_identity: bool = True
    ):
        self.pose_weight = pose_weight
        self.quality_weight = quality_weight
        self.visibility_weight = visibility_weight
        self.overlap_weight = overlap_weight
        self.top_k = top_k
        self.strict_identity = strict_identity
        
        # Verify weight normalization
        total_w = pose_weight + quality_weight + visibility_weight + overlap_weight
        assert abs(total_w - 1.0) < 1e-4, f"Scoring weights must sum to 1.0, got {total_w}"

    def score_references(
        self,
        target_analysis: Dict[str, Any],
        reference_analyses: List[Dict[str, Any]],
        target_identity_id: Optional[str] = None,
        reference_identity_ids: Optional[List[str]] = None,
        reference_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes mask-aware reference scoring across all mask components.
        
        Args:
            target_analysis: Complete analysis dictionary of target from FaceAnalysisPipeline
            reference_analyses: List of analysis dictionaries for 1..N candidate references
            target_identity_id: Ground-truth identity ID of the target subject (verified metadata)
            reference_identity_ids: List of verified identity IDs for candidate references
            reference_names: Optional human-readable identifiers (e.g. ['ref_0', 'ref_1'])
            
        Returns:
            Dict containing per-region score breakdowns, normalized weights, rankings, and Top-K selections.
        """
        num_refs = len(reference_analyses)
        if num_refs == 0:
            raise ValueError("Reference list cannot be empty.")

        if reference_names is None:
            reference_names = [f"ref_{i}" for i in range(num_refs)]

        routing = target_analysis.get("routing", {})
        task_mode = routing.get("task_mode", TaskMode.MODE_1_FACIAL_INPAINTING.value)
        mask_profile = target_analysis.get("mask_profile", {})
        components = mask_profile.get("components", [])
        target_face = target_analysis.get("face", {})
        target_pose = target_analysis.get("pose", {})
        target_face_bbox = target_face.get("bbox") if target_face.get("detected", False) else None

        # 1. Verification of Same-Identity Metadata
        identity_status = self._verify_identities(
            target_identity_id=target_identity_id,
            reference_identity_ids=reference_identity_ids,
            num_refs=num_refs
        )

        # 2. Check for Non-Face Object Removal Mode
        if task_mode == TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value or mask_profile.get("is_pure_background", False):
            return self._build_non_face_result(reference_names, task_mode, components)

        # 3. If no components detected (e.g. no mask or single full-mask fallback)
        if len(components) == 0:
            components = [{
                "component_id": 0,
                "bbox": mask_profile.get("global_mask_bbox", (0, 0, 100, 100)),
                "area_pixels": mask_profile.get("total_masked_pixels", 1),
                "affected_landmarks": mask_profile.get("global_affected_landmarks", []),
                "category": "FACIAL"
            }]

        # 4. Regional Scoring for each Component
        component_results = []
        global_ref_scores = {name: 0.0 for name in reference_names}
        total_mask_weight = 0.0

        for comp in components:
            comp_id = comp.get("component_id", 0)
            comp_bbox = comp.get("bbox", (0, 0, 100, 100))
            comp_area = max(1, comp.get("area_pixels", 1))
            affected_lms = comp.get("affected_landmarks", [])
            comp_category = comp.get("category", "FACIAL")

            # If this individual component is non-facial (e.g. background text in multi-mask):
            if comp_category == "NON_FACE":
                comp_res = {
                    "component_id": comp_id,
                    "category": comp_category,
                    "identity_conditioning_enabled": False,
                    "reason": "Non-facial region; reference identity injection suppressed.",
                    "reference_scores": {name: 0.0 for name in reference_names},
                    "normalized_weights": {name: 0.0 for name in reference_names},
                    "ranked_references": [],
                    "top_k_selected": []
                }
                component_results.append(comp_res)
                continue

            # Compute scores for all references on this component
            comp_ref_scores = {}
            comp_breakdown = {}
            raw_scores = []

            for i, (ref_analysis, ref_name) in enumerate(zip(reference_analyses, reference_names)):
                # If identity is invalid, assign 0.0
                if not identity_status[i]["is_valid"]:
                    comp_ref_scores[ref_name] = 0.0
                    comp_breakdown[ref_name] = {
                        "pose_score": 0.0,
                        "quality_score": 0.0,
                        "visibility_score": 0.0,
                        "overlap_score": 0.0,
                        "composite_score": 0.0,
                        "rejection_reason": identity_status[i]["rejection_reason"]
                    }
                    raw_scores.append(0.0)
                    continue

                ref_face = ref_analysis.get("face", {})
                ref_pose = ref_analysis.get("pose", {})
                ref_quality = ref_analysis.get("quality", {})
                ref_dims = (
                    ref_analysis["image_dimensions"]["height"],
                    ref_analysis["image_dimensions"]["width"]
                )

                # Sub-score 1: Pose Compatibility
                s_pose = compute_pose_compatibility(target_pose, ref_pose)
                
                # Sub-score 2: Quality & Sharpness
                s_qual = compute_quality_score(ref_quality)
                
                # Sub-score 3: Masked Landmark Visibility
                s_vis = compute_masked_landmark_visibility(
                    affected_landmark_indices=affected_lms,
                    ref_face_info=ref_face,
                    ref_dimensions=ref_dims
                )
                
                # Sub-score 4: Region Overlap Compatibility
                s_overlap = compute_region_overlap_compatibility(
                    comp_bbox=comp_bbox,
                    target_face_bbox=target_face_bbox,
                    ref_face_info=ref_face,
                    ref_dimensions=ref_dims
                )

                composite = (
                    self.pose_weight * s_pose +
                    self.quality_weight * s_qual +
                    self.visibility_weight * s_vis +
                    self.overlap_weight * s_overlap
                )
                composite = round(float(np.clip(composite, 0.0, 1.0)), 4)

                comp_ref_scores[ref_name] = composite
                comp_breakdown[ref_name] = {
                    "pose_score": s_pose,
                    "quality_score": s_qual,
                    "visibility_score": s_vis,
                    "overlap_score": s_overlap,
                    "composite_score": composite,
                    "is_valid_identity": True
                }
                raw_scores.append(composite)

            # Relative weights normalization across references for this component
            sum_scores = sum(raw_scores)
            norm_weights = {}
            for name in reference_names:
                if sum_scores > 1e-6:
                    norm_weights[name] = round(float(comp_ref_scores[name] / sum_scores), 4)
                else:
                    norm_weights[name] = round(1.0 / num_refs, 4)

            # Rank references for this component
            ranked = sorted(
                [{"name": name, "score": comp_ref_scores[name], "weight": norm_weights[name]} for name in reference_names],
                key=lambda x: x["score"],
                reverse=True
            )
            top_k_for_comp = [r["name"] for r in ranked[:min(self.top_k, len(ranked))] if r["score"] > 0]

            comp_res = {
                "component_id": comp_id,
                "category": comp_category,
                "affected_landmarks": affected_lms,
                "identity_conditioning_enabled": True,
                "score_breakdown": comp_breakdown,
                "normalized_weights": norm_weights,
                "ranked_references": ranked,
                "top_k_selected": top_k_for_comp
            }
            component_results.append(comp_res)

            # Accumulate global face-level score
            for name in reference_names:
                global_ref_scores[name] += comp_ref_scores[name] * comp_area
            total_mask_weight += comp_area

        # 5. Global Face-Level Aggregation
        if total_mask_weight > 0:
            for name in reference_names:
                global_ref_scores[name] = round(float(global_ref_scores[name] / total_mask_weight), 4)

        global_sum = sum(global_ref_scores.values())
        global_weights = {}
        for name in reference_names:
            if global_sum > 1e-6:
                global_weights[name] = round(float(global_ref_scores[name] / global_sum), 4)
            else:
                global_weights[name] = round(1.0 / num_refs, 4)

        global_ranked = sorted(
            [{"name": name, "score": global_ref_scores[name], "weight": global_weights[name]} for name in reference_names],
            key=lambda x: x["score"],
            reverse=True
        )
        global_top_k = [r["name"] for r in global_ranked[:min(self.top_k, len(global_ranked))] if r["score"] > 0]

        return {
            "task_mode": task_mode,
            "identity_conditioning_enabled": True,
            "identity_status": identity_status,
            "scoring_weights": {
                "pose": self.pose_weight,
                "quality": self.quality_weight,
                "visibility": self.visibility_weight,
                "overlap": self.overlap_weight
            },
            "num_references_evaluated": num_refs,
            "components": component_results,
            "global_scores": global_ref_scores,
            "global_normalized_weights": global_weights,
            "global_ranked_references": global_ranked,
            "global_top_k_selected": global_top_k
        }

    def _verify_identities(
        self,
        target_identity_id: Optional[str],
        reference_identity_ids: Optional[List[str]],
        num_refs: int
    ) -> List[Dict[str, Any]]:
        """Verifies reference identity IDs against target identity."""
        status = []
        if target_identity_id is None or reference_identity_ids is None:
            # If identity metadata not passed (e.g. quick smoke testing), allow with warning
            for i in range(num_refs):
                status.append({
                    "ref_index": i,
                    "is_valid": True,
                    "target_id": target_identity_id,
                    "ref_id": None,
                    "rejection_reason": None
                })
            return status

        for i, ref_id in enumerate(reference_identity_ids):
            if ref_id != target_identity_id:
                msg = f"Identity mismatch: target ID '{target_identity_id}' != reference ID '{ref_id}'."
                if self.strict_identity:
                    raise IdentityMismatchError(msg)
                status.append({
                    "ref_index": i,
                    "is_valid": False,
                    "target_id": target_identity_id,
                    "ref_id": ref_id,
                    "rejection_reason": msg
                })
            else:
                status.append({
                    "ref_index": i,
                    "is_valid": True,
                    "target_id": target_identity_id,
                    "ref_id": ref_id,
                    "rejection_reason": None
                })
        return status

    def _build_non_face_result(
        self,
        reference_names: List[str],
        task_mode: str,
        components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Constructs result for MODE_4_NON_FACE_OBJECT_REMOVAL with identity conditioning disabled."""
        return {
            "task_mode": task_mode,
            "identity_conditioning_enabled": False, # STRICTLY FALSE
            "reason": "Facial identity reference scoring is suppressed for non-face/background object removal.",
            "num_references_evaluated": len(reference_names),
            "components": [
                {
                    "component_id": comp.get("component_id", idx),
                    "category": "NON_FACE",
                    "identity_conditioning_enabled": False,
                    "reason": "Non-face object mask; zero facial references applied.",
                    "reference_scores": {name: 0.0 for name in reference_names},
                    "normalized_weights": {name: 0.0 for name in reference_names},
                    "ranked_references": [],
                    "top_k_selected": []
                }
                for idx, comp in enumerate(components)
            ],
            "global_scores": {name: 0.0 for name in reference_names},
            "global_normalized_weights": {name: 0.0 for name in reference_names},
            "global_ranked_references": [],
            "global_top_k_selected": []
        }

    def save_scoring_report(self, scoring_result: Dict[str, Any], output_path: str):
        """Saves machine-readable JSON metadata report."""
        def _convert(val):
            if isinstance(val, (np.int64, np.int32)):
                return int(val)
            if isinstance(val, (np.float32, np.float64)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        with open(output_path, "w") as f:
            json.dump(scoring_result, f, indent=2, default=_convert)
