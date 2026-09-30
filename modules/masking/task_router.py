"""
Unified Task Routing Layer.
Automatically inspects the mask profile and face analysis to route the input into
one of the four distinct research task modes:
1. MODE_1_FACIAL_INPAINTING
2. MODE_2_FACIAL_RESTORATION
3. MODE_3_MULTI_MASK
4. MODE_4_NON_FACE_OBJECT_REMOVAL
"""

from enum import Enum
from typing import Dict, Any, List, Optional


class TaskMode(str, Enum):
    MODE_1_FACIAL_INPAINTING = "MODE_1_FACIAL_INPAINTING"
    MODE_2_FACIAL_RESTORATION = "MODE_2_FACIAL_RESTORATION"
    MODE_3_MULTI_MASK = "MODE_3_MULTI_MASK"
    MODE_4_NON_FACE_OBJECT_REMOVAL = "MODE_4_NON_FACE_OBJECT_REMOVAL"


class TaskRouter:
    """
    Routes an image and mask to the correct operational pipeline.
    Prevents injecting facial identity embeddings into non-face backgrounds
    and distinguishes localized skin degradation from missing-organ inpainting.
    """

    def __init__(
        self,
        degradation_max_face_coverage: float = 0.12,
        multi_mask_min_components: int = 2
    ):
        self.degradation_max_face_coverage = degradation_max_face_coverage
        self.multi_mask_min_components = multi_mask_min_components

    def route(
        self,
        mask_profile: Dict[str, Any],
        face_info: Optional[Dict[str, Any]] = None,
        explicit_mode: Optional[str] = None,
        is_degradation_flag: bool = False
    ) -> Dict[str, Any]:
        """
        Determines the execution mode and configuration.
        
        Args:
            mask_profile: Output from MaskRegionProfiler
            face_info: Output from FaceDetector (or None if no face detected)
            explicit_mode: User or experiment override mode
            is_degradation_flag: Flag indicating user specified skin degradation/restoration
            
        Returns:
            Dict containing assigned mode, identity_conditioning_enabled, and routing rationale.
        """
        num_components = mask_profile.get("num_components", 0)
        face_overlap_pixels = mask_profile.get("face_overlap_pixels", 0)
        face_coverage = mask_profile.get("face_coverage", 0.0)
        has_face = face_info is not None and face_info.get("face_detected", False)

        # 1. Handle explicit override if provided
        if explicit_mode is not None:
            mode = TaskMode(explicit_mode)
            id_enabled = mode in [
                TaskMode.MODE_1_FACIAL_INPAINTING,
                TaskMode.MODE_2_FACIAL_RESTORATION,
                TaskMode.MODE_3_MULTI_MASK
            ] and has_face
            return {
                "task_mode": mode.value,
                "identity_conditioning_enabled": id_enabled,
                "structural_guidance_enabled": has_face,
                "routing_reason": f"Explicitly configured mode: {explicit_mode}",
                "component_routing": self._route_individual_components(mask_profile.get("components", []))
            }

        # 2. Case: Non-face object removal (Mask does not touch detected face or no face exists)
        if not has_face or face_overlap_pixels == 0:
            return {
                "task_mode": TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value,
                "identity_conditioning_enabled": False, # STRICTLY FALSE: No face tokens in background
                "structural_guidance_enabled": False,
                "routing_reason": "Mask is located in background/object region outside face (IoU_face == 0).",
                "component_routing": self._route_individual_components(mask_profile.get("components", []))
            }

        # 3. Case: Multiple disconnected mask components
        if num_components >= self.multi_mask_min_components:
            return {
                "task_mode": TaskMode.MODE_3_MULTI_MASK.value,
                "identity_conditioning_enabled": True,
                "structural_guidance_enabled": True,
                "routing_reason": f"Mask consists of {num_components} disconnected components requiring coordinated reconstruction.",
                "component_routing": self._route_individual_components(mask_profile.get("components", []))
            }

        # 4. Case: Localized facial degradation / restoration
        if is_degradation_flag or (face_coverage <= self.degradation_max_face_coverage and mask_profile.get("num_affected_landmarks", 0) <= 2):
            return {
                "task_mode": TaskMode.MODE_2_FACIAL_RESTORATION.value,
                "identity_conditioning_enabled": True,
                "structural_guidance_enabled": True,
                "routing_reason": f"Localized skin modification detected (face coverage={face_coverage*100:.1f}%). Underlying geometry preserved.",
                "component_routing": self._route_individual_components(mask_profile.get("components", []))
            }

        # 5. Case: Standard / Severe facial inpainting
        return {
            "task_mode": TaskMode.MODE_1_FACIAL_INPAINTING.value,
            "identity_conditioning_enabled": True,
            "structural_guidance_enabled": True,
            "routing_reason": f"Facial region missing/occluded (face coverage={face_coverage*100:.1f}%, affected landmarks={mask_profile.get('num_affected_landmarks')}).",
            "component_routing": self._route_individual_components(mask_profile.get("components", []))
        }

    def _route_individual_components(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determines per-component routing in multi-mask scenarios."""
        routed = []
        for comp in components:
            cat = comp.get("category", "FACIAL")
            if cat == "NON_FACE":
                comp_mode = TaskMode.MODE_4_NON_FACE_OBJECT_REMOVAL.value
                use_id = False
            else:
                comp_mode = TaskMode.MODE_1_FACIAL_INPAINTING.value
                use_id = True
                
            routed.append({
                "component_id": comp.get("component_id"),
                "category": cat,
                "sub_task_mode": comp_mode,
                "use_identity_conditioning": use_id
            })
        return routed
