"""
Geometric transformation, aspect-ratio preserving padding/resizing,
ROI extraction, and pixel-invariance verification.
Ensures output resolution strictly preserves original input dimensions.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional


def resize_and_pad(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    interpolation: int = cv2.INTER_LINEAR,
    pad_value: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resizes image to fit target_size (H_t, W_t) while maintaining original aspect ratio.
    Applies symmetric zero-padding.
    
    Returns:
        padded_image: shape (H_t, W_t, C) or (H_t, W_t)
        pad_info: dictionary containing scale and padding offsets for exact inverse restoration.
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    if image.ndim == 3:
        padded = np.pad(
            resized,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=pad_value
        )
    else:
        padded = np.pad(
            resized,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value
        )
        
    pad_info = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "scale": scale,
    }
    return padded, pad_info


def unpad_and_restore(
    padded_image: np.ndarray,
    pad_info: Dict[str, Any],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Reverses padding and scales image back to original dimensions (orig_h, orig_w).
    """
    pad_top = pad_info["pad_top"]
    new_h = pad_info["new_h"]
    pad_left = pad_info["pad_left"]
    new_w = pad_info["new_w"]
    orig_h = pad_info["orig_h"]
    orig_w = pad_info["orig_w"]
    
    # Crop out the unpadded region
    cropped = padded_image[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    
    # Resize back to exact original dimensions
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=interpolation)
    return restored


def extract_roi(
    image: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float = 0.35,
    square: bool = True
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Extracts bounding box around mask with contextual margin.
    
    Returns:
        roi_image: Cropped contextual region
        roi_mask: Cropped mask within ROI
        bbox: (x1, y1, x2, y2) coordinates in original image space
    """
    h, w = image.shape[:2]
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback to full image if mask is empty
        return image.copy(), mask.copy(), (0, 0, w, h)
        
    all_points = np.vstack(contours)
    x, y, bw, bh = cv2.boundingRect(all_points)
    
    # Expand with margin
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)
    
    if square:
        rw = x2 - x1
        rh = y2 - y1
        side = max(rw, rh)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)
        # Re-adjust if bounded by image edge
        if (x2 - x1) < side and x1 > 0:
            x1 = max(0, x2 - side)
        if (y2 - y1) < side and y1 > 0:
            y1 = max(0, y2 - side)

    roi_image = image[y1:y2, x1:x2].copy()
    roi_mask = mask[y1:y2, x1:x2].copy()
    return roi_image, roi_mask, (x1, y1, x2, y2)


def paste_roi(
    base_image: np.ndarray,
    reconstructed_roi: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask_roi: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Pastes reconstructed ROI back into base image at bounding box coordinates.
    If mask_roi is provided, only updates pixels where mask_roi > 0.5.
    """
    x1, y1, x2, y2 = bbox
    result = base_image.copy()
    
    roi_h, roi_w = y2 - y1, x2 - x1
    if reconstructed_roi.shape[:2] != (roi_h, roi_w):
        reconstructed_roi = cv2.resize(reconstructed_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        
    if mask_roi is not None:
        if mask_roi.shape[:2] != (roi_h, roi_w):
            mask_roi = cv2.resize(mask_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        m = (mask_roi > 0.5).astype(np.float32)
        if base_image.ndim == 3 and m.ndim == 2:
            m = np.expand_dims(m, axis=-1)
        patch = (reconstructed_roi * m + result[y1:y2, x1:x2] * (1.0 - m)).astype(result.dtype)
        result[y1:y2, x1:x2] = patch
    else:
        result[y1:y2, x1:x2] = reconstructed_roi
        
    return result


def verify_unmasked_invariance(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    tolerance: float = 1e-5
) -> Tuple[bool, float, int]:
    """
    Verifies that unmasked pixels (where mask == 0) are strictly unchanged.
    Returns:
        passed: True if maximum unmasked difference <= tolerance
        max_diff: Maximum absolute difference on unmasked pixels
        violated_pixels: Number of unmasked pixels modified
    """
    assert original.shape == reconstructed.shape, "Shape mismatch between original and reconstructed"
    unmasked = (mask < 0.5).astype(bool)
    if original.ndim == 3 and unmasked.ndim == 2:
        unmasked = np.repeat(np.expand_dims(unmasked, -1), original.shape[-1], axis=-1)
        
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    unmasked_diff = diff[unmasked]
    
    if unmasked_diff.size == 0:
        return True, 0.0, 0
        
    max_diff = float(np.max(unmasked_diff))
    violated = int(np.sum(unmasked_diff > tolerance))
    passed = max_diff <= tolerance
    return passed, max_diff, violated
