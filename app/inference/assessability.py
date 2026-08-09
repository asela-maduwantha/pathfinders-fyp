"""
Light Tree-Usability Safety Gate & Prominent Foreground Protection Module.
Evaluates proposals from YOLO26s-seg model trained on cleaned dataset.
Rejects only clearly unusable candidates (tiny, extreme thin sliver, severely fragmented,
almost zero mask area, obviously malformed mask) while strictly protecting prominent
foreground trunks regardless of border contact, partial clipping, or unusual dimensions.
"""
from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np

from app.config import (
    PROMINENT_MIN_WIDTH_RATIO,
    PROMINENT_MIN_HEIGHT_RATIO,
    PROMINENT_MIN_BBOX_AREA_RATIO,
    MIN_TINY_ISLAND_AREA_PX
)


def remove_tiny_mask_islands(
    mask: np.ndarray,
    min_area: int = MIN_TINY_ISLAND_AREA_PX
) -> np.ndarray:
    """
    Remove tiny segmentation island components around leaves/background,
    preserving the primary connected trunk mask.
    """
    if mask is None or not mask.any():
        return mask

    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 2:  # Background + 1 component
        return mask

    cleaned_mask = np.zeros_like(mask, dtype=bool)
    largest_label = 1
    max_area = 0

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            largest_label = label_idx

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area or label_idx == largest_label:
            cleaned_mask[labels == label_idx] = True

    return cleaned_mask


def calculate_foreground_importance_score(
    width_ratio: float,
    height_ratio: float,
    bbox_area_ratio: float,
    largest_component_ratio: float
) -> float:
    """
    Calculate composite foreground importance score for ranking and diagnostics.
    """
    score = (
        0.35 * min(width_ratio / 0.20, 1.0)
        + 0.30 * min(height_ratio / 0.70, 1.0)
        + 0.20 * min(bbox_area_ratio / 0.10, 1.0)
        + 0.15 * min(largest_component_ratio, 1.0)
    )
    return float(score)


def assess_trunk_visibility(
    mask: np.ndarray,
    orig_w: int,
    orig_h: int
) -> Dict[str, Any]:
    """
    Evaluates light bark-usability safety gate for a candidate tree trunk mask:
    - Calculates bbox_width_ratio, bbox_height_ratio, bbox_area_ratio, mask_area_ratio,
      mask_occupancy, largest_component_ratio, median_mask_width_ratio, touches_frame_edge.
    - Protects prominent foreground trunks from rejection due to border contact, partial clipping,
      moderately low occupancy, or unusual dimensions.
    - Rejects only clearly unusable candidates (tiny, thin sliver, fragmented, empty/malformed).
    """
    if mask is None or not mask.any() or orig_w <= 0 or orig_h <= 0:
        return {
            "passed": False,
            "is_prominent_foreground": False,
            "touches_frame_edge": False,
            "foreground_score": 0.0,
            "cleaned_mask": mask,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "bbox_area_ratio": 0.0,
            "mask_area_ratio": 0.0,
            "median_mask_width_ratio": 0.0,
            "mask_occupancy": 0.0,
            "largest_component_ratio": 0.0,
            "reasons": ["empty-mask"],
            "reason_str": "empty-mask"
        }

    # Clean tiny isolated mask islands while preserving primary trunk component
    cleaned_mask = remove_tiny_mask_islands(mask)
    ys, xs = np.where(cleaned_mask)

    if len(xs) == 0:
        return {
            "passed": False,
            "is_prominent_foreground": False,
            "touches_frame_edge": False,
            "foreground_score": 0.0,
            "cleaned_mask": cleaned_mask,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "bbox_area_ratio": 0.0,
            "mask_area_ratio": 0.0,
            "median_mask_width_ratio": 0.0,
            "mask_occupancy": 0.0,
            "largest_component_ratio": 0.0,
            "reasons": ["empty-mask"],
            "reason_str": "empty-mask"
        }

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    bbox_width = float(x2 - x1)
    bbox_height = float(y2 - y1)
    bbox_area = float(bbox_width * bbox_height)
    mask_area = float(cleaned_mask.sum())
    total_img_area = float(orig_w * orig_h)

    width_ratio = bbox_width / float(orig_w)
    height_ratio = bbox_height / float(orig_h)
    bbox_area_ratio = bbox_area / total_img_area
    mask_area_ratio = mask_area / total_img_area

    # Frame-edge contact check (diagnostics only — NOT a rejection criterion)
    touches_frame_edge = bool(x1 <= 2 or y1 <= 2 or x2 >= orig_w - 2 or y2 >= orig_h - 2)

    # Median Mask Width calculation across active rows
    row_widths = []
    for y in range(y1, y2):
        row_xs = xs[ys == y]
        if len(row_xs) > 0:
            row_widths.append(float(row_xs.max() - row_xs.min() + 1))

    median_mask_width = float(np.median(row_widths)) if row_widths else 0.0
    median_mask_width_ratio = median_mask_width / float(orig_w)

    # Mask Occupancy
    mask_occupancy = mask_area / max(bbox_area, 1.0)

    # Mask Continuity (Largest Component Ratio)
    mask_u8 = cleaned_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_area = float(component_areas.max())
    else:
        largest_component_area = 0.0

    largest_component_ratio = largest_component_area / max(mask_area, 1.0)

    # Composite Foreground Importance Score
    fg_score = calculate_foreground_importance_score(
        width_ratio=width_ratio,
        height_ratio=height_ratio,
        bbox_area_ratio=bbox_area_ratio,
        largest_component_ratio=largest_component_ratio
    )

    # Check Prominent Foreground Trunk Protection Rule (Requirement 7)
    is_prominent_foreground = bool(
        (width_ratio >= PROMINENT_MIN_WIDTH_RATIO and height_ratio >= PROMINENT_MIN_HEIGHT_RATIO)
        or bbox_area_ratio >= PROMINENT_MIN_BBOX_AREA_RATIO
    )

    reasons = []

    # Light Usability Safety Gate Rejection Rules
    if is_prominent_foreground:
        # Prominent trunks: Only reject if clearly invalid or unusable
        if mask_area < 100:
            reasons.append("invalid-mask")
        elif largest_component_ratio < 0.30:
            reasons.append("fragmented")
        elif mask_occupancy < 0.03:
            reasons.append("invalid-mask")
    else:
        # Normal candidates: Reject only clearly unusable predictions
        if mask_area < 100 or mask_area_ratio < 0.0003:
            reasons.append("tiny")
        elif width_ratio < 0.015 and bbox_area_ratio < 0.001:
            reasons.append("tiny")
        elif median_mask_width_ratio < 0.008 or (median_mask_width_ratio < 0.01 and width_ratio < 0.02):
            reasons.append("thin")
        elif largest_component_ratio < 0.40:
            reasons.append("fragmented")
        elif mask_occupancy < 0.05:
            reasons.append("invalid-mask")

    passed = len(reasons) == 0

    return {
        "passed": passed,
        "is_prominent_foreground": is_prominent_foreground,
        "touches_frame_edge": touches_frame_edge,
        "foreground_score": round(fg_score, 4),
        "cleaned_mask": cleaned_mask,
        "width_ratio": round(width_ratio, 4),
        "height_ratio": round(height_ratio, 4),
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "mask_area_ratio": round(mask_area_ratio, 4),
        "median_mask_width_ratio": round(median_mask_width_ratio, 4),
        "mask_occupancy": round(mask_occupancy, 4),
        "largest_component_ratio": round(largest_component_ratio, 4),
        "reasons": reasons,
        "reason_str": "; ".join(reasons) if reasons else "Passed"
    }


def evaluate_assessability(
    mask: np.ndarray,
    orig_w: int,
    orig_h: int
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates assessability of a candidate tree trunk mask (backward-compatible interface).
    """
    res = assess_trunk_visibility(mask, orig_w, orig_h)
    return res["passed"], res
