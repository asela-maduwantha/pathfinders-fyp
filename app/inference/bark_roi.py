"""
Multiple Bark Candidate Extraction & Quality Gate Module.
Extracts candidate bark ROIs isolating the clearly visible trunk section
with considerable maximum width in the binary mask, and evaluates quality criteria.
"""
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

from app.config import (
    QUALITY_MIN_OCCUPANCY,
    QUALITY_MIN_SHARPNESS,
    QUALITY_MIN_CONTRAST,
    QUALITY_MIN_BRIGHTNESS,
    QUALITY_MAX_BRIGHTNESS,
    CANDIDATE_TARGET_COUNT,
    MIN_BARK_WIDTH_RATIO,
    MIN_BARK_HEIGHT_RATIO
)

NEUTRAL_PAD_RGB = (127, 127, 127)


def neutral_masked_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Create a crop with neutral gray background where mask is False.
    """
    output = np.empty_like(image_rgb)
    output[:, :] = np.asarray(NEUTRAL_PAD_RGB, dtype=np.uint8)
    output[mask] = image_rgb[mask]
    return output


def calculate_bark_quality(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate bark quality metrics on a candidate bark region.
    """
    if crop_rgb.size == 0 or not crop_mask.any():
        return {
            "occupancy": 0.0,
            "sharpness": 0.0,
            "contrast": 0.0,
            "brightness": 0.0,
            "quality_score": 0.0,
        }

    occupancy = float(crop_mask.mean())
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    pixels = gray[crop_mask]

    if len(pixels) == 0:
        brightness = 0.0
        contrast = 0.0
    else:
        brightness = float(pixels.mean())
        contrast = float(pixels.std())

    # Laplacian sharpness calculation on eroded mask to reduce edge artifacts
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    eroded = cv2.erode(
        crop_mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1
    ).astype(bool)

    sharp_mask = eroded if eroded.any() else crop_mask
    sharpness = float(laplacian[sharp_mask].var()) if sharp_mask.any() else 0.0

    exposure_score = float(np.clip(1.0 - abs(brightness - 127.5) / 127.5, 0.0, 1.0))

    # Composite Quality Score Formula
    quality_score = (
        0.35 * occupancy
        + 0.30 * float(np.clip(sharpness / 150.0, 0.0, 1.0))
        + 0.20 * float(np.clip(contrast / 60.0, 0.0, 1.0))
        + 0.15 * exposure_score
    )

    return {
        "occupancy": round(occupancy, 4),
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "brightness": round(brightness, 2),
        "quality_score": round(quality_score, 4),
    }


def evaluate_quality_gate(
    metrics: Dict[str, float],
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    orig_w: Optional[int] = None,
    orig_h: Optional[int] = None,
    min_occupancy: float = QUALITY_MIN_OCCUPANCY,
    min_sharpness: float = QUALITY_MIN_SHARPNESS,
    min_contrast: float = QUALITY_MIN_CONTRAST,
    min_brightness: float = QUALITY_MIN_BRIGHTNESS,
    max_brightness: float = QUALITY_MAX_BRIGHTNESS,
    min_bark_w_ratio: float = MIN_BARK_WIDTH_RATIO,
    min_bark_h_ratio: float = MIN_BARK_HEIGHT_RATIO
) -> Tuple[bool, Optional[str]]:
    """
    Check if candidate metrics and ROI dimensions satisfy all mandatory quality conditions.
    """
    reasons = []
    if crop_box and orig_w and orig_h and orig_w > 0 and orig_h > 0:
        w_ratio = float(crop_box[2] - crop_box[0]) / float(orig_w)
        h_ratio = float(crop_box[3] - crop_box[1]) / float(orig_h)
        if w_ratio < min_bark_w_ratio:
            reasons.append(f"Bark ROI too narrow ({w_ratio:.3f} < {min_bark_w_ratio:.3f})")
        if h_ratio < min_bark_h_ratio:
            reasons.append(f"Bark ROI too short ({h_ratio:.3f} < {min_bark_h_ratio:.3f})")

    if metrics["occupancy"] < min_occupancy:
        reasons.append(f"Low occupancy ({metrics['occupancy']:.2f} < {min_occupancy:.2f})")
    if metrics["sharpness"] < min_sharpness:
        reasons.append(f"Low sharpness ({metrics['sharpness']:.1f} < {min_sharpness:.1f})")
    if metrics["contrast"] < min_contrast:
        reasons.append(f"Low contrast ({metrics['contrast']:.1f} < {min_contrast:.1f})")
    if metrics["brightness"] < min_brightness:
        reasons.append(f"Too dark ({metrics['brightness']:.1f} < {min_brightness:.1f})")
    elif metrics["brightness"] > max_brightness:
        reasons.append(f"Too bright ({metrics['brightness']:.1f} > {max_brightness:.1f})")

    passed = len(reasons) == 0
    reason_str = "; ".join(reasons) if not passed else None
    return passed, reason_str


def get_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Get min/max coordinates (x1, y1, x2, y2) of mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def find_widest_clear_visible_trunk_span(
    mask: np.ndarray,
    min_width_ratio: float = 0.55
) -> Optional[Tuple[int, int, int, int]]:
    """
    Locates the clear visible trunk section having considerable maximum width within the binary mask.
    Calculates row-by-row mask width W(y), identifies maximum width W_max,
    and returns bounding box (x1, y1, x2, y2) of the primary continuous vertical band
    where W(y) >= min_width_ratio * W_max.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    y_min, y_max = ys.min(), ys.max()
    num_rows = y_max - y_min + 1
    row_widths = np.zeros(num_rows, dtype=np.float32)

    for y in range(y_min, y_max + 1):
        row_xs = xs[ys == y]
        if len(row_xs) > 0:
            row_widths[y - y_min] = len(row_xs)

    if len(row_widths) == 0 or row_widths.max() == 0:
        return get_mask_bbox(mask)

    # 1D Gaussian smoothing to reduce row width profile noise
    kernel_size = max(5, int(num_rows * 0.05) | 1)
    smoothed_widths = cv2.GaussianBlur(row_widths.reshape(-1, 1), (1, kernel_size), 0).flatten()

    w_max = float(smoothed_widths.max())
    w_thresh = w_max * min_width_ratio

    # Find rows meeting considerable maximum width threshold
    valid_indices = np.where(smoothed_widths >= w_thresh)[0]
    if len(valid_indices) == 0:
        return (int(xs.min()), int(y_min), int(xs.max()) + 1, int(y_max) + 1)

    # Find the largest continuous vertical band of maximum width
    split_indices = np.where(np.diff(valid_indices) > 1)[0]
    bands = np.split(valid_indices, split_indices + 1)
    largest_band = max(bands, key=len)

    band_y1 = y_min + int(largest_band[0])
    band_y2 = y_min + int(largest_band[-1]) + 1

    # Extract horizontal x bounds within this continuous widest span
    band_mask = mask[band_y1:band_y2]
    _, band_xs = np.where(band_mask)

    if len(band_xs) == 0:
        return (int(xs.min()), int(band_y1), int(xs.max()) + 1, int(band_y2))

    x1 = int(band_xs.min())
    x2 = int(band_xs.max()) + 1

    return (x1, band_y1, x2, band_y2)


def generate_candidate_rois(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_count: int = CANDIDATE_TARGET_COUNT
) -> List[Dict[str, Any]]:
    """
    Generate 3-5 candidate bark ROIs vertically inside the segmented trunk,
    prioritizing the clear visible section with considerable maximum width.
    """
    full_bbox = get_mask_bbox(mask)
    if full_bbox is None:
        return []

    img_height, img_width = mask.shape[:2]

    # 1. Primary Candidate: Clear Visible Trunk Section with Considerable Maximum Width
    widest_bbox = find_widest_clear_visible_trunk_span(mask, min_width_ratio=0.55) or full_bbox
    x1_w, y1_w, x2_w, y2_w = widest_bbox

    widest_crop_rgb = image_rgb[y1_w:y2_w, x1_w:x2_w].copy()
    widest_crop_mask = mask[y1_w:y2_w, x1_w:x2_w].copy()
    widest_masked = neutral_masked_crop(widest_crop_rgb, widest_crop_mask)
    widest_metrics = calculate_bark_quality(widest_masked, widest_crop_mask)
    widest_pass, widest_reason = evaluate_quality_gate(widest_metrics, crop_box=widest_bbox, orig_w=img_width, orig_h=img_height)

    raw_candidates = [{
        "candidate_id": 1,
        "type": "widest_clear_trunk_crop",
        "box": widest_bbox,
        "crop_rgb": widest_crop_rgb,
        "masked_crop": widest_masked,
        "crop_mask": widest_crop_mask,
        "metrics": widest_metrics,
        "accepted": widest_pass,
        "reason": widest_reason
    }]

    # 2. Vertical Sub-window Candidates centered within the clear visible maximum-width span
    span_height = y2_w - y1_w

    top_limit = y1_w + int(0.05 * span_height)
    bottom_limit = y2_w - int(0.05 * span_height)
    usable_height = max(1, bottom_limit - top_limit)

    window_height = min(usable_height, max(48, int(0.35 * span_height)))

    first_center = top_limit + window_height / 2.0
    last_center = bottom_limit - window_height / 2.0

    if first_center >= last_center:
        centers = [(top_limit + bottom_limit) / 2.0]
    else:
        centers = np.linspace(first_center, last_center, target_count)

    cand_idx = 2
    for center_y in centers:
        y1 = max(0, int(round(center_y - window_height / 2.0)))
        y2 = min(img_height, y1 + window_height)

        _, local_x = np.where(mask[y1:y2])
        if len(local_x) < 20:
            continue

        x5 = float(np.percentile(local_x, 5))
        x95 = float(np.percentile(local_x, 95))
        local_width = max(1.0, x95 - x5)
        local_center = float(np.median(local_x))

        crop_width = min(img_width, max(48, int(round(local_width * 1.10))))
        x1 = max(0, min(int(round(local_center - crop_width / 2.0)), img_width - crop_width))
        x2 = min(img_width, x1 + crop_width)

        w_box = (x1, y1, x2, y2)
        w_crop_rgb = image_rgb[y1:y2, x1:x2].copy()
        w_crop_mask = mask[y1:y2, x1:x2].copy()

        w_masked = neutral_masked_crop(w_crop_rgb, w_crop_mask)
        w_metrics = calculate_bark_quality(w_masked, w_crop_mask)
        w_pass, w_reason = evaluate_quality_gate(w_metrics, crop_box=w_box, orig_w=img_width, orig_h=img_height)

        raw_candidates.append({
            "candidate_id": cand_idx,
            "type": f"vertical_window_{cand_idx-1}",
            "box": w_box,
            "crop_rgb": w_crop_rgb,
            "masked_crop": w_masked,
            "crop_mask": w_crop_mask,
            "metrics": w_metrics,
            "accepted": w_pass,
            "reason": w_reason
        })
        cand_idx += 1

    return raw_candidates


def select_top_bark_candidates(
    candidates: List[Dict[str, Any]],
    max_select: int = 3
) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """
    Filter and rank accepted candidates by quality score.
    Returns:
    - bark_ready (bool)
    - selected_candidates (List, top 1-3)
    - failure_reason (Optional[str])
    """
    accepted = [c for c in candidates if c["accepted"]]

    if not accepted:
        return False, [], "No bark region passed the quality gate."

    # Rank accepted candidates by quality_score descending
    accepted_sorted = sorted(accepted, key=lambda c: c["metrics"]["quality_score"], reverse=True)
    selected = accepted_sorted[:max_select]

    return True, selected, None
