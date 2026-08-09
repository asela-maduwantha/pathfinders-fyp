from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

from app.maturity.config import (
    CLOSEUP_ROW_COUNT,
    MAX_HALF_ANGLE_DEG,
    MAX_PLAUSIBLE_DBH_CM,
    MAX_RUN_WIDTH_FRACTION,
    MIN_MASK_PIXELS,
    MIN_ROW_DEPTH_FRACTION,
    MIN_RUN_WIDTH_PX,
    MIN_VALID_ROWS,
    NORMAL_ROW_COUNT,
)


def as_numpy(value):
    import torch

    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def as_2d(value) -> np.ndarray:
    value = np.squeeze(as_numpy(value))
    if value.ndim != 2:
        raise ValueError(f"Expected HxW array, received {value.shape}")
    return value.astype(np.float32)


def as_k(value) -> np.ndarray:
    value = np.squeeze(as_numpy(value)).astype(np.float32)
    if value.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics, received {value.shape}")
    return value


def row_runs(row_mask, min_width: int = 1) -> list[tuple[int, int]]:
    values = np.asarray(row_mask, dtype=np.int16)
    padded = np.pad(values, (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start + 1 >= min_width
    ]


def largest_component(mask, target_x: float | None = None) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    if count <= 1:
        return mask.astype(bool)

    height, width = mask.shape
    if target_x is None:
        target_x = width / 2.0

    best_id = None
    best_score = -np.inf

    for component_id in range(1, count):
        _, _, _component_width, component_height, area = stats[component_id]
        centre_x, _ = centroids[component_id]
        area_score = min(area / max(0.35 * width * height, 1), 1.0)
        vertical_score = min(component_height / max(0.65 * height, 1), 1.0)
        centre_score = 1.0 - min(abs(centre_x - target_x) / max(width / 2.0, 1), 1.0)
        score = 0.50 * area_score + 0.30 * vertical_score + 0.20 * centre_score

        if score > best_score:
            best_score = score
            best_id = component_id

    return labels == best_id


def clean_mask(mask, target_x: float | None = None) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    height, width = mask.shape
    kernel_size = max(3, int(round(min(height, width) * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    open_size = max(3, kernel_size // 2 * 2 + 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return largest_component(mask, target_x=target_x).astype(bool)


def mask_quality(mask) -> dict:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    ys, xs = np.where(mask)

    if len(xs) < MIN_MASK_PIXELS:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "too_few_pixels",
            "close_up": False,
            "median_width_fraction": 0.0,
        }

    y_top = int(ys.min())
    y_bottom = int(ys.max())
    centres = []
    widths = []
    nonempty = 0
    multiple = 0

    for y in range(y_top, y_bottom + 1):
        runs = row_runs(mask[y], min_width=2)
        if not runs:
            continue

        nonempty += 1
        if len(runs) > 1:
            multiple += 1

        run = max(runs, key=lambda item: item[1] - item[0] + 1)
        centres.append((run[0] + run[1]) / 2.0)
        widths.append(run[1] - run[0] + 1)

    vertical_coverage = (y_bottom - y_top + 1) / float(height)
    row_continuity = nonempty / float(max(y_bottom - y_top + 1, 1))
    multi_fraction = multiple / float(max(nonempty, 1))
    median_width_fraction = float(np.median(widths) / width) if widths else 0.0

    if len(centres) >= 5:
        centres_array = np.asarray(centres, dtype=np.float32)
        smooth = cv2.GaussianBlur(centres_array.reshape(-1, 1), (1, 5), 0).reshape(-1)
        roughness = float(np.median(np.abs(centres_array - smooth)) / max(width, 1))
    else:
        roughness = 1.0

    close_up = bool(
        median_width_fraction >= 0.38
        or mask[:, 0].mean() > 0.04
        or mask[:, -1].mean() > 0.04
    )

    area_fraction = float(mask.mean())
    reasons = []
    if not (0.004 <= area_fraction <= 0.92):
        reasons.append("area")
    if vertical_coverage < 0.28:
        reasons.append("vertical")
    if row_continuity < 0.62:
        reasons.append("continuity")
    if multi_fraction > 0.48:
        reasons.append("fragmentation")
    if roughness > 0.20:
        reasons.append("roughness")
    if not (0.01 <= median_width_fraction <= 0.94):
        reasons.append("width")

    score = (
        0.25 * min(vertical_coverage / 0.75, 1.0)
        + 0.25 * row_continuity
        + 0.20 * (1.0 - min(multi_fraction, 1.0))
        + 0.15 * (1.0 - min(roughness / 0.18, 1.0))
        + 0.15 * (1.0 - min(abs(median_width_fraction - 0.30) / 0.68, 1.0))
    )

    return {
        "pass": len(reasons) == 0,
        "score": float(np.clip(score, 0.0, 1.0)),
        "reason": "ok" if not reasons else "|".join(reasons),
        "close_up": close_up,
        "area_fraction": area_fraction,
        "vertical_coverage": vertical_coverage,
        "row_continuity": row_continuity,
        "multi_run_fraction": multi_fraction,
        "roughness": roughness,
        "median_width_fraction": median_width_fraction,
    }


def fallback_box(width: int, height: int) -> list[int]:
    return [
        int(0.04 * width),
        int(0.01 * height),
        int(0.96 * width),
        int(0.99 * height),
    ]


def expand_box(
    box,
    width: int,
    height: int,
    x_fraction: float = 0.10,
    y_fraction: float = 0.05,
) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    return [
        max(0.0, x1 - x_fraction * box_width),
        max(0.0, y1 - y_fraction * box_height),
        min(width - 1.0, x2 + x_fraction * box_width),
        min(height - 1.0, y2 + y_fraction * box_height),
    ]


def detection_score(box, confidence: float, width: int, height: int) -> float:
    x1, y1, x2, y2 = map(float, box)
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    area = box_width * box_height / max(width * height, 1)
    centre_x = (x1 + x2) / 2.0
    centre_score = 1.0 - min(abs(centre_x - width / 2.0) / max(width / 2.0, 1), 1.0)
    vertical_score = min(box_height / max(height * 0.70, 1), 1.0)
    area_score = 1.0 - min(abs(area - 0.35) / 0.68, 1.0)
    return 0.40 * float(confidence) + 0.25 * centre_score + 0.20 * vertical_score + 0.15 * area_score


def normalise_intrinsics(K, width: int, height: int) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32).copy()
    if K[0, 0] < 10:
        K[0, 0] *= width
        K[0, 2] *= width
    if K[1, 1] < 10:
        K[1, 1] *= height
        K[1, 2] *= height
    return K


def intrinsics_diagnostics(K, width: int, height: int) -> tuple[np.ndarray, bool, float]:
    K = normalise_intrinsics(K, width, height)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    valid = fx > 0 and fy > 0
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * fx))) if valid else np.nan
    valid = (
        valid
        and 15.0 <= fov_x <= 135.0
        and -0.15 * width <= cx <= 1.15 * width
        and -0.15 * height <= cy <= 1.15 * height
    )
    return K, valid, fov_x


def expected_trunk_centre(mask) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    centres = []
    rows = []

    for y in range(height):
        runs = row_runs(mask[y], min_width=MIN_RUN_WIDTH_PX)
        if not runs:
            continue
        run = max(runs, key=lambda item: item[1] - item[0] + 1)
        centres.append((run[0] + run[1]) / 2.0)
        rows.append(y)

    if len(centres) < 4:
        return np.full(height, width / 2.0, dtype=np.float32)

    centres = np.asarray(centres, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.float32)
    degree = 2 if len(centres) >= 12 else 1

    try:
        coefficients = np.polyfit(rows, centres, degree)
        predicted = np.polyval(coefficients, np.arange(height))
    except Exception:
        predicted = np.interp(np.arange(height), rows, centres)

    return np.clip(predicted, 0, width - 1).astype(np.float32)


def select_run_near_centre(row_mask, centre_x: float) -> tuple[int, int] | None:
    runs = row_runs(row_mask, min_width=MIN_RUN_WIDTH_PX)
    if not runs:
        return None

    containing = [run for run in runs if run[0] <= centre_x <= run[1]]
    if containing:
        return max(containing, key=lambda item: item[1] - item[0] + 1)
    return min(runs, key=lambda run: abs((run[0] + run[1]) / 2.0 - centre_x))


def robust_mad_keep(values, threshold: float = 3.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    keep = valid.copy()

    if valid.sum() < 4:
        return keep

    median = np.median(values[valid])
    mad = np.median(np.abs(values[valid] - median))
    if mad <= 1e-9:
        return keep

    robust_z = 0.6745 * np.abs(values - median) / mad
    keep &= robust_z <= threshold
    return keep


def robust_cylinder_dbh(depth, mask, K) -> dict:
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    K, K_valid, fov_x = intrinsics_diagnostics(K, width, height)
    mask_info = mask_quality(mask)
    centreline = expected_trunk_centre(mask)
    ys, _ = np.where(mask)

    if len(ys) < MIN_MASK_PIXELS:
        return {"success": False, "reason": "insufficient mask", "rows": []}

    y_top = int(ys.min())
    y_bottom = int(ys.max())
    close_up = bool(mask_info["close_up"])

    if close_up:
        start_fraction = 0.24
        end_fraction = 0.76
        row_count = CLOSEUP_ROW_COUNT
    else:
        start_fraction = 0.36
        end_fraction = 0.68
        row_count = NORMAL_ROW_COUNT

    candidate_rows = np.unique(
        np.rint(
            np.linspace(
                y_top + start_fraction * (y_bottom - y_top),
                y_top + end_fraction * (y_bottom - y_top),
                row_count,
            )
        ).astype(int)
    )

    fx = float(K[0, 0])
    rows = []

    for y in candidate_rows:
        if y < 0 or y >= height:
            continue

        run = select_run_near_centre(mask[y], centreline[y])
        if run is None:
            continue

        left, right = run
        width_px = float(right - left + 1)
        width_fraction = width_px / float(width)
        if width_px < MIN_RUN_WIDTH_PX or width_fraction > MAX_RUN_WIDTH_FRACTION:
            continue

        inset = max(1, int(round(0.28 * width_px)))
        centre_left = min(right, left + inset)
        centre_right = max(centre_left + 1, right - inset)
        depth_values = depth[y, centre_left : centre_right + 1]
        valid_depth = np.isfinite(depth_values) & (depth_values > 0)
        valid_fraction = float(valid_depth.mean()) if depth_values.size else 0.0
        if valid_fraction < MIN_ROW_DEPTH_FRACTION:
            continue

        front_depth_m = float(np.median(depth_values[valid_depth]))
        half_angle = math.atan(width_px / max(2.0 * fx, 1e-6))
        half_angle_deg = math.degrees(half_angle)
        sine = math.sin(half_angle)
        cylinder_dbh_cm = 200.0 * front_depth_m * sine / max(1.0 - sine, 1e-6)

        rows.append(
            {
                "y": int(y),
                "left": int(left),
                "right": int(right),
                "width_px": width_px,
                "front_depth_m": front_depth_m,
                "cylinder_dbh_cm": cylinder_dbh_cm,
                "half_angle_deg": half_angle_deg,
                "valid_depth_fraction": valid_fraction,
            }
        )

    if len(rows) < MIN_VALID_ROWS:
        return {"success": False, "reason": "too few valid rows", "rows": rows}

    row_df = pd.DataFrame(rows)
    keep = np.ones(len(row_df), dtype=bool)
    keep &= robust_mad_keep(np.log(np.clip(row_df["width_px"], 1e-6, None)))
    keep &= robust_mad_keep(np.log(np.clip(row_df["front_depth_m"], 1e-6, None)))
    keep &= robust_mad_keep(np.log(np.clip(row_df["cylinder_dbh_cm"], 1e-6, None)))
    keep &= row_df["half_angle_deg"].to_numpy() <= MAX_HALF_ANGLE_DEG
    kept = row_df.loc[keep].copy()

    if len(kept) < MIN_VALID_ROWS:
        return {"success": False, "reason": "rows rejected by quality", "rows": rows}

    dbh_values = kept["cylinder_dbh_cm"].to_numpy(dtype=float)
    distance_values = kept["front_depth_m"].to_numpy(dtype=float)
    predicted_dbh_cm = float(np.median(dbh_values))
    predicted_distance_m = float(np.median(distance_values))
    row_cv = float(np.std(dbh_values) / max(np.mean(dbh_values), 1e-6))

    quality_reasons = []
    if not mask_info["pass"]:
        quality_reasons.append("mask")
    if not K_valid:
        quality_reasons.append("intrinsics")
    if row_cv > 0.30:
        quality_reasons.append("row spread")
    if not (1.0 <= predicted_dbh_cm <= MAX_PLAUSIBLE_DBH_CM):
        quality_reasons.append("implausible DBH")

    return {
        "success": True,
        "reason": "ok",
        "quality": "high" if not quality_reasons else "review",
        "quality_reasons": "ok" if not quality_reasons else "|".join(quality_reasons),
        "close_up": close_up,
        "predicted_dbh_cm": predicted_dbh_cm,
        "predicted_distance_m": predicted_distance_m,
        "trunk_width_px": float(np.median(kept["width_px"])),
        "rows_before": int(len(row_df)),
        "rows_used": int(len(kept)),
        "row_cv": row_cv,
        "dbh_row_std_cm": float(np.std(dbh_values)),
        "dbh_row_p10_cm": float(np.percentile(dbh_values, 10)),
        "dbh_row_p90_cm": float(np.percentile(dbh_values, 90)),
        "distance_row_std_m": float(np.std(distance_values)),
        "half_angle_deg": float(np.median(kept["half_angle_deg"])),
        "fov_x_deg": fov_x,
        "mask_score": mask_info["score"],
        "rows": kept.to_dict(orient="records"),
    }
