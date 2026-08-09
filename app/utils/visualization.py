"""
Visualization Module for Annotated Tree Trunk Segmentation & Species Overlays.
Renders translucent masks, bounding boxes, and status labels on original forest images.
"""
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

# Curated harmonious visual color palette (RGB)
TREE_COLORS = [
    (46, 204, 113),   # Emerald Green (Species Identified)
    (52, 152, 219),   # Peter River Blue
    (155, 89, 182),   # Amethyst Purple
    (241, 196, 15),   # Sun Yellow (Unknown / Open-Set Rejected)
    (230, 126, 34),   # Carrot Orange
    (231, 76, 60),    # Alizarin Red (Quality Failed / Not Assessable)
    (26, 188, 156),   # Turquoise
    (54, 215, 183),   # Medium Aquamarine
]

STATUS_COLORS = {
    "known": (46, 204, 113),               # Green
    "open_set_rejected": (241, 196, 15),   # Yellow
    "quality_failed": (230, 126, 34),      # Orange
    "not_assessable": (231, 76, 60),       # Red
}


def draw_annotated_results(
    original_image_rgb: np.ndarray,
    trees_data: List[Dict[str, Any]],
    alpha: float = 0.40
) -> np.ndarray:
    """
    Overlay segmented tree trunk translucent masks, polygon contour outlines, and status tags onto original image.
    Uses precise mask polygon boundaries instead of rectangular bounding boxes.
    """
    annotated = original_image_rgb.copy()
    overlay = original_image_rgb.copy()
    img_h, img_w = original_image_rgb.shape[:2]

    # Base font scaling according to image resolution
    scale_factor = max(0.5, min(img_w, img_h) / 1000.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale_factor
    thickness = max(1, int(2 * scale_factor))
    line_thickness = max(2, int(3 * scale_factor))

    for idx, tree in enumerate(trees_data):
        mask = tree.get("mask")
        tree_id = tree.get("tree_id", idx + 1)
        final_species = tree.get("final_species", "Unknown")
        status = tree.get("status", "known")

        if mask is None or not mask.any():
            continue

        # Pick color based on status or color cycle
        color_rgb = STATUS_COLORS.get(status, TREE_COLORS[idx % len(TREE_COLORS)])
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # Render translucent mask overlay
        overlay[mask] = color_rgb

        # Extract precise polygon contours from binary mask
        mask_u8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Draw exact segmentation polygon contours (no rectangular boxes)
            cv2.drawContours(annotated, contours, -1, color_bgr, line_thickness, cv2.LINE_AA)

            # Find top-most vertex of polygon contour to place label tag
            all_pts = np.vstack(contours)
            top_pt_idx = np.argmin(all_pts[:, 0, 1])
            anchor_x = int(all_pts[top_pt_idx, 0, 0])
            anchor_y = int(all_pts[top_pt_idx, 0, 1])

            # Formulate concise text label
            if status == "known" and final_species:
                label_text = f"Tree {tree_id} - {final_species.capitalize()}"
            elif status == "open_set_rejected":
                label_text = f"Tree {tree_id} - Unknown"
            elif status == "quality_failed":
                label_text = f"Tree {tree_id} - Quality Failed"
            elif status == "not_assessable":
                label_text = f"Tree {tree_id} - Not Assessable"
            else:
                label_text = f"Tree {tree_id}"

            # Calculate text background box dimensions
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            bg_x1 = max(0, min(anchor_x, img_w - text_w - 14))
            bg_y1 = max(0, anchor_y - text_h - baseline - 10)
            bg_y2 = max(text_h + 10, anchor_y)

            cv2.rectangle(
                annotated,
                (bg_x1, bg_y1),
                (bg_x1 + text_w + 12, bg_y2),
                color_bgr,
                -1
            )
            # Text label in crisp white
            cv2.putText(
                annotated,
                label_text,
                (bg_x1 + 6, bg_y2 - baseline - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    # Blend translucent mask overlay with original image
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    return annotated


def draw_debug_after_duplicates(
    original_image_rgb: np.ndarray,
    dedup_candidates: List[Dict[str, Any]],
    alpha: float = 0.35
) -> np.ndarray:
    """
    Renders diagnostic overlay of candidates after duplicate suppression.
    Saved to debug_after_duplicates.jpg when DEBUG_TREE_DETECTION is True.
    """
    annotated = original_image_rgb.copy()
    overlay = original_image_rgb.copy()
    img_h, img_w = original_image_rgb.shape[:2]

    scale_factor = max(0.4, min(img_w, img_h) / 1000.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale_factor
    thickness = max(1, int(1 * scale_factor))

    for idx, cand in enumerate(dedup_candidates, start=1):
        mask = cand.get("mask")
        conf = cand.get("confidence", 0.0)
        box = cand.get("bbox", [0, 0, 0, 0])

        color_rgb = (155, 89, 182)  # Purple
        color_bgr = (182, 89, 155)

        if mask is not None and mask.any():
            overlay[mask] = color_rgb
            mask_u8 = (mask.astype(np.uint8)) * 255
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(annotated, cnts, -1, color_bgr, 2, cv2.LINE_AA)

        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)

        label = f"DEDUP {idx}: conf={conf*100:.1f}%"
        cv2.putText(annotated, label, (max(5, x1), max(15, y1 - 5)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    return annotated


def draw_debug_final_trunks(
    original_image_rgb: np.ndarray,
    accepted_trees: List[Dict[str, Any]],
    rejected_trees: List[Dict[str, Any]],
    alpha: float = 0.35
) -> np.ndarray:
    """
    Renders diagnostic overlay of final trunk candidates (Green = Accepted, Red = Rejected).
    Displays short reason (tiny, thin, fragmented, invalid-mask) for rejected predictions.
    Saved to debug_final_trunks.jpg when DEBUG_TREE_DETECTION is True.
    """
    annotated = original_image_rgb.copy()
    overlay = original_image_rgb.copy()
    img_h, img_w = original_image_rgb.shape[:2]

    scale_factor = max(0.4, min(img_w, img_h) / 1000.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale_factor
    thickness = max(1, int(1 * scale_factor))

    # 1. Render Rejected Candidates in Red with short reason
    for idx, cand in enumerate(rejected_trees, start=1):
        mask = cand.get("mask")
        reasons = cand.get("assessability", {}).get("reasons", ["rejected"])
        reason_str = reasons[0] if reasons else "rejected"

        if mask is None or not mask.any():
            continue

        color_rgb = (231, 76, 60)  # Red
        color_bgr = (60, 76, 231)

        overlay[mask] = color_rgb

        mask_u8 = (mask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(annotated, cnts, -1, color_bgr, 2, cv2.LINE_AA)
            all_pts = np.vstack(cnts)
            top_idx = np.argmin(all_pts[:, 0, 1])
            ax = int(all_pts[top_idx, 0, 0])
            ay = int(all_pts[top_idx, 0, 1])

            label = f"REJECT: {reason_str}"
            cv2.putText(annotated, label, (max(5, ax), max(15, ay - 5)), font, font_scale, color_bgr, thickness, cv2.LINE_AA)

    # 2. Render Accepted Candidates in Green
    for tree in accepted_trees:
        mask = tree.get("mask")
        t_id = tree.get("tree_id", 0)

        if mask is None or not mask.any():
            continue

        color_rgb = (46, 204, 113)  # Green
        color_bgr = (113, 204, 46)

        overlay[mask] = color_rgb

        mask_u8 = (mask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(annotated, cnts, -1, color_bgr, 3, cv2.LINE_AA)
            all_pts = np.vstack(cnts)
            top_idx = np.argmin(all_pts[:, 0, 1])
            ax = int(all_pts[top_idx, 0, 0])
            ay = int(all_pts[top_idx, 0, 1])

            label = f"Tree {t_id} (ACCEPTED)"
            cv2.putText(annotated, label, (max(5, ax), max(15, ay - 5)), font, font_scale, (0, 255, 0), thickness + 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    return annotated


def draw_debug_all_candidates(
    original_image_rgb: np.ndarray,
    accepted_trees: List[Dict[str, Any]],
    rejected_trees: List[Dict[str, Any]],
    alpha: float = 0.35
) -> np.ndarray:
    """Alias for draw_debug_final_trunks."""
    return draw_debug_final_trunks(original_image_rgb, accepted_trees, rejected_trees, alpha=alpha)


def draw_debug_raw_yolo(
    original_image_rgb: np.ndarray,
    raw_candidates: List[Dict[str, Any]],
    alpha: float = 0.35
) -> np.ndarray:
    """
    Renders diagnostic overlay of EVERY raw YOLO proposal candidate BEFORE filtering.
    Saved to debug_raw_yolo.jpg when DEBUG_TREE_DETECTION is True.
    """
    annotated = original_image_rgb.copy()
    overlay = original_image_rgb.copy()
    img_h, img_w = original_image_rgb.shape[:2]

    scale_factor = max(0.4, min(img_w, img_h) / 1000.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale_factor
    thickness = max(1, int(1 * scale_factor))

    for idx, cand in enumerate(raw_candidates, start=1):
        mask = cand.get("mask")
        conf = cand.get("confidence", 0.0)
        box = cand.get("bbox", [0, 0, 0, 0])

        color_rgb = (52, 152, 219)  # Blue
        color_bgr = (219, 152, 52)

        if mask is not None and mask.any():
            overlay[mask] = color_rgb
            mask_u8 = (mask.astype(np.uint8)) * 255
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(annotated, cnts, -1, color_bgr, 2, cv2.LINE_AA)

        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)

        label = f"RAW {idx}: conf={conf*100:.1f}%"
        cv2.putText(annotated, label, (max(5, x1), max(15, y1 - 5)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    return annotated

