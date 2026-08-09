"""
Tree Trunk Detector module wrapping Ultralytics YOLO model.
Loads models/tree_trunk_detection_best.pt.
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from app.config import (
    TREE_IMGSZ,
    TREE_CONFIDENCE,
    TREE_PROPOSAL_CONF,
    TREE_NMS_IOU,
    TREE_IOU,
    TREE_MAX_DET,
    DUPLICATE_IOU_THRESHOLD,
    PROMINENT_MIN_WIDTH_RATIO,
    PROMINENT_MIN_HEIGHT_RATIO,
    PROMINENT_MIN_BBOX_AREA_RATIO,
    DEBUG_TREE_DETECTION
)
from app.inference.assessability import remove_tiny_mask_islands, calculate_foreground_importance_score
from app.utils.image_utils import load_pil_image_safe

logger = logging.getLogger(__name__)


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return float(inter_area / max(union_area, 1e-8))


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate IoU between two binary masks."""
    if mask1 is None or mask2 is None:
        return 0.0
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / max(union, 1.0))


def calculate_candidate_priority(candidate: Dict[str, Any], orig_w: int, orig_h: int) -> float:
    """
    Calculate candidate priority ranking score for duplicate suppression.
    Ranks candidates using:
      1. usable mask area
      2. foreground prominence
      3. mask continuity
      4. YOLO confidence
    Prefers large complete trunks over small overlapping fragments.
    """
    box = candidate["bbox"]
    mask = candidate["mask"]
    conf = candidate["confidence"]

    bw = float(box[2] - box[0])
    bh = float(box[3] - box[1])
    b_area = bw * bh
    img_area = float(orig_w * orig_h)

    width_ratio = bw / float(orig_w)
    height_ratio = bh / float(orig_h)
    bbox_area_ratio = b_area / max(img_area, 1.0)
    mask_area = float(mask.sum())
    mask_area_ratio = mask_area / max(img_area, 1.0)

    # Compute largest component ratio for continuity
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels > 1:
        largest_comp_area = float(stats[1:, cv2.CC_STAT_AREA].max())
        comp_ratio = largest_comp_area / max(mask_area, 1.0)
    else:
        comp_ratio = 1.0

    is_prominent = bool(
        (width_ratio >= PROMINENT_MIN_WIDTH_RATIO and height_ratio >= PROMINENT_MIN_HEIGHT_RATIO)
        or bbox_area_ratio >= PROMINENT_MIN_BBOX_AREA_RATIO
    )

    prominence_score = 1.0 if is_prominent else min(1.0, (width_ratio / 0.08 + height_ratio / 0.40) / 2.0)
    area_score = min(1.0, mask_area_ratio / 0.08)
    continuity_score = comp_ratio

    # Weighted priority score
    priority_score = (
        0.35 * area_score +
        0.30 * prominence_score +
        0.20 * continuity_score +
        0.15 * conf
    )
    return float(priority_score)


def suppress_duplicate_detections(
    detections: List[Dict[str, Any]],
    orig_w: int,
    orig_h: int,
    iou_threshold: float = DUPLICATE_IOU_THRESHOLD
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Suppress duplicate predictions that strongly overlap on the same trunk.
    Prefers large foreground trunks over smaller high-confidence overlapping fragments.
    Returns (kept_detections, suppressed_count).
    """
    if len(detections) <= 1:
        return detections, 0

    for det in detections:
        det["priority_score"] = calculate_candidate_priority(det, orig_w, orig_h)

    sorted_dets = sorted(
        detections,
        key=lambda d: (d["priority_score"], d["confidence"]),
        reverse=True
    )

    kept = []
    suppressed_count = 0

    for candidate in sorted_dets:
        is_duplicate = False
        cand_box = candidate["bbox"]
        cand_mask = candidate["mask"]

        for existing in kept:
            ex_box = existing["bbox"]
            ex_mask = existing["mask"]

            box_iou = compute_box_iou(cand_box, ex_box)
            mask_iou = compute_mask_iou(cand_mask, ex_mask)

            if max(box_iou, mask_iou) >= iou_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            candidate["duplicate_status"] = "SUPPRESSED"
            suppressed_count += 1
        else:
            candidate["duplicate_status"] = "KEPT"
            kept.append(candidate)

    return kept, suppressed_count


class TreeDetector:
    """Ultralytics YOLO Tree Trunk Instance-Segmentation Model Wrapper."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tree detection model checkpoint not found: {self.model_path}")

        if YOLO is None:
            raise ImportError("Ultralytics package is required to load tree detection model.")

        logger.info(f"Loading tree detector from {self.model_path} onto {self.device}...")
        self.model = YOLO(str(self.model_path))
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        logger.info("Tree detection model successfully loaded.")

    def run_detection_diagnostics(self, img_np: np.ndarray, thresholds: List[float] = [0.30, 0.20, 0.15, 0.10, 0.05]):
        """Diagnostic count printing across confidence thresholds."""
        print(f"\n[DIAGNOSTIC] --- Tree Detection Counts across Confidence Thresholds ---")
        for conf_val in thresholds:
            res = self.model.predict(
                source=img_np,
                imgsz=TREE_IMGSZ,
                conf=conf_val,
                iou=TREE_NMS_IOU,
                max_det=TREE_MAX_DET,
                verbose=False,
                device=self.device
            )
            count = len(res[0].boxes) if (res and res[0].boxes is not None) else 0
            print(f"  - Conf Threshold {conf_val:.2f}: {count} trunk(s) detected")
        print(f"[DIAGNOSTIC] ---------------------------------------------------------\n")

    def predict(
        self,
        image_path_or_array: Any,
        imgsz: int = TREE_IMGSZ,
        conf: float = TREE_PROPOSAL_CONF,
        iou: float = TREE_NMS_IOU,
        max_det: int = TREE_MAX_DET
    ) -> List[Dict[str, Any]]:
        """
        Run instance segmentation on target image.
        Identifies clear visible trunk section with considerable maximum width in mask.
        Returns detected trees sorted deterministically left-to-right by horizontal centroid.
        """
        if self.model is None:
            raise RuntimeError("Tree detection model is not loaded.")

        if isinstance(image_path_or_array, (str, Path, Image.Image, bytes)):
            img = load_pil_image_safe(image_path_or_array)
            img_np = np.array(img)
        elif isinstance(image_path_or_array, np.ndarray):
            img_np = image_path_or_array
        else:
            raise ValueError("Invalid image input type for TreeDetector.")

        orig_h, orig_w = img_np.shape[:2]

        if DEBUG_TREE_DETECTION:
            self.run_detection_diagnostics(img_np)

        results = self.model.predict(
            source=img_np,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            device=self.device
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        boxes = result.boxes
        masks = result.masks

        if boxes is None or len(boxes) == 0:
            return []

        raw_detections = []

        boxes_xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
        confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()

        has_masks = masks is not None and len(masks) > 0
        if has_masks:
            mask_data = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else masks.data.numpy()
            xy_polygons = masks.xy if hasattr(masks, "xy") else None
        else:
            mask_data = None
            xy_polygons = None

        from app.inference.bark_roi import find_widest_clear_visible_trunk_span

        for idx in range(len(boxes_xyxy)):
            box = boxes_xyxy[idx]
            confidence = float(confidences[idx])

            x1, y1, x2, y2 = [float(v) for v in box]
            bbox = [x1, y1, x2, y2]

            # Reconstruct full resolution binary mask
            if has_masks and mask_data is not None and idx < len(mask_data):
                raw_mask = mask_data[idx]
                if raw_mask.shape[:2] != (orig_h, orig_w):
                    binary_mask = cv2.resize(
                        raw_mask.astype(np.uint8),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    binary_mask = raw_mask.astype(bool)
            else:
                binary_mask = np.zeros((orig_h, orig_w), dtype=bool)
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                binary_mask[iy1:iy2, ix1:ix2] = True

            # Clean tiny mask islands around trunk contour
            binary_mask = remove_tiny_mask_islands(binary_mask)

            polygon = None
            if xy_polygons is not None and idx < len(xy_polygons):
                poly_pts = xy_polygons[idx]
                if len(poly_pts) > 0:
                    polygon = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in poly_pts]

            if not polygon and binary_mask is not None and binary_mask.any():
                mask_u8 = (binary_mask.astype(np.uint8)) * 255
                cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    largest_cnt = max(cnts, key=cv2.contourArea)
                    polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in largest_cnt]

            # Identify clear visible section with considerable maximum width
            widest_bbox = find_widest_clear_visible_trunk_span(binary_mask, min_width_ratio=0.55) or (int(x1), int(y1), int(x2), int(y2))

            # Calculate centroid for deterministic ordering
            ys, xs = np.where(binary_mask)
            if len(xs) > 0:
                centroid_x = float(xs.mean())
                centroid_y = float(ys.mean())
            else:
                centroid_x = (x1 + x2) / 2.0
                centroid_y = (y1 + y2) / 2.0

            raw_detections.append({
                "bbox": bbox,
                "widest_trunk_bbox": list(widest_bbox),
                "confidence": confidence,
                "mask": binary_mask,
                "polygon": polygon,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "raw_index": idx
            })

        # Apply Duplicate Suppression with Foreground Priority (IoU >= 0.65)
        dedup_detections, suppressed_count = suppress_duplicate_detections(raw_detections, orig_w, orig_h)
        if DEBUG_TREE_DETECTION:
            logger.info(f"Duplicate Suppression: {len(raw_detections)} raw -> {len(dedup_detections)} candidates ({suppressed_count} suppressed)")

        # Order candidates deterministically from left to right by centroid X
        dedup_detections.sort(key=lambda d: (d["centroid_x"], d["centroid_y"]))

        return dedup_detections
