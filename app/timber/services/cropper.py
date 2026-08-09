from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


@dataclass
class CropCandidate:
    crop_id: str
    label: str
    filename: str
    bbox: list[int]
    score: float
    area_ratio: float
    preview: str
    image_bytes: bytes
    method: str
    visible_ratio: float | None = None
    confidence: float | None = None
    sharpness_score: float | None = None


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _log_step(log: list[dict[str, Any]], stage: str, started_at: float, **details: Any) -> None:
    entry: dict[str, Any] = {"stage": stage, "elapsed_ms": _elapsed_ms(started_at)}
    if details:
        entry["details"] = details
    log.append(entry)


def image_to_data_url(image: Image.Image, max_size: tuple[int, int] = (760, 520), quality: int = 86) -> str:
    preview = image.copy()
    preview.thumbnail(max_size)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_to_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


class TrunkCropper:
    """Notebook-aligned cropper.

    Accurate multi-trunk quantity comes from TimberVision class-0 `cut`
    segmentation. Only mostly visible, sharp cut faces are cropped for grading.
    The classical code is kept only for environments without TimberVision.
    """

    CUT_CLASS_ID = 0

    def __init__(
        self,
        timbervision_seg_path: Path | None = None,
        max_dim: int = 1280,
        min_area_ratio: float = 0.008,
        max_crops: int = 40,
        tv_conf: float = 0.25,
        pass_threshold: float = 0.70,
        min_visible_ratio: float = 0.85,
        min_sharpness_score: float = 120.0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.max_dim = max_dim
        self.min_area_ratio = min_area_ratio
        self.max_crops = max_crops
        self.tv_conf = tv_conf
        self.pass_threshold = pass_threshold
        self.min_visible_ratio = min_visible_ratio
        self.min_sharpness_score = min_sharpness_score
        self.tv_model: Any = None
        self.tv_status: dict[str, Any] = {
            "available": False,
            "path": str(timbervision_seg_path) if timbervision_seg_path else None,
            "reason": "TimberVision segmentation path was not configured",
        }

        if timbervision_seg_path is not None:
            self._load_timbervision(timbervision_seg_path)

    def _load_timbervision(self, path: Path) -> None:
        if YOLO is None:
            self.tv_status = {"available": False, "path": str(path), "reason": "ultralytics is not installed"}
            return
        if not path.exists():
            self.tv_status = {"available": False, "path": str(path), "reason": "weight file does not exist"}
            return
        started = time.perf_counter()
        try:
            self.tv_model = YOLO(str(path))
            self.tv_status = {
                "available": True,
                "path": str(path),
                "load_elapsed_ms": _elapsed_ms(started),
                "classes": self.tv_model.names,
                "conf": self.tv_conf,
                "pass_threshold": self.pass_threshold,
                "min_visible_ratio": self.min_visible_ratio,
                "min_sharpness_score": self.min_sharpness_score,
            }
        except Exception as exc:
            self.tv_model = None
            self.tv_status = {"available": False, "path": str(path), "reason": f"{type(exc).__name__}: {exc}"}

    def status(self) -> dict[str, Any]:
        return {
            "timbervision": self.tv_status,
            "device": self.device,
            "fallback": "notebook foreground/watershed, conservative; used only when TimberVision is unavailable",
            "max_dim": self.max_dim,
            "min_area_ratio": self.min_area_ratio,
            "max_crops": self.max_crops,
            "pass_threshold": self.pass_threshold,
            "min_visible_ratio": self.min_visible_ratio,
            "min_sharpness_score": self.min_sharpness_score,
        }

    def detect(self, filename: str, image_bytes: bytes, crop_prefix: str) -> dict[str, Any]:
        total_started = time.perf_counter()
        log: list[dict[str, Any]] = []

        started = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        _log_step(log, "decode image", started, width=image.width, height=image.height)

        started = time.perf_counter()
        working_rgb, scale_to_original = self._resize_for_processing(np.array(image))
        working_bgr = cv2.cvtColor(working_rgb, cv2.COLOR_RGB2BGR)
        _log_step(
            log,
            "resize for CPU detection",
            started,
            width=int(working_rgb.shape[1]),
            height=int(working_rgb.shape[0]),
            scale_to_original=round(scale_to_original, 5),
        )

        crops: list[CropCandidate] = []
        detection_stats = self._empty_detection_stats()
        if self.tv_model is not None:
            started = time.perf_counter()
            tv_result = self._timbervision_cut_candidates(working_bgr)
            tv_candidates = tv_result["accepted"]
            detection_stats = tv_result["stats"]
            _log_step(
                log,
                "TimberVision CUT segmentation",
                started,
                detected=detection_stats["trunks_detected"],
                accepted_before_merge=len(tv_candidates),
                rejected=detection_stats["trunks_rejected"],
                conf=self.tv_conf,
                pass_threshold=self.pass_threshold,
                min_visible_ratio=self.min_visible_ratio,
                min_sharpness_score=self.min_sharpness_score,
            )
            started = time.perf_counter()
            before_merge = len(tv_candidates)
            tv_candidates = self._merge_candidates(tv_candidates, working_bgr.shape, iou_threshold=0.70)
            duplicates = before_merge - len(tv_candidates)
            if duplicates:
                detection_stats["trunks_rejected"] += duplicates
                detection_stats["rejected"].append({"reason": "duplicate overlap after merging", "count": duplicates})
            detection_stats["trunks_passed_for_grading"] = len(tv_candidates)
            _log_step(log, "filter TimberVision candidates", started, candidates=len(tv_candidates), duplicates_removed=duplicates)
            started = time.perf_counter()
            crops = self._make_crops(filename, image, tv_candidates, crop_prefix, scale_to_original)
            _log_step(log, "crop TimberVision candidates", started, crops=len(crops))
        else:
            log.append({"stage": "TimberVision CUT segmentation", "elapsed_ms": 0.0, "details": self.tv_status})
            fallback_crops, fallback_stats = self._detect_with_classical_fallback(
                filename, image, working_bgr, crop_prefix, scale_to_original, log
            )
            crops = fallback_crops
            detection_stats = fallback_stats

        if not crops:
            log.append(
                {
                    "stage": "no grading-ready trunk crops",
                    "elapsed_ms": 0.0,
                    "details": {
                        "reason": "no CUT masks passed the confidence, 85% visible-face, and blur-quality filters",
                        "trunks_detected": detection_stats["trunks_detected"],
                        "trunks_passed_for_grading": detection_stats["trunks_passed_for_grading"],
                    },
                }
            )

        log.append({"stage": "total detection and cropping", "elapsed_ms": _elapsed_ms(total_started)})

        return {
            "filename": filename,
            "original_preview": image_to_data_url(image),
            **detection_stats,
            "crops": [self._public_crop(crop) for crop in crops],
            "stored_crops": crops,
            "log": log,
        }

    def _empty_detection_stats(self) -> dict[str, Any]:
        return {
            "trunks_detected": 0,
            "trunks_passed_for_grading": 0,
            "trunks_rejected": 0,
            "rejected": [],
            "detector_conf_threshold": self.tv_conf,
            "grading_confidence_rule": f"confidence >= {self.pass_threshold:.2f}",
            "grading_visibility_rule": f"visible face ratio >= {self.min_visible_ratio:.2f}",
            "grading_sharpness_rule": f"sharpness score >= {self.min_sharpness_score:.1f}",
        }

    def _reject_record(
        self,
        reason: str,
        *,
        index: int | None = None,
        confidence: float | None = None,
        visible_ratio: float | None = None,
        bbox: list[int] | None = None,
        sharpness_score: float | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {"reason": reason}
        if index is not None:
            record["detector_index"] = index
        if confidence is not None:
            record["confidence"] = round(float(confidence), 5)
        if visible_ratio is not None:
            record["visible_ratio"] = round(float(visible_ratio), 5)
        if bbox is not None:
            record["bbox"] = bbox
        if sharpness_score is not None:
            record["sharpness_score"] = round(float(sharpness_score), 2)
        return record

    def _timbervision_cut_candidates(self, img_bgr: np.ndarray) -> dict[str, Any]:
        assert self.tv_model is not None
        H, W = img_bgr.shape[:2]
        stats = self._empty_detection_stats()
        try:
            results = self.tv_model.predict(
                img_bgr,
                verbose=False,
                conf=self.tv_conf,
                retina_masks=True,
                imgsz=1024,
                device=self.device,
            )
        except Exception as exc:
            stats["rejected"].append({"reason": f"TimberVision inference failed: {type(exc).__name__}: {exc}"})
            return {"accepted": [], "stats": stats}
        if not results:
            return {"accepted": [], "stats": stats}
        result = results[0]
        if result.masks is None or result.boxes is None:
            return {"accepted": [], "stats": stats}

        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        names = getattr(self.tv_model, "names", {}) or {}

        candidates: list[dict[str, Any]] = []
        for idx, mask_data in enumerate(masks):
            confidence = float(confidences[idx])
            class_id = int(classes[idx])
            class_name = str(names.get(class_id, class_id)).lower()
            if class_id != self.CUT_CLASS_ID and "cut" not in class_name:
                continue

            stats["trunks_detected"] += 1
            mask = (mask_data > 0.5).astype(np.uint8) * 255
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = self._largest_component(mask)
            candidate, reason = self._candidate_from_mask(mask, img_bgr.shape, "TimberVision cut segmentation", confidence)
            if candidate is None:
                stats["trunks_rejected"] += 1
                stats["rejected"].append(
                    self._reject_record(reason or "invalid trunk region", index=idx + 1, confidence=confidence)
                )
                continue

            candidate["sharpness_score"] = self._sharpness_score(img_bgr, mask, candidate["bbox"])

            if confidence < self.pass_threshold:
                stats["trunks_rejected"] += 1
                stats["rejected"].append(
                    self._reject_record(
                        f"confidence below pass threshold {self.pass_threshold:.2f}",
                        index=idx + 1,
                        confidence=confidence,
                        visible_ratio=candidate.get("visible_ratio"),
                        bbox=candidate.get("bbox"),
                        sharpness_score=candidate.get("sharpness_score"),
                    )
                )
                continue

            visible_ratio = float(candidate.get("visible_ratio", 0.0))
            if visible_ratio < self.min_visible_ratio:
                stats["trunks_rejected"] += 1
                stats["rejected"].append(
                    self._reject_record(
                        f"visible face ratio below {self.min_visible_ratio:.2f}",
                        index=idx + 1,
                        confidence=confidence,
                        visible_ratio=visible_ratio,
                        bbox=candidate.get("bbox"),
                        sharpness_score=candidate.get("sharpness_score"),
                    )
                )
                continue

            sharpness_score = float(candidate.get("sharpness_score", 0.0))
            if sharpness_score < self.min_sharpness_score:
                stats["trunks_rejected"] += 1
                stats["rejected"].append(
                    self._reject_record(
                        f"sharpness below blur threshold {self.min_sharpness_score:.1f}",
                        index=idx + 1,
                        confidence=confidence,
                        visible_ratio=visible_ratio,
                        bbox=candidate.get("bbox"),
                        sharpness_score=sharpness_score,
                    )
                )
                continue

            candidates.append(candidate)

        stats["trunks_passed_for_grading"] = len(candidates)
        return {"accepted": candidates, "stats": stats}

    def _detect_with_classical_fallback(
        self,
        filename: str,
        image: Image.Image,
        working_bgr: np.ndarray,
        crop_prefix: str,
        scale_to_original: float,
        log: list[dict[str, Any]],
    ) -> tuple[list[CropCandidate], dict[str, Any]]:
        started = time.perf_counter()
        strict_mask = self._strict_wood_mask(working_bgr)
        strict_coverage = cv2.countNonZero(strict_mask) / float(strict_mask.size)
        _log_step(log, "fallback build strict wood mask", started, coverage=round(strict_coverage, 5))

        started = time.perf_counter()
        foreground_mask = self._notebook_foreground_mask(working_bgr)
        foreground_coverage = cv2.countNonZero(foreground_mask) / float(foreground_mask.size)
        _log_step(log, "fallback build notebook foreground mask", started, coverage=round(foreground_coverage, 5))

        candidates: list[dict[str, Any]] = []
        started = time.perf_counter()
        candidates.extend(self._component_candidates(strict_mask, working_bgr.shape, "fallback strict wood component"))
        _log_step(log, "fallback strict wood components", started, candidates=len(candidates))

        started = time.perf_counter()
        strict_split = self._watershed_candidates(working_bgr, strict_mask, "fallback strict watershed")
        candidates.extend(strict_split)
        _log_step(log, "fallback strict watershed", started, candidates=len(strict_split))

        if 0.02 <= foreground_coverage <= 0.72:
            started = time.perf_counter()
            foreground_split = self._watershed_candidates(working_bgr, foreground_mask, "fallback notebook watershed")
            candidates.extend(foreground_split)
            _log_step(log, "fallback notebook watershed", started, candidates=len(foreground_split))
        else:
            log.append(
                {
                    "stage": "fallback skip notebook foreground split",
                    "elapsed_ms": 0.0,
                    "details": {"reason": "foreground coverage is too broad/narrow", "coverage": round(foreground_coverage, 5)},
                }
            )

        started = time.perf_counter()
        raw_candidates = len(candidates)
        rejected = [candidate for candidate in candidates if float(candidate.get("visible_ratio", 0.0)) < self.min_visible_ratio]
        candidates = [candidate for candidate in candidates if float(candidate.get("visible_ratio", 0.0)) >= self.min_visible_ratio]
        candidates = self._merge_candidates(candidates, working_bgr.shape, iou_threshold=0.48)
        duplicate_or_bbox_rejects = max(0, raw_candidates - len(rejected) - len(candidates))
        _log_step(
            log,
            "fallback filter candidates",
            started,
            candidates=len(candidates),
            raw_candidates=raw_candidates,
            visible_rejected=len(rejected),
            other_rejected=duplicate_or_bbox_rejects,
            min_visible_ratio=self.min_visible_ratio,
        )

        started = time.perf_counter()
        crops = self._make_crops(filename, image, candidates, crop_prefix, scale_to_original)
        _log_step(log, "fallback crop candidates", started, crops=len(crops))

        stats = self._empty_detection_stats()
        stats.update(
            {
                "trunks_detected": raw_candidates,
                "trunks_passed_for_grading": len(crops),
                "trunks_rejected": max(0, raw_candidates - len(crops)),
                "rejected": [],
            }
        )
        if rejected:
            stats["rejected"].append(
                {"reason": f"visible face ratio below {self.min_visible_ratio:.2f}", "count": len(rejected)}
            )
        if duplicate_or_bbox_rejects:
            stats["rejected"].append({"reason": "duplicate overlap or invalid crop bounds", "count": duplicate_or_bbox_rejects})
        return crops, stats

    def _resize_for_processing(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = rgb.shape[:2]
        largest = max(width, height)
        if largest <= self.max_dim:
            return rgb, 1.0
        scale = self.max_dim / float(largest)
        resized = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale

    def _normalize_illumination(self, img_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness_eq = clahe.apply(lightness)
        illumination = cv2.GaussianBlur(lightness_eq, (0, 0), sigmaX=max(1.0, lightness.shape[1] * 0.05))
        illumination = np.clip(illumination, 1, 255).astype(np.float32)
        lightness_norm = np.clip((lightness_eq.astype(np.float32) / illumination) * 128, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([lightness_norm, a, b]), cv2.COLOR_LAB2BGR)

    def _strict_wood_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        img_norm = self._normalize_illumination(img_bgr)
        hsv = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 24, 28), (38, 255, 245))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        return self._remove_tiny_components(mask, min_area_ratio=self.min_area_ratio * 0.55)

    def _notebook_foreground_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        img_norm = self._normalize_illumination(img_bgr)
        hsv = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
        wood_range = cv2.inRange(hsv, (0, 15, 30), (40, 255, 235))
        grey_bark = cv2.inRange(hsv, (0, 0, 40), (180, 40, 200))
        mask = cv2.bitwise_or(wood_range, grey_bark)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        return self._remove_tiny_components(mask, min_area_ratio=0.01)

    def _remove_tiny_components(self, mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        min_area = max(80.0, min_area_ratio * mask.shape[0] * mask.shape[1])
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255
        return clean

    def _watershed_candidates(self, img_bgr: np.ndarray, mask: np.ndarray, method: str) -> list[dict[str, Any]]:
        if cv2.countNonZero(mask) < 0.02 * mask.size:
            return []
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return []

        all_candidates: list[dict[str, Any]] = []
        for threshold in (0.62, 0.52, 0.42):
            _, sure_fg = cv2.threshold(dist, threshold * dist.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            num_markers, markers = cv2.connectedComponents(sure_fg)
            if num_markers <= 1:
                continue
            sure_bg = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(img_bgr, markers)
            for label in np.unique(markers):
                if label <= 1:
                    continue
                region = np.uint8(markers == label) * 255
                all_candidates.extend(self._component_candidates(region, img_bgr.shape, f"{method} {threshold:.2f}"))
        return all_candidates

    def _component_candidates(self, mask: np.ndarray, img_shape: tuple[int, ...], method: str) -> list[dict[str, Any]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, Any]] = []
        for contour in contours:
            clean = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.drawContours(clean, [contour], -1, 255, thickness=cv2.FILLED)
            candidate, _ = self._candidate_from_mask(clean, img_shape, method, None)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _candidate_from_mask(
        self, mask: np.ndarray, img_shape: tuple[int, ...], method: str, confidence: float | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        mask = self._largest_component(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "empty mask"
        contour = max(contours, key=cv2.contourArea)
        score, metrics = self._score_contour(contour, img_shape)
        if metrics is None or not self._valid_region(metrics):
            return None, "invalid trunk region size/shape"
        x, y, w, h = cv2.boundingRect(contour)
        visible_ratio, touches_border = self._visible_face_ratio(contour, img_shape)
        pad = round(max(w, h) * 0.08)
        bbox = self._clip_bbox([x - pad, y - pad, w + 2 * pad, h + 2 * pad], img_shape[1], img_shape[0])
        if not self._valid_bbox(bbox, img_shape):
            return None, "invalid crop bounds"
        final_score = float(score if confidence is None else (2.0 * confidence + 0.35 * score))
        return {
            "bbox": bbox,
            "score": final_score,
            "area_ratio": float(metrics["area_frac"]),
            "method": method,
            "visible_ratio": visible_ratio,
            "touches_border": touches_border,
            "confidence": confidence,
            "sharpness_score": None,
        }, None

    def _visible_face_ratio(self, contour: np.ndarray, img_shape: tuple[int, ...]) -> tuple[float, bool]:
        H, W = img_shape[:2]
        area = cv2.contourArea(contour)
        points = contour.reshape(-1, 2)
        touches_border = bool(
            np.any(points[:, 0] <= 1)
            or np.any(points[:, 0] >= W - 2)
            or np.any(points[:, 1] <= 1)
            or np.any(points[:, 1] >= H - 2)
        )
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_ratio = area / hull_area if hull_area > 0 else 0.0
        ellipse_ratio = hull_ratio
        if len(contour) >= 5:
            try:
                (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
                ellipse_area = np.pi * (axis_a / 2.0) * (axis_b / 2.0)
                if ellipse_area > 0:
                    ellipse_ratio = area / ellipse_area
            except cv2.error:
                ellipse_ratio = hull_ratio
        ratio = max(0.0, min(1.0, min(hull_ratio, ellipse_ratio)))
        if touches_border:
            ratio = min(ratio, 0.70)
        return round(float(ratio), 5), touches_border

    def _sharpness_score(self, img_bgr: np.ndarray, mask: np.ndarray, bbox: list[int]) -> float:
        H, W = img_bgr.shape[:2]
        x, y, w, h = bbox
        x0 = max(0, min(W, int(x)))
        y0 = max(0, min(H, int(y)))
        x1 = max(x0, min(W, int(x + w)))
        y1 = max(y0, min(H, int(y + h)))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return 0.0
        roi = img_bgr[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1] > 0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        values = laplacian[mask_roi] if np.count_nonzero(mask_roi) >= 64 else laplacian.reshape(-1)
        if values.size < 64:
            return 0.0
        return round(float(np.var(values)), 2)

    def _score_contour(self, contour: np.ndarray, img_shape: tuple[int, ...]) -> tuple[float, dict[str, float] | None]:
        H, W = img_shape[:2]
        area = cv2.contourArea(contour)
        if area <= 0:
            return -999.0, None
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        extent = area / float(max(1, w * h))
        aspect = max(w / max(1, h), h / max(1, w))
        area_frac = area / float(H * W)
        score = 1.25 * min(circularity, 1.0) + 0.55 * min(extent, 1.0) + 0.35 * min(area_frac / 0.16, 1.0) - 0.20 * max(0.0, aspect - 2.2)
        return score, {"area_frac": area_frac, "circularity": min(circularity, 1.0), "extent": extent, "aspect": aspect, "w": float(w), "h": float(h)}

    def _valid_region(self, metrics: dict[str, float]) -> bool:
        if metrics["area_frac"] < self.min_area_ratio or metrics["area_frac"] > 0.78:
            return False
        if metrics["w"] <= 24 or metrics["h"] <= 24:
            return False
        if metrics["aspect"] > 3.3:
            return False
        if metrics["extent"] < 0.20:
            return False
        return True

    def _valid_bbox(self, bbox: list[int], img_shape: tuple[int, ...]) -> bool:
        H, W = img_shape[:2]
        _, _, w, h = bbox
        if w <= 20 or h <= 20:
            return False
        area_frac = (w * h) / float(H * W)
        if area_frac > 0.86:
            return False
        if w >= 0.96 * W and h >= 0.96 * H:
            return False
        if max(w / max(1, h), h / max(1, w)) > 3.7:
            return False
        return True

    def _merge_candidates(self, candidates: list[dict[str, Any]], img_shape: tuple[int, ...], iou_threshold: float) -> list[dict[str, Any]]:
        filtered = [candidate for candidate in candidates if self._valid_bbox(candidate["bbox"], img_shape)]
        filtered.sort(key=lambda item: item["score"], reverse=True)
        kept: list[dict[str, Any]] = []
        for candidate in filtered:
            if any(self._iou(candidate["bbox"], existing["bbox"]) >= iou_threshold for existing in kept):
                continue
            kept.append(candidate)
            if len(kept) >= self.max_crops:
                break
        kept.sort(key=lambda item: (item["bbox"][0] + item["bbox"][2] / 2.0, item["bbox"][1] + item["bbox"][3] / 2.0))
        return kept

    def _make_crops(self, filename: str, image: Image.Image, candidates: list[dict[str, Any]], crop_prefix: str, scale_to_original: float) -> list[CropCandidate]:
        crops: list[CropCandidate] = []
        for index, candidate in enumerate(candidates, start=1):
            x, y, w, h = candidate["bbox"]
            ox = max(0, round(x * scale_to_original))
            oy = max(0, round(y * scale_to_original))
            ow = max(1, round(w * scale_to_original))
            oh = max(1, round(h * scale_to_original))
            ox2 = min(image.width, ox + ow)
            oy2 = min(image.height, oy + oh)
            crop = image.crop((ox, oy, ox2, oy2))
            crops.append(
                CropCandidate(
                    crop_id=f"{crop_prefix}-{index:02d}",
                    label=f"T{index:02d}",
                    filename=filename,
                    bbox=[ox, oy, ox2 - ox, oy2 - oy],
                    score=round(float(candidate["score"]), 5),
                    area_ratio=round(float(candidate["area_ratio"]), 5),
                    preview=image_to_data_url(crop, max_size=(420, 320)),
                    image_bytes=image_to_jpeg_bytes(crop),
                    method=str(candidate["method"]),
                    visible_ratio=round(float(candidate["visible_ratio"]), 5) if candidate.get("visible_ratio") is not None else None,
                    confidence=round(float(candidate["confidence"]), 5) if candidate.get("confidence") is not None else None,
                    sharpness_score=round(float(candidate["sharpness_score"]), 2) if candidate.get("sharpness_score") is not None else None,
                )
            )
        return crops

    def _public_crop(self, crop: CropCandidate) -> dict[str, Any]:
        return {
            "crop_id": crop.crop_id,
            "label": crop.label,
            "filename": crop.filename,
            "bbox": crop.bbox,
            "score": crop.score,
            "area_ratio": crop.area_ratio,
            "method": crop.method,
            "visible_ratio": crop.visible_ratio,
            "confidence": crop.confidence,
            "sharpness_score": crop.sharpness_score,
            "preview": crop.preview,
        }

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
        return clean

    def _clip_bbox(self, bbox: list[int], width: int, height: int) -> list[int]:
        x, y, w, h = bbox
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(width, int(round(x + w)))
        y1 = min(height, int(round(y + h)))
        return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]

    def _iou(self, a: list[int], b: list[int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return 0.0 if union <= 0 else inter / union
