from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from app.maturity.config import (
    DETECTOR_BACKEND,
    DINO_ID,
    FASTSAM_CONFIDENCE,
    FASTSAM_IMAGE_SIZE,
    FASTSAM_IOU,
    FASTSAM_MODEL,
    FASTSAM_RETINA_MASKS,
    GROUNDING_BOX_THRESHOLD,
    GROUNDING_TEXT_THRESHOLD,
    MIN_MASK_PIXELS,
    SAM_ID,
    SEGMENTER_BACKEND,
    TEXT_LABELS,
    YOLO_WORLD_CONFIDENCE,
    YOLO_WORLD_IMAGE_SIZE,
    YOLO_WORLD_LABELS,
    YOLO_WORLD_MODEL,
)
from app.maturity.services.geometry import (
    clean_mask,
    detection_score,
    expand_box,
    fallback_box,
    mask_quality,
)
from app.maturity.services.image_io import load_rgb
from app.maturity.services.runtime import autocast_context, clear_device, log_elapsed, model_load_kwargs


_DINO_CACHE: dict[tuple[str, str], tuple[object, object]] = {}
_SAM_CACHE: dict[tuple[str, str], tuple[object, object]] = {}
_YOLO_WORLD_CACHE: dict[tuple[str, str, tuple[str, ...]], object] = {}
_FASTSAM_CACHE: dict[tuple[str, str], object] = {}


def detection_stage_label() -> str:
    if DETECTOR_BACKEND == "yolo_world":
        return "YOLO-World detection"
    return "Grounding DINO detection"


def run_trunk_detection(
    image_records: dict[str, dict],
    device: torch.device,
    log: Callable[[str], None],
) -> dict[str, dict]:
    if DETECTOR_BACKEND == "yolo_world":
        return run_yolo_world_detection(image_records, device, log)
    return run_grounding_dino(image_records, device, log)


def build_detections_from_module1(
    image_records: dict[str, dict],
    external_detections_by_tree: dict[str, dict],
    log: Callable[[str], None],
) -> dict[str, dict]:
    """
    Skips Module 2's own independent trunk re-detection (used only by the full
    "Detection + Classification + Maturity Assessment" pipeline): converts the
    bounding box Module 1 already detected and classified into the same
    {image_key: {"box", "method", "confidence"}} shape run_trunk_detection
    would have produced, so the rest of the pipeline (segmentation onward) is
    guaranteed to measure the exact same tree Module 1 identified, instead of
    Module 2's own open-vocabulary detector possibly picking a different one.
    """
    detections: dict[str, dict] = {}
    for image_key, record in image_records.items():
        external = external_detections_by_tree[record["tree_id"]]
        scale = record["resize_scale"]
        original_box = external["bbox"]
        processing_box = [float(v) * scale for v in original_box]
        padded_box = expand_box(processing_box, record["width"], record["height"])

        detections[image_key] = {
            "box": padded_box,
            "method": "module1_handoff",
            "confidence": float(external.get("confidence", float("nan"))),
        }
        log(
            f"Reusing Module 1's detected trunk for {record['tree_id']} "
            f"(bbox={[round(v, 1) for v in padded_box]}) instead of re-detecting"
        )
    return detections


def get_grounding_dino_model(device: torch.device, log: Callable[[str], None]) -> tuple[object, object]:
    cache_key = (DINO_ID, str(device))
    cached = _DINO_CACHE.get(cache_key)
    if cached is not None:
        log("[timing] Grounding DINO model load: cached")
        return cached

    stage_started_at = perf_counter()
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(DINO_ID)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_ID,
            **model_load_kwargs(device),
        )
        .to(device)
        .eval()
    )
    log_elapsed(log, "Grounding DINO model load", stage_started_at)

    _DINO_CACHE[cache_key] = (processor, model)
    return processor, model


def get_sam_model(device: torch.device, log: Callable[[str], None]) -> tuple[object, object]:
    cache_key = (SAM_ID, str(device))
    cached = _SAM_CACHE.get(cache_key)
    if cached is not None:
        log("[timing] SAM 2.1 model load: cached")
        return cached

    stage_started_at = perf_counter()
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(SAM_ID)
    model = (
        Sam2Model.from_pretrained(SAM_ID, **model_load_kwargs(device))
        .to(device)
        .eval()
    )
    log_elapsed(log, "SAM 2.1 model load", stage_started_at)

    _SAM_CACHE[cache_key] = (processor, model)
    return processor, model


def run_grounding_dino(
    image_records: dict[str, dict],
    device: torch.device,
    log: Callable[[str], None],
) -> dict[str, dict]:
    log("[1/4] Detecting trunks with Grounding DINO")
    processor, model = get_grounding_dino_model(device, log)

    detections: dict[str, dict] = {}
    total_images = len(image_records)

    try:
        for image_index, (image_key, record) in enumerate(image_records.items(), start=1):
            image_label = f"Grounding DINO image {image_index}/{total_images}"
            image_started_at = perf_counter()

            step_started_at = perf_counter()
            image = load_rgb(record["processing_path"])
            width, height = image.size
            log_elapsed(log, f"{image_label} load", step_started_at)

            try:
                step_started_at = perf_counter()
                inputs = processor(images=image, text=TEXT_LABELS, return_tensors="pt").to(device)
                log_elapsed(log, f"{image_label} preprocess", step_started_at)

                step_started_at = perf_counter()
                with torch.inference_mode(), autocast_context(device):
                    outputs = model(**inputs)
                log_elapsed(log, f"{image_label} model inference", step_started_at)

                step_started_at = perf_counter()
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=GROUNDING_BOX_THRESHOLD,
                    text_threshold=GROUNDING_TEXT_THRESHOLD,
                    target_sizes=[(height, width)],
                    text_labels=TEXT_LABELS,
                )[0]
                boxes = result["boxes"].detach().float().cpu().numpy()
                scores = result["scores"].detach().float().cpu().numpy()

                if len(boxes):
                    ranked = [
                        (
                            detection_score(candidate_box, score, width, height),
                            candidate_box,
                            score,
                        )
                        for candidate_box, score in zip(boxes, scores)
                    ]
                    ranked.sort(key=lambda item: item[0], reverse=True)
                    _, box, confidence = ranked[0]
                    box = expand_box(box, width, height)
                    method = "Grounding DINO Base"
                else:
                    box = fallback_box(width, height)
                    confidence = np.nan
                    method = "wide close-up fallback"
                log_elapsed(log, f"{image_label} postprocess", step_started_at)

            except Exception as error:
                box = fallback_box(width, height)
                confidence = np.nan
                method = f"wide fallback after DINO error: {type(error).__name__}"

            detections[image_key] = {
                "box": list(map(float, box)),
                "method": method,
                "confidence": float(confidence) if np.isfinite(confidence) else np.nan,
            }
            log(f"  Detected: {record['tree_id']} / {record['display_name']} ({method})")
            log_elapsed(log, f"{image_label} total", image_started_at)

    finally:
        clear_device(device)

    return detections


def _ultralytics_device(device: torch.device) -> str | int:
    if device.type == "cuda":
        return 0
    if device.type == "mps":
        return "mps"
    return "cpu"


def get_yolo_world_model(device: torch.device, log: Callable[[str], None]) -> object:
    cache_key = (YOLO_WORLD_MODEL, str(device), tuple(YOLO_WORLD_LABELS))
    cached = _YOLO_WORLD_CACHE.get(cache_key)
    if cached is not None:
        log("[timing] YOLO-World model load: cached")
        return cached

    stage_started_at = perf_counter()
    from ultralytics import YOLOWorld

    model = YOLOWorld(YOLO_WORLD_MODEL)
    model.set_classes(YOLO_WORLD_LABELS)
    log_elapsed(log, "YOLO-World model load", stage_started_at)

    _YOLO_WORLD_CACHE[cache_key] = model
    return model


def run_yolo_world_detection(
    image_records: dict[str, dict],
    device: torch.device,
    log: Callable[[str], None],
) -> dict[str, dict]:
    log("[1/4] Detecting trunks with YOLO-World")
    log(f"YOLO-World model: {YOLO_WORLD_MODEL} | imgsz={YOLO_WORLD_IMAGE_SIZE} | conf={YOLO_WORLD_CONFIDENCE}")
    log(f"YOLO-World labels: {', '.join(YOLO_WORLD_LABELS)}")
    model = get_yolo_world_model(device, log)

    detections: dict[str, dict] = {}
    total_images = len(image_records)

    for image_index, (image_key, record) in enumerate(image_records.items(), start=1):
        image_label = f"YOLO-World image {image_index}/{total_images}"
        image_started_at = perf_counter()

        step_started_at = perf_counter()
        image = load_rgb(record["processing_path"])
        width, height = image.size
        image_array = np.asarray(image)
        log_elapsed(log, f"{image_label} load", step_started_at)

        try:
            step_started_at = perf_counter()
            results = model.predict(
                source=image_array,
                imgsz=YOLO_WORLD_IMAGE_SIZE,
                conf=YOLO_WORLD_CONFIDENCE,
                device=_ultralytics_device(device),
                verbose=False,
            )
            log_elapsed(log, f"{image_label} model inference", step_started_at)

            step_started_at = perf_counter()
            result = results[0]
            boxes_obj = result.boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                boxes = boxes_obj.xyxy.detach().float().cpu().numpy()
                scores = boxes_obj.conf.detach().float().cpu().numpy()
                class_ids = boxes_obj.cls.detach().long().cpu().numpy()
                ranked = [
                    (
                        detection_score(candidate_box, score, width, height),
                        candidate_box,
                        score,
                        class_id,
                    )
                    for candidate_box, score, class_id in zip(boxes, scores, class_ids)
                ]
                ranked.sort(key=lambda item: item[0], reverse=True)
                _, box, confidence, class_id = ranked[0]
                box = expand_box(box, width, height)
                label = YOLO_WORLD_LABELS[int(class_id)] if int(class_id) < len(YOLO_WORLD_LABELS) else "object"
                method = f"YOLO-World ({label})"
            else:
                box = fallback_box(width, height)
                confidence = np.nan
                method = "wide close-up fallback after YOLO-World no detection"
            log_elapsed(log, f"{image_label} postprocess", step_started_at)

        except Exception as error:
            box = fallback_box(width, height)
            confidence = np.nan
            method = f"wide fallback after YOLO-World error: {type(error).__name__}"

        detections[image_key] = {
            "box": list(map(float, box)),
            "method": method,
            "confidence": float(confidence) if np.isfinite(confidence) else np.nan,
        }
        log(f"  Detected: {record['tree_id']} / {record['display_name']} ({method})")
        log_elapsed(log, f"{image_label} total", image_started_at)

    clear_device(device)
    return detections


def segmentation_stage_label() -> str:
    if SEGMENTER_BACKEND == "fastsam":
        return "FastSAM segmentation"
    return "SAM 2.1 segmentation"


def get_fastsam_model(device: torch.device, log: Callable[[str], None]) -> object:
    cache_key = (FASTSAM_MODEL, str(device))
    cached = _FASTSAM_CACHE.get(cache_key)
    if cached is not None:
        log("[timing] FastSAM model load: cached")
        return cached

    stage_started_at = perf_counter()
    from ultralytics import FastSAM

    model = FastSAM(FASTSAM_MODEL)
    log_elapsed(log, "FastSAM model load", stage_started_at)

    _FASTSAM_CACHE[cache_key] = model
    return model


def run_trunk_segmentation(
    image_records: dict[str, dict],
    detections: dict[str, dict],
    folders: dict[str, Path],
    device: torch.device,
    log: Callable[[str], None],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if SEGMENTER_BACKEND == "fastsam":
        return run_fastsam_segmentation(image_records, detections, folders, device, log)
    return run_sam_segmentation(image_records, detections, folders, device, log)


def run_fastsam_segmentation(
    image_records: dict[str, dict],
    detections: dict[str, dict],
    folders: dict[str, Path],
    device: torch.device,
    log: Callable[[str], None],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    log("[2/4] Segmenting trunks with FastSAM")
    log(
        f"FastSAM model: {FASTSAM_MODEL} | imgsz={FASTSAM_IMAGE_SIZE} | "
        f"conf={FASTSAM_CONFIDENCE} | iou={FASTSAM_IOU} | retina_masks={FASTSAM_RETINA_MASKS}"
    )
    model = get_fastsam_model(device, log)

    trunk_masks: dict[str, np.ndarray] = {}
    mask_rows = []
    total_images = len(image_records)

    for image_index, (image_key, record) in enumerate(image_records.items(), start=1):
        image_label = f"FastSAM image {image_index}/{total_images}"
        image_started_at = perf_counter()

        step_started_at = perf_counter()
        image = load_rgb(record["processing_path"])
        width, height = image.size
        image_array = np.asarray(image)
        box = list(map(float, detections[image_key]["box"]))
        target_x = (box[0] + box[2]) / 2.0
        log_elapsed(log, f"{image_label} load", step_started_at)

        step_started_at = perf_counter()
        results = model(
            source=image_array,
            bboxes=[box],
            device=_ultralytics_device(device),
            retina_masks=FASTSAM_RETINA_MASKS,
            imgsz=FASTSAM_IMAGE_SIZE,
            conf=FASTSAM_CONFIDENCE,
            iou=FASTSAM_IOU,
            verbose=False,
        )
        log_elapsed(log, f"{image_label} model inference", step_started_at)

        step_started_at = perf_counter()
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            raise RuntimeError("FastSAM did not return a trunk mask.")

        mask_data = result.masks.data.detach().float().cpu().numpy()
        box_confidences = []
        if result.boxes is not None and len(result.boxes) > 0:
            box_confidences = result.boxes.conf.detach().float().cpu().numpy().tolist()

        candidates = []
        for candidate_index, raw_mask in enumerate(mask_data):
            if raw_mask.shape != (height, width):
                raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_LINEAR)

            candidate_mask = clean_mask(raw_mask > 0.5, target_x=target_x)
            quality = mask_quality(candidate_mask)
            confidence = (
                float(box_confidences[candidate_index])
                if candidate_index < len(box_confidences)
                else float(detections[image_key].get("confidence", 0.0) or 0.0)
            )
            score = (
                0.50 * confidence
                + 0.50 * quality["score"]
                - (0.18 if not quality["pass"] else 0.0)
            )
            candidates.append((score, confidence, candidate_mask, quality))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected_score, selected_confidence, selected_mask, selected_quality = candidates[0]
        log_elapsed(log, f"{image_label} mask postprocess", step_started_at)

        if selected_mask.sum() < MIN_MASK_PIXELS:
            raise RuntimeError(f"FastSAM mask contains only {int(selected_mask.sum())} pixels.")

        step_started_at = perf_counter()
        trunk_masks[image_key] = selected_mask
        mask_path = folders["masks_dir"] / f"{image_key}.png"
        cv2.imwrite(str(mask_path), selected_mask.astype(np.uint8) * 255)
        log_elapsed(log, f"{image_label} mask save", step_started_at)

        mask_rows.append(
            {
                "image_key": image_key,
                "tree_id": record["tree_id"],
                "image": record["display_name"],
                "segmenter_backend": "fastsam",
                "SAM_iou": np.nan,
                "FastSAM_confidence": selected_confidence,
                "combined_score": selected_score,
                **selected_quality,
            }
        )
        log(f"  Segmented: {record['tree_id']} / {record['display_name']} (FastSAM)")
        log_elapsed(log, f"{image_label} total", image_started_at)

    clear_device(device)
    return trunk_masks, pd.DataFrame(mask_rows)

def run_sam_segmentation(
    image_records: dict[str, dict],
    detections: dict[str, dict],
    folders: dict[str, Path],
    device: torch.device,
    log: Callable[[str], None],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    log("[2/4] Segmenting trunks with SAM 2.1")
    log(f"SAM model: {SAM_ID}")
    processor, model = get_sam_model(device, log)

    trunk_masks: dict[str, np.ndarray] = {}
    mask_rows = []
    total_images = len(image_records)

    try:
        for image_index, (image_key, record) in enumerate(image_records.items(), start=1):
            image_label = f"SAM 2.1 image {image_index}/{total_images}"
            image_started_at = perf_counter()

            step_started_at = perf_counter()
            image = load_rgb(record["processing_path"])
            box = detections[image_key]["box"]
            log_elapsed(log, f"{image_label} load", step_started_at)

            step_started_at = perf_counter()
            inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt").to(device)
            original_sizes = inputs["original_sizes"].detach().cpu()
            log_elapsed(log, f"{image_label} preprocess", step_started_at)

            step_started_at = perf_counter()
            with torch.inference_mode(), autocast_context(device):
                outputs = model(**inputs, multimask_output=True)
            log_elapsed(log, f"{image_label} model inference", step_started_at)

            step_started_at = perf_counter()
            masks = processor.post_process_masks(
                outputs.pred_masks.detach().float().cpu(),
                original_sizes,
            )[0]
            mask_array = np.asarray(masks > 0)

            while mask_array.ndim > 3:
                mask_array = mask_array[0]

            iou_values = outputs.iou_scores.detach().float().cpu().numpy().reshape(-1)
            target_x = (box[0] + box[2]) / 2.0
            candidates = []

            for candidate_index in range(mask_array.shape[0]):
                candidate = clean_mask(mask_array[candidate_index], target_x=target_x)
                quality = mask_quality(candidate)
                predicted_iou = (
                    float(iou_values[candidate_index])
                    if candidate_index < len(iou_values)
                    else 0.0
                )
                score = (
                    0.55 * predicted_iou
                    + 0.45 * quality["score"]
                    - (0.18 if not quality["pass"] else 0.0)
                )
                candidates.append((score, predicted_iou, candidate, quality))

            candidates.sort(key=lambda item: item[0], reverse=True)
            selected_score, selected_iou, selected_mask, selected_quality = candidates[0]
            log_elapsed(log, f"{image_label} mask postprocess", step_started_at)

            if selected_mask.sum() < MIN_MASK_PIXELS:
                raise RuntimeError(f"Mask contains only {int(selected_mask.sum())} pixels.")

            step_started_at = perf_counter()
            trunk_masks[image_key] = selected_mask
            mask_path = folders["masks_dir"] / f"{image_key}.png"
            cv2.imwrite(str(mask_path), selected_mask.astype(np.uint8) * 255)
            log_elapsed(log, f"{image_label} mask save", step_started_at)

            mask_rows.append(
                {
                    "image_key": image_key,
                    "tree_id": record["tree_id"],
                    "image": record["display_name"],
                    "SAM_iou": selected_iou,
                    "combined_score": selected_score,
                    **selected_quality,
                }
            )
            log(f"  Segmented: {record['tree_id']} / {record['display_name']}")
            log_elapsed(log, f"{image_label} total", image_started_at)

    finally:
        clear_device(device)

    return trunk_masks, pd.DataFrame(mask_rows)


def save_segmentation_overlays(
    image_records: dict[str, dict],
    detections: dict[str, dict],
    trunk_masks: dict[str, np.ndarray],
    folders: dict[str, Path],
) -> list[Path]:
    overlay_paths = []

    for image_key, record in image_records.items():
        image = np.asarray(load_rgb(record["processing_path"]))
        mask = trunk_masks[image_key]
        box = detections[image_key]["box"]
        overlay = image.copy()
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        overlay[mask] = (0.62 * overlay[mask] + 0.38 * green[mask]).astype(np.uint8)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            max(2, int(min(record["width"], record["height"]) * 0.004)),
        )

        path = folders["overlays_dir"] / f"{image_key}_segmentation.png"
        Image.fromarray(overlay).save(path)
        overlay_paths.append(path)

    return overlay_paths
