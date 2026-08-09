from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
import torch

from app.maturity.config import (
    INTRINSICS_MODE,
    TUNED_CHECKPOINT_PATH,
    UNIDEPTH_MODEL_ID,
    UNIDEPTH_RESOLUTION_LEVEL,
)
from app.maturity.services.geometry import as_2d, as_k, intrinsics_diagnostics
from app.maturity.services.image_io import load_rgb
from app.maturity.services.runtime import autocast_context, clear_device, log_elapsed


_ORIGINAL_UNIDEPTH_CACHE: dict[tuple[str, str], object] = {}
_FINE_TUNED_UNIDEPTH_CACHE: dict[tuple[str, str, str], tuple[object, dict]] = {}


def get_original_unidepth_model(device: torch.device, log: Callable[[str], None]) -> object:
    cache_key = (UNIDEPTH_MODEL_ID, str(device))
    cached = _ORIGINAL_UNIDEPTH_CACHE.get(cache_key)
    if cached is not None:
        log("[timing] Original UniDepth model load: cached")
        return cached

    stage_started_at = perf_counter()
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained(UNIDEPTH_MODEL_ID).to(device).eval()
    log_elapsed(log, "Original UniDepth model load", stage_started_at)

    _ORIGINAL_UNIDEPTH_CACHE[cache_key] = model
    return model


def get_fine_tuned_unidepth_model(device: torch.device, log: Callable[[str], None]) -> tuple[object, dict]:
    cache_key = (UNIDEPTH_MODEL_ID, str(TUNED_CHECKPOINT_PATH), str(device))
    cached = _FINE_TUNED_UNIDEPTH_CACHE.get(cache_key)
    if cached is not None:
        log("[timing] Fine-tuned UniDepth base model load: cached")
        log("[timing] Fine-tuned UniDepth checkpoint load: cached")
        return cached

    stage_started_at = perf_counter()
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained(UNIDEPTH_MODEL_ID).to(device).eval()
    log_elapsed(log, "Fine-tuned UniDepth base model load", stage_started_at)

    stage_started_at = perf_counter()
    checkpoint = load_tuned_checkpoint(model, TUNED_CHECKPOINT_PATH, log)
    log_elapsed(log, "Fine-tuned UniDepth checkpoint load", stage_started_at)

    _FINE_TUNED_UNIDEPTH_CACHE[cache_key] = (model, checkpoint)
    return model, checkpoint


def load_tuned_checkpoint(model, path: Path, log: Callable[[str], None]) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("tuned_state_dict", checkpoint)
    load_info = model.load_state_dict(state, strict=False)
    log(f"Loaded tuned checkpoint: {path}")
    log(f"  Tuned tensors: {len(state)} | unexpected: {len(load_info.unexpected_keys)}")
    return checkpoint


def choose_intrinsics(
    model_K: np.ndarray,
    exif_intrinsics: dict,
    width: int,
    height: int,
) -> tuple[np.ndarray, str, str]:
    _, model_valid, _ = intrinsics_diagnostics(model_K, width, height)
    exif_K = exif_intrinsics.get("K")
    exif_available = bool(exif_intrinsics.get("available")) and exif_K is not None
    exif_valid = False

    if exif_available:
        _, exif_valid, _ = intrinsics_diagnostics(exif_K, width, height)

    if INTRINSICS_MODE == "prefer_exif" and exif_valid:
        return np.asarray(exif_K, dtype=np.float32), exif_intrinsics["source"], "EXIF preferred by configuration"

    if INTRINSICS_MODE == "exif_fallback" and (not model_valid) and exif_valid:
        return np.asarray(exif_K, dtype=np.float32), exif_intrinsics["source"], "EXIF used because UniDepth intrinsics failed diagnostics"

    if exif_available:
        note = f"EXIF metadata available ({exif_intrinsics['source']}); UniDepth intrinsics retained"
    else:
        note = exif_intrinsics.get("reason", "No EXIF intrinsics available")
    return np.asarray(model_K, dtype=np.float32), "unidepth", note


def infer_images(
    model,
    image_records: dict[str, dict],
    label: str,
    folders: dict[str, Path],
    device: torch.device,
    log: Callable[[str], None],
) -> dict[str, dict]:
    predictions = {}
    model.eval()
    model.resolution_level = UNIDEPTH_RESOLUTION_LEVEL
    total_images = len(image_records)
    stage_label = "Original UniDepth" if label == "original" else "Fine-tuned UniDepth"

    for image_index, (image_key, record) in enumerate(image_records.items(), start=1):
        image_label = f"{stage_label} image {image_index}/{total_images}"
        image_started_at = perf_counter()

        step_started_at = perf_counter()
        image = load_rgb(record["processing_path"])
        log_elapsed(log, f"{image_label} load", step_started_at)

        step_started_at = perf_counter()
        rgb = (
            torch.from_numpy(np.asarray(image).copy())
            .permute(2, 0, 1)
            .contiguous()
            .to(device)
        )
        log_elapsed(log, f"{image_label} tensor prep", step_started_at)

        step_started_at = perf_counter()
        with torch.inference_mode(), autocast_context(device):
            prediction = model.infer(rgb)
        log_elapsed(log, f"{image_label} model inference", step_started_at)

        step_started_at = perf_counter()
        depth = as_2d(prediction["depth"])
        model_K = as_k(prediction["intrinsics"])

        if depth.shape != (record["height"], record["width"]):
            depth = cv2.resize(
                depth,
                (record["width"], record["height"]),
                interpolation=cv2.INTER_LINEAR,
            )

        K, K_source, K_note = choose_intrinsics(
            model_K,
            record.get("exif_intrinsics", {}),
            record["width"],
            record["height"],
        )

        predictions[image_key] = {
            "depth": depth.astype(np.float32),
            "K": K.astype(np.float32),
            "K_source": K_source,
            "K_note": K_note,
        }
        log_elapsed(log, f"{image_label} postprocess", step_started_at)

        step_started_at = perf_counter()
        np.savez_compressed(
            folders["predictions_dir"] / f"{image_key}_{label}.npz",
            depth=depth.astype(np.float16),
            K=K.astype(np.float32),
            K_source=np.asarray([K_source]),
        )
        log_elapsed(log, f"{image_label} prediction save", step_started_at)

        del prediction, rgb
        clear_device(device)
        log(f"  Inferred: {record['tree_id']} / {record['display_name']} | intrinsics={K_source}")
        log_elapsed(log, f"{image_label} total", image_started_at)

    return predictions


def run_original_unidepth(
    image_records: dict[str, dict],
    folders: dict[str, Path],
    device: torch.device,
    log: Callable[[str], None],
) -> dict[str, dict]:
    log("[3/4] Running original UniDepthV2 for final DBH")
    model = get_original_unidepth_model(device, log)

    try:
        predictions = infer_images(model, image_records, "original", folders, device, log)
    finally:
        clear_device(device)

    return predictions


def run_fine_tuned_unidepth(
    image_records: dict[str, dict],
    folders: dict[str, Path],
    device: torch.device,
    log: Callable[[str], None],
) -> tuple[dict[str, dict], dict]:
    log("[4/4] Running fine-tuned UniDepthV2 for internal comparison only")
    model, checkpoint = get_fine_tuned_unidepth_model(device, log)

    try:
        predictions = infer_images(
            model,
            image_records,
            "fine_tuned_internal",
            folders,
            device,
            log,
        )
    finally:
        clear_device(device)

    return predictions, checkpoint
