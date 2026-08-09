"""
API routes for the Timber Grading module (SmartTimber-LK Module 3), vendored as an
independent module: TimberVision cut-face detection/cropping, then a ResNet-18 +
tabular fusion model that grades each accepted crop. Does not share any state, data,
or models with the tree detection/classification/maturity pipeline elsewhere in the app.
"""
from __future__ import annotations

import math
import time
import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.timber.services.cropper import TrunkCropper
from app.timber.services.fusion_model import TimberFusionService

router = APIRouter()

# In-memory session store for detected crops awaiting grading, same convention as
# app.main's jobs_store (single-process, no persistence/TTL - a server restart
# between detect and grade requires re-running detection).
_crop_sessions: dict[str, dict[str, Any]] = {}


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _volume_from_mid_girth(length_m: float, mid_girth_m: float) -> float:
    return length_m * (mid_girth_m**2) / (4.0 * math.pi)


class CropMeasurement(BaseModel):
    crop_id: str
    Length_M: float
    Mid_Girth_M: float


class GradeRequest(BaseModel):
    session_id: str
    measurements: list[CropMeasurement]


def _get_cropper() -> TrunkCropper:
    from app.main import models_store

    if "timber_cropper" not in models_store:
        raise HTTPException(status_code=503, detail="Timber Grading models are still initializing.")
    return models_store["timber_cropper"]


def _get_fusion_service() -> TimberFusionService:
    from app.main import models_store

    if "timber_fusion" not in models_store:
        raise HTTPException(status_code=503, detail="Timber Grading models are still initializing.")
    return models_store["timber_fusion"]


@router.get("/api/timber/status")
def timber_status() -> dict[str, Any]:
    fusion_service = _get_fusion_service()
    cropper = _get_cropper()
    status = fusion_service.status()
    status["volume_formula"] = "Length_M * Mid_Girth_M^2 / (4*pi)"
    status["cropper"] = cropper.status()
    return status


@router.post("/api/timber/detect")
async def timber_detect(files: list[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")

    cropper = _get_cropper()
    session_id = uuid.uuid4().hex
    request_started = time.perf_counter()
    images: list[dict[str, Any]] = []
    stored_crops: dict[str, dict[str, Any]] = {}

    for image_index, upload in enumerate(files, start=1):
        read_started = time.perf_counter()
        image_bytes = await upload.read()
        filename = upload.filename or f"image_{image_index}"
        crop_prefix = f"{session_id}-{image_index:02d}"

        try:
            image_result = cropper.detect(filename, image_bytes, crop_prefix)
            image_result["log"].insert(
                0,
                {
                    "stage": "read upload",
                    "elapsed_ms": _elapsed_ms(read_started),
                    "details": {"bytes": len(image_bytes), "content_type": upload.content_type},
                },
            )
            for crop in image_result.pop("stored_crops"):
                stored_crops[crop.crop_id] = {
                    "image_bytes": crop.image_bytes,
                    "filename": crop.filename,
                    "label": crop.label,
                    "bbox": crop.bbox,
                    "score": crop.score,
                    "area_ratio": crop.area_ratio,
                    "method": crop.method,
                    "visible_ratio": crop.visible_ratio,
                    "confidence": crop.confidence,
                    "sharpness_score": crop.sharpness_score,
                    "preview": crop.preview,
                }
        except Exception as exc:
            image_result = {
                "filename": filename,
                "original_preview": "",
                "crops": [],
                "log": [
                    {
                        "stage": "read upload",
                        "elapsed_ms": _elapsed_ms(read_started),
                        "details": {"bytes": len(image_bytes), "content_type": upload.content_type},
                    },
                    {"stage": "detection failed", "elapsed_ms": 0, "details": {"error": str(exc)}},
                ],
            }
        images.append(image_result)

    _crop_sessions[session_id] = {
        "created_at": time.time(),
        "crops": stored_crops,
    }
    response = {
        "session_id": session_id,
        "total_elapsed_ms": _elapsed_ms(request_started),
        "images": images,
        "crop_count": len(stored_crops),
    }
    return JSONResponse(response)


@router.post("/api/timber/grade")
async def timber_grade(payload: GradeRequest) -> JSONResponse:
    session = _crop_sessions.get(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Crop session not found; run detection again.")

    fusion_service = _get_fusion_service()
    request_started = time.perf_counter()
    results: list[dict[str, Any]] = []

    for measurement in payload.measurements:
        crop = session["crops"].get(measurement.crop_id)
        if crop is None:
            results.append(
                {
                    "crop_id": measurement.crop_id,
                    "prediction": {"status": "error", "reason": "crop id was not found in this session"},
                    "log": [],
                }
            )
            continue

        length_m = float(measurement.Length_M)
        mid_girth_m = float(measurement.Mid_Girth_M)
        if length_m <= 0 or mid_girth_m <= 0:
            results.append(
                {
                    "crop_id": measurement.crop_id,
                    "prediction": {"status": "error", "reason": "length and girth must be positive numbers"},
                    "log": [],
                }
            )
            continue

        volume = _volume_from_mid_girth(length_m, mid_girth_m)
        metadata = {
            "Length_M": length_m,
            "Mid_Girth_M": mid_girth_m,
            "Volume": volume,
        }
        result = fusion_service.process_image(
            f"{crop['filename']} {crop['label']}",
            crop["image_bytes"],
            metadata,
        )
        result.update(
            {
                "crop_id": measurement.crop_id,
                "label": crop["label"],
                "source_filename": crop["filename"],
                "bbox": crop["bbox"],
                "score": crop["score"],
                "area_ratio": crop["area_ratio"],
                "method": crop.get("method"),
                "visible_ratio": crop.get("visible_ratio"),
                "confidence": crop.get("confidence"),
                "sharpness_score": crop.get("sharpness_score"),
                "measurements": {
                    "Length_M": round(length_m, 5),
                    "Mid_Girth_M": round(mid_girth_m, 5),
                    "Volume": round(volume, 6),
                    "volume_formula": "Length_M * Mid_Girth_M^2 / (4*pi)",
                },
            }
        )
        results.append(result)

    response = {
        "session_id": payload.session_id,
        "device": "cpu",
        "total_elapsed_ms": _elapsed_ms(request_started),
        "results": results,
    }
    return JSONResponse(response)
