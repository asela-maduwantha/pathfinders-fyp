"""
API routes for Module 2 (DBH + maturity estimation) and the auto-integrated
Module 1 -> Module 2 flow (species identification feeding straight into DBH/maturity).
"""
from __future__ import annotations

import io
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.config import UPLOADS_DIR, RESULTS_DIR
from app.inference_lock import INFERENCE_LOCK
from app.maturity.config import ALLOWED_IMAGE_EXTENSIONS, OUTPUTS_DIR
from app.maturity.services.maturity import get_maturity_predictor
from app.maturity.services.pipeline import run_merged_analysis
from app.utils.image_utils import MAX_FILE_SIZE_BYTES

logger = logging.getLogger("maturity.routes")

router = APIRouter()


# ---------------------------------------------------------------------------
# Response-shaping helpers (ported from timberResearchMaturity-main/app/main.py)
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _dataframe_records(df) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _output_url(path: Path) -> str:
    relative = Path(path).resolve().relative_to(OUTPUTS_DIR.resolve()).as_posix()
    return f"/outputs/{relative}"


def _artifact_gallery(
    run_dir: Path,
    folder_name: str,
    include_tokens: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    folder = run_dir / folder_name
    if not folder.exists():
        return []

    items = []
    for path in sorted(folder.glob("*.png")):
        if include_tokens and not any(token in path.name for token in include_tokens):
            continue

        if "maturity_curve" in path.name:
            caption = "Maturity DBH model"
        elif "dbh_rows" in path.name:
            caption = "DBH measurement rows"
        elif "dbh_consistency" in path.name:
            caption = "Original-model DBH consistency"
        else:
            caption = "Trunk segmentation"
        items.append({"url": _output_url(path), "caption": caption, "name": path.name})
    return items


def _build_maturity_response(result: dict[str, Any], log_lines: list[str]) -> dict[str, Any]:
    run_dir = Path(result["folders"]["run_dir"])
    maturity_plots = _artifact_gallery(run_dir, "plots", include_tokens=("maturity_curve",))
    return {
        "summary": _dataframe_records(result["tree_summary"]),
        "tree_results": _json_safe(result["tree_results"]),
        "per_image_internal": _dataframe_records(result["per_image_internal"]),
        "segmentation_quality": _dataframe_records(result["segmentation_quality"]),
        "plots": maturity_plots,
        "maturity_plots": maturity_plots,
        "measurement_guides": _artifact_gallery(run_dir, "measurements"),
        "overlays": _artifact_gallery(run_dir, "overlays"),
        "zip_url": _output_url(Path(result["zip_path"])),
        "run_id": result["folders"]["run_id"],
        "run_dir": str(run_dir),
        "log": "\n".join(log_lines),
    }


def _run_merged_analysis_locked(tree_inputs: list[dict[str, Any]], log_lines: list[str]) -> dict[str, Any]:
    def log(message: str) -> None:
        log_lines.append(str(message))

    with INFERENCE_LOCK:
        return run_merged_analysis(tree_inputs, log=log)


# ---------------------------------------------------------------------------
# GET /api/maturity/config
# ---------------------------------------------------------------------------

@router.get("/api/maturity/config")
def maturity_config():
    predictor = get_maturity_predictor()
    category_options = {
        column: predictor.category_options(column)
        for column in predictor.optional_categorical_features
    }
    return {
        "species": predictor.species_values,
        "optional_numeric_features": predictor.optional_numeric_features,
        "optional_categorical_features": predictor.optional_categorical_features,
        "category_options": category_options,
        "maturity_model_version": predictor.version,
        "maturity_architecture": predictor.architecture,
        "maturity_threshold": predictor.maturity_threshold,
    }


# ---------------------------------------------------------------------------
# POST /api/maturity/analyze - Complete Pipeline (multi-tree queue) and
# Single Estimation (one-tree payload) both call this same endpoint.
# ---------------------------------------------------------------------------

@router.post("/api/maturity/analyze")
async def maturity_analyze(payload: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload JSON: {exc}") from exc

    trees = parsed.get("trees")
    if not isinstance(trees, list) or not trees:
        raise HTTPException(status_code=400, detail="Add at least one tree before running analysis.")

    file_bytes = []
    for upload in files:
        file_bytes.append(
            {
                "name": upload.filename or "image.jpg",
                "content": await upload.read(),
            }
        )

    tree_inputs = []
    for tree in trees:
        file_indexes = tree.get("file_indexes") or []
        uploaded_items = []
        for index in file_indexes:
            if not isinstance(index, int) or index < 0 or index >= len(file_bytes):
                raise HTTPException(status_code=400, detail="Payload references an uploaded file that is missing.")
            uploaded_items.append(file_bytes[index])

        if not uploaded_items:
            raise HTTPException(status_code=400, detail=f"Tree '{tree.get('tree_id', '')}' has no images.")

        tree_inputs.append(
            {
                "tree_id": str(tree.get("tree_id") or f"Tree_{len(tree_inputs) + 1:02d}").strip(),
                "species": str(tree.get("species") or "").strip(),
                "uploaded_items": uploaded_items,
                "optional_inputs": dict(tree.get("optional_inputs") or {}),
            }
        )

    log_lines: list[str] = []

    try:
        result = await run_in_threadpool(_run_merged_analysis_locked, tree_inputs, log_lines)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(exc),
                "log": "\n".join(log_lines),
            },
        ) from exc

    return _build_maturity_response(result, log_lines)


# ---------------------------------------------------------------------------
# POST /api/maturity/analyze/auto - the new orchestration endpoint:
# Module 1 identifies the species from the uploaded photo, then that species
# (with the same photo) is fed straight into Module 2's DBH/maturity pipeline.
# ---------------------------------------------------------------------------

_NUMERIC_OPTIONAL_FIELDS = (
    "age_years",
    "height_m",
    "annual_rainfall_mm",
    "elevation_m",
    "stems_per_ha",
)
_CATEGORICAL_OPTIONAL_FIELDS = (
    "climatic_zone",
    "spacing",
    "site_class",
    "management_intensity",
    "intended_product",
)


def _pick_identified_tree(module1_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    candidates = [t for t in module1_result.get("trees", []) if t.get("status") == "known"]
    if not candidates:
        return None
    candidates.sort(key=lambda t: t.get("detection", {}).get("confidence", 0.0), reverse=True)
    return candidates[0]


def _match_species(species_name: str, available: list[str]) -> Optional[str]:
    target = species_name.strip().lower()
    for candidate in available:
        if candidate.strip().lower() == target:
            return candidate
    return None


def _validate_auto_flow_image(file_bytes: bytes, filename: str) -> tuple[bool, Optional[str]]:
    """
    Same checks as image_utils.validate_image_file, but allows HEIC/HEIF too -
    this endpoint's second stage (maturity assessment) already supports those formats,
    and pillow_heif's global Pillow opener is registered via services.image_io.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({len(file_bytes) / 1024 / 1024:.1f} MB) exceeds maximum allowed size (20 MB)."

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception as exc:
        return False, f"Corrupted or invalid image content: {exc}"

    return True, None


# The bark classifier can output "rubber", but the maturity model has no Rubber data
# (its species set is Teak/Mahogany/Pine/Eucalyptus). Rubber and Eucalyptus DBH/maturity
# curves are treated as interchangeable for this pipeline: Eucalyptus is queried internally,
# but every user-facing label is forced back to "Rubber" so the substitution stays invisible.
_RUBBER_SPECIES_NAME = "rubber"
_RUBBER_MATURITY_SUBSTITUTE = "eucalyptus"


def _relabel_species_in_maturity_response(maturity_response: dict[str, Any], substituted_species: str, display_species: str) -> None:
    target = substituted_species.strip().lower()
    for tree_result in maturity_response.get("tree_results", []):
        if str(tree_result.get("species", "")).strip().lower() == target:
            tree_result["species"] = display_species
    for row in maturity_response.get("summary", []):
        if str(row.get("species", "")).strip().lower() == target:
            row["species"] = display_species


@router.post("/api/maturity/analyze/auto")
async def maturity_analyze_auto(
    file: UploadFile = File(...),
    age_years: Optional[float] = Form(None),
    height_m: Optional[float] = Form(None),
    annual_rainfall_mm: Optional[float] = Form(None),
    elevation_m: Optional[float] = Form(None),
    stems_per_ha: Optional[float] = Form(None),
    climatic_zone: Optional[str] = Form(None),
    spacing: Optional[str] = Form(None),
    site_class: Optional[str] = Form(None),
    management_intensity: Optional[str] = Form(None),
    intended_product: Optional[str] = Form(None),
):
    # Deferred import: app.main imports this router, so importing at module load
    # time would create a circular import.
    from app.main import models_store
    from app.inference.integrated_pipeline import IntegratedPipeline
    from app.inference.tree_detector import TreeDetector  # noqa: F401  (type context only)

    if "pipeline" not in models_store:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please retry in a few seconds.")

    file_bytes = await file.read()
    valid, err_msg = _validate_auto_flow_image(file_bytes, file.filename)
    if not valid:
        raise HTTPException(status_code=400, detail=err_msg)

    job_id = str(uuid.uuid4())
    upload_ext = Path(file.filename).suffix.lower()
    upload_path = UPLOADS_DIR / f"{job_id}{upload_ext}"
    upload_path.write_bytes(file_bytes)

    pipeline: IntegratedPipeline = models_store["pipeline"]

    def run_identification_locked() -> dict[str, Any]:
        with INFERENCE_LOCK:
            return pipeline.run_pipeline(
                job_id=job_id,
                image_path=upload_path,
                results_dir=RESULTS_DIR,
                progress_callback=None,
            )

    try:
        module1_result = await run_in_threadpool(run_identification_locked)
    except Exception as exc:
        logger.error(f"Auto-flow identification failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Species identification failed: {exc}") from exc

    identified_tree = _pick_identified_tree(module1_result)
    if identified_tree is None:
        return {
            "species_resolved": False,
            "identification": module1_result,
            "message": (
                "Species could not be identified automatically from this photo. "
                "Use the Maturity: Single Estimation tab to specify the species manually."
            ),
        }

    raw_species = identified_tree["raglo"]["final_species"]
    is_rubber = raw_species.strip().lower() == _RUBBER_SPECIES_NAME
    species_query = _RUBBER_MATURITY_SUBSTITUTE if is_rubber else raw_species
    predictor = get_maturity_predictor()
    matched_species = _match_species(species_query, predictor.species_values)

    if matched_species is None:
        return {
            "species_resolved": False,
            "identification": module1_result,
            "message": (
                f"Species '{raw_species}' was identified, but the maturity model has no data for it. "
                "Use the Maturity: Single Estimation tab to enter species and details manually."
            ),
        }

    optional_inputs: dict[str, Any] = {}
    for field_name, value in (
        ("age_years", age_years),
        ("height_m", height_m),
        ("annual_rainfall_mm", annual_rainfall_mm),
        ("elevation_m", elevation_m),
        ("stems_per_ha", stems_per_ha),
    ):
        if value is not None:
            optional_inputs[field_name] = value
    for field_name, value in (
        ("climatic_zone", climatic_zone),
        ("spacing", spacing),
        ("site_class", site_class),
        ("management_intensity", management_intensity),
        ("intended_product", intended_product),
    ):
        if value and value not in {"Blank", "Unknown"}:
            optional_inputs[field_name] = value

    tree_inputs = [
        {
            "tree_id": f"Tree_{identified_tree['tree_id']:02d}",
            "species": matched_species,
            "uploaded_items": [{"name": file.filename or "image.jpg", "content": file_bytes}],
            "optional_inputs": optional_inputs,
        }
    ]

    log_lines: list[str] = []
    try:
        result = await run_in_threadpool(_run_merged_analysis_locked, tree_inputs, log_lines)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(exc),
                "log": "\n".join(log_lines),
            },
        ) from exc

    maturity_response = _build_maturity_response(result, log_lines)
    if is_rubber:
        _relabel_species_in_maturity_response(maturity_response, matched_species, raw_species.strip().capitalize())

    return {
        "species_resolved": True,
        "identification": module1_result,
        "maturity": maturity_response,
    }
