from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from app.maturity.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    DETECTOR_BACKEND,
    SEGMENTER_BACKEND,
    INTRINSICS_MODE,
    MATURITY_MODEL_PATH,
    MAX_LONG_SIDE,
    OUTPUTS_DIR,
    RUN_FINE_TUNED_INTERNAL,
    RUNS_DIR,
    TUNED_CHECKPOINT_PATH,
    UNIDEPTH_MODEL_ID,
)
from app.maturity.services.depth import run_fine_tuned_unidepth, run_original_unidepth
from app.maturity.services.geometry import robust_cylinder_dbh, robust_mad_keep
from app.maturity.services.image_io import (
    estimate_intrinsics_from_metadata,
    extract_camera_metadata,
    load_rgb,
    resize_for_processing,
    safe_name,
)
from app.maturity.services.maturity import get_maturity_predictor
from app.maturity.services.runtime import log_elapsed, select_device
from app.maturity.services.segmentation import (
    detection_stage_label,
    run_trunk_detection,
    run_trunk_segmentation,
    segmentation_stage_label,
    save_segmentation_overlays,
)


def _default_log(message: str) -> None:
    print(message)



def validate_required_files() -> None:
    if not MATURITY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Maturity deployment bundle not found: {MATURITY_MODEL_PATH}")
    if RUN_FINE_TUNED_INTERNAL and not TUNED_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Fine-tuned UniDepth checkpoint not found: {TUNED_CHECKPOINT_PATH}")


def prepare_run_directories() -> dict[str, Path | str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id

    folders: dict[str, Path | str] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "run_dir": run_dir,
        "uploads_dir": run_dir / "uploads",
        "processing_dir": run_dir / "processing",
        "masks_dir": run_dir / "masks",
        "overlays_dir": run_dir / "overlays",
        "measurements_dir": run_dir / "measurements",
        "plots_dir": run_dir / "plots",
        "predictions_dir": run_dir / "predictions",
        "exports_dir": run_dir / "exports",
    }

    for key, folder in folders.items():
        if key.endswith("_dir") and isinstance(folder, Path):
            folder.mkdir(parents=True, exist_ok=True)

    return folders


def build_image_records(tree_inputs: list[dict[str, Any]], folders: dict[str, Path | str]) -> dict[str, dict]:
    records: dict[str, dict] = {}

    for tree_position, tree in enumerate(tree_inputs, start=1):
        tree_id = tree["tree_id"]
        for image_position, item in enumerate(tree["uploaded_items"], start=1):
            original_name = item.get("name") or f"image_{image_position}.jpg"
            extension = Path(original_name).suffix.lower()
            if extension not in ALLOWED_IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image type: {original_name}")

            unique_name = (
                f"{tree_position:02d}_{safe_name(tree_id)}_"
                f"{image_position:02d}_{safe_name(original_name)}"
            )
            raw_path = Path(folders["uploads_dir"]) / unique_name
            raw_path.write_bytes(item["content"])

            original = load_rgb(raw_path)
            camera_metadata = extract_camera_metadata(raw_path, original.width, original.height)
            processing, scale = resize_for_processing(original, MAX_LONG_SIDE)
            processing_path = Path(folders["processing_dir"]) / f"{Path(unique_name).stem}.png"
            processing.save(processing_path)

            exif_intrinsics = estimate_intrinsics_from_metadata(
                camera_metadata,
                processing.width,
                processing.height,
            )

            image_key = f"tree_{tree_position:02d}__image_{image_position:02d}"
            records[image_key] = {
                "image_key": image_key,
                "display_name": original_name,
                "tree_id": tree_id,
                "tree_position": tree_position,
                "image_position": image_position,
                "raw_path": raw_path,
                "processing_path": processing_path,
                "original_width": original.width,
                "original_height": original.height,
                "width": processing.width,
                "height": processing.height,
                "resize_scale": scale,
                "camera_metadata": camera_metadata,
                "exif_intrinsics": exif_intrinsics,
            }

    if not records:
        raise RuntimeError("No valid images were prepared.")

    return records


def measure_all_images(
    image_records: dict[str, dict],
    trunk_masks: dict[str, np.ndarray],
    original_predictions: dict[str, dict],
    fine_predictions: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    measurement_details: dict[str, dict] = {}

    for image_key, record in image_records.items():
        mask = trunk_masks[image_key]
        original_result = robust_cylinder_dbh(
            original_predictions[image_key]["depth"],
            mask,
            original_predictions[image_key]["K"],
        )
        measurement_details[image_key] = {
            "original": original_result,
        }
        camera = record.get("camera_metadata", {})

        row = {
            "image_key": image_key,
            "tree_id": record["tree_id"],
            "image": record["display_name"],
            "camera_make": camera.get("make"),
            "camera_model": camera.get("model"),
            "camera_lens_model": camera.get("lens_model"),
            "camera_focal_length_mm": camera.get("focal_length_mm"),
            "camera_focal_length_35mm": camera.get("focal_length_35mm"),
            "camera_has_focal_metadata": camera.get("has_focal_metadata"),
            "original_intrinsics_source": original_predictions[image_key].get("K_source"),
            "original_intrinsics_note": original_predictions[image_key].get("K_note"),
            "original_success": original_result.get("success", False),
            "original_dbh_cm": original_result.get("predicted_dbh_cm", np.nan),
            "original_distance_m": original_result.get("predicted_distance_m", np.nan),
            "original_quality": original_result.get("quality", "failed"),
            "original_quality_reasons": original_result.get(
                "quality_reasons",
                original_result.get("reason", "unknown"),
            ),
            "original_rows_used": original_result.get("rows_used", 0),
            "original_row_cv": original_result.get("row_cv", np.nan),
        }

        if fine_predictions is not None:
            fine_prediction = fine_predictions.get(image_key)
            if fine_prediction is None:
                fine_result = {"success": False, "reason": "fine-tuned prediction missing"}
                fine_intrinsics_source = "missing"
            else:
                fine_result = robust_cylinder_dbh(
                    fine_prediction["depth"],
                    mask,
                    fine_prediction["K"],
                )
                fine_intrinsics_source = fine_prediction.get("K_source")

            measurement_details[image_key]["fine_tuned_internal"] = fine_result
            row.update(
                {
                    "fine_tuned_intrinsics_source_internal": fine_intrinsics_source,
                    "fine_tuned_success_internal": fine_result.get("success", False),
                    "fine_tuned_dbh_cm_internal": fine_result.get("predicted_dbh_cm", np.nan),
                    "fine_tuned_distance_m_internal": fine_result.get("predicted_distance_m", np.nan),
                    "fine_tuned_quality_internal": fine_result.get("quality", "failed"),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows), measurement_details


def aggregate_original_tree_dbh(tree_id: str, image_df: pd.DataFrame) -> dict[str, Any]:
    tree_rows = image_df[image_df["tree_id"].astype(str).eq(str(tree_id))].copy()
    successful = tree_rows[tree_rows["original_success"].fillna(False)].copy()

    if successful.empty:
        return {
            "success": False,
            "reason": "No successful original-model DBH measurement",
            "images_uploaded": int(len(tree_rows)),
            "successful_images": 0,
        }

    high_quality = successful[successful["original_quality"].eq("high")].copy()
    preferred = high_quality if not high_quality.empty else successful
    values = pd.to_numeric(preferred["original_dbh_cm"], errors="coerce").to_numpy(dtype=float)
    keep = robust_mad_keep(values)

    if keep.sum() >= 1:
        preferred = preferred.iloc[np.flatnonzero(keep)].copy()
        values = pd.to_numeric(preferred["original_dbh_cm"], errors="coerce").to_numpy(dtype=float)

    final_dbh = float(np.median(values))
    dbh_std = float(np.std(values))
    dbh_range = float(np.max(values) - np.min(values))
    relative_spread = float(dbh_std / max(final_dbh, 1e-6))
    quality_reasons = []

    if high_quality.empty:
        quality_reasons.append("review-quality images used")
    if len(values) >= 2 and relative_spread > 0.15:
        quality_reasons.append("multi-image DBH spread")
    if len(values) == 1:
        quality_reasons.append("single successful image")

    return {
        "success": True,
        "final_dbh_cm": final_dbh,
        "final_quality": "high" if not quality_reasons else "review",
        "quality_reasons": "ok" if not quality_reasons else " | ".join(quality_reasons),
        "images_uploaded": int(len(tree_rows)),
        "successful_images": int(len(successful)),
        "images_used_for_final": int(len(preferred)),
        "dbh_std_cm": dbh_std,
        "dbh_range_cm": dbh_range,
        "relative_spread": relative_spread,
        "used_image_keys": preferred["image_key"].tolist(),
    }


def save_dbh_consistency_plot(
    tree_id: str,
    image_df: pd.DataFrame,
    aggregation: dict[str, Any],
    output_path: Path,
) -> None:
    tree_rows = image_df[image_df["tree_id"].astype(str).eq(str(tree_id))].copy().reset_index(drop=True)
    plt.figure(figsize=(max(8, len(tree_rows) * 1.7), 5.5))
    positions = np.arange(len(tree_rows))
    original_values = pd.to_numeric(tree_rows["original_dbh_cm"], errors="coerce")
    plt.plot(positions, original_values, marker="o", linewidth=2, label="Original UniDepthV2 (used)")

    if aggregation.get("success") or aggregation.get("dbh_success"):
        plt.axhline(
            aggregation["final_dbh_cm"],
            linestyle="--",
            linewidth=1.5,
            label=f"Final median DBH {aggregation['final_dbh_cm']:.2f} cm",
        )

    plt.xticks(positions, tree_rows["image"], rotation=25, ha="right")
    plt.ylabel("DBH (cm)")
    plt.title(f"{tree_id} - original-model DBH consistency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_dbh_measurement_row_overlays(
    image_records: dict[str, dict],
    trunk_masks: dict[str, np.ndarray],
    measurement_details: dict[str, dict],
    folders: dict[str, Path | str],
) -> list[Path]:
    output_paths = []
    output_dir = Path(folders["measurements_dir"])

    for image_key, record in image_records.items():
        image = np.asarray(load_rgb(record["processing_path"])).copy()
        mask = np.asarray(trunk_masks[image_key], dtype=bool)

        green = np.zeros_like(image)
        green[..., 1] = 255
        image[mask] = (0.72 * image[mask] + 0.28 * green[mask]).astype(np.uint8)

        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        line_width = max(2, int(round(min(width, height) * 0.004)))
        endpoint_radius = max(3, int(round(line_width * 1.4)))

        rows = (
            measurement_details.get(image_key, {})
            .get("original", {})
            .get("rows", [])
        )
        for row in rows:
            y = int(round(row.get("y", 0)))
            left = int(round(row.get("left", 0)))
            right = int(round(row.get("right", 0)))
            if y < 0 or y >= height or right <= left:
                continue

            left = max(0, min(left, width - 1))
            right = max(0, min(right, width - 1))
            draw.line([(left, y), (right, y)], fill=(255, 216, 64), width=line_width)
            for x in (left, right):
                draw.ellipse(
                    [
                        x - endpoint_radius,
                        y - endpoint_radius,
                        x + endpoint_radius,
                        y + endpoint_radius,
                    ],
                    fill=(255, 255, 255),
                    outline=(255, 216, 64),
                    width=max(1, line_width // 2),
                )

        path = output_dir / f"{image_key}_dbh_rows.png"
        canvas.save(path)
        output_paths.append(path)

    return output_paths


def save_maturity_plot(
    tree_id: str,
    curve_df: pd.DataFrame,
    maturity_result: dict[str, Any],
    maturity_threshold: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(curve_df["dbh_cm"], curve_df["maturity_score"], linewidth=2.5, label="Maturity curve")
    plt.axhline(
        maturity_threshold,
        linestyle="--",
        linewidth=1,
        label=f"Mature threshold ({maturity_threshold:.0f}%)",
    )
    plt.scatter(
        [maturity_result["dbh_cm"]],
        [maturity_result["maturity_score"]],
        s=105,
        zorder=5,
        label="Current tree",
    )
    plt.axvline(maturity_result["dbh_cm"], linestyle=":", linewidth=1)
    plt.xlabel("DBH (cm)")
    plt.ylabel("Maturity (%)")
    plt.title(
        f"{tree_id} - {maturity_result['maturity_class']} "
        f"({maturity_result['maturity_score']:.1f}%)"
    )
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def run_merged_analysis(
    tree_inputs: list[dict[str, Any]],
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    log = log or _default_log
    total_started_at = perf_counter()
    validate_required_files()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    setup_started_at = perf_counter()
    folders = prepare_run_directories()
    device = select_device(log)
    log("=" * 72)
    log("MODULE 2 MERGED ANALYSIS: DBH + MATURITY")
    log("=" * 72)
    log(f"Trees: {len(tree_inputs)}")
    log(f"Run folder: {folders['run_dir']}")
    log(f"Final DBH branch: original UniDepthV2 ({UNIDEPTH_MODEL_ID})")
    if RUN_FINE_TUNED_INTERNAL:
        log("Fine-tuned UniDepthV2: internal comparison enabled")
    else:
        log("Fine-tuned UniDepthV2: skipped for runtime speed")
    log(f"Intrinsics mode: {INTRINSICS_MODE}")
    log(f"Processing max long side: {MAX_LONG_SIDE}px")
    log(f"Detector backend: {DETECTOR_BACKEND}")
    log(f"Segmenter backend: {SEGMENTER_BACKEND}")
    log_elapsed(log, "Runtime setup", setup_started_at)

    stage_started_at = perf_counter()
    image_records = build_image_records(tree_inputs, folders)
    log_elapsed(log, "Image preparation", stage_started_at)
    log(f"Images: {len(image_records)}")

    focal_count = sum(
        1
        for record in image_records.values()
        if record["camera_metadata"].get("has_focal_metadata")
    )
    log(f"Images with focal EXIF metadata: {focal_count} of {len(image_records)}")

    stage_started_at = perf_counter()
    detections = run_trunk_detection(image_records, device, log)
    log_elapsed(log, detection_stage_label(), stage_started_at)

    stage_started_at = perf_counter()
    trunk_masks, mask_df = run_trunk_segmentation(image_records, detections, folders, device, log)
    log_elapsed(log, segmentation_stage_label(), stage_started_at)

    stage_started_at = perf_counter()
    save_segmentation_overlays(image_records, detections, trunk_masks, folders)
    log_elapsed(log, "Segmentation overlay export", stage_started_at)

    stage_started_at = perf_counter()
    original_predictions = run_original_unidepth(image_records, folders, device, log)
    log_elapsed(log, "Original UniDepth inference", stage_started_at)

    fine_predictions = None
    if RUN_FINE_TUNED_INTERNAL:
        stage_started_at = perf_counter()
        fine_predictions, _checkpoint = run_fine_tuned_unidepth(image_records, folders, device, log)
        log_elapsed(log, "Fine-tuned UniDepth inference", stage_started_at)
    else:
        log("[4/4] Skipping fine-tuned UniDepthV2 internal comparison")
        log("[timing] Fine-tuned UniDepth inference: skipped")

    stage_started_at = perf_counter()
    per_image_df, measurement_details = measure_all_images(
        image_records,
        trunk_masks,
        original_predictions,
        fine_predictions,
    )
    log_elapsed(log, "DBH geometry measurement", stage_started_at)

    stage_started_at = perf_counter()
    save_dbh_measurement_row_overlays(image_records, trunk_masks, measurement_details, folders)
    log_elapsed(log, "DBH measurement row overlay export", stage_started_at)

    stage_started_at = perf_counter()
    predictor = get_maturity_predictor()
    log_elapsed(log, "Maturity model load", stage_started_at)

    tree_results = []
    curve_tables: dict[str, pd.DataFrame] = {}

    stage_started_at = perf_counter()
    for tree in tree_inputs:
        tree_id = tree["tree_id"]
        aggregation = aggregate_original_tree_dbh(tree_id, per_image_df)

        if not aggregation.get("success"):
            tree_results.append(
                {
                    "tree_id": tree_id,
                    "species": tree["species"],
                    "dbh_success": False,
                    "reason": aggregation.get("reason", "Unknown DBH error"),
                    "images_uploaded": aggregation.get("images_uploaded", 0),
                    "successful_images": aggregation.get("successful_images", 0),
                }
            )
            continue

        final_dbh = float(aggregation["final_dbh_cm"])
        curve_df, maturity_result = predictor.generate_maturity_curve(
            species=tree["species"],
            dbh_cm=final_dbh,
            **tree["optional_inputs"],
        )
        curve_tables[tree_id] = curve_df

        tree_result = {
            "tree_id": tree_id,
            "species": tree["species"],
            "dbh_success": True,
            "final_dbh_cm": final_dbh,
            "dbh_quality": aggregation["final_quality"],
            "dbh_quality_reasons": aggregation["quality_reasons"],
            "images_uploaded": aggregation["images_uploaded"],
            "successful_images": aggregation["successful_images"],
            "images_used_for_final": aggregation["images_used_for_final"],
            "dbh_std_cm": aggregation["dbh_std_cm"],
            "dbh_range_cm": aggregation["dbh_range_cm"],
            "maturity_score": maturity_result["maturity_score"],
            "maturity_class": maturity_result["maturity_class"],
            "mature": maturity_result["mature"],
            "optional_fields_used": maturity_result["optional_fields_used"],
            "curve_monotonic_violations": maturity_result["curve_monotonic_violations"],
        }
        tree_results.append(tree_result)

    log_elapsed(log, "Final maturity prediction", stage_started_at)

    stage_started_at = perf_counter()
    tree_summary_df = pd.DataFrame(tree_results)
    exports_dir = Path(folders["exports_dir"])
    public_per_image_df = per_image_df.drop(
        columns=[column for column in per_image_df.columns if "fine_tuned" in column]
    )

    per_image_df.to_csv(exports_dir / "per_image_dbh_internal.csv", index=False)
    public_per_image_df.to_csv(exports_dir / "per_image_original_dbh.csv", index=False)
    tree_summary_df.to_csv(exports_dir / "tree_summary.csv", index=False)
    mask_df.to_csv(exports_dir / "segmentation_quality.csv", index=False)

    for tree_id, curve_df in curve_tables.items():
        curve_df.to_csv(exports_dir / f"{safe_name(tree_id)}_maturity_curve.csv", index=False)
    log_elapsed(log, "CSV export", stage_started_at)

    stage_started_at = perf_counter()
    plots_dir = Path(folders["plots_dir"])
    for tree_result in tree_results:
        if not tree_result.get("dbh_success"):
            continue
        tree_id = tree_result["tree_id"]
        save_maturity_plot(
            tree_id,
            curve_tables[tree_id],
            {
                "dbh_cm": tree_result["final_dbh_cm"],
                "maturity_score": tree_result["maturity_score"],
                "maturity_class": tree_result["maturity_class"],
            },
            predictor.maturity_threshold,
            plots_dir / f"{safe_name(tree_id)}_maturity_curve.png",
        )
    log_elapsed(log, "Plot export", stage_started_at)

    stage_started_at = perf_counter()
    report = {
        "created_at": datetime.now().isoformat(),
        "local_runtime": True,
        "original_dbh_model": UNIDEPTH_MODEL_ID,
        "fine_tuned_checkpoint": str(TUNED_CHECKPOINT_PATH),
        "fine_tuned_executed": RUN_FINE_TUNED_INTERNAL,
        "fine_tuned_usage": (
            "internal prediction only; not shown in final output and not used by maturity"
            if RUN_FINE_TUNED_INTERNAL
            else "skipped; code retained behind SMARTTIMBER_RUN_FINE_TUNED_INTERNAL"
        ),
        "maturity_bundle": str(MATURITY_MODEL_PATH),
        "intrinsics_mode": INTRINSICS_MODE,
        "tree_results": tree_results,
    }
    (exports_dir / "merged_run_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    log_elapsed(log, "Report export", stage_started_at)

    stage_started_at = perf_counter()
    archive_base = OUTPUTS_DIR / f"module2_merged_results_{folders['run_id']}"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=Path(folders["run_dir"])))
    zip_path = Path(folders["run_dir"]) / "module2_merged_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.move(str(archive_path), zip_path)
    log_elapsed(log, "ZIP archive", stage_started_at)
    log_elapsed(log, "Total analysis", total_started_at)

    log("Saved local results")
    log(str(folders["run_dir"]))
    log(f"ZIP: {zip_path}")
    if RUN_FINE_TUNED_INTERNAL:
        log("Fine-tuned DBH values were saved only in exports/per_image_dbh_internal.csv")
    else:
        log("Fine-tuned DBH branch was not executed for this run")

    return {
        "tree_summary": tree_summary_df,
        "per_image_internal": per_image_df,
        "segmentation_quality": mask_df,
        "measurement_details": measurement_details,
        "tree_results": tree_results,
        "curve_tables": curve_tables,
        "folders": folders,
        "zip_path": zip_path,
    }
