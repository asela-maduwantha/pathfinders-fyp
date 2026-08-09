"""
Integration Logger Module.
Logs candidate inference results, quality metrics, and open-set z-scores to CSV for research auditing.
"""
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from app.config import LOGS_DIR

CSV_FILE_PATH = LOGS_DIR / "raglo_integration_results.csv"

CSV_HEADERS = [
    "timestamp",
    "job_id",
    "tree_id",
    "candidate_id",
    "quality_score",
    "occupancy",
    "sharpness",
    "contrast",
    "brightness",
    "raw_predicted_class",
    "raw_confidence",
    "prototype_distance",
    "prototype_z",
    "energy",
    "energy_z",
    "combined_unknownness",
    "threshold",
    "decision_margin",
    "open_set_rejected",
    "consensus_status",
    "final_species"
]


def log_candidate_inference(
    job_id: str,
    tree_id: int,
    candidate_id: int,
    quality_metrics: Dict[str, float],
    clf_dict: Dict[str, Any],
    open_set_dict: Dict[str, Any],
    consensus_status: str,
    final_species: str
):
    """
    Append single candidate inference record to research integration CSV log.
    """
    file_exists = CSV_FILE_PATH.exists()

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "job_id": job_id,
        "tree_id": tree_id,
        "candidate_id": candidate_id,

        "quality_score": quality_metrics.get("quality_score", 0.0),
        "occupancy": quality_metrics.get("occupancy", 0.0),
        "sharpness": quality_metrics.get("sharpness", 0.0),
        "contrast": quality_metrics.get("contrast", 0.0),
        "brightness": quality_metrics.get("brightness", 0.0),

        "raw_predicted_class": clf_dict.get("raw_predicted_class", ""),
        "raw_confidence": clf_dict.get("raw_confidence", 0.0),

        "prototype_distance": open_set_dict.get("prototype_distance", 0.0),
        "prototype_z": open_set_dict.get("prototype_z", 0.0),

        "energy": open_set_dict.get("energy", 0.0),
        "energy_z": open_set_dict.get("energy_z", 0.0),

        "combined_unknownness": open_set_dict.get("combined_unknownness", 0.0),
        "threshold": open_set_dict.get("threshold", 0.0),
        "decision_margin": open_set_dict.get("decision_margin", 0.0),

        "open_set_rejected": open_set_dict.get("open_set_rejected", False),
        "consensus_status": consensus_status,
        "final_species": final_species
    }

    try:
        with open(CSV_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[WARN] Failed to write integration log row: {e}")
