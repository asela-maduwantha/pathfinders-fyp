"""
Open-Set Calibration Evaluation Tool.
Evaluates candidate open-set thresholds on a validation dataset without retraining the model.
Generates calibration metrics CSV and summary JSON reports.
Run with:
    python tools/calibrate_open_set.py --val-dir integration_validation
"""
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import CLASSIFIER_MODEL_PATH
from app.inference.raglo_classifier import RAGLOClassifier
from app.inference.raglo_preprocessing import prepare_raglo_images, prepare_raglo_tensors
from app.utils.image_utils import load_pil_image_safe

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def evaluate_dataset(classifier: RAGLOClassifier, val_dir: Path) -> List[Dict[str, Any]]:
    records = []
    classes = ["mahogany", "pine", "rubber", "teak", "unknown"]

    for class_folder in classes:
        folder = val_dir / class_folder
        if not folder.exists():
            continue

        images = [f for f in folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        print(f"Evaluating {len(images)} images in category '{class_folder}'...")

        for img_path in images:
            try:
                pil_img = load_pil_image_safe(img_path)
                g_img, l_imgs, _ = prepare_raglo_images(pil_img)
                g_t, l_t = prepare_raglo_tensors(g_img, l_imgs)

                res = classifier.classify_candidate(g_t, l_t)
                open_set = res["open_set"]

                records.append({
                    "filename": img_path.name,
                    "true_class": class_folder,
                    "is_known_ground_truth": (class_folder != "unknown"),
                    "raw_predicted_class": res["classification"]["raw_predicted_class"],
                    "raw_confidence": res["classification"]["raw_confidence"],
                    "prototype_distance": open_set["prototype_distance"],
                    "prototype_z": open_set["prototype_z"],
                    "energy": open_set["energy"],
                    "energy_z": open_set["energy_z"],
                    "combined_unknownness": open_set["combined_unknownness"]
                })
            except Exception as e:
                print(f"  [WARN] Skipping {img_path.name}: {e}")

    return records


def calibrate_thresholds(records: List[Dict[str, Any]], candidates: List[float]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    calib_rows = []
    best_summary = {}

    known_records = [r for r in records if r["is_known_ground_truth"]]
    unknown_records = [r for r in records if not r["is_known_ground_truth"]]

    num_known = max(len(known_records), 1)
    num_unknown = max(len(unknown_records), 1)

    print(f"\nTotal Ground-Truth Known Images: {len(known_records)}")
    print(f"Total Ground-Truth Unknown Images: {len(unknown_records)}")

    for thresh in candidates:
        known_accepted = sum(1 for r in known_records if r["combined_unknownness"] < thresh)
        known_false_rejected = len(known_records) - known_accepted

        unknown_rejected = sum(1 for r in unknown_records if r["combined_unknownness"] >= thresh)
        unknown_false_accepted = len(unknown_records) - unknown_rejected

        known_accept_rate = known_accepted / num_known
        known_false_reject_rate = known_false_rejected / num_known
        unknown_reject_rate = unknown_rejected / num_unknown
        unknown_false_accept_rate = unknown_false_accepted / num_unknown

        balanced_acc = (known_accept_rate + unknown_reject_rate) / 2.0

        row = {
            "threshold": round(thresh, 4),
            "known_accept_rate": round(known_accept_rate, 4),
            "known_false_reject_rate": round(known_false_reject_rate, 4),
            "unknown_reject_rate": round(unknown_reject_rate, 4),
            "unknown_false_accept_rate": round(unknown_false_accept_rate, 4),
            "balanced_accuracy": round(balanced_acc, 4)
        }
        calib_rows.append(row)

    if calib_rows:
        best_row = max(calib_rows, key=lambda r: r["balanced_accuracy"])
        best_summary = {
            "dataset_total_samples": len(records),
            "known_samples": len(known_records),
            "unknown_samples": len(unknown_records),
            "optimal_candidate": best_row
        }

    return calib_rows, best_summary


def main():
    parser = argparse.ArgumentParser(description="Open-Set Calibration Evaluation Tool")
    parser.add_argument("--val-dir", type=str, default="integration_validation", help="Validation dataset directory")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)

    print("=" * 65)
    print("      OPEN-SET CALIBRATION EVALUATION TOOL")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device}")

    classifier = RAGLOClassifier(CLASSIFIER_MODEL_PATH, device=device)

    if not val_dir.exists():
        print(f"\n[INFO] Validation directory '{val_dir}' not found.")
        print("Creating directory structure for validation datasets:")
        print(f"  {val_dir}/mahogany/")
        print(f"  {val_dir}/pine/")
        print(f"  {val_dir}/rubber/")
        print(f"  {val_dir}/teak/")
        print(f"  {val_dir}/unknown/")

        for sub in ["mahogany", "pine", "rubber", "teak", "unknown"]:
            (val_dir / sub).mkdir(parents=True, exist_ok=True)

        print("\nPlaceholder validation directory created. Add validation images and rerun tool.")
        records = []
    else:
        records = evaluate_dataset(classifier, val_dir)

    candidate_thresholds = np.linspace(-2.0, 2.0, 41).tolist()
    calib_rows, summary = calibrate_thresholds(records, candidate_thresholds)

    # Save CSV Report
    csv_path = REPORTS_DIR / "open_set_calibration.csv"
    if calib_rows:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
            writer.writeheader()
            writer.writerows(calib_rows)
        print(f"\nSaved CSV calibration report to: {csv_path}")

    # Save JSON Summary
    json_path = REPORTS_DIR / "open_set_calibration_summary.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON calibration summary to: {json_path}")

    print("\n" + "=" * 65)
    print(" CALIBRATION COMPLETE: Review reports before adjusting threshold.")
    print("=" * 65)


if __name__ == "__main__":
    main()
