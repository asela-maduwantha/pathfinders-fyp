"""
End-to-End Integration Validation Tool.
Runs full-scene labeled photos through the REAL deployed pipeline (YOLO trunk
segmentation -> assessability -> bark ROI candidates -> quality gate -> RAGLO
classification -> calibrated open-set -> multi-ROI consensus), exactly as
app/main.py does for a real upload, and reports per-tree/per-class outcomes
against the folder-name ground truth in integration_validation/<species>/.

Unlike tools/calibrate_open_set.py (which feeds a whole image directly into
prepare_raglo_images as if it were already a clean bark ROI), this tool does
NOT bypass detection/segmentation -- it measures what a real user actually
gets, which matters here because integration_validation/ currently contains
full forest scene photos, not pre-cropped bark patches.

Writes each image's rows to reports/integration_validation_results.csv
immediately (flushed to disk), and skips images whose job_id is already in
that file on a re-run -- safe to interrupt and resume.

Run with:
    .venv\\Scripts\\python.exe tools/evaluate_integration_validation.py --val-dir integration_validation
    .venv\\Scripts\\python.exe tools/evaluate_integration_validation.py --summary-only   # just re-print stats
"""
import sys
import csv
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import Counter, defaultdict

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
OUT_CSV = REPORTS_DIR / "integration_validation_results.csv"

CLASSES = ["mahogany", "pine", "rubber", "teak", "unknown"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
FIELDNAMES = ["true_class", "image", "job_id", "tree_id", "tree_status",
              "final_species", "known_votes", "candidate_count",
              "agreement_ratio", "reason", "elapsed_s"]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_existing_job_ids() -> Set[str]:
    if not OUT_CSV.exists():
        return set()
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        return {row["job_id"] for row in csv.DictReader(f)}


def append_rows(rows: List[Dict[str, Any]]) -> None:
    is_new = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def rows_for_image(true_class: str, img_path: Path, job_id: str,
                    response: Dict[str, Any], elapsed: float) -> List[Dict[str, Any]]:
    trees = response.get("trees", [])
    if not trees:
        return [{
            "true_class": true_class, "image": img_path.name, "job_id": job_id,
            "tree_id": None, "tree_status": "no_trees_detected", "final_species": None,
            "known_votes": None, "candidate_count": None, "agreement_ratio": None,
            "reason": None, "elapsed_s": elapsed,
        }]
    out = []
    for tree in trees:
        raglo = tree.get("raglo", {})
        consensus = raglo.get("consensus", {})
        out.append({
            "true_class": true_class, "image": img_path.name, "job_id": job_id,
            "tree_id": tree.get("tree_id"), "tree_status": tree.get("status"),
            "final_species": raglo.get("final_species"),
            "known_votes": consensus.get("known_votes"),
            "candidate_count": consensus.get("candidate_count"),
            "agreement_ratio": consensus.get("agreement_ratio"),
            "reason": consensus.get("reason") or tree.get("bark", {}).get("reason"),
            "elapsed_s": elapsed,
        })
    return out


def run_evaluation(val_dir: Path, results_dir: Path, limit_per_class: int) -> None:
    from app.config import DETECTOR_MODEL_PATH, CLASSIFIER_MODEL_PATH
    from app.inference.tree_detector import TreeDetector
    from app.inference.raglo_classifier import RAGLOClassifier
    from app.inference.integrated_pipeline import IntegratedPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Inference device: {device}")

    tree_detector = TreeDetector(DETECTOR_MODEL_PATH, device=device)
    raglo_classifier = RAGLOClassifier(CLASSIFIER_MODEL_PATH, device=device)
    pipeline = IntegratedPipeline(tree_detector, raglo_classifier)

    done_job_ids = load_existing_job_ids()
    if done_job_ids:
        log(f"Resuming: {len(done_job_ids)} image(s) already recorded, will be skipped.")

    total_images = 0
    total_start = time.perf_counter()

    for true_class in CLASSES:
        folder = val_dir / true_class
        if not folder.exists():
            continue
        images = sorted(p for p in folder.iterdir() if p.suffix in IMAGE_EXTS)
        if limit_per_class > 0:
            images = images[:limit_per_class]
        if not images:
            continue

        log(f"\n=== {true_class}: {len(images)} image(s) ===")

        for img_path in images:
            job_id = f"{true_class}__{img_path.stem}"
            if job_id in done_job_ids:
                continue

            started = time.perf_counter()
            try:
                response = pipeline.run_pipeline(job_id, img_path, results_dir)
                elapsed = round(time.perf_counter() - started, 2)
                rows = rows_for_image(true_class, img_path, job_id, response, elapsed)
                log(f"  {img_path.name}: {len(response.get('trees', []))} tree(s), "
                    f"{elapsed}s | summary={response.get('summary')}")
            except Exception as e:
                elapsed = round(time.perf_counter() - started, 2)
                rows = [{
                    "true_class": true_class, "image": img_path.name, "job_id": job_id,
                    "tree_id": None, "tree_status": "pipeline_error", "final_species": None,
                    "known_votes": None, "candidate_count": None, "agreement_ratio": None,
                    "reason": str(e), "elapsed_s": elapsed,
                }]
                log(f"  [ERROR] {img_path.name}: {e}")

            append_rows(rows)
            total_images += 1

    total_elapsed = time.perf_counter() - total_start
    log(f"\nProcessed {total_images} new image(s) in {total_elapsed/60:.1f} min.")


def summarize() -> None:
    if not OUT_CSV.exists():
        log("No results yet.")
        return
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["true_class"]].append(r)

    log("\n" + "=" * 70)
    log(" PER-CLASS TREE OUTCOME BREAKDOWN")
    log("=" * 70)
    for true_class, class_rows in by_class.items():
        status_counts = Counter(r["tree_status"] for r in class_rows)
        log(f"\n{true_class} (n={len(class_rows)} rows):")
        for status, count in status_counts.most_common():
            log(f"    {status:22s} {count:4d}  ({count/len(class_rows)*100:.1f}%)")

    log("\n" + "=" * 70)
    log(" CONFUSION MATRIX (trees that reached final status = 'known')")
    log("=" * 70)
    confusion = defaultdict(Counter)
    for r in rows:
        if r["tree_status"] == "known" and r["final_species"]:
            confusion[r["true_class"]][r["final_species"].lower()] += 1

    predicted_labels = sorted({sp for c in confusion.values() for sp in c})
    header = "true\\pred".ljust(12) + "".join(p.ljust(12) for p in predicted_labels)
    log(header)
    for true_class in CLASSES:
        if true_class not in confusion:
            continue
        line = true_class.ljust(12)
        for p in predicted_labels:
            line += str(confusion[true_class].get(p, 0)).ljust(12)
        log(line)

    log("\n" + "=" * 70)
    log(" KNOWN-CLASS OUTCOME RATE (excludes 'unknown' folder)")
    log("=" * 70)
    for true_class, class_rows in by_class.items():
        if true_class == "unknown":
            continue
        bark_ready_rows = [r for r in class_rows if r["tree_status"] in
                            ("known", "uncertain", "open_set_rejected")]
        if not bark_ready_rows:
            continue
        rejected = sum(1 for r in bark_ready_rows if r["tree_status"] == "open_set_rejected")
        uncertain = sum(1 for r in bark_ready_rows if r["tree_status"] == "uncertain")
        known = sum(1 for r in bark_ready_rows if r["tree_status"] == "known")
        n = len(bark_ready_rows)
        log(f"{true_class:12s} n={n:3d}  known={known:3d} ({known/n*100:5.1f}%)  "
            f"uncertain={uncertain:3d} ({uncertain/n*100:5.1f}%)  "
            f"open_set_rejected={rejected:3d} ({rejected/n*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Integration Validation Tool")
    parser.add_argument("--val-dir", type=str, default="integration_validation")
    parser.add_argument("--results-dir", type=str, default="reports/integration_eval_runs")
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 = no limit")
    parser.add_argument("--summary-only", action="store_true",
                         help="Skip inference; just re-summarize the existing CSV")
    args = parser.parse_args()

    log("=" * 70)
    log("      END-TO-END INTEGRATION VALIDATION TOOL")
    log("=" * 70)

    if not args.summary_only:
        val_dir = Path(args.val_dir)
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        run_evaluation(val_dir, results_dir, args.limit_per_class)

    summarize()


if __name__ == "__main__":
    main()
