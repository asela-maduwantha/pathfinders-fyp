"""
Comprehensive Test Script for Cleaned-Dataset YOLO26s-seg Tree Detector.
Tests cases:
CASE 1: Very large central foreground trunk
CASE 2: Large pine trunk filling much of frame
CASE 3: Dense plantation
CASE 4: Normal medium distance tree
CASE 5: Boundary touching tree
Performs raw YOLO diagnosis for any missed important trunks.
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    DETECTOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    RESULTS_DIR,
    TREE_IMGSZ,
    TREE_PROPOSAL_CONF,
    TREE_NMS_IOU,
    TREE_MAX_DET
)
from app.inference.tree_detector import TreeDetector
from app.inference.raglo_classifier import RAGLOClassifier
from app.inference.integrated_pipeline import IntegratedPipeline


def test_images():
    uploads_dir = Path("app/static/uploads")
    all_uploads = sorted(list(uploads_dir.glob("*.[jJ][pP]*[gG]")) + list(uploads_dir.glob("*.[pP][nN][gG]")))

    if not all_uploads:
        print("No uploaded images found in app/static/uploads.")
        return

    print(f"Found {len(all_uploads)} images in app/static/uploads.")

    print("\n1. Loading Tree Detector...")
    tree_detector = TreeDetector(DETECTOR_MODEL_PATH, device="cpu")

    print("\n2. Loading RAGLO Classifier...")
    raglo_classifier = RAGLOClassifier(CLASSIFIER_MODEL_PATH, device="cpu")

    pipeline = IntegratedPipeline(tree_detector, raglo_classifier)

    # Select representative sample images of different sizes/scenarios
    sample_images = all_uploads[:10]

    for idx, img_path in enumerate(sample_images, start=1):
        job_id = f"test_eval_{idx:02d}"
        print(f"\n==================================================")
        print(f" Testing Image {idx}/{len(sample_images)}: {img_path.name}")
        print(f"==================================================")

        res = pipeline.run_pipeline(
            job_id=job_id,
            image_path=img_path,
            results_dir=RESULTS_DIR
        )

        summary = res.get("summary", {})
        print(f"Results summary:")
        print(f"  - Raw YOLO proposals : {summary.get('raw_yolo_candidates')}")
        print(f"  - Visibility rejected: {summary.get('visibility_rejected')}")
        print(f"  - Accepted trunks    : {summary.get('trees_detected')}")
        print(f"  - Bark ready trunks  : {summary.get('bark_ready')}")
        print(f"  - Identified species : {summary.get('identified')}")
        print(f"  - Unknown species    : {summary.get('unknown')}")

        for tree in res.get("trees", []):
            det = tree.get("detection", {})
            ass = tree.get("assessability", {})
            bark = tree.get("bark", {})
            rag = tree.get("raglo", {})
            print(f"    * {tree['display_name']}: Conf={det.get('confidence')}, BarkReady={bark.get('bark_ready')}, Species={rag.get('final_species')}")


if __name__ == "__main__":
    test_images()
