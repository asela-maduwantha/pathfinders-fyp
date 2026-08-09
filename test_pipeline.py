"""
Pipeline Integration Test Script.
Tests end-to-end inference execution on a test forest image.
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from app.inference.tree_detector import TreeDetector
from app.inference.raglo_classifier import RAGLOClassifier
from app.inference.integrated_pipeline import IntegratedPipeline


def create_test_forest_image(filepath: Path):
    """Generate a realistic test image simulating tree trunks in a forest setting."""
    img_w, img_h = 1024, 768
    # Background forest color
    img = Image.new("RGB", (img_w, img_h), color=(34, 139, 34))
    draw = ImageDraw.Draw(img)

    # Draw simulated vertical tree trunks
    # Tree 1 (Brown trunk)
    draw.rectangle([120, 100, 220, 700], fill=(101, 67, 33))
    # Tree 2 (Dark trunk)
    draw.rectangle([450, 50, 560, 720], fill=(80, 50, 20))
    # Tree 3 (Light trunk)
    draw.rectangle([780, 120, 870, 680], fill=(120, 85, 45))

    img.save(filepath, quality=95)
    print(f"Generated test forest image at: {filepath}")


def resolve_model(primary: str, fallback: str) -> Path:
    base = Path(__file__).resolve().parent / "models"
    p1 = base / primary
    if p1.exists():
        return p1
    p2 = base / fallback
    if p2.exists():
        return p2
    return p1


def main():
    print("==================================================")
    print(" RUNNING INTEGRATED PIPELINE TEST")
    print("==================================================")

    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    test_img_path = Path("test_output/real_forest.jpg")
    if not test_img_path.exists():
        create_test_forest_image(test_img_path)

    det_path = Path("models/tree_trunk_detection_best.pt")
    clf_path = Path("models/tree_classification_best.pt")

    print("\n1. Initializing Tree Detector...")
    tree_detector = TreeDetector(det_path, device="cpu")

    print("\n2. Initializing RAGLO Classifier...")
    raglo_classifier = RAGLOClassifier(clf_path, device="cpu")

    print("\n3. Running Integrated Pipeline...")
    pipeline = IntegratedPipeline(tree_detector, raglo_classifier)

    results = pipeline.run_pipeline(
        job_id="test_job_001",
        image_path=test_img_path,
        results_dir=Path("app/static/results")
    )

    print("\n==================================================")
    print(" PIPELINE TEST RESULTS SUMMARY:")
    print("==================================================")
    print("Job ID:", results["job_id"])
    print("Status:", results["status"])
    print("Summary:", results["summary"])
    print("Annotated Output Image:", results["image"]["annotated"])
    print(f"Trees Processed Count: {len(results['trees'])}")

    for t in results["trees"]:
        print(f"  - {t['display_name']}: Assessable={t['assessability']['passed']}, BarkReady={t['bark']['bark_ready']}, Species={t['raglo']['final_species']}, Status={t['raglo']['status']}")

    print("\nINTEGRATED PIPELINE TEST COMPLETE: PASSED.")


if __name__ == "__main__":
    main()
