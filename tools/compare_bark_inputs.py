"""
Same-Tree Diagnostic Tool.
Compares 3 bark image representations of the same tree:
A. Manually cropped clean bark
B. Full tree/trunk crop
C. Automatically selected bark ROI

Evaluates RAGLO class probabilities, prototype distance, energy, combined unknownness,
threshold, decision margin, and open-set decisions across all three.
Run with:
    python tools/compare_bark_inputs.py --clean path/to/clean.jpg --trunk path/to/trunk.jpg --auto path/to/auto.jpg
"""
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image

# Ensure project root is in Python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import CLASSIFIER_MODEL_PATH
from app.inference.raglo_classifier import RAGLOClassifier
from app.inference.raglo_preprocessing import prepare_raglo_images, prepare_raglo_tensors
from app.utils.image_utils import load_pil_image_safe


def evaluate_single_input(classifier: RAGLOClassifier, img_path: Path, label: str):
    print("-" * 65)
    print(f" Representation [{label}]: {img_path.name}")
    print("-" * 65)

    if not img_path.exists():
        print(f"  [ERROR] File missing at {img_path}")
        return

    pil_img = load_pil_image_safe(img_path)
    global_img, local_imgs, _ = prepare_raglo_images(pil_img)
    global_t, local_t = prepare_raglo_tensors(global_img, local_imgs)

    res = classifier.classify_candidate(global_t, local_t, candidate_id=1, tree_id=1)

    clf = res["classification"]
    open_set = res["open_set"]

    print(f"  Raw Prediction          : {clf['raw_predicted_class']} ({clf['raw_confidence']*100:.1f}%)")
    print(f"  Class Probabilities     : {clf['class_probabilities']}")
    print(f"  Prototype Distance      : {open_set['prototype_distance']:.4f} (Z: {open_set['prototype_z']:.4f})")
    print(f"  Energy Score            : {open_set['energy']:.4f} (Z: {open_set['energy_z']:.4f})")
    print(f"  Combined Unknownness    : {open_set['combined_unknownness']:.4f}")
    print(f"  Open-Set Threshold      : {open_set['threshold']:.4f}")
    print(f"  Decision Margin         : {open_set['decision_margin']:.4f}")
    print(f"  Open-Set Decision       : {'REJECTED' if open_set['open_set_rejected'] else 'KNOWN'}")
    print(f"  Final Species Output    : {res['final_species']}")


def main():
    parser = argparse.ArgumentParser(description="Same-Tree Bark Input Diagnostic Tool")
    parser.add_argument("--clean", type=str, help="Path to manually cropped clean bark image")
    parser.add_argument("--trunk", type=str, help="Path to full tree/trunk crop image")
    parser.add_argument("--auto", type=str, help="Path to automatically selected bark ROI image")

    args = parser.parse_args()

    print("=" * 65)
    print("      SAME-TREE BARK INPUT DIAGNOSTIC COMPARISON TOOL")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device}")

    classifier = RAGLOClassifier(CLASSIFIER_MODEL_PATH, device=device)

    inputs = [
        ("A. Clean Bark Crop", args.clean),
        ("B. Full Trunk Crop", args.trunk),
        ("C. Auto Bark ROI", args.auto)
    ]

    has_run = False
    for label, path_str in inputs:
        if path_str:
            has_run = True
            evaluate_single_input(classifier, Path(path_str), label)

    if not has_run:
        print("\nNo image paths specified. Running self-diagnostic test on sample dataset image if available...")
        sample_img = BASE_DIR / "test_output" / "real_forest.jpg"
        if sample_img.exists():
            evaluate_single_input(classifier, sample_img, "Sample Diagnostic Image")
        else:
            print("Usage example:")
            print("  python tools/compare_bark_inputs.py --clean data/clean.jpg --trunk data/trunk.jpg --auto data/auto.jpg")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
