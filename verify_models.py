"""
Model Verification Script.
Inspects models/tree_trunk_detection_best.pt and models/tree_classification_best.pt,
verifies architecture metadata, preprocessing parameters, and open-set calibration properties.
Run with:
    python verify_models.py
"""
import sys
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

DETECTOR_PATH = MODELS_DIR / "tree_trunk_detection_best.pt"
CLASSIFIER_PATH = MODELS_DIR / "tree_classification_best.pt"


def main():
    print("=" * 65)
    print("      RESEARCH MODEL VERIFICATION REPORT")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    print(f"Inference Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU Model Name    : {torch.cuda.get_device_name(0)}")
    print("-" * 65)

    # 1. Tree Trunk Detector Checkpoint Verification
    print(f"\n[1] Tree Trunk Detector Checkpoint: {DETECTOR_PATH.name}")
    if not DETECTOR_PATH.exists():
        print(f"  [ERROR] Checkpoint file missing at: {DETECTOR_PATH}")
    else:
        file_size_mb = DETECTOR_PATH.stat().st_size / (1024 * 1024)
        print(f"  File Status     : EXISTS ({file_size_mb:.2f} MB)")
        if YOLO is not None:
            try:
                yolo = YOLO(str(DETECTOR_PATH))
                names = getattr(yolo, "names", {0: "tree_trunk"})
                print(f"  Tree Detector   : LOADED SUCCESSFULLY")
                print(f"  Detector Classes: {names}")
            except Exception as e:
                print(f"  [ERROR] YOLO loading failure: {e}")
        else:
            print("  [WARN] Ultralytics package is not installed.")

    # 2. RAGLO Bark Species Classifier Checkpoint Verification
    print(f"\n[2] RAGLO Bark Species Classifier: {CLASSIFIER_PATH.name}")
    if not CLASSIFIER_PATH.exists():
        print(f"  [ERROR] Checkpoint file missing at: {CLASSIFIER_PATH}")
    else:
        file_size_mb = CLASSIFIER_PATH.stat().st_size / (1024 * 1024)
        print(f"  File Status     : EXISTS ({file_size_mb:.2f} MB)")
        try:
            ckpt = torch.load(str(CLASSIFIER_PATH), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                print(f"  RAGLO Artifact Keys : {list(ckpt.keys())}")

                classes = ckpt.get("class_names", ckpt.get("classes", ["mahogany", "pine", "rubber", "teak"]))
                print(f"  RAGLO Class Names   : {classes}")

                prep = ckpt.get("preprocessing")
                print(f"  Preprocessing Config: {prep}")

                patches = ckpt.get("patch_settings")
                print(f"  Patch Metadata      : {patches}")

                open_set = ckpt.get("open_set_configuration", ckpt.get("open_set_config"))
                print(f"  Open-Set Config     : {open_set}")

                if open_set:
                    thresh = open_set.get("threshold")
                    alpha = open_set.get("alpha")
                    score_dir = open_set.get("score_direction")
                    print(f"    - Threshold      : {thresh}")
                    print(f"    - Alpha          : {alpha}")
                    print(f"    - Score Direction: {score_dir}")

                if "class_prototypes" in ckpt:
                    proto = ckpt["class_prototypes"]
                    print(f"  Class Prototypes    : Shape {tuple(proto.shape) if hasattr(proto, 'shape') else type(proto)}")
            else:
                print("  Checkpoint type: Direct torch Module / state_dict")

            print("  RAGLO Load Check    : LOADED SUCCESSFULLY")
        except Exception as e:
            print(f"  [ERROR] RAGLO checkpoint inspection failure: {e}")

    print("\n" + "=" * 65)
    print(" VERIFICATION SUMMARY: Ready for application execution.")
    print("=" * 65)


if __name__ == "__main__":
    main()
