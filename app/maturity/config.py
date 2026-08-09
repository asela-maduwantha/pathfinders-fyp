"""
Configuration for Module 2 (DBH + maturity estimation), vendored from SmartTimber-LK.
Paths are rebased onto pathfinders-fyp's project root; SMARTTIMBER_* env vars are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

from app.config import BASE_DIR as ROOT_DIR

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models" / "maturity"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
STATIC_DIR = ROOT_DIR / "app" / "static"

MATURITY_MODEL_PATH = Path(
    os.getenv(
        "SMARTTIMBER_MATURITY_MODEL",
        MODELS_DIR / "smooth_direct_residual_deployment_v3.joblib",
    )
)
MATURITY_DATASET_PATH = Path(
    os.getenv(
        "SMARTTIMBER_MATURITY_DATASET",
        DATA_DIR / "Combined_Tree_Maturity_Dataset_v2.csv",
    )
)
TUNED_CHECKPOINT_PATH = Path(
    os.getenv(
        "SMARTTIMBER_TUNED_CHECKPOINT",
        MODELS_DIR / "unidepth_finetuned" / "best_tuned_weights.pt",
    )
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


RUN_FINE_TUNED_INTERNAL = _env_flag("SMARTTIMBER_RUN_FINE_TUNED_INTERNAL", False)

UNIDEPTH_MODEL_ID = os.getenv(
    "SMARTTIMBER_UNIDEPTH_MODEL_ID",
    "lpiccinelli/unidepth-v2-vitb14",
)
UNIDEPTH_RESOLUTION_LEVEL = int(os.getenv("SMARTTIMBER_UNIDEPTH_RESOLUTION_LEVEL", "6"))
DINO_ID = os.getenv("SMARTTIMBER_DINO_ID", "IDEA-Research/grounding-dino-base")
SAM_ID = os.getenv("SMARTTIMBER_SAM_ID", "facebook/sam2.1-hiera-tiny")
SEGMENTER_BACKEND = os.getenv("SMARTTIMBER_SEGMENTER_BACKEND", "fastsam").strip().lower()
if SEGMENTER_BACKEND not in {"sam2", "fastsam"}:
    SEGMENTER_BACKEND = "fastsam"

FASTSAM_MODEL = os.getenv("SMARTTIMBER_FASTSAM_MODEL", "FastSAM-s.pt")
FASTSAM_IMAGE_SIZE = int(os.getenv("SMARTTIMBER_FASTSAM_IMAGE_SIZE", "640"))
FASTSAM_CONFIDENCE = float(os.getenv("SMARTTIMBER_FASTSAM_CONFIDENCE", "0.4"))
FASTSAM_IOU = float(os.getenv("SMARTTIMBER_FASTSAM_IOU", "0.9"))
FASTSAM_RETINA_MASKS = _env_flag("SMARTTIMBER_FASTSAM_RETINA_MASKS", True)

DETECTOR_BACKEND = os.getenv("SMARTTIMBER_DETECTOR_BACKEND", "yolo_world").strip().lower()
if DETECTOR_BACKEND not in {"dino", "yolo_world"}:
    DETECTOR_BACKEND = "yolo_world"

YOLO_WORLD_MODEL = os.getenv("SMARTTIMBER_YOLO_WORLD_MODEL", "yolov8s-worldv2.pt")
YOLO_WORLD_IMAGE_SIZE = int(os.getenv("SMARTTIMBER_YOLO_WORLD_IMAGE_SIZE", "640"))
YOLO_WORLD_CONFIDENCE = float(os.getenv("SMARTTIMBER_YOLO_WORLD_CONFIDENCE", "0.01"))
YOLO_WORLD_LABELS = [
    label.strip()
    for label in os.getenv(
        "SMARTTIMBER_YOLO_WORLD_LABELS",
        "tree|trunk|tree trunk|stem|wood",
    ).split("|")
    if label.strip()
]
if not YOLO_WORLD_LABELS:
    YOLO_WORLD_LABELS = ["tree"]


DEVICE_MODE = os.getenv("SMARTTIMBER_DEVICE", "auto").strip().lower()
INTRINSICS_MODE = os.getenv("SMARTTIMBER_INTRINSICS_MODE", "unidepth").strip().lower()
if INTRINSICS_MODE not in {"unidepth", "exif_fallback", "prefer_exif"}:
    INTRINSICS_MODE = "unidepth"

MAX_LONG_SIDE = int(os.getenv("SMARTTIMBER_MAX_LONG_SIDE", "1024"))
GROUNDING_BOX_THRESHOLD = float(os.getenv("SMARTTIMBER_GROUNDING_BOX_THRESHOLD", "0.18"))
GROUNDING_TEXT_THRESHOLD = float(os.getenv("SMARTTIMBER_GROUNDING_TEXT_THRESHOLD", "0.16"))
MIN_MASK_PIXELS = int(os.getenv("SMARTTIMBER_MIN_MASK_PIXELS", "250"))

NORMAL_ROW_COUNT = int(os.getenv("SMARTTIMBER_NORMAL_ROW_COUNT", "21"))
CLOSEUP_ROW_COUNT = int(os.getenv("SMARTTIMBER_CLOSEUP_ROW_COUNT", "25"))
MIN_VALID_ROWS = int(os.getenv("SMARTTIMBER_MIN_VALID_ROWS", "7"))
MIN_RUN_WIDTH_PX = int(os.getenv("SMARTTIMBER_MIN_RUN_WIDTH_PX", "6"))
MAX_RUN_WIDTH_FRACTION = float(os.getenv("SMARTTIMBER_MAX_RUN_WIDTH_FRACTION", "0.94"))
MIN_ROW_DEPTH_FRACTION = float(os.getenv("SMARTTIMBER_MIN_ROW_DEPTH_FRACTION", "0.55"))
MAX_HALF_ANGLE_DEG = float(os.getenv("SMARTTIMBER_MAX_HALF_ANGLE_DEG", "43.0"))
MAX_PLAUSIBLE_DBH_CM = float(os.getenv("SMARTTIMBER_MAX_PLAUSIBLE_DBH_CM", "300.0"))

TEXT_LABELS = [[
    "tree trunk",
    "tree stem",
    "main tree trunk",
    "bark-covered trunk",
]]

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
}

for _directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, RUNS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
