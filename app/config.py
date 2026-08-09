"""
Application Configuration Module.
Centralizes default parameters, detection thresholds, quality limits, and diagnostic flags.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"

# Ensure runtime directories exist
for directory in (LOGS_DIR, UPLOADS_DIR, RESULTS_DIR, TEMPLATES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Exact Model Filenames
DETECTOR_MODEL_NAME = "tree_trunk_detection_best.pt"
CLASSIFIER_MODEL_NAME = "tree_classification_best.pt"

DETECTOR_MODEL_PATH = MODELS_DIR / DETECTOR_MODEL_NAME
CLASSIFIER_MODEL_PATH = MODELS_DIR / CLASSIFIER_MODEL_NAME

# Section 4: Tree Trunk Detection & Proposal Parameters
TREE_IMGSZ = 1024
TREE_PROPOSAL_CONF = 0.10
TREE_CONFIDENCE = 0.10
TREE_NMS_IOU = 0.70
TREE_IOU = 0.70
TREE_MAX_DET = 300

# Minimum detection confidence required for a trunk to be shown in Tree Detection
# results and to proceed onward into classification. Kept separate from
# TREE_PROPOSAL_CONF/TREE_CONFIDENCE above (which stay low so the model still
# generates enough raw candidates to filter down from).
TREE_MIN_CONFIDENCE_FOR_CLASSIFICATION = 0.55

# Prominent Foreground Trunk Protection Thresholds
PROMINENT_MIN_WIDTH_RATIO = 0.08
PROMINENT_MIN_HEIGHT_RATIO = 0.40
PROMINENT_MIN_BBOX_AREA_RATIO = 0.025

# Aliases for backward compatibility
FOREGROUND_MIN_WIDTH_RATIO = PROMINENT_MIN_WIDTH_RATIO
FOREGROUND_MIN_HEIGHT_RATIO = PROMINENT_MIN_HEIGHT_RATIO
FOREGROUND_MIN_AREA_RATIO = PROMINENT_MIN_BBOX_AREA_RATIO

# Section 7: Diagnostic Mode Flags
DEBUG_TREE_DETECTION = False
DEBUG_ASSESSABILITY = False
DEBUG_TREE_FILTERING = False
DEBUG_RAGLO = False

# Section 8: Duplicate Suppression & Mask Cleanup Parameters
DUPLICATE_IOU_THRESHOLD = 0.65
MIN_TINY_ISLAND_AREA_PX = 100

# Section 10: Bark Quality Gate Thresholds & Relative Dimension Gate
MIN_BARK_WIDTH_RATIO = 0.04
MIN_BARK_HEIGHT_RATIO = 0.12
QUALITY_MIN_OCCUPANCY = 0.40
QUALITY_MIN_SHARPNESS = 30.0
QUALITY_MIN_CONTRAST = 15.0
QUALITY_MIN_BRIGHTNESS = 25.0
QUALITY_MAX_BRIGHTNESS = 230.0
CANDIDATE_TARGET_COUNT = 5

# Section 18: Application Uncertainty Zone Review Margin
OPEN_SET_REVIEW_MARGIN = None

