"""
Configuration for the Timber Grading module (SmartTimber-LK Module 3), vendored
as an independent module. Paths are rebased onto pathfinders-fyp's project root.
"""
from __future__ import annotations

from pathlib import Path

from app.config import BASE_DIR as ROOT_DIR

MODELS_DIR = ROOT_DIR / "models" / "timber"

FUSION_MODEL_PATH = MODELS_DIR / "best_timber_model.pth"
GRADE_CLASSES_PATH = MODELS_DIR / "grade_classes.json"
TABULAR_SCALER_PATH = MODELS_DIR / "tabular_scaler.pkl"
TIMBERVISION_MODEL_PATH = MODELS_DIR / "timbervision" / "yolov8l-1024-seg.pt"
