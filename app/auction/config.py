"""
Configuration for the Auction Bid Price module (SmartTimber-LK Module 4), vendored as
an independent module. Integrated with Timber Grading (Module 3) only at the UI layer
(the frontend pre-fills this module's form from a graded crop) - no backend coupling.
Paths are rebased onto pathfinders-fyp's project root.
"""
from __future__ import annotations

from app.config import BASE_DIR as ROOT_DIR

MODELS_DIR = ROOT_DIR / "models" / "auction"
MODEL_PATH = MODELS_DIR / "xgb_value_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

OUTPUTS_DIR = ROOT_DIR / "outputs" / "auction"
DB_PATH = OUTPUTS_DIR / "predictions.db"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
