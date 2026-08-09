from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.maturity.config import MATURITY_DATASET_PATH, MATURITY_MODEL_PATH


SPECIES_COLUMN = "species_group"
DBH_COLUMN = "dbh_cm"
MISSING_CATEGORY = "__MISSING__"

FALLBACK_CATEGORY_OPTIONS = {
    "climatic_zone": [
        "Dry",
        "Intermediate",
        "Wet",
        "Up-country",
    ],
    "spacing": [
        "1.75 x 1.75 m",
        "2.0 x 2.0 m",
        "2.5 x 2.5 m",
        "3.0 x 3.0 m",
        "4.0 x 4.0 m",
    ],
    "site_class": [
        "I",
        "II",
        "III",
        "IV",
        "V",
    ],
    "management_intensity": [
        "Intensive",
        "Standard",
        "Less intensive",
    ],
    "intended_product": [
        "General commercial timber",
        "Pulp / smallwood",
        "General sawlog",
        "Large timber / furniture",
        "Fuelwood / pulp",
        "Transmission poles",
        "Sawn timber / sleepers",
    ],
}


class MaturityPredictor:
    required_bundle_keys = {
        "version",
        "architecture",
        "species_values",
        "base_models",
        "base_metadata",
        "residual_model",
        "exact_alpha_map",
        "count_alpha_map",
        "optional_numeric_features",
        "optional_categorical_features",
        "residual_model_columns",
        "residual_limit",
        "maturity_threshold",
    }

    def __init__(self, model_path: Path = MATURITY_MODEL_PATH, dataset_path: Path = MATURITY_DATASET_PATH):
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        deployment = joblib.load(self.model_path)
        missing = sorted(self.required_bundle_keys.difference(deployment))
        if missing:
            raise ValueError("The maturity bundle is incomplete: " + ", ".join(missing))

        self.deployment = deployment
        self.version = deployment["version"]
        self.architecture = deployment["architecture"]
        self.species_values = list(deployment["species_values"])
        self.base_models = deployment["base_models"]
        self.base_metadata = deployment["base_metadata"]
        self.residual_model = deployment["residual_model"]
        self.exact_alpha_map = deployment["exact_alpha_map"]
        self.count_alpha_map = deployment["count_alpha_map"]
        self.optional_numeric_features = list(deployment["optional_numeric_features"])
        self.optional_categorical_features = list(deployment["optional_categorical_features"])
        self.optional_features = self.optional_numeric_features + self.optional_categorical_features
        self.residual_model_columns = list(deployment["residual_model_columns"])
        self.residual_limit = float(deployment["residual_limit"])
        self.maturity_threshold = float(deployment["maturity_threshold"])
        self.category_source_df = self._load_category_source()

    def _load_category_source(self) -> pd.DataFrame | None:
        if not self.dataset_path.exists():
            return None
        columns = [SPECIES_COLUMN, *self.optional_categorical_features]
        available = pd.read_csv(self.dataset_path, nrows=0).columns
        usecols = [column for column in columns if column in available]
        if not usecols:
            return None
        return pd.read_csv(self.dataset_path, usecols=usecols, low_memory=False)

    def category_options(self, column: str, species: str | None = None) -> list[str]:
        values: list[str] = []
        if self.category_source_df is not None and column in self.category_source_df.columns:
            source = self.category_source_df
            if species is not None and column == "intended_product" and SPECIES_COLUMN in source.columns:
                source = source[source[SPECIES_COLUMN].astype(str).eq(str(species))]
            values = (
                source[column]
                .dropna()
                .astype(str)
                .loc[lambda series: ~series.isin(["", "Unknown", "nan", "None"])]
                .unique()
                .tolist()
            )

        if not values:
            values = FALLBACK_CATEGORY_OPTIONS.get(column, [])
        return ["Blank"] + sorted(set(values))

    def maturity_class(self, score: float) -> str:
        if score < 40:
            return "Immature"
        if score < self.maturity_threshold:
            return "Approaching maturity"
        return "Mature"

    @staticmethod
    def pattern_key(selected_features: list[str]) -> str:
        selected = sorted(set(selected_features))
        return "|".join(selected) if selected else "__NONE__"

    @staticmethod
    def count_bucket(optional_count: int) -> str:
        if optional_count <= 0:
            return "0"
        if optional_count == 1:
            return "1"
        if optional_count == 2:
            return "2"
        if optional_count <= 4:
            return "3-4"
        if optional_count <= 7:
            return "5-7"
        return "8-10"

    @staticmethod
    def is_available_numeric(value: Any) -> bool:
        if value is None:
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return not np.isnan(number)

    @staticmethod
    def is_available_category(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        return bool(text) and text not in {"Unknown", "Blank", MISSING_CATEGORY}

    def selected_optional_features(self, context: dict[str, Any]) -> list[str]:
        selected = []
        for feature in self.optional_numeric_features:
            if self.is_available_numeric(context.get(feature)):
                selected.append(feature)
        for feature in self.optional_categorical_features:
            if self.is_available_category(context.get(feature)):
                selected.append(feature)
        return selected

    def alpha_for_pattern(self, species: str, features: list[str]) -> float:
        if not features:
            return 0.0
        exact_key = self.pattern_key(features)
        species_exact = self.exact_alpha_map.get(species, {})
        if exact_key in species_exact:
            return float(species_exact[exact_key])
        bucket = self.count_bucket(len(set(features)))
        return float(self.count_alpha_map.get(species, {}).get(bucket, 0.0))

    def prepare_optional_row(self, context: dict[str, Any], features: list[str]) -> pd.DataFrame:
        selected = set(features)
        row: dict[str, Any] = {SPECIES_COLUMN: str(context[SPECIES_COLUMN])}

        for feature in self.optional_numeric_features:
            row[feature] = float(context[feature]) if feature in selected else np.nan

        for feature in self.optional_categorical_features:
            row[feature] = str(context[feature]) if feature in selected else MISSING_CATEGORY

        available_count = 0
        for feature in self.optional_numeric_features:
            indicator = int(self.is_available_numeric(row[feature]))
            row[f"has__{feature}"] = indicator
            available_count += indicator

        for feature in self.optional_categorical_features:
            indicator = int(row[feature] != MISSING_CATEGORY)
            row[f"has__{feature}"] = indicator
            available_count += indicator

        row["available_optional_count"] = available_count
        prepared = pd.DataFrame([row])
        return prepared[self.residual_model_columns]

    def predict_tree_maturity(self, species: str, dbh_cm: float, **optional_inputs) -> dict[str, Any]:
        species = str(species).strip()
        if species not in self.species_values:
            raise ValueError(f"Unsupported species '{species}'. Choose one of: {self.species_values}")

        dbh_cm = float(dbh_cm)
        if not np.isfinite(dbh_cm) or dbh_cm <= 0:
            raise ValueError("dbh_cm must be a positive number.")

        context = {
            SPECIES_COLUMN: species,
            DBH_COLUMN: dbh_cm,
        }
        for feature in self.optional_features:
            if feature in optional_inputs:
                context[feature] = optional_inputs[feature]

        metadata = self.base_metadata[species]
        model_dbh = float(np.clip(dbh_cm, metadata["dbh_min"], metadata["dbh_max"]))
        base_score = float(
            np.clip(
                self.base_models[species].predict(np.array([[model_dbh]]))[0],
                0,
                100,
            )
        )

        used_features = self.selected_optional_features(context)
        raw_residual = 0.0
        safety_weight = 0.0
        correction = 0.0

        if used_features:
            prepared = self.prepare_optional_row(context, used_features)
            raw_residual = float(
                np.clip(
                    self.residual_model.predict(prepared)[0],
                    -self.residual_limit,
                    self.residual_limit,
                )
            )
            safety_weight = self.alpha_for_pattern(species, used_features)
            correction = float(safety_weight * raw_residual)

        final_score = float(np.clip(base_score + correction, 0, 100))
        return {
            "species": species,
            "dbh_cm": dbh_cm,
            "maturity_score": round(final_score, 2),
            "maturity_class": self.maturity_class(final_score),
            "mature": bool(final_score >= self.maturity_threshold),
            "optional_fields_used": used_features,
            "base_curve_score_internal": base_score,
            "raw_optional_residual_internal": raw_residual,
            "safety_weight_internal": safety_weight,
            "applied_correction_internal": correction,
        }

    def generate_maturity_curve(
        self,
        species: str,
        dbh_cm: float,
        points: int = 320,
        **optional_inputs,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        result = self.predict_tree_maturity(species=species, dbh_cm=dbh_cm, **optional_inputs)
        metadata = self.base_metadata[species]
        actual_dbh = float(dbh_cm)
        dbh_min = min(float(metadata["dbh_min"]), actual_dbh)
        dbh_max = max(float(metadata["dbh_max"]), actual_dbh)
        dbh_grid = np.linspace(dbh_min, dbh_max, points)
        base_curve = np.clip(
            self.base_models[species].predict(dbh_grid.reshape(-1, 1)),
            0,
            100,
        )
        correction = float(result["applied_correction_internal"])
        final_curve = np.clip(base_curve + correction, 0, 100)
        final_curve = np.maximum.accumulate(final_curve)
        curve_df = pd.DataFrame({"dbh_cm": dbh_grid, "maturity_score": final_curve})
        result["curve_monotonic_violations"] = int((np.diff(final_curve) < -1e-8).sum())
        return curve_df, result


@lru_cache(maxsize=1)
def get_maturity_predictor() -> MaturityPredictor:
    return MaturityPredictor()
