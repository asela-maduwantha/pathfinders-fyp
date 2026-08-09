"""
Timber auction bid-price valuation service (SmartTimber-LK Module 4), vendored as an
independent module. XGBoost regression + scikit-learn ColumnTransformer preprocessing,
plus the original app's uncertainty-interval, bidding-strategy, and SHAP-style local
attribution heuristics - ported essentially unchanged (no hardcoded paths in the source).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

EXPECTED_COLUMNS = [
    'species', 'region', 'season', 'auction_type', 'competition_level',
    'diameter_cm', 'length_m', 'volume_m3', 'straightness_score',
    'taper_score', 'visible_defects_score', 'internal_defect_risk',
    'density_kg_m3', 'moisture_content', 'quality_grade',
    'market_demand_index', 'supply_index', 'avg_market_price_species',
    'price_volatility', 'export_demand_index', 'num_expected_bidders',
]


def calculate_uncertainty_interval(predicted_value: float, inputs: dict[str, Any]) -> tuple[float, float]:
    """Simulates a heteroscedastic 90% conformal prediction interval."""
    volatility = float(inputs['price_volatility'])
    risk = float(inputs['internal_defect_risk'])
    error_sd_pct = 0.08 + 0.15 * volatility + 0.12 * risk
    margin = predicted_value * (1.645 * error_sd_pct)
    low_bound = max(0.0, predicted_value - margin)
    high_bound = predicted_value + margin
    return low_bound, high_bound


def optimize_bidding_strategy(predicted_value: float, inputs: dict[str, Any], low_bound: float) -> tuple[float, float]:
    """Recommended starting bid and expected final price based on competitive intensity."""
    comp = inputs['competition_level']
    bidders = int(inputs['num_expected_bidders'])
    if comp == 'high' or bidders > 8:
        start_ratio = 0.85
        expected_final = predicted_value * (1.0 + 0.015 * bidders)
    elif comp == 'low' or bidders < 3:
        start_ratio = 0.70
        expected_final = predicted_value
    else:
        start_ratio = 0.80
        expected_final = predicted_value * (1.0 + 0.005 * bidders)
    starting_bid = low_bound * start_ratio
    return starting_bid, expected_final


def calculate_success_probability(inputs: dict[str, Any]) -> float:
    """Auction success probability via a logistic approximation on demand/supply/grade."""
    demand = float(inputs['market_demand_index'])
    supply = float(inputs['supply_index'])
    grade = int(inputs['quality_grade'])
    volatility = float(inputs['price_volatility'])
    z = -1.0 + 2.5 * demand - 1.2 * supply + 0.4 * (grade - 3) - 1.5 * volatility
    prob = 1.0 / (1.0 + math.exp(-z))
    return min(0.99, max(0.01, prob))


class AuctionValuationService:
    def __init__(self, model_path: Path, preprocessor_path: Path) -> None:
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model: Any = None
        self.preprocessor: Any = None
        self.available = False
        self.error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists() or not self.preprocessor_path.exists():
            self.error = "Model or preprocessor artifacts are missing."
            return
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "model_path": str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path),
        }

    def _compute_local_attributions(self, input_df: pd.DataFrame, base_pred: float) -> dict[str, float]:
        """Approximates SHAP values by perturbing key inputs to standard baselines."""
        attributions: dict[str, float] = {}
        baselines = {
            'diameter_cm': 25.0,
            'quality_grade': 3,
            'avg_market_price_species': 100.0,
            'market_demand_index': 1.0,
            'internal_defect_risk': 0.5,
        }
        friendly_names = {
            'diameter_cm': 'Log Diameter',
            'quality_grade': 'Quality Grade',
            'avg_market_price_species': 'Species Base Price',
            'market_demand_index': 'Market Demand',
            'internal_defect_risk': 'Internal Defect Risk',
        }
        for feature, baseline in baselines.items():
            temp_df = input_df.copy()
            temp_df[feature] = baseline
            temp_trans = self.preprocessor.transform(temp_df)
            temp_pred = float(self.model.predict(temp_trans)[0])
            attributions[friendly_names[feature]] = base_pred - temp_pred
        return attributions

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.error or "Auction valuation model is unavailable.")

        row = {col: [inputs[col]] for col in EXPECTED_COLUMNS}
        input_df = pd.DataFrame(row)[EXPECTED_COLUMNS]

        transformed = self.preprocessor.transform(input_df)
        raw_predicted_value = float(self.model.predict(transformed)[0])
        predicted_value = max(0.0, raw_predicted_value)
        below_price_floor = raw_predicted_value < 0.0

        low_bound, high_bound = calculate_uncertainty_interval(predicted_value, inputs)
        starting_bid, expected_final_price = optimize_bidding_strategy(predicted_value, inputs, low_bound)
        sale_probability = calculate_success_probability(inputs)
        # Attribute against the raw (unclamped) model output, not the floored display
        # value - otherwise a floored Rs 0 estimate makes every contribution look
        # positive even when the underlying signal was strongly negative.
        shap_values = self._compute_local_attributions(input_df, raw_predicted_value)

        min_scale = max(0.0, low_bound * 0.8)
        max_scale = high_bound * 1.2
        scale_range = max_scale - min_scale if (max_scale - min_scale) > 0 else 1.0
        bar_metrics = {
            'low_bound': low_bound,
            'high_bound': high_bound,
            'range_left_pct': min(100.0, max(0.0, ((low_bound - min_scale) / scale_range) * 100)),
            'range_width_pct': min(100.0, max(0.0, ((high_bound - low_bound) / scale_range) * 100)),
            'point_pct': min(100.0, max(0.0, ((predicted_value - min_scale) / scale_range) * 100)),
        }

        quality_profile = {
            'labels': ['Straightness', 'Taper (Inverted)', 'Density Rating', 'Moisture Dryness', 'Defect-Free Index'],
            'user_values': [
                float(inputs['straightness_score']),
                float(max(0.1, 10.0 - inputs['taper_score'])),
                float((inputs['density_kg_m3'] / 1000.0) * 10.0),
                float(max(0.1, 100.0 - inputs['moisture_content']) / 10.0),
                float(max(0.1, 10.0 - inputs['visible_defects_score'])),
            ],
            'market_avg': [7.0, 7.5, 6.5, 8.5, 8.0],
        }

        price_comparison = {
            'labels': ['Base Market Price', 'Rec. Starting Bid', 'Point Estimate', 'Expected Final'],
            'values': [
                float(inputs['avg_market_price_species']),
                starting_bid,
                predicted_value,
                expected_final_price,
            ],
        }

        return {
            "predicted_value": predicted_value,
            "raw_predicted_value": raw_predicted_value,
            "below_price_floor": below_price_floor,
            "low_bound": low_bound,
            "high_bound": high_bound,
            "starting_bid": starting_bid,
            "expected_final_price": expected_final_price,
            "sale_probability": sale_probability,
            "shap": [{"label": k, "value": v} for k, v in shap_values.items()],
            "quality_profile": quality_profile,
            "price_comparison": price_comparison,
            "bar_metrics": bar_metrics,
        }
