"""
API routes for the Auction Bid Price module (SmartTimber-LK Module 4), vendored as an
independent module. The only integration point with Timber Grading (Module 3) is at
the UI layer - the frontend pre-fills this form from a graded crop. Nothing here
imports from or calls into app.timber, app.maturity, or app.inference.
"""
from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.auction.services import history
from app.auction.services.pdf_report import build_report_pdf

router = APIRouter()

CATEGORICAL_OPTIONS = {
    "species": ["Eucalyptus", "Mahogany", "Pine", "Teak"],
    "region": ["Anuradhapura", "Colombo", "Galle", "Gampaha", "Kandy", "Kurunegala"],
    "season": ["Dry", "Inter-monsoon", "Wet"],
    "auction_type": ["open", "sealed"],
    "competition_level": ["high", "medium", "low"],
}

FIELD_DEFAULTS: dict[str, Any] = {
    # Module 4's original defaults (diameter 35cm/length 5m/Teak/Colombo/Dry, etc.)
    # produce a raw negative valuation from this model, floored to Rs. 0 - a
    # confusing out-of-the-box result. This profile was empirically verified
    # against the actual model to reliably score positive at a realistic
    # moderate log size, while remaining fully editable like every other field.
    "species": "Teak",
    "region": "Gampaha",
    "season": "Wet",
    "auction_type": "open",
    "competition_level": "high",
    "diameter_cm": 45.0,
    "length_m": 6.0,
    "volume_m3": 0.9543,
    "straightness_score": 9.0,
    "taper_score": 1.5,
    "visible_defects_score": 0.5,
    "internal_defect_risk": 0.05,
    "density_kg_m3": 700,
    "moisture_content": 12.0,
    "quality_grade": 5,
    "avg_market_price_species": 400.0,
    "price_volatility": 0.1,
    "market_demand_index": 1.50,
    "supply_index": 0.70,
    "export_demand_index": 1.50,
    "num_expected_bidders": 8,
}


class ValuationRequest(BaseModel):
    species: str
    region: str
    season: str
    auction_type: str
    competition_level: str
    diameter_cm: float = Field(gt=0)
    length_m: float = Field(gt=0)
    volume_m3: float = Field(gt=0)
    straightness_score: float = Field(ge=0, le=10)
    taper_score: float = Field(ge=0, le=10)
    visible_defects_score: float = Field(ge=0, le=10)
    internal_defect_risk: float = Field(ge=0, le=1)
    density_kg_m3: float = Field(gt=0)
    moisture_content: float = Field(ge=0, le=100)
    quality_grade: int = Field(ge=1, le=5)
    market_demand_index: float = Field(gt=0)
    supply_index: float = Field(gt=0)
    avg_market_price_species: float = Field(gt=0)
    price_volatility: float = Field(ge=0, le=1)
    export_demand_index: float = Field(gt=0)
    num_expected_bidders: int = Field(ge=1)


def _get_service():
    from app.main import models_store

    if "auction_valuation" not in models_store:
        raise HTTPException(status_code=503, detail="Auction valuation model is still initializing.")
    service = models_store["auction_valuation"]
    if not service.available:
        raise HTTPException(status_code=503, detail=service.error or "Auction valuation model is unavailable.")
    return service


@router.get("/api/auction/config")
def auction_config() -> dict[str, Any]:
    return {
        "categorical_options": CATEGORICAL_OPTIONS,
        "field_defaults": FIELD_DEFAULTS,
    }


@router.post("/api/auction/predict")
def auction_predict(payload: ValuationRequest) -> JSONResponse:
    service = _get_service()
    inputs = payload.model_dump()

    try:
        result = service.predict(inputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Valuation failed: {exc}") from exc

    prediction_id = history.save_prediction(inputs, result)
    result["prediction_id"] = prediction_id
    return JSONResponse(result)


@router.get("/api/auction/history")
def auction_history() -> dict[str, Any]:
    return {"history": history.get_history(limit=5)}


@router.get("/api/auction/report/{prediction_id}")
def auction_report(prediction_id: int) -> StreamingResponse:
    row = history.get_prediction_by_id(prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Valuation report record not found.")
    pdf_bytes = build_report_pdf(dict(row))
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Timber_Valuation_Report_{prediction_id}.pdf"},
    )
