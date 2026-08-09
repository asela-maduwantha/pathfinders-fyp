from __future__ import annotations

import base64
import io
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

TABULAR_COLS = ["Length_M", "Mid_Girth_M", "Volume"]
LOW_UPPER = 30
MED_UPPER = 60


class TimberFusionNet(nn.Module):
    def __init__(self, num_classes: int, num_tabular_features: int = 3, tabular_hidden: int = 16) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.image_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_branch = backbone
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, tabular_hidden),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.image_feature_dim + tabular_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_features = self.image_branch(image)
        tabular_features = self.tabular_branch(tabular)
        return self.classifier(torch.cat([image_features, tabular_features], dim=1))


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _log_step(log: list[dict[str, Any]], stage: str, started_at: float, **details: Any) -> None:
    entry: dict[str, Any] = {"stage": stage, "elapsed_ms": _elapsed_ms(started_at)}
    if details:
        entry["details"] = details
    log.append(entry)


def _classes_from_json(meta: dict[str, Any]) -> list[str]:
    if meta.get("REAL_GRADE_CLASSES"):
        return list(meta["REAL_GRADE_CLASSES"])
    if meta.get("grade_to_idx"):
        return [grade for grade, _ in sorted(meta["grade_to_idx"].items(), key=lambda item: int(item[1]))]
    raise ValueError("grade_classes.json contains neither REAL_GRADE_CLASSES nor grade_to_idx")


def grade_to_ordinal(grade: str | None) -> int | None:
    if grade is None:
        return None
    digits = "".join(ch for ch in str(grade).strip() if ch.isdigit())
    return int(digits) if digits else -10


def grade_to_bucket(grade: str | None, low_upper: int = LOW_UPPER, med_upper: int = MED_UPPER) -> str | None:
    ordinal = grade_to_ordinal(grade)
    if ordinal is None:
        return None
    if ordinal < low_upper:
        return "Low"
    if ordinal < med_upper:
        return "Medium"
    return "High"


class TimberFusionService:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        self.load_log: list[dict[str, Any]] = []
        self.available = False
        self.missing: list[str] = []
        self.classes: list[str] = []
        self.idx_to_grade: dict[int, str] = {}
        self.low_upper = LOW_UPPER
        self.med_upper = MED_UPPER
        self.scaler: Any = None
        self.model: TimberFusionNet | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        torch.set_grad_enabled(False)
        self._load_bundle()

    @property
    def artifact_paths(self) -> dict[str, Path]:
        return {
            "fusion_model": self.model_dir / "best_timber_model.pth",
            "grade_classes": self.model_dir / "grade_classes.json",
            "tabular_scaler": self.model_dir / "tabular_scaler.pkl",
        }

    def _load_bundle(self) -> None:
        load_started = time.perf_counter()
        paths = self.artifact_paths
        self.missing = [str(path) for path in paths.values() if not path.exists()]
        if self.missing:
            self.load_log.append(
                {
                    "stage": "artifact check",
                    "elapsed_ms": _elapsed_ms(load_started),
                    "details": {"status": "missing", "missing": self.missing},
                }
            )
            return

        started = time.perf_counter()
        with paths["grade_classes"].open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self.classes = _classes_from_json(meta)
        self.idx_to_grade = {idx: grade for idx, grade in enumerate(self.classes)}
        self.low_upper = int(meta.get("LOW_UPPER", LOW_UPPER))
        self.med_upper = int(meta.get("MED_UPPER", MED_UPPER))
        _log_step(
            self.load_log,
            "load grade_classes.json",
            started,
            classes=len(self.classes),
            low_upper=self.low_upper,
            med_upper=self.med_upper,
        )

        started = time.perf_counter()
        with paths["tabular_scaler"].open("rb") as handle:
            self.scaler = pickle.load(handle)
        _log_step(self.load_log, "load tabular_scaler.pkl", started, scaler_type=type(self.scaler).__name__)

        started = time.perf_counter()
        self.model = TimberFusionNet(num_classes=len(self.classes)).to(self.device)
        state = self._torch_load(paths["fusion_model"])
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        _log_step(self.load_log, "load best_timber_model.pth", started, device=str(self.device))

        self.available = True
        self.load_log.append(
            {
                "stage": "model bundle ready",
                "elapsed_ms": _elapsed_ms(load_started),
                "details": {"device": str(self.device), "input_mode": "uploaded image + tabular measurements"},
            }
        )

    def _torch_load(self, path: Path) -> Any:
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "device": str(self.device),
            "model_dir": str(self.model_dir),
            "artifacts": {name: {"path": str(path), "exists": path.exists()} for name, path in self.artifact_paths.items()},
            "classes": self.classes,
            "bucket_thresholds": {"low_upper": self.low_upper, "med_upper": self.med_upper},
            "tabular_columns": TABULAR_COLS,
            "load_log": self.load_log,
            "missing": self.missing,
        }

    def process_image(self, filename: str, image_bytes: bytes, metadata: dict[str, Any] | None) -> dict[str, Any]:
        total_started = time.perf_counter()
        log: list[dict[str, Any]] = []

        started = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        _log_step(log, "decode image", started, width=image.width, height=image.height, mode=image.mode)

        started = time.perf_counter()
        preview = self._make_preview(image)
        _log_step(log, "create preview", started)

        started = time.perf_counter()
        tabular_values, metadata_status = self._coerce_metadata(metadata)
        _log_step(log, "validate tabular measurements", started, status=metadata_status["status"])

        if not self.available:
            return self._finish_result(
                filename,
                preview,
                log,
                total_started,
                {"status": "error", "reason": "model artifacts are unavailable", "missing": self.missing},
                metadata_status,
            )

        if tabular_values is None:
            return self._finish_result(
                filename,
                preview,
                log,
                total_started,
                {"status": "skipped", "reason": metadata_status["reason"]},
                metadata_status,
            )

        started = time.perf_counter()
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        _log_step(log, "preprocess image tensor", started, shape=list(image_tensor.shape))

        started = time.perf_counter()
        tabular_array = np.array([[tabular_values[col] for col in TABULAR_COLS]], dtype=np.float32)
        scaled_tabular = self.scaler.transform(tabular_array).astype(np.float32)
        tabular_tensor = torch.tensor(scaled_tabular, dtype=torch.float32, device=self.device)
        _log_step(log, "scale tabular features", started, columns=TABULAR_COLS)

        started = time.perf_counter()
        assert self.model is not None
        with torch.inference_mode():
            logits = self.model(image_tensor, tabular_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        _log_step(log, "ResNet-18 tabular fusion inference", started, device=str(self.device))

        started = time.perf_counter()
        prediction = self._prediction_from_probabilities(probabilities)
        _log_step(log, "postprocess probabilities", started)

        return self._finish_result(filename, preview, log, total_started, prediction, metadata_status)

    def _finish_result(
        self,
        filename: str,
        preview: str,
        log: list[dict[str, Any]],
        total_started: float,
        prediction: dict[str, Any],
        metadata_status: dict[str, Any],
    ) -> dict[str, Any]:
        log.append({"stage": "total image processing", "elapsed_ms": _elapsed_ms(total_started)})
        return {
            "filename": filename,
            "preview": preview,
            "prediction": prediction,
            "metadata_status": metadata_status,
            "log": log,
        }

    def _coerce_metadata(self, metadata: dict[str, Any] | None) -> tuple[dict[str, float] | None, dict[str, Any]]:
        if not metadata:
            return None, {"status": "missing", "reason": "Length_M, Mid_Girth_M, and Volume were not provided"}

        values: dict[str, float] = {}
        missing: list[str] = []
        invalid: list[str] = []
        for col in TABULAR_COLS:
            raw = metadata.get(col)
            if raw is None or str(raw).strip() == "":
                missing.append(col)
                continue
            try:
                values[col] = float(raw)
            except (TypeError, ValueError):
                invalid.append(col)

        if invalid:
            return None, {"status": "invalid", "reason": "invalid numeric fields: " + ", ".join(invalid)}
        if missing:
            return None, {"status": "missing", "reason": "missing fields: " + ", ".join(missing)}
        return values, {"status": "ok", "values": values}

    def _prediction_from_probabilities(self, probabilities: np.ndarray) -> dict[str, Any]:
        pred_idx = int(np.argmax(probabilities))
        pred_grade = self.idx_to_grade[pred_idx]
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(self.classes))]
        top_predictions = [
            {
                "grade": self.idx_to_grade[int(idx)],
                "bucket": grade_to_bucket(self.idx_to_grade[int(idx)], self.low_upper, self.med_upper),
                "probability": round(float(probabilities[int(idx)]), 4),
            }
            for idx in top_indices
        ]
        return {
            "status": "ok",
            "grade": pred_grade,
            "bucket": grade_to_bucket(pred_grade, self.low_upper, self.med_upper),
            "confidence": round(float(probabilities[pred_idx]), 4),
            "top_predictions": top_predictions,
            "classes": self.classes,
        }

    def _make_preview(self, image: Image.Image) -> str:
        preview = image.copy()
        preview.thumbnail((640, 480))
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=86, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
