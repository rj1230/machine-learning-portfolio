#!/usr/bin/env python3
"""
FastAPI inference service for the Heart Disease Prediction project.

Endpoints
---------
GET  /health          Service and model health check
GET  /model-info      Loaded model, threshold, features, and metadata
POST /predict         Score one patient
POST /predict-batch   Score multiple patients

Run locally (Windows PowerShell)
-------------------------------
    pip install fastapi uvicorn pydantic
    python backend.py

Then open:
    http://127.0.0.1:8000/docs

Example request
---------------
POST http://127.0.0.1:8000/predict
{
  "age": 58,
  "sex": "male",
  "cp": "asymptomatic",
  "trestbps": 140,
  "chol": 240,
  "fbs": false,
  "restecg": "normal",
  "thalch": 150,
  "exang": true,
  "oldpeak": 2.3,
  "slope": "flat",
  "ca": 2,
  "thal": "reversible_defect"
}

This service is for educational/portfolio use only and is not a medical device.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

# Limit BLAS/OpenMP threads before NumPy is imported.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

LOGGER = logging.getLogger("heart_disease_backend")

MODEL_PATH = Path(os.getenv("HEART_MODEL_PATH", "artifacts/model.joblib"))

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
BINARY_FEATURES = ["fbs", "exang"]
CATEGORICAL_FEATURES = ["sex", "cp", "restecg", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

PLAUSIBLE_RANGES = {
    "age": (18, 110),
    "trestbps": (50, 260),
    "chol": (80, 700),
    "thalch": (50, 250),
    "oldpeak": (-3.5, 10),
    "ca": (0, 4),
}

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "?", "<na>"}
TRUE_TOKENS = {"true", "1", "1.0", "yes", "y", "t"}
FALSE_TOKENS = {"false", "0", "0.0", "no", "n", "f"}


class PredictionRequest(BaseModel):
    """Input schema for a single patient."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=110)
    sex: Literal["male", "female"]
    cp: str
    trestbps: float | None = Field(default=None, ge=50, le=260)
    chol: float | None = Field(default=None, ge=80, le=700)
    fbs: bool | None = None
    restecg: str | None = None
    thalch: float | None = Field(default=None, ge=50, le=250)
    exang: bool | None = None
    oldpeak: float | None = Field(default=None, ge=-3.5, le=10)
    slope: str | None = None
    ca: int | None = Field(default=None, ge=0, le=4)
    thal: str | None = None

    @field_validator("sex", "cp", "restecg", "slope", "thal")
    @classmethod
    def normalise_text(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalised = str(value).strip().lower()
        return None if normalised in MISSING_TOKENS else normalised


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    probability: float = Field(ge=0.0, le=1.0)
    prediction: int = Field(ge=0, le=1)
    risk_label: Literal["low_risk", "high_risk"]
    threshold: float
    model_name: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse]
    count: int


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""

    model_name: str
    threshold: float
    features: list[str]
    sklearn_version: str
    include_site: bool
    model_path: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the serialized model bundle once when the service starts."""
    global BUNDLE

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model bundle not found: {MODEL_PATH}. "
            "Run `python main.py train --data heart_disease.csv` first."
        )

    BUNDLE = joblib.load(MODEL_PATH)
    LOGGER.info("Loaded model '%s' from %s", BUNDLE["model_name"], MODEL_PATH)

    yield

    BUNDLE = None


BUNDLE: dict[str, Any] | None = None

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description=(
        "Predicts the probability of angiographic heart disease from clinical "
        "features. Educational/portfolio project; not a medical device."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def normalise_text(value: Any) -> Any:
    """Convert missing-like strings to None and lowercase categorical values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip().lower()
    return None if text in MISSING_TOKENS else text


def to_binary(value: Any) -> float | None:
    """Convert booleans and common textual binary values to 0/1."""
    if value is None:
        return None

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    normalised = normalise_text(value)

    if normalised in TRUE_TOKENS:
        return 1.0
    if normalised in FALSE_TOKENS:
        return 0.0

    return None


def request_to_feature_row(request: PredictionRequest) -> dict[str, Any]:
    """Convert a validated API request into the model's expected feature schema."""
    row: dict[str, Any] = {}

    for feature in NUMERIC_FEATURES:
        value = getattr(request, feature)
        row[feature] = float(value) if value is not None else np.nan

    for feature in BINARY_FEATURES:
        row[feature] = to_binary(getattr(request, feature))

    for feature in CATEGORICAL_FEATURES:
        row[feature] = normalise_text(getattr(request, feature))

    return row


def validate_feature_values(row: dict[str, Any]) -> None:
    """Reject values outside medically plausible ranges before scoring."""
    for feature, (low, high) in PLAUSIBLE_RANGES.items():
        value = row[feature]

        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        if not low <= value <= high:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid value for '{feature}': {value}. "
                    f"Expected a value between {low} and {high}."
                ),
            )


def predict_row(row: dict[str, Any]) -> PredictionResponse:
    """Run preprocessing and inference for one patient."""
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    validate_feature_values(row)

    features = pd.DataFrame([row])[BUNDLE["features"]]
    probability = float(BUNDLE["pipeline"].predict_proba(features)[:, 1][0])
    threshold = float(BUNDLE["threshold"])
    prediction = int(probability >= threshold)

    return PredictionResponse(
        probability=round(probability, 4),
        prediction=prediction,
        risk_label="high_risk" if prediction == 1 else "low_risk",
        threshold=round(threshold, 4),
        model_name=BUNDLE["model_name"],
    )


@app.get("/", tags=["Service"])
def root() -> dict[str, str]:
    """Basic service information."""
    return {
        "service": "Heart Disease Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Service"])
def health() -> dict[str, str]:
    """Check whether the API and model are available."""
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    return {
        "status": "ok",
        "model": BUNDLE["model_name"],
        "sklearn_version": sklearn.__version__,
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Service"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    if BUNDLE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    return ModelInfoResponse(
        model_name=BUNDLE["model_name"],
        threshold=BUNDLE["threshold"],
        features=BUNDLE["features"],
        sklearn_version=BUNDLE["sklearn_version"],
        include_site=BUNDLE["include_site"],
        model_path=str(MODEL_PATH),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict heart-disease risk for one patient."""
    row = request_to_feature_row(request)
    return predict_row(row)


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict_batch(
    requests: list[PredictionRequest],
) -> BatchPredictionResponse:
    """Predict heart-disease risk for multiple patients."""
    if not requests:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request must contain at least one patient.",
        )

    if len(requests) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 1,000 patients.",
        )

    predictions = [predict_row(request_to_feature_row(request)) for request in requests]

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
