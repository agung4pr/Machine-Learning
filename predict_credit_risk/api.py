from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from predict_credit_risk.training import (
    FEATURE_COLUMNS,
    MODEL_OUTPUT_PATH,
    load_model_artifact,
    predict_from_probabilities,
)


app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
    description="Predict loan default risk from applicant and loan features.",
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=120)
    person_income: float = Field(..., ge=0)
    person_home_ownership: Literal["MORTGAGE", "OTHER", "OWN", "RENT"]
    person_emp_length: float | None = Field(default=None, ge=0)
    loan_intent: Literal[
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., ge=0)
    loan_int_rate: float | None = Field(default=None, ge=0)
    loan_percent_income: float = Field(..., ge=0)
    cb_person_default_on_file: Literal["N", "Y"]
    cb_person_cred_hist_length: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_business_rules(self):
        if (
            self.person_emp_length is not None
            and self.person_emp_length > self.person_age - 15
        ):
            raise ValueError(
                "person_emp_length cannot be greater than person_age - 15."
            )
        return self


class PredictionRequest(BaseModel):
    record: LoanApplication
    strategy: Literal["aggressive", "balanced", "conservative"] = "balanced"


class BatchPredictionRequest(BaseModel):
    records: list[LoanApplication] = Field(..., min_length=1)
    strategy: Literal["aggressive", "balanced", "conservative"] = "balanced"


class PredictionResponse(BaseModel):
    predicted_loan_status: int
    predicted_risk_probability: float
    decision_threshold: float
    strategy: str
    risk_label: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


@lru_cache
def load_model():
    if not MODEL_OUTPUT_PATH.exists():
        raise RuntimeError(
            f"Saved model not found at {MODEL_OUTPUT_PATH}. "
            "Run `python -m predict_credit_risk.training` first."
        )
    return load_model_artifact(MODEL_OUTPUT_PATH)


def to_feature_frame(records: list[LoanApplication]) -> pd.DataFrame:
    data = [record.model_dump() for record in records]
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def score_records(records: list[LoanApplication], strategy: str = "balanced") -> list[PredictionResponse]:
    artifact = load_model_artifact()
    model = artifact["model"]
    thresholds = artifact.get("thresholds", {})
    
    if strategy not in thresholds:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{strategy}'. Must be one of: {list(thresholds.keys())}"
        )
    
    threshold = thresholds[strategy]
    features = to_feature_frame(records)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = predict_from_probabilities(probabilities, threshold)

    results = []
    for predicted_status, probability in zip(predictions, probabilities):
        results.append(
            PredictionResponse(
                predicted_loan_status=int(predicted_status),
                predicted_risk_probability=round(float(probability), 6),
                decision_threshold=round(threshold, 6),
                strategy=strategy,
                risk_label="high_risk" if int(predicted_status) == 1 else "low_risk",
            )
        )

    return results


@app.get("/")
def root():
    ui_file = STATIC_DIR / "index.html"
    if ui_file.exists():
        return FileResponse(ui_file, media_type="text/html")
    return {
        "message": "Credit risk prediction API is running.",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "predict_batch_url": "/predict-batch",
    }


@app.get("/health")
def health_check():
    try:
        artifact = load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "model_path": str(MODEL_OUTPUT_PATH),
        "model_type": type(artifact["model"]).__name__,
        "model_version": artifact.get("model_version", "unknown"),
        "decision_threshold": artifact["decision_threshold"],
        "threshold_strategy": artifact["threshold_strategy"],
        "available_strategies": artifact.get("thresholds", {}).keys(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return score_records([request.record], request.strategy)[0]


@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    return BatchPredictionResponse(predictions=score_records(request.records, request.strategy))
