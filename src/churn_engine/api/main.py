"""Phase 6: FastAPI Production Serving Layer for Churn Prediction & Retention."""

import os
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_engine.api.schemas import (
    BatchCustomerPayload,
    CustomerPayload,
    RetentionResponse,
)
from churn_engine.features.preprocessor import ChurnFeaturePreprocessor
from churn_engine.retention.engine import RetentionEngine
from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Customer Churn & Retention Engine API",
    description="End-to-End MLOps API for real-time customer churn scoring & retention action plans.",
    version="0.1.0",
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/models/champion_model.joblib"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "artifacts/models/preprocessor.joblib"))

_STATE: Dict[str, Any] = {
    "preprocessor": None,
    "model_bundle": None,
    "retention_engine": RetentionEngine(),
}


def _load_artifacts_if_available() -> None:
    """Lazy-load trained model and preprocessor artifacts if present on disk."""
    if _STATE["preprocessor"] is None and PREPROCESSOR_PATH.exists():
        _STATE["preprocessor"] = ChurnFeaturePreprocessor.load(PREPROCESSOR_PATH)
        logger.info(f"Loaded preprocessor from {PREPROCESSOR_PATH}")
    if _STATE["model_bundle"] is None and MODEL_PATH.exists():
        _STATE["model_bundle"] = joblib.load(MODEL_PATH)
        logger.info(f"Loaded champion model bundle from {MODEL_PATH}")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Liveness and model readiness check."""
    _load_artifacts_if_available()
    model_ready = _STATE["preprocessor"] is not None and _STATE["model_bundle"] is not None
    return {
        "status": "healthy",
        "model_loaded": model_ready,
        "champion_model": (
            _STATE["model_bundle"]["model_name"] if _STATE["model_bundle"] else None
        ),
    }


@app.post("/predict", response_model=RetentionResponse)
def predict_single(payload: CustomerPayload) -> RetentionResponse:
    """Score a single customer and return churn probability + retention playbook."""
    _load_artifacts_if_available()
    if _STATE["preprocessor"] is None or _STATE["model_bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded yet. Run the training pipeline first.",
        )

    preprocessor: ChurnFeaturePreprocessor = _STATE["preprocessor"]
    bundle: Dict[str, Any] = _STATE["model_bundle"]
    retention_engine: RetentionEngine = _STATE["retention_engine"]

    raw_df = pd.DataFrame([payload.features])
    X_trans = preprocessor.transform(raw_df)
    churn_prob = float(bundle["model"].predict_proba(X_trans)[:, 1][0])

    plan = retention_engine.generate_plan(
        customer_id=payload.customer_id,
        churn_prob=churn_prob,
        customer_features=payload.features,
        decision_threshold=float(bundle.get("threshold", 0.50)),
    )

    return RetentionResponse(**plan.to_dict(), model_name=bundle["model_name"])


@app.post("/predict/batch", response_model=List[RetentionResponse])
def predict_batch(payload: BatchCustomerPayload) -> List[RetentionResponse]:
    """Score a batch of customers and return retention plans."""
    return [predict_single(customer) for customer in payload.customers]

