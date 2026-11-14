"""Phase 6: FastAPI Production Serving Layer for Churn Prediction & Retention."""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from churn_engine.api.schemas import (
    BatchCustomerPayload,
    CustomerPayload,
    RetentionResponse,
)
from churn_engine.features.preprocessor import ChurnFeaturePreprocessor
from churn_engine.retention.engine import RetentionEngine
from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]


def _resolve_path(env_key: str, default_rel: str) -> Path:
    raw = Path(os.getenv(env_key, default_rel))
    return raw if raw.is_absolute() else (ROOT_DIR / raw)


app = FastAPI(
    title="Customer Churn & Retention Engine API",
    description="End-to-End MLOps API for real-time customer churn scoring, Evidently drift monitoring, & retention action plans.",
    version="0.2.0",
)

MODEL_PATH = _resolve_path("MODEL_PATH", "artifacts/models/champion_model.joblib")
PREPROCESSOR_PATH = _resolve_path("PREPROCESSOR_PATH", "artifacts/models/preprocessor.joblib")
MONITORING_ACTIONS_PATH = _resolve_path(
    "MONITORING_ACTIONS_PATH", "artifacts/reports/monitoring_actions.json"
)
EVALUATION_METRICS_PATH = _resolve_path(
    "EVALUATION_METRICS_PATH", "artifacts/reports/evaluation_metrics.json"
)
EVIDENTLY_HTML_PATH = _resolve_path(
    "EVIDENTLY_HTML_PATH", "artifacts/reports/evidently_drift_report.html"
)
REFERENCE_DATA_PATH = _resolve_path("REFERENCE_DATA_PATH", "data/raw/customer_churn.csv")
PUBLIC_INDEX_PATH = ROOT_DIR / "public" / "index.html"

PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total customer churn predictions served",
    ["endpoint", "risk_tier", "will_churn"],
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Churn prediction latency in seconds",
    ["endpoint"],
)
PREDICTION_PROBABILITY = Histogram(
    "churn_predicted_probability",
    "Distribution of predicted churn probabilities",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_LOADED_GAUGE = Gauge(
    "churn_model_loaded",
    "Whether champion model and preprocessor are loaded (1=yes, 0=no)",
)
DRIFT_FEATURES_GAUGE = Gauge(
    "churn_drift_detected_features",
    "Count of drifted columns reported by the latest Evidently monitoring check",
)
DRIFT_SHARE_GAUGE = Gauge(
    "churn_drift_share",
    "Share of drifted columns (0.0-1.0) reported by Evidently monitoring",
)
RETRAIN_RECOMMENDED_GAUGE = Gauge(
    "churn_retrain_recommended",
    "Whether monitoring triggered a retraining recommendation (1=yes, 0=no)",
)
MONITORING_SEVERITY_GAUGE = Gauge(
    "churn_monitoring_alert_severity",
    "Monitoring alert severity level (0=OK, 1=WARNING, 2=CRITICAL)",
)

_STATE: Dict[str, Any] = {
    "preprocessor": None,
    "model_bundle": None,
    "retention_engine": RetentionEngine(),
    "monitor": None,
}


def _sync_monitoring_gauges(actions_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Update Prometheus drift & alert gauges from latest monitoring_actions.json."""
    payload = actions_payload
    if payload is None and MONITORING_ACTIONS_PATH.exists():
        try:
            with open(MONITORING_ACTIONS_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed reading {MONITORING_ACTIONS_PATH}: {exc}")
            payload = {}
    payload = payload or {}

    DRIFT_FEATURES_GAUGE.set(float(payload.get("num_drifted_columns", 0)))
    DRIFT_SHARE_GAUGE.set(float(payload.get("drift_share", 0.0)))
    RETRAIN_RECOMMENDED_GAUGE.set(1.0 if payload.get("retrain_recommended", False) else 0.0)
    MONITORING_SEVERITY_GAUGE.set(float(payload.get("severity_level", 0)))
    return payload


def _load_artifacts_if_available() -> None:
    """Lazy-load trained model and preprocessor artifacts if present on disk."""
    if _STATE["preprocessor"] is None and PREPROCESSOR_PATH.exists():
        _STATE["preprocessor"] = ChurnFeaturePreprocessor.load(PREPROCESSOR_PATH)
        logger.info(f"Loaded preprocessor from {PREPROCESSOR_PATH}")
    if _STATE["model_bundle"] is None and MODEL_PATH.exists():
        _STATE["model_bundle"] = joblib.load(MODEL_PATH)
        logger.info(f"Loaded champion model bundle from {MODEL_PATH}")
    ready = _STATE["preprocessor"] is not None and _STATE["model_bundle"] is not None
    MODEL_LOADED_GAUGE.set(1.0 if ready else 0.0)


def _score_customer(payload: CustomerPayload, endpoint_label: str) -> RetentionResponse:
    """Shared scoring and Prometheus instrumentation helper for single & batch endpoints."""
    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start
    PREDICTION_LATENCY.labels(endpoint=endpoint_label).observe(elapsed)
    PREDICTION_PROBABILITY.observe(churn_prob)
    PREDICTION_COUNTER.labels(
        endpoint=endpoint_label,
        risk_tier=plan.risk_tier,
        will_churn=str(plan.will_churn).lower(),
    ).inc()
    return RetentionResponse(**plan.to_dict(), model_name=bundle["model_name"])


@app.get("/", include_in_schema=False)
def serve_home() -> Response:
    """Serve the Vercel/FastAPI web frontend when present, or API metadata JSON."""
    if PUBLIC_INDEX_PATH.exists():
        return FileResponse(PUBLIC_INDEX_PATH)
    return JSONResponse(
        {
            "service": "Customer Churn & Retention Engine API",
            "docs_url": "/docs",
            "health_url": "/health",
        }
    )


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


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """Return champion metadata, test leaderboard, and top feature importances."""
    if not EVALUATION_METRICS_PATH.exists():
        return {}
    try:
        return json.loads(EVALUATION_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed reading {EVALUATION_METRICS_PATH}: {exc}")
        return {}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Expose Prometheus telemetry for Grafana scraping."""
    _load_artifacts_if_available()
    _sync_monitoring_gauges()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/status")
def monitoring_status() -> Dict[str, Any]:
    """Return latest Evidently AI monitoring status and triggered actions."""
    payload = _sync_monitoring_gauges()
    return {
        "status": payload.get("severity", "UNKNOWN"),
        "evidently_html_available": EVIDENTLY_HTML_PATH.exists(),
        "monitoring_actions": payload,
    }


@app.post("/monitoring/check")
def run_monitoring_check(payload: BatchCustomerPayload) -> Dict[str, Any]:
    """Run an on-demand Evidently AI drift check on an incoming batch and update Prometheus gauges."""
    if not payload.customers:
        raise HTTPException(status_code=400, detail="Batch payload must contain at least one customer.")
    if not REFERENCE_DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Reference dataset not found at {REFERENCE_DATA_PATH}.",
        )

    _load_artifacts_if_available()
    ref_df = (
        pd.read_parquet(REFERENCE_DATA_PATH)
        if REFERENCE_DATA_PATH.suffix == ".parquet"
        else pd.read_csv(REFERENCE_DATA_PATH)
    )
    cur_df = pd.DataFrame([c.features for c in payload.customers])

    try:
        from churn_engine.monitoring.evidently_monitor import EvidentlyMonitor
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Evidently AI runtime is not installed in lightweight serverless deployment; run batch monitoring via CLI or full container.",
        ) from exc

    if _STATE["monitor"] is None:
        _STATE["monitor"] = EvidentlyMonitor()
    monitor: EvidentlyMonitor = _STATE["monitor"]
    summary = monitor.run_monitoring(
        ref_df,
        cur_df,
        html_output_path=EVIDENTLY_HTML_PATH,
        actions_output_path=MONITORING_ACTIONS_PATH,
    )
    actions_payload = _sync_monitoring_gauges(EvidentlyMonitor.build_actions_record(summary))
    return {
        "severity": actions_payload["severity"],
        "num_columns_checked": summary["num_columns_checked"],
        "num_drifted_columns": summary["num_drifted_columns"],
        "drift_share": summary["drift_share"],
        "drifted_columns": summary["drifted_columns"],
        "action_plan": summary["action_plan"],
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
    return _score_customer(payload, endpoint_label="/predict")


@app.post("/predict/batch", response_model=List[RetentionResponse])
def predict_batch(payload: BatchCustomerPayload) -> List[RetentionResponse]:
    """Score a batch of customers and return retention plans."""
    _load_artifacts_if_available()
    if _STATE["preprocessor"] is None or _STATE["model_bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded yet. Run the training pipeline first.",
        )
    return [_score_customer(customer, endpoint_label="/predict/batch") for customer in payload.customers]

