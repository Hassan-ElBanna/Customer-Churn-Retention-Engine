"""Unit and Integration Tests across all MLOps phases."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from churn_engine.api.main import app, _STATE
from churn_engine.data.validation import DataValidator
from churn_engine.features.preprocessor import ChurnFeaturePreprocessor
from churn_engine.models.evaluator import ModelEvaluator
from churn_engine.models.trainer import ChurnModelTrainer
from churn_engine.monitoring.drift import DriftDetector
from churn_engine.retention.engine import RetentionEngine


def _make_synthetic_churn_df(n_rows: int = 240) -> pd.DataFrame:
    """Create a realistic synthetic customer churn dataset for testing."""
    rng = np.random.RandomState(42)
    tenure = rng.randint(1, 72, size=n_rows)
    monthly_charges = np.round(rng.uniform(20.0, 115.0, size=n_rows), 2)
    total_charges = np.round(tenure * monthly_charges + rng.normal(0, 15, size=n_rows), 2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20]
    )
    tech_support = rng.choice(["Yes", "No"], size=n_rows, p=[0.35, 0.65])
    payment_method = rng.choice(
        ["Electronic check", "Credit card (automatic)", "Bank transfer (automatic)"],
        size=n_rows,
    )

    # Generate realistic churn probability signal
    logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month").astype(float)
        - 0.03 * tenure
        + 0.015 * monthly_charges
        + 0.6 * (tech_support == "No").astype(float)
    )
    probs = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.uniform(0, 1, size=n_rows) < probs, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(n_rows)],
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": [str(x) if i % 15 != 0 else " " for i, x in enumerate(total_charges)],
            "Contract": contract,
            "TechSupport": tech_support,
            "PaymentMethod": payment_method,
            "Churn": churn,
        }
    )


def test_data_validation_passes_on_valid_data():
    df = _make_synthetic_churn_df(120)
    validator = DataValidator(target_column="Churn", min_rows=50)
    report = validator.validate(df)
    assert report.passed is True
    assert report.num_rows == 120
    assert report.target_present is True
    assert len(report.errors) == 0


def test_preprocessor_no_leakage_and_domain_features():
    df = _make_synthetic_churn_df(200)
    prep = ChurnFeaturePreprocessor(target_column="Churn", random_state=42)
    train_df, val_df, test_df = prep.split_data(df, test_size=0.15, val_size=0.15)

    X_train = prep.fit_transform(train_df)
    X_val = prep.transform(val_df)
    X_test = prep.transform(test_df)

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()


def test_trainer_and_retention_engine_end_to_end():
    df = _make_synthetic_churn_df(240)
    prep = ChurnFeaturePreprocessor(target_column="Churn", random_state=42)
    train_df, val_df, test_df = prep.split_data(df)

    X_train = prep.fit_transform(train_df)
    X_val = prep.transform(val_df)
    X_test = prep.transform(test_df)

    trainer = ChurnModelTrainer(evaluator=ModelEvaluator(), random_state=42)
    champ_name, champ_model, summary = trainer.train_and_select_champion(
        X_train,
        train_df["Churn"],
        X_val,
        val_df["Churn"],
        X_test,
        test_df["Churn"],
    )

    assert champ_name in (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "voting_ensemble",
    )
    assert summary["champion_test_metrics"]["roc_auc"] > 0.60

    # Test Retention Engine
    engine = RetentionEngine()
    plan = engine.generate_plan(
        customer_id="CUST-9999",
        churn_prob=0.85,
        customer_features={
            "tenure": 2,
            "MonthlyCharges": 95.0,
            "Contract": "Month-to-month",
            "TechSupport": "No",
        },
    )
    assert plan.risk_tier == "Critical"
    assert plan.will_churn is True
    assert plan.revenue_at_risk > 0
    assert len(plan.recommended_actions) >= 1

    # Test FastAPI endpoint with in-memory artifacts
    _STATE["preprocessor"] = prep
    _STATE["model_bundle"] = {
        "model_name": champ_name,
        "model": champ_model,
        "threshold": trainer.champion_threshold,
    }
    client = TestClient(app)
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["model_loaded"] is True

    pred_res = client.post(
        "/predict",
        json={
            "customer_id": "CUST-0042",
            "features": {
                "tenure": 2,
                "MonthlyCharges": 99.0,
                "TotalCharges": 198.0,
                "Contract": "Month-to-month",
                "TechSupport": "No",
                "PaymentMethod": "Electronic check",
            },
        },
    )
    assert pred_res.status_code == 200
    body = pred_res.json()
    assert "churn_probability" in body
    assert "recommended_actions" in body


def test_drift_detector_identifies_shifted_distribution():
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame({"f1": rng.normal(0, 1, 300), "f2": rng.normal(10, 2, 300)})
    cur_df = pd.DataFrame({"f1": rng.normal(2.5, 1.5, 300), "f2": rng.normal(10, 2, 300)})

    detector = DriftDetector(psi_threshold=0.20, ks_alpha=0.05)
    report = detector.evaluate_drift(ref_df, cur_df)
    assert "f1" in report["drifted_features"]
    assert "f2" not in report["drifted_features"]

