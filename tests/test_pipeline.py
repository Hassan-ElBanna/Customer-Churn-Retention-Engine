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
from churn_engine.monitoring.evidently_monitor import EvidentlyMonitor
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


def test_evidently_monitor_detects_drift_and_executes_retrain_action(tmp_path):
    rng = np.random.RandomState(42)
    ref_df = pd.DataFrame(
        {
            "tenure": rng.randint(12, 72, size=200),
            "MonthlyCharges": rng.normal(65.0, 10.0, size=200),
            "Contract": rng.choice(["One year", "Two year"], size=200),
            "Churn": rng.choice([0, 1], size=200, p=[0.75, 0.25]),
        }
    )
    ref_df["pred_prob"] = np.clip(ref_df["Churn"] * 0.7 + rng.uniform(0.05, 0.25, 200), 0.0, 1.0)
    ref_df["pred_label"] = (ref_df["pred_prob"] >= 0.5).astype(int)

    # Identical / stable current batch should stay OK
    stable_df = ref_df.sample(n=180, random_state=7).reset_index(drop=True)
    monitor = EvidentlyMonitor(drift_share_threshold=0.25, psi_threshold=0.20)
    html_ok = tmp_path / "ok.html"
    actions_ok = tmp_path / "actions_ok.json"
    ok_res = monitor.run_monitoring(
        ref_df,
        stable_df,
        target_col="Churn",
        pred_label_col="pred_label",
        pred_prob_col="pred_prob",
        html_output_path=html_ok,
        actions_output_path=actions_ok,
    )
    assert html_ok.exists()
    assert actions_ok.exists()
    assert ok_res["action_plan"]["severity"] == "OK"
    assert ok_res["action_plan"]["retrain_recommended"] is False

    # Severely drifted batch + auto_retrain=True should execute callback
    drifted_df = ref_df.copy()
    drifted_df["tenure"] = rng.randint(1, 5, size=200)
    drifted_df["MonthlyCharges"] = rng.normal(115.0, 5.0, size=200)
    drifted_df["Contract"] = "Month-to-month"

    retrain_calls = []
    html_crit = tmp_path / "crit.html"
    actions_crit = tmp_path / "actions_crit.json"
    crit_res = monitor.run_monitoring(
        ref_df,
        drifted_df,
        target_col="Churn",
        pred_label_col="pred_label",
        pred_prob_col="pred_prob",
        html_output_path=html_crit,
        actions_output_path=actions_crit,
        auto_retrain=True,
        retrain_callback=lambda: retrain_calls.append("triggered"),
    )
    assert crit_res["action_plan"]["severity"] == "CRITICAL"
    assert crit_res["action_plan"]["retrain_recommended"] is True
    assert crit_res["action_plan"]["retrain_executed"] is True
    assert retrain_calls == ["triggered"]


def test_api_prometheus_metrics_and_monitoring_endpoints(tmp_path, monkeypatch):
    import json
    import churn_engine.api.main as api_main

    raw_df = _make_synthetic_churn_df(160)
    prep = ChurnFeaturePreprocessor(target_column="Churn")
    train_df, val_df, test_df = prep.split_data(raw_df, test_size=0.2, val_size=0.2)
    X_train = prep.fit_transform(train_df)
    X_val = prep.transform(val_df)
    X_test = prep.transform(test_df)

    trainer = ChurnModelTrainer(evaluator=ModelEvaluator(), primary_metric="roc_auc")
    champ_name, champ_model, _ = trainer.train_and_select_champion(
        X_train, train_df["Churn"], X_val, val_df["Churn"], X_test, test_df["Churn"]
    )
    _STATE["preprocessor"] = prep
    _STATE["model_bundle"] = {
        "model_name": champ_name,
        "model": champ_model,
        "threshold": trainer.champion_threshold,
    }

    actions_file = tmp_path / "monitoring_actions.json"
    actions_file.write_text(
        json.dumps(
            {
                "severity": "WARNING",
                "severity_level": 1,
                "retrain_recommended": False,
                "num_drifted_columns": 2,
                "drift_share": 0.12,
                "actions": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main, "MONITORING_ACTIONS_PATH", actions_file)

    client = TestClient(app)
    pred_res = client.post(
        "/predict",
        json={
            "customer_id": "CUST-PROM-01",
            "features": {
                "tenure": 3,
                "MonthlyCharges": 95.0,
                "TotalCharges": 285.0,
                "Contract": "Month-to-month",
                "TechSupport": "No",
                "PaymentMethod": "Electronic check",
            },
        },
    )
    assert pred_res.status_code == 200

    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert "churn_predictions_total" in metrics_res.text
    assert "churn_prediction_latency_seconds" in metrics_res.text
    assert "churn_drift_detected_features 2.0" in metrics_res.text
    assert "churn_monitoring_alert_severity 1.0" in metrics_res.text

    status_res = client.get("/monitoring/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "WARNING"

    home_res = client.get("/")
    assert home_res.status_code == 200
    assert "Customer Churn" in home_res.text

    info_res = client.get("/model-info")
    assert info_res.status_code == 200


def test_vercel_entrypoint_exports_fastapi_app():
    from api.index import app as vercel_app

    vercel_client = TestClient(vercel_app)
    res = vercel_client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_vercel_build_builds_and_verifies_compatible_artifacts(tmp_path, monkeypatch):
    import scripts.vercel_build as vercel_build

    model_path = tmp_path / "champion_model.joblib"
    prep_path = tmp_path / "preprocessor.joblib"
    metrics_path = tmp_path / "evaluation_metrics.json"
    sample_df = _make_synthetic_churn_df(240)

    monkeypatch.setattr(vercel_build, "MODEL_PATH", model_path)
    monkeypatch.setattr(vercel_build, "PREP_PATH", prep_path)
    monkeypatch.setattr(vercel_build, "EVAL_METRICS_PATH", metrics_path)
    monkeypatch.setattr(vercel_build, "_load_training_dataframe", lambda _cfg: sample_df)

    assert not vercel_build._verify_artifacts(model_path, prep_path)

    cfg = {
        "project": {"random_state": 42},
        "data": {"target_column": "Churn", "test_size": 0.15, "val_size": 0.15},
        "features": {"max_categorical_cardinality": 50},
        "training": {"business_cost_fn": 300.0, "business_cost_fp": 75.0, "primary_metric": "accuracy"},
    }
    vercel_build._build_compatible_artifacts(cfg)

    assert model_path.exists()
    assert prep_path.exists()
    assert metrics_path.exists()
    assert vercel_build._verify_artifacts(model_path, prep_path)


