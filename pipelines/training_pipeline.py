"""End-to-End MLOps Orchestration Pipeline for Customer Churn & Retention Engine."""

import argparse
import json
from pathlib import Path
from typing import Optional

from churn_engine.config import load_config
from churn_engine.data.eda import ChurnEDAAnalyzer
from churn_engine.data.ingestion import DataIngestor
from churn_engine.data.validation import DataValidator
from churn_engine.features.preprocessor import ChurnFeaturePreprocessor
from churn_engine.models.evaluator import ModelEvaluator
from churn_engine.models.trainer import ChurnModelTrainer
from churn_engine.monitoring.drift import DriftDetector
from churn_engine.retention.engine import RetentionEngine
from churn_engine.utils.logger import get_logger
from churn_engine.utils.tracking import ExperimentTracker

logger = get_logger("pipeline")


def run_pipeline(
    config_path: str = "configs/config.yaml",
    dataset_url: Optional[str] = None,
    target_column: Optional[str] = None,
    stage: str = "all",
) -> dict:
    """Execute the full MLOps lifecycle from ingestion to retention & drift baseline."""
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    artifacts_cfg = cfg["artifacts"]

    url = dataset_url or data_cfg.get("dataset_url") or None
    target_col = target_column or data_cfg["target_column"]
    raw_path = Path(data_cfg["raw_data_path"])
    processed_dir = Path(data_cfg["processed_dir"])
    model_dir = Path(artifacts_cfg["model_dir"])
    reports_dir = Path(artifacts_cfg["reports_dir"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Phase 1: Data Ingestion & Validation
    # -------------------------------------------------------------------------
    logger.info("=== PHASE 1: Data Ingestion & Quality Validation ===")
    ingestor = DataIngestor(raw_data_path=raw_path)
    raw_df = ingestor.load_data(url=url)

    validator = DataValidator(
        target_column=target_col,
        min_rows=data_cfg.get("min_rows", 50),
        max_missing_ratio=data_cfg.get("max_missing_ratio", 0.40),
    )
    val_report = validator.validate(raw_df)
    val_report.save(reports_dir / "data_validation_report.json")
    if not val_report.passed:
        raise ValueError(f"Data validation failed: {val_report.errors}")

    # -------------------------------------------------------------------------
    # Phase 1b: Comprehensive Exploratory Data Analysis & Visualizations
    # -------------------------------------------------------------------------
    logger.info("=== PHASE 1b: Exploratory Data Analysis & Visualizations ===")
    eda_analyzer = ChurnEDAAnalyzer(figures_dir=reports_dir / "figures")
    eda_insights = eda_analyzer.run_eda(raw_df, target_col=target_col)
    with open(reports_dir / "eda_insights.json", "w", encoding="utf-8") as f:
        json.dump(eda_insights, f, indent=2)

    # -------------------------------------------------------------------------
    # Phase 2: Leakage-Free Split & Feature Engineering
    # -------------------------------------------------------------------------
    logger.info("=== PHASE 2: Feature Engineering & Preprocessing ===")
    preprocessor = ChurnFeaturePreprocessor(
        target_column=target_col,
        max_categorical_cardinality=cfg["features"]["max_categorical_cardinality"],
        random_state=cfg["project"]["random_state"],
    )
    train_df, val_df, test_df = preprocessor.split_data(
        raw_df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
    )

    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df[target_col]

    X_val = preprocessor.transform(val_df)
    y_val = val_df[target_col]

    X_test = preprocessor.transform(test_df)
    y_test = test_df[target_col]

    # Save processed splits
    X_train.assign(**{target_col: y_train.values}).to_parquet(processed_dir / "train.parquet", index=False)
    X_val.assign(**{target_col: y_val.values}).to_parquet(processed_dir / "val.parquet", index=False)
    X_test.assign(**{target_col: y_test.values}).to_parquet(processed_dir / "test.parquet", index=False)

    preprocessor.save(model_dir / "preprocessor.joblib")

    # -------------------------------------------------------------------------
    # Phase 3: Model Training, Evaluation & Champion Selection
    # -------------------------------------------------------------------------
    logger.info("=== PHASE 3: Model Training, Threshold Optimization & Selection ===")
    evaluator = ModelEvaluator(
        cost_fn=train_cfg["business_cost_fn"],
        cost_fp=train_cfg["business_cost_fp"],
    )
    tracker = ExperimentTracker(
        experiment_name=train_cfg["experiment_name"],
        tracking_uri=train_cfg.get("mlflow_tracking_uri"),
        dagshub_owner=train_cfg.get("dagshub_repo_owner"),
        dagshub_repo=train_cfg.get("dagshub_repo_name"),
    )
    trainer = ChurnModelTrainer(
        evaluator=evaluator,
        tracker=tracker,
        primary_metric=train_cfg["primary_metric"],
        random_state=cfg["project"]["random_state"],
    )

    champion_name, champion_model, summary = trainer.train_and_select_champion(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    summary["top_feature_importances"] = trainer.get_feature_importances(
        preprocessor.feature_names_out_, top_k=15
    )

    eval_viz_summary = eda_analyzer.plot_model_evaluation_and_retention(
        trained_models=trainer.trained_models,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_raw_df=raw_df.loc[test_df.index],
        champion_name=champion_name,
        champion_threshold=trainer.champion_threshold,
        feature_importances=summary["top_feature_importances"],
        cost_fn=train_cfg["business_cost_fn"],
        cost_fp=train_cfg["business_cost_fp"],
    )
    summary.update(eval_viz_summary)

    trainer.save_champion(
        model_path=model_dir / "champion_model.joblib",
        metrics_path=reports_dir / "evaluation_metrics.json",
        summary=summary,
    )

    # -------------------------------------------------------------------------
    # Phase 4: Retention Playbook Simulation on Test Set
    # -------------------------------------------------------------------------
    logger.info("=== PHASE 4: Customer Retention Engine Simulation ===")
    ret_cfg = cfg["retention"]
    retention_engine = RetentionEngine(
        low_threshold=ret_cfg["risk_thresholds"]["low"],
        medium_threshold=ret_cfg["risk_thresholds"]["medium"],
        high_threshold=ret_cfg["risk_thresholds"]["high"],
        default_annual_revenue=ret_cfg["default_annual_revenue"],
        max_offer_budget_ratio=ret_cfg["max_offer_budget_ratio"],
    )

    test_probs = champion_model.predict_proba(X_test)[:, 1]
    sample_plans = []
    for idx, prob in zip(test_df.index[:5], test_probs[:5]):
        row_dict = test_df.loc[idx].to_dict()
        cid = str(row_dict.get(data_cfg.get("id_column", "customerID"), f"CUST-{idx}"))
        plan = retention_engine.generate_plan(
            customer_id=cid,
            churn_prob=float(prob),
            customer_features=row_dict,
            decision_threshold=trainer.champion_threshold,
        )
        sample_plans.append(plan.to_dict())

    summary["sample_retention_plans"] = sample_plans

    # -------------------------------------------------------------------------
    # Phase 5: Reference vs Test Data Drift Check
    # -------------------------------------------------------------------------
    logger.info("=== PHASE 5: Data Drift Baseline Verification ===")
    drift_detector = DriftDetector(
        psi_threshold=cfg["monitoring"]["psi_threshold"],
        ks_alpha=cfg["monitoring"]["ks_alpha"],
    )
    drift_report = drift_detector.evaluate_drift(X_train, X_test)
    drift_detector.save_report(drift_report, reports_dir / "drift_report.json")

    tracker.log_pipeline_summary(
        champion_name=champion_name,
        champion_threshold=trainer.champion_threshold,
        test_metrics=summary.get("champion_test_metrics", {}),
        feature_importances=summary.get("top_feature_importances", {}),
        artifact_paths=[
            reports_dir / "data_validation_report.json",
            reports_dir / "eda_insights.json",
            reports_dir / "evaluation_metrics.json",
            reports_dir / "drift_report.json",
            reports_dir / "figures",
        ],
    )

    logger.info(f"=== PIPELINE COMPLETED SUCCESSFULLY (Champion: {champion_name}) ===")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Customer Churn & Retention MLOps Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--url", type=str, default=None, help="Dataset URL to download")
    parser.add_argument("--target", type=str, default=None, help="Target column name override")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "ingest", "preprocess", "train"],
        help="Pipeline stage to execute",
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        dataset_url=args.url,
        target_column=args.target,
        stage=args.stage,
    )

