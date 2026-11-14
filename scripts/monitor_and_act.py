"""Automated Evidently AI monitoring check & MLOps action execution CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional
import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
for _path in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from churn_engine.config import load_config
from churn_engine.features.preprocessor import ChurnFeaturePreprocessor
from churn_engine.monitoring.evidently_monitor import EvidentlyMonitor
from churn_engine.utils.logger import get_logger
from pipelines.training_pipeline import run_pipeline

logger = get_logger("monitor_and_act")


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def run_monitoring_check(
    config_path: str = "configs/config.yaml",
    reference_path: Optional[str] = None,
    current_path: Optional[str] = None,
    auto_retrain: Optional[bool] = None,
    html_output_path: Optional[str] = None,
    json_output_path: Optional[str] = None,
    actions_output_path: Optional[str] = None,
) -> dict:
    """Compare production/current batch against reference data and trigger MLOps actions."""
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    mon_cfg = cfg["monitoring"]
    artifacts_cfg = cfg["artifacts"]

    target_col = data_cfg["target_column"]
    id_col = data_cfg.get("id_column", "customerID")

    ref_file = Path(reference_path or data_cfg["raw_data_path"])
    cur_file = Path(current_path or data_cfg["raw_data_path"])

    ref_df = _load_frame(ref_file)
    cur_df = _load_frame(cur_file)

    preprocessor_path = Path(artifacts_cfg["model_dir"]) / "preprocessor.joblib"
    model_path = Path(artifacts_cfg["model_dir"]) / "champion_model.joblib"

    has_model = preprocessor_path.exists() and model_path.exists()
    pred_prob_col: Optional[str] = None
    pred_label_col: Optional[str] = None
    active_target: Optional[str] = target_col if target_col in ref_df.columns and target_col in cur_df.columns else None

    if active_target:
        ref_df[target_col] = ChurnFeaturePreprocessor.encode_target(ref_df[target_col])
        cur_df[target_col] = ChurnFeaturePreprocessor.encode_target(cur_df[target_col])

    if has_model and ref_file.suffix.lower() != ".parquet" and cur_file.suffix.lower() != ".parquet":
        preprocessor = ChurnFeaturePreprocessor.load(preprocessor_path)
        bundle = joblib.load(model_path)
        threshold = float(bundle.get("threshold", 0.50))

        X_ref = preprocessor.transform(ref_df)
        X_cur = preprocessor.transform(cur_df)
        ref_df = ChurnFeaturePreprocessor.clean_and_engineer_domain_features(
            ref_df.drop(columns=[c for c in (id_col,) if c in ref_df.columns])
        )
        cur_df = ChurnFeaturePreprocessor.clean_and_engineer_domain_features(
            cur_df.drop(columns=[c for c in (id_col,) if c in cur_df.columns])
        )
        ref_df["pred_prob"] = bundle["model"].predict_proba(X_ref)[:, 1]
        ref_df["pred_label"] = (ref_df["pred_prob"] >= threshold).astype(int)
        cur_df["pred_prob"] = bundle["model"].predict_proba(X_cur)[:, 1]
        cur_df["pred_label"] = (cur_df["pred_prob"] >= threshold).astype(int)
        pred_prob_col = "pred_prob"
        pred_label_col = "pred_label"
    else:
        ref_df = ref_df.drop(columns=[c for c in (id_col,) if c in ref_df.columns])
        cur_df = cur_df.drop(columns=[c for c in (id_col,) if c in cur_df.columns])

    should_retrain = (
        bool(auto_retrain)
        if auto_retrain is not None
        else bool(mon_cfg.get("auto_retrain_on_drift", False))
    )

    monitor = EvidentlyMonitor(
        drift_share_threshold=mon_cfg.get("drift_share_threshold", 0.25),
        psi_threshold=mon_cfg["psi_threshold"],
        severe_psi_threshold=mon_cfg.get("severe_psi_threshold", 0.35),
        min_accuracy_threshold=mon_cfg.get("min_accuracy_threshold", 0.75),
        min_roc_auc_threshold=mon_cfg.get("min_roc_auc_threshold", 0.78),
    )
    summary = monitor.run_monitoring(
        ref_df,
        cur_df,
        target_col=active_target,
        pred_label_col=pred_label_col,
        pred_prob_col=pred_prob_col,
        html_output_path=html_output_path or mon_cfg.get("evidently_html_path", "artifacts/reports/evidently_drift_report.html"),
        json_output_path=json_output_path or mon_cfg.get("evidently_json_path", "artifacts/reports/evidently_summary.json"),
        actions_output_path=actions_output_path or mon_cfg.get("actions_report_path", "artifacts/reports/monitoring_actions.json"),
        auto_retrain=should_retrain,
        retrain_callback=lambda: run_pipeline(config_path=config_path, stage="all"),
    )
    action_plan = summary["action_plan"]
    logger.info(
        f"Monitoring check finished | Severity={action_plan['severity']} | "
        f"Retrain recommended={action_plan['retrain_recommended']} | "
        f"Retrain executed={action_plan['retrain_executed']}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidently AI monitoring & automated MLOps actions")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--reference", type=str, default=None, help="Path to reference CSV/Parquet dataset")
    parser.add_argument("--current", type=str, default=None, help="Path to current production batch CSV/Parquet")
    parser.add_argument(
        "--auto-retrain",
        action="store_true",
        help="Automatically trigger full training pipeline if CRITICAL drift/degradation occurs.",
    )
    parser.add_argument("--html-out", type=str, default=None, help="Optional override for HTML report path")
    parser.add_argument("--json-out", type=str, default=None, help="Optional override for JSON summary path")
    parser.add_argument("--actions-out", type=str, default=None, help="Optional override for actions JSON path")
    args = parser.parse_args()
    run_monitoring_check(
        config_path=args.config,
        reference_path=args.reference,
        current_path=args.current,
        auto_retrain=args.auto_retrain,
        html_output_path=args.html_out,
        json_output_path=args.json_out,
        actions_output_path=args.actions_out,
    )
