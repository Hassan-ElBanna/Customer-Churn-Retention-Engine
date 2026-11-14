"""Evidently AI monitoring and automated action engine for Customer Churn MLOps."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import urllib.request

from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset
from evidently.tests import gt
import numpy as np
import pandas as pd

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)

_SEVERITY_RANK = {"OK": 0, "WARNING": 1, "CRITICAL": 2}


def _normalize_status(raw_status: Any) -> str:
    """Convert an Evidently TestStatus enum or value into a plain uppercase string."""
    if raw_status is None:
        return "UNKNOWN"
    value = getattr(raw_status, "value", None)
    if value is not None:
        return str(value).upper()
    text = str(raw_status).upper()
    if "." in text:
        text = text.split(".")[-1]
    return text


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for NumPy scalars, Paths, and Evidently objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class EvidentlyMonitor:
    """Runs Evidently AI data/prediction drift and classification quality checks with automated actions."""

    def __init__(
        self,
        drift_share_threshold: float = 0.25,
        psi_threshold: float = 0.20,
        severe_psi_threshold: float = 0.35,
        min_accuracy_threshold: float = 0.75,
        min_roc_auc_threshold: float = 0.78,
    ) -> None:
        self.drift_share_threshold = float(drift_share_threshold)
        self.psi_threshold = float(psi_threshold)
        self.severe_psi_threshold = float(severe_psi_threshold)
        self.min_accuracy_threshold = float(min_accuracy_threshold)
        self.min_roc_auc_threshold = float(min_roc_auc_threshold)

    @staticmethod
    def _infer_column_types(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        exclude_cols: set[str],
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> tuple[List[str], List[str]]:
        common_cols = [
            col for col in reference_df.columns
            if col in current_df.columns and col not in exclude_cols
        ]
        if numerical_columns is not None or categorical_columns is not None:
            num_cols = [c for c in (numerical_columns or []) if c in common_cols]
            cat_cols = [c for c in (categorical_columns or []) if c in common_cols and c not in num_cols]
            return num_cols, cat_cols

        num_cols: List[str] = []
        cat_cols: List[str] = []
        for col in common_cols:
            ref_dtype = reference_df[col].dtype
            if pd.api.types.is_bool_dtype(ref_dtype):
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(ref_dtype):
                num_cols.append(col)
            else:
                sample = reference_df[col].dropna().astype(str).str.strip()
                sample = sample[sample != ""]
                if len(sample) > 0 and pd.to_numeric(sample, errors="coerce").notna().mean() >= 0.90:
                    num_cols.append(col)
                else:
                    cat_cols.append(col)
        return num_cols, cat_cols

    @staticmethod
    def _sanitize_frame(
        df: pd.DataFrame,
        selected_cols: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str],
        target_col: Optional[str],
        pred_label_col: Optional[str],
        pred_prob_col: Optional[str],
    ) -> pd.DataFrame:
        out = df[selected_cols].copy()
        for col in categorical_columns:
            if col in out.columns:
                out[col] = out[col].astype(str)
        for col in numerical_columns:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if target_col and target_col in out.columns:
            out[target_col] = pd.to_numeric(out[target_col], errors="coerce").fillna(0).astype(int)
        if pred_label_col and pred_label_col in out.columns:
            out[pred_label_col] = pd.to_numeric(out[pred_label_col], errors="coerce").fillna(0).astype(int)
        if pred_prob_col and pred_prob_col in out.columns:
            out[pred_prob_col] = pd.to_numeric(out[pred_prob_col], errors="coerce").clip(0.0, 1.0)
        return out

    @staticmethod
    def _extract_metrics_and_tests(
        snapshot_dict: Dict[str, Any],
        psi_threshold: float,
    ) -> Dict[str, Any]:
        drifted_columns_from_tests: set[str] = set()
        drifted_columns_fallback: set[str] = set()
        column_drift_scores: Dict[str, float] = {}
        psi_drift_scores: Dict[str, float] = {}
        classification_metrics: Dict[str, float] = {}
        evidently_drift_count: Optional[int] = None
        evidently_drift_share: Optional[float] = None

        for metric in snapshot_dict.get("metrics", []):
            metric_name = str(metric.get("metric_name", ""))
            config = metric.get("config", {}) or {}
            value = metric.get("value")

            if metric_name.startswith("DriftedColumnsCount") and isinstance(value, dict):
                if "count" in value:
                    evidently_drift_count = int(value["count"])
                if "share" in value:
                    evidently_drift_share = float(value["share"])
            elif metric_name.startswith("ValueDrift"):
                col = config.get("column")
                method = str(config.get("method", "")).lower()
                threshold = float(config.get("threshold") or psi_threshold)
                if col and isinstance(value, (int, float)):
                    score = float(value)
                    col_name = str(col)
                    column_drift_scores[col_name] = round(score, 4)
                    if "psi" in method:
                        psi_drift_scores[col_name] = round(score, 4)
                    is_p_value = "p_value" in method or "test" in method
                    if (is_p_value and score < threshold) or (not is_p_value and score >= threshold):
                        drifted_columns_fallback.add(col_name)
            elif "Accuracy" in metric_name and isinstance(value, (int, float)):
                classification_metrics["accuracy"] = round(float(value), 4)
            elif "Precision" in metric_name and isinstance(value, (int, float)):
                classification_metrics["precision"] = round(float(value), 4)
            elif "Recall" in metric_name and isinstance(value, (int, float)):
                classification_metrics["recall"] = round(float(value), 4)
            elif "F1Score" in metric_name and isinstance(value, (int, float)):
                classification_metrics["f1"] = round(float(value), 4)
            elif "RocAuc" in metric_name and isinstance(value, (int, float)):
                classification_metrics["roc_auc"] = round(float(value), 4)

        parsed_tests: List[Dict[str, Any]] = []
        passed_count = 0
        failed_tests: List[Dict[str, Any]] = []
        for test in snapshot_dict.get("tests", []):
            status = _normalize_status(test.get("status"))
            test_id = str(test.get("id", ""))
            test_name = str(test.get("name", ""))
            metric_params = ((test.get("metric_config") or {}).get("params") or {})
            col_name = metric_params.get("column")
            if test_id == "drift" and col_name and status in {"FAIL", "ERROR"}:
                drifted_columns_from_tests.add(str(col_name))

            is_mlops_sla_test = (
                test_id == "drift"
                or test_name.startswith("Share of Drifted Columns")
                or test_name.startswith("Accuracy metric:")
                or test_name.startswith("RocAuc metric:")
            )
            if not is_mlops_sla_test:
                continue

            item = {
                "id": test_id,
                "name": test_name,
                "description": str(test.get("description", "")),
                "status": status,
                "column": str(col_name) if col_name else None,
            }
            parsed_tests.append(item)
            if status == "SUCCESS":
                passed_count += 1
            elif status in {"FAIL", "ERROR"}:
                failed_tests.append(item)

        drifted_columns = sorted(drifted_columns_from_tests or drifted_columns_fallback)
        num_checked = len(column_drift_scores)
        num_drifted = (
            evidently_drift_count if evidently_drift_count is not None else len(drifted_columns)
        )
        drift_share = (
            evidently_drift_share
            if evidently_drift_share is not None
            else (num_drifted / max(num_checked, 1))
        )

        return {
            "num_columns_checked": num_checked,
            "num_drifted_columns": num_drifted,
            "drift_share": round(float(drift_share), 4),
            "drifted_columns": drifted_columns,
            "column_drift_scores": column_drift_scores,
            "psi_drift_scores": psi_drift_scores,
            "classification_metrics": classification_metrics,
            "tests_total": len(parsed_tests),
            "tests_passed": passed_count,
            "tests_failed": len(failed_tests),
            "failed_tests": failed_tests,
            "tests": parsed_tests,
        }

    def evaluate_actions(
        self,
        summary: Dict[str, Any],
        auto_retrain: bool = False,
        retrain_callback: Optional[Callable[[], Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Determine alert severity and execute/record appropriate MLOps actions."""
        drift_share = float(summary.get("drift_share", 0.0))
        num_drifted = int(summary.get("num_drifted_columns", 0))
        drifted_cols = list(summary.get("drifted_columns", []))
        psi_scores = summary.get("psi_drift_scores", {}) or {}
        cls_metrics = summary.get("classification_metrics", {}) or {}
        failed_tests = list(summary.get("failed_tests", []))

        max_psi = max((float(v) for v in psi_scores.values()), default=0.0)
        accuracy = cls_metrics.get("accuracy")
        roc_auc = cls_metrics.get("roc_auc")

        reasons: List[str] = []
        severity = "OK"

        if drift_share >= self.drift_share_threshold:
            severity = "CRITICAL"
            reasons.append(
                f"Drift share {drift_share:.2%} >= threshold {self.drift_share_threshold:.2%}"
            )
        elif num_drifted > 0:
            severity = "WARNING"
            reasons.append(
                f"{num_drifted} column(s) drifted below global share threshold ({drift_share:.2%})"
            )

        if max_psi >= self.severe_psi_threshold:
            severity = "CRITICAL"
            reasons.append(
                f"Severe single-column PSI shift detected (max PSI={max_psi:.3f} >= {self.severe_psi_threshold:.2f})"
            )

        if accuracy is not None and float(accuracy) < self.min_accuracy_threshold:
            severity = "CRITICAL"
            reasons.append(
                f"Classification accuracy dropped to {accuracy:.4f} (< {self.min_accuracy_threshold:.4f})"
            )

        if roc_auc is not None and float(roc_auc) < self.min_roc_auc_threshold:
            severity = "CRITICAL"
            reasons.append(
                f"Classification ROC-AUC dropped to {roc_auc:.4f} (< {self.min_roc_auc_threshold:.4f})"
            )

        cls_failed = [
            t for t in failed_tests if "Accuracy" in t.get("name", "") or "RocAuc" in t.get("name", "")
        ]
        if cls_failed:
            severity = "CRITICAL"
            reasons.append(f"{len(cls_failed)} Evidently classification quality test(s) failed")
        elif failed_tests and severity == "OK":
            severity = "WARNING"
            reasons.append(f"{len(failed_tests)} Evidently suite test(s) failed")

        retrain_recommended = severity == "CRITICAL"
        actions: List[Dict[str, str]] = []
        retrain_executed = False
        retrain_status = "not_required"

        if severity == "OK":
            actions.append(
                {
                    "action": "CONTINUE_PRODUCTION_SERVING",
                    "priority": "LOW",
                    "status": "EXECUTED",
                    "detail": "All Evidently drift and quality tests passed; champion model remains active.",
                }
            )
        elif severity == "WARNING":
            actions.append(
                {
                    "action": "LOG_DRIFT_WARNING_ALERT",
                    "priority": "MEDIUM",
                    "status": "EXECUTED",
                    "detail": f"Shifted columns ({', '.join(drifted_cols[:8]) or 'see report'}) logged for investigation.",
                }
            )
            actions.append(
                {
                    "action": "INCREASE_MONITORING_FREQUENCY",
                    "priority": "MEDIUM",
                    "status": "RECOMMENDED",
                    "detail": "Schedule tighter batch drift audits until shifted features stabilize.",
                }
            )
        else:
            actions.append(
                {
                    "action": "HALT_SILENT_MODEL_PROMOTION",
                    "priority": "HIGH",
                    "status": "EXECUTED",
                    "detail": "Flagged active deployment as degraded in monitoring_actions.json and Prometheus gauges.",
                }
            )
            actions.append(
                {
                    "action": "UPDATE_PROMETHEUS_ALERT_STATE",
                    "priority": "HIGH",
                    "status": "EXECUTED",
                    "detail": "Set churn_retrain_recommended=1 and churn_monitoring_alert_severity=2.",
                }
            )

            if auto_retrain and retrain_callback is not None:
                try:
                    logger.warning("CRITICAL drift/degradation detected -> triggering automated retraining pipeline.")
                    retrain_callback()
                    retrain_executed = True
                    retrain_status = "executed_success"
                    actions.append(
                        {
                            "action": "TRIGGER_AUTOMATED_RETRAINING",
                            "priority": "CRITICAL",
                            "status": "EXECUTED",
                            "detail": "Executed training pipeline automatically after CRITICAL drift trigger.",
                        }
                    )
                except Exception as exc:
                    retrain_status = f"executed_failed: {exc}"
                    actions.append(
                        {
                            "action": "TRIGGER_AUTOMATED_RETRAINING",
                            "priority": "CRITICAL",
                            "status": "FAILED",
                            "detail": f"Automated retraining failed: {exc}",
                        }
                    )
            else:
                retrain_status = "recommended_manual_or_scheduled"
                actions.append(
                    {
                        "action": "TRIGGER_AUTOMATED_RETRAINING",
                        "priority": "CRITICAL",
                        "status": "RECOMMENDED",
                        "detail": "Run `python -m pipelines.training_pipeline` or click Retrain Now in Streamlit.",
                    }
                )

        target_webhook = webhook_url or os.getenv("MONITORING_WEBHOOK_URL", "").strip()
        if severity in {"WARNING", "CRITICAL"} and target_webhook:
            webhook_status = self._dispatch_webhook(
                target_webhook,
                {"severity": severity, "reasons": reasons, "drift_share": drift_share, "drifted_columns": drifted_cols},
            )
            actions.append(
                {
                    "action": "SEND_WEBHOOK_NOTIFICATION",
                    "priority": "HIGH",
                    "status": webhook_status,
                    "detail": f"Dispatched alert payload to configured monitoring webhook ({webhook_status}).",
                }
            )

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "severity_level": _SEVERITY_RANK[severity],
            "retrain_recommended": retrain_recommended,
            "retrain_executed": retrain_executed,
            "retrain_status": retrain_status,
            "reasons": reasons or ["All monitored distributions and test checks within thresholds."],
            "actions": actions,
        }

    @staticmethod
    def _dispatch_webhook(url: str, payload: Dict[str, Any]) -> str:
        """Best-effort webhook delivery for Slack/Teams/PagerDuty alerting."""
        try:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return f"EXECUTED_HTTP_{resp.status}"
        except Exception as exc:
            logger.warning(f"Monitoring webhook delivery failed: {exc}")
            return "FAILED"

    def run_monitoring(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        *,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        pred_label_col: Optional[str] = None,
        pred_prob_col: Optional[str] = None,
        html_output_path: Optional[str | Path] = None,
        json_output_path: Optional[str | Path] = None,
        actions_output_path: Optional[str | Path] = None,
        auto_retrain: bool = False,
        retrain_callback: Optional[Callable[[], Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Evidently DataSummary + DataDrift (+ Classification if labels/preds exist) and evaluate actions."""
        if reference_df.empty or current_df.empty:
            raise ValueError("Reference and current DataFrames must both be non-empty.")

        special_cols = {c for c in (target_col, pred_label_col, pred_prob_col) if c}
        has_classification = bool(
            target_col
            and (pred_label_col or pred_prob_col)
            and target_col in reference_df.columns
            and target_col in current_df.columns
            and (
                (pred_label_col and pred_label_col in reference_df.columns and pred_label_col in current_df.columns)
                or (pred_prob_col and pred_prob_col in reference_df.columns and pred_prob_col in current_df.columns)
            )
        )

        num_cols, cat_cols = self._infer_column_types(
            reference_df,
            current_df,
            exclude_cols=special_cols,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
        drift_columns = num_cols + cat_cols
        selected_cols = list(drift_columns)
        if has_classification:
            for col in (target_col, pred_label_col, pred_prob_col):
                if col and col in reference_df.columns and col in current_df.columns and col not in selected_cols:
                    selected_cols.append(col)

        ref_clean = self._sanitize_frame(
            reference_df, selected_cols, cat_cols, num_cols,
            target_col if has_classification else None,
            pred_label_col if has_classification else None,
            pred_prob_col if has_classification else None,
        )
        cur_clean = self._sanitize_frame(
            current_df, selected_cols, cat_cols, num_cols,
            target_col if has_classification else None,
            pred_label_col if has_classification else None,
            pred_prob_col if has_classification else None,
        )

        classification_def = None
        if has_classification:
            classification_def = [
                BinaryClassification(
                    target=target_col,
                    prediction_labels=pred_label_col if pred_label_col in cur_clean.columns else None,
                    prediction_probas=pred_prob_col if pred_prob_col in cur_clean.columns else None,
                    pos_label=1,
                )
            ]

        data_def = DataDefinition(
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
            classification=classification_def,
        )
        ds_ref = Dataset.from_pandas(ref_clean, data_definition=data_def)
        ds_cur = Dataset.from_pandas(cur_clean, data_definition=data_def)

        presets: List[Any] = [
            DataSummaryPreset(),
            DataDriftPreset(
                columns=drift_columns,
                drift_share=self.drift_share_threshold,
                num_method="psi",
                num_threshold=self.psi_threshold,
                include_tests=True,
            ),
        ]
        if has_classification:
            presets.append(
                ClassificationPreset(
                    accuracy_tests=[gt(self.min_accuracy_threshold)],
                    rocauc_tests=[gt(self.min_roc_auc_threshold)],
                    include_tests=True,
                )
            )

        report = Report(metrics=presets, include_tests=True)
        snapshot = report.run(current_data=ds_cur, reference_data=ds_ref)

        if html_output_path is not None:
            html_out = Path(html_output_path)
            html_out.parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(str(html_out))
            logger.info(f"Saved Evidently interactive HTML report to {html_out}")

        snapshot_dict = snapshot.dict()
        summary = self._extract_metrics_and_tests(snapshot_dict, psi_threshold=self.psi_threshold)
        summary["reference_rows"] = int(len(ref_clean))
        summary["current_rows"] = int(len(cur_clean))
        summary["classification_monitored"] = has_classification
        summary["html_report_path"] = str(html_output_path) if html_output_path else None

        action_plan = self.evaluate_actions(
            summary,
            auto_retrain=auto_retrain,
            retrain_callback=retrain_callback,
            webhook_url=webhook_url,
        )
        summary["action_plan"] = action_plan

        if json_output_path is not None:
            self.save_json(summary, json_output_path)
        if actions_output_path is not None:
            self.save_json(self.build_actions_record(summary), actions_output_path)

        logger.info(
            f"Evidently monitoring complete | Severity={action_plan['severity']} | "
            f"Drifted={summary['num_drifted_columns']}/{summary['num_columns_checked']} "
            f"({summary['drift_share']:.2%}) | Tests={summary['tests_passed']}/{summary['tests_total']} passed"
        )
        return summary

    @staticmethod
    def build_actions_record(summary: Dict[str, Any]) -> Dict[str, Any]:
        """Build the flat monitoring action record schema shared across API, CLI, and disk reports."""
        action_plan = summary["action_plan"]
        return {
            "severity": action_plan["severity"],
            "severity_level": action_plan["severity_level"],
            "retrain_recommended": action_plan["retrain_recommended"],
            "retrain_executed": action_plan["retrain_executed"],
            "retrain_status": action_plan["retrain_status"],
            "timestamp_utc": action_plan["timestamp_utc"],
            "drift_share": summary["drift_share"],
            "num_drifted_columns": summary["num_drifted_columns"],
            "num_columns_checked": summary["num_columns_checked"],
            "drifted_columns": summary["drifted_columns"],
            "classification_metrics": summary.get("classification_metrics", {}),
            "tests_total": summary.get("tests_total", 0),
            "tests_passed": summary.get("tests_passed", 0),
            "tests_failed": summary.get("tests_failed", 0),
            "reasons": action_plan["reasons"],
            "actions": action_plan["actions"],
        }

    @staticmethod
    def save_json(payload: Dict[str, Any], path: str | Path) -> None:
        """Persist a monitoring report or action payload cleanly as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
