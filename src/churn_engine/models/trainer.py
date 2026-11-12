"""Phase 3a: Multi-Model Training, Experiment Tracking (MLflow), and Champion Selection."""

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression

from churn_engine.models.evaluator import ModelEvaluator
from churn_engine.utils.logger import get_logger
from churn_engine.utils.tracking import ExperimentTracker

logger = get_logger(__name__)


class ChurnModelTrainer:
    """
    Trains baseline and ensemble models, logs runs to MLflow / DagsHub,
    compares validation metrics, and promotes the champion model.
    """

    def __init__(
        self,
        evaluator: Optional[ModelEvaluator] = None,
        tracker: Optional[ExperimentTracker] = None,
        primary_metric: str = "roc_auc",
        random_state: int = 42,
    ):
        self.evaluator = evaluator or ModelEvaluator()
        self.tracker = tracker
        self.primary_metric = primary_metric
        self.random_state = random_state
        self.trained_models: Dict[str, Any] = {}
        self.leaderboard: Dict[str, Dict[str, Any]] = {}
        self.champion_name: Optional[str] = None
        self.champion_model: Optional[Any] = None
        self.champion_threshold: float = 0.50

    def _build_candidate_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Instantiate tuned candidate models optimized for high classification accuracy & ROC-AUC."""
        lr_tuned = LogisticRegression(
            C=0.25,
            penalty="l2",
            solver="lbfgs",
            max_iter=1500,
            random_state=self.random_state,
        )
        rf_tuned = RandomForestClassifier(
            n_estimators=400,
            max_depth=9,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
        )
        gb_tuned = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            min_samples_split=15,
            min_samples_leaf=10,
            subsample=0.85,
            random_state=self.random_state,
        )

        candidates: Dict[str, Any] = {
            "naive_baseline": DummyClassifier(strategy="most_frequent", random_state=self.random_state),
            "logistic_regression": lr_tuned,
            "random_forest": rf_tuned,
            "gradient_boosting": gb_tuned,
        }

        voting_estimators = [
            ("lr", lr_tuned),
            ("rf", rf_tuned),
            ("gb", gb_tuned),
        ]

        try:
            xgb_module = importlib.import_module("xgboost")
            XGBClassifier = getattr(xgb_module, "XGBClassifier")

            xgb_tuned = XGBClassifier(
                n_estimators=350,
                learning_rate=0.025,
                max_depth=4,
                min_child_weight=6,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=2.5,
                gamma=0.15,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
            candidates["xgboost"] = xgb_tuned
            voting_estimators.append(("xgb", xgb_tuned))
        except (ImportError, AttributeError):
            logger.warning("XGBoost not installed; skipping XGBClassifier candidate.")

        candidates["voting_ensemble"] = VotingClassifier(
            estimators=voting_estimators,
            voting="soft",
            n_jobs=-1,
        )

        return candidates

    def train_and_select_champion(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """Train all candidates, evaluate on validation set, and evaluate champion on test set."""
        candidates = self._build_candidate_models(X_train, y_train)
        best_score = -float("inf")

        for name, model in candidates.items():
            logger.info(f"Training candidate model: {name} ...")
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]
            val_metrics = self.evaluator.evaluate(y_val, val_probs)

            self.trained_models[name] = model
            self.leaderboard[name] = val_metrics

            if self.tracker is not None:
                self.tracker.log_candidate_run(
                    model_name=name,
                    model=model,
                    params=model.get_params() if hasattr(model, "get_params") else {},
                    val_metrics=val_metrics,
                )

            logger.info(
                f"[{name}] Val Accuracy: {val_metrics['accuracy']:.4f} | "
                f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"PR-AUC: {val_metrics['pr_auc']:.4f} | "
                f"F1: {val_metrics['f1_score']:.4f} | "
                f"Optimal Thresh: {val_metrics['optimal_threshold']:.3f}"
            )

            score = val_metrics[self.primary_metric]
            # Tie-breaker: use ROC-AUC when primary_metric (e.g., accuracy) is tied
            best_auc = (
                self.leaderboard[self.champion_name]["roc_auc"]
                if self.champion_name in self.leaderboard
                else -1.0
            )
            if name != "naive_baseline" and (
                score > best_score
                or (abs(score - best_score) < 1e-6 and val_metrics["roc_auc"] > best_auc)
            ):
                best_score = score
                self.champion_name = name
                self.champion_model = model
                self.champion_threshold = val_metrics["optimal_threshold"]

        if self.champion_model is None or self.champion_name is None:
            raise RuntimeError("No champion model could be selected.")

        summary: Dict[str, Any] = {
            "champion_model": self.champion_name,
            "champion_threshold": self.champion_threshold,
            "validation_leaderboard": self.leaderboard,
        }

        if X_test is not None and y_test is not None:
            test_probs = self.champion_model.predict_proba(X_test)[:, 1]
            test_metrics = self.evaluator.evaluate(
                y_test, test_probs, threshold=self.champion_threshold
            )
            summary["champion_test_metrics"] = test_metrics
            logger.info(
                f"Champion '{self.champion_name}' Test Accuracy: {test_metrics['accuracy']:.4f} | "
                f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} "
                f"(95% CI: {test_metrics['roc_auc_95ci']})"
            )

            # Also evaluate all candidate models on the test set for complete comparison
            test_leaderboard: Dict[str, Dict[str, Any]] = {}
            for c_name, c_model in self.trained_models.items():
                c_probs = c_model.predict_proba(X_test)[:, 1]
                c_thresh = self.leaderboard[c_name]["optimal_threshold"]
                test_leaderboard[c_name] = self.evaluator.evaluate(
                    y_test, c_probs, threshold=c_thresh
                )
            summary["test_leaderboard"] = test_leaderboard

        return self.champion_name, self.champion_model, summary

    def get_feature_importances(self, feature_names: list[str], top_k: int = 10) -> Dict[str, float]:
        """Extract top-K feature importances from the champion model (including VotingClassifier)."""
        if self.champion_model is None:
            return {}

        if hasattr(self.champion_model, "feature_importances_"):
            importances = np.asarray(self.champion_model.feature_importances_, dtype=float)
        elif hasattr(self.champion_model, "coef_"):
            importances = np.abs(np.asarray(self.champion_model.coef_[0], dtype=float))
        elif hasattr(self.champion_model, "named_estimators_"):
            # Average normalized importances across sub-estimators in VotingClassifier
            sub_imps = []
            for est in self.champion_model.named_estimators_.values():
                if hasattr(est, "feature_importances_"):
                    arr = np.asarray(est.feature_importances_, dtype=float)
                    sub_imps.append(arr / max(arr.sum(), 1e-9))
                elif hasattr(est, "coef_"):
                    arr = np.abs(np.asarray(est.coef_[0], dtype=float))
                    sub_imps.append(arr / max(arr.sum(), 1e-9))
            if not sub_imps:
                return {}
            importances = np.mean(sub_imps, axis=0)
        else:
            return {}

        paired = sorted(
            zip(feature_names, importances), key=lambda item: item[1], reverse=True
        )[:top_k]
        return {k: round(float(v), 4) for k, v in paired}

    def save_champion(
        self,
        model_path: str | Path,
        metrics_path: str | Path,
        summary: Dict[str, Any],
    ) -> None:
        """Persist champion model bundle and evaluation metrics JSON."""
        model_out = Path(model_path)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_name": self.champion_name,
            "model": self.champion_model,
            "threshold": self.champion_threshold,
        }
        joblib.dump(bundle, model_out)

        metrics_out = Path(metrics_path)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved champion model to {model_out} and evaluation report to {metrics_out}")

