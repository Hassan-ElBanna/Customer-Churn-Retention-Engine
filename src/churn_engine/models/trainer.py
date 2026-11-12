"""Phase 3a: Multi-Model Training, Experiment Tracking (MLflow), and Champion Selection."""

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
        """Instantiate candidate models (Naive baseline, Logistic Regression, Random Forest, XGBoost)."""
        pos_count = max(int((y_train == 1).sum()), 1)
        neg_count = max(int((y_train == 0).sum()), 1)
        scale_pos_weight = round(neg_count / pos_count, 2)

        candidates: Dict[str, Any] = {
            "naive_baseline": DummyClassifier(strategy="prior", random_state=self.random_state),
            "logistic_regression": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=self.random_state,
            ),
        }

        try:
            xgb_module = importlib.import_module("xgboost")
            XGBClassifier = getattr(xgb_module, "XGBClassifier")

            candidates["xgboost"] = XGBClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        except (ImportError, AttributeError):
            logger.warning("XGBoost not installed; skipping XGBClassifier candidate.")

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
                f"[{name}] Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"PR-AUC: {val_metrics['pr_auc']:.4f} | "
                f"F1: {val_metrics['f1_score']:.4f} | "
                f"Optimal Thresh: {val_metrics['optimal_threshold']:.3f}"
            )

            if name != "naive_baseline" and val_metrics[self.primary_metric] > best_score:
                best_score = val_metrics[self.primary_metric]
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
                f"Champion '{self.champion_name}' Test ROC-AUC: {test_metrics['roc_auc']:.4f} "
                f"(95% CI: {test_metrics['roc_auc_95ci']})"
            )

        return self.champion_name, self.champion_model, summary

    def get_feature_importances(self, feature_names: list[str], top_k: int = 10) -> Dict[str, float]:
        """Extract top-K feature importances from the champion model."""
        if self.champion_model is None:
            return {}

        if hasattr(self.champion_model, "feature_importances_"):
            importances = np.asarray(self.champion_model.feature_importances_, dtype=float)
        elif hasattr(self.champion_model, "coef_"):
            importances = np.abs(np.asarray(self.champion_model.coef_[0], dtype=float))
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

