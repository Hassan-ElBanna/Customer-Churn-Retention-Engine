"""Phase 3b: Model Evaluation, Business Threshold Optimization & Confidence Intervals."""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates churn models using statistical, calibration, and business-cost metrics."""

    def __init__(
        self,
        cost_fn: float = 300.0,
        cost_fp: float = 75.0,
        optimize_for: str = "accuracy",
    ):
        self.cost_fn = cost_fn  # Cost of a missed churner (False Negative)
        self.cost_fp = cost_fp  # Cost of an unnecessary retention offer (False Positive)
        self.optimize_for = optimize_for

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, float]:
        """Find probability threshold that maximizes accuracy (or minimizes business cost)."""
        thresholds = np.linspace(0.20, 0.80, 121)
        best_threshold = 0.50
        best_acc = -1.0
        best_f1 = -1.0
        min_cost = float("inf")

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            total_cost = float(fn * self.cost_fn + fp * self.cost_fp)
            acc = float((tp + tn) / max(len(y_true), 1))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))

            if self.optimize_for == "accuracy":
                # Require meaningful minority-class recall (>= 54%) to prevent degenerate high thresholds
                if rec < 0.54 and thresh > 0.35:
                    continue
                if (acc > best_acc) or (abs(acc - best_acc) < 1e-6 and f1 > best_f1):
                    best_acc = acc
                    best_f1 = f1
                    best_threshold = float(thresh)
                    min_cost = total_cost
            else:
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_threshold = float(thresh)

        return round(best_threshold, 3), round(min_cost, 2)

    @staticmethod
    def bootstrap_roc_auc_ci(
        y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 200, random_state: int = 42
    ) -> Tuple[float, float]:
        """Compute 95% confidence interval for ROC-AUC via bootstrapping."""
        rng = np.random.RandomState(random_state)
        scores = []
        n = len(y_true)
        for _ in range(n_bootstraps):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if not scores:
            return 0.0, 0.0
        return round(float(np.percentile(scores, 2.5)), 4), round(
            float(np.percentile(scores, 97.5)), 4
        )

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_prob: np.ndarray,
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive classification, calibration, and business metrics."""
        y_true_arr = np.asarray(y_true).astype(int)
        y_prob_arr = np.asarray(y_prob).astype(float)

        opt_threshold, min_business_cost = self.find_optimal_threshold(y_true_arr, y_prob_arr)
        active_threshold = threshold if threshold is not None else opt_threshold
        y_pred = (y_prob_arr >= active_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        ci_low, ci_high = self.bootstrap_roc_auc_ci(y_true_arr, y_prob_arr)

        metrics = {
            "roc_auc": round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4),
            "roc_auc_95ci": [ci_low, ci_high],
            "pr_auc": round(float(average_precision_score(y_true_arr, y_prob_arr)), 4),
            "f1_score": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
            "brier_score": round(float(brier_score_loss(y_true_arr, y_prob_arr)), 4),
            "optimal_threshold": opt_threshold,
            "expected_business_cost": min_business_cost,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
        return metrics

