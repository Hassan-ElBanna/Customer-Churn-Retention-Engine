"""Phase 5: Data & Prediction Drift Monitoring (PSI & Two-Sample KS Test)."""

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift tests."""

    def __init__(self, psi_threshold: float = 0.20, ks_alpha: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    @staticmethod
    def calculate_psi(
        reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4
    ) -> float:
        """Compute Population Stability Index (PSI) between reference and current distributions."""
        ref_clean = reference[~np.isnan(reference)]
        cur_clean = current[~np.isnan(current)]
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return 0.0

        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.unique(np.percentile(ref_clean, quantiles))
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

        ref_pct = np.clip(ref_counts / max(len(ref_clean), 1), eps, None)
        cur_pct = np.clip(cur_counts / max(len(cur_clean), 1), eps, None)

        psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(psi_value, 4)

    def evaluate_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare numerical features between reference training data and production batches."""
        feature_reports: Dict[str, Any] = {}
        drifted_features: list[str] = []

        common_num_cols = [
            col
            for col in reference_df.select_dtypes(include=[np.number]).columns
            if col in current_df.columns
        ]

        for col in common_num_cols:
            ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
            cur_vals = current_df[col].dropna().to_numpy(dtype=float)
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue

            psi = self.calculate_psi(ref_vals, cur_vals)
            ks_stat, ks_p_value = ks_2samp(ref_vals, cur_vals)

            is_drifted = bool(psi >= self.psi_threshold or ks_p_value < self.ks_alpha)
            if is_drifted:
                drifted_features.append(col)

            feature_reports[col] = {
                "psi": psi,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_p_value), 6),
                "drift_detected": is_drifted,
            }

        drift_share = len(drifted_features) / max(len(common_num_cols), 1)
        retrain_recommended = drift_share >= 0.25 or any(
            r["psi"] >= 0.35 for r in feature_reports.values()
        )

        report = {
            "num_features_checked": len(common_num_cols),
            "num_drifted_features": len(drifted_features),
            "drift_share": round(drift_share, 4),
            "drifted_features": drifted_features,
            "retrain_recommended": retrain_recommended,
            "features": feature_reports,
        }

        logger.info(
            f"Drift check complete: {len(drifted_features)}/{len(common_num_cols)} features drifted "
            f"(Retrain recommended: {retrain_recommended})"
        )
        return report

    @staticmethod
    def save_report(report: Dict[str, Any], path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

