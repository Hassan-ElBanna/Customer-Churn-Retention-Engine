"""Phase 1b: Data Quality & Schema Validation."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Structured report of dataset health and quality checks."""

    passed: bool
    num_rows: int
    num_columns: int
    target_column: str
    target_present: bool
    target_distribution: Dict[str, float]
    duplicate_rows: int
    missing_summary: Dict[str, float]
    high_missing_columns: List[str]
    constant_columns: List[str]
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class DataValidator:
    """Validates raw customer churn datasets prior to feature engineering."""

    def __init__(
        self,
        target_column: str = "Churn",
        min_rows: int = 50,
        max_missing_ratio: float = 0.40,
    ):
        self.target_column = target_column
        self.min_rows = min_rows
        self.max_missing_ratio = max_missing_ratio

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run comprehensive data quality checks."""
        errors: List[str] = []
        warnings: List[str] = []

        num_rows, num_columns = df.shape
        if num_rows < self.min_rows:
            errors.append(f"Row count ({num_rows}) is below minimum threshold ({self.min_rows}).")

        # Auto-detect target column case-insensitively if exact match is missing
        target_present = self.target_column in df.columns
        if not target_present:
            errors.append(
                f"Target column '{self.target_column}' not found in columns: {list(df.columns)}"
            )

        target_dist: Dict[str, float] = {}
        if target_present:
            null_targets = int(df[self.target_column].isna().sum())
            if null_targets > 0:
                warnings.append(f"Target column '{self.target_column}' has {null_targets} missing values.")
            vc = df[self.target_column].dropna().value_counts(normalize=True)
            target_dist = {str(k): round(float(v), 4) for k, v in vc.items()}
            if len(target_dist) < 2:
                errors.append(
                    f"Target column '{self.target_column}' must have at least 2 classes, found: {target_dist}"
                )

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate rows.")

        missing_ratios = (df.isna().sum() / max(num_rows, 1)).round(4)
        missing_summary = {col: float(ratio) for col, ratio in missing_ratios.items() if ratio > 0}
        high_missing = [
            col for col, ratio in missing_summary.items() if ratio > self.max_missing_ratio
        ]
        if high_missing:
            warnings.append(
                f"Columns exceeding {self.max_missing_ratio:.0%} missingness: {high_missing}"
            )

        constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
        if constant_cols:
            warnings.append(f"Constant or zero-variance columns detected: {constant_cols}")

        passed = len(errors) == 0
        report = ValidationReport(
            passed=passed,
            num_rows=num_rows,
            num_columns=num_columns,
            target_column=self.target_column,
            target_present=target_present,
            target_distribution=target_dist,
            duplicate_rows=duplicate_rows,
            missing_summary=missing_summary,
            high_missing_columns=high_missing,
            constant_columns=constant_cols,
            warnings=warnings,
            errors=errors,
        )

        if passed:
            logger.info(
                f"Data validation PASSED ({num_rows:,} rows, {num_columns} cols, target dist: {target_dist})"
            )
        else:
            logger.error(f"Data validation FAILED with errors: {errors}")

        return report

