"""Phase 2: Feature Engineering & Strict Leakage-Free Preprocessing."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)


class ChurnFeaturePreprocessor:
    """
    Handles domain feature engineering, train/val/test splitting BEFORE fitting,
    missing value imputation, categorical encoding, and numerical scaling.
    """

    def __init__(
        self,
        target_column: str = "Churn",
        id_columns: Optional[List[str]] = None,
        max_categorical_cardinality: int = 50,
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.id_columns = id_columns or [
            "customerID",
            "CustomerID",
            "customer_id",
            "id",
            "RowNumber",
            "Surname",
        ]
        self.max_categorical_cardinality = max_categorical_cardinality
        self.random_state = random_state

        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.dropped_cols: List[str] = []
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_out_: List[str] = []

    @staticmethod
    def encode_target(series: pd.Series) -> pd.Series:
        """Convert binary churn target (Yes/No, True/False, 1/0, Churned/Retained) to 0/1."""
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(int)
        cleaned = series.astype(str).str.strip().str.lower()
        mapping = {
            "yes": 1,
            "true": 1,
            "1": 1,
            "1.0": 1,
            "churn": 1,
            "churned": 1,
            "exited": 1,
            "no": 0,
            "false": 0,
            "0": 0,
            "0.0": 0,
            "retained": 0,
            "stayed": 0,
        }
        mapped = cleaned.map(mapping)
        if mapped.isna().any():
            unmapped = series[mapped.isna()].unique()
            raise ValueError(f"Unrecognized target values in '{series.name}': {unmapped}")
        return mapped.astype(int)

    @staticmethod
    def clean_and_engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean common data quirks (e.g., blank strings in numeric columns like TotalCharges)
        and engineer churn-specific behavioral & monetary ratios.
        """
        df = df.copy()

        # Coerce object columns that are actually numeric with whitespace (e.g. Telco TotalCharges)
        for col in df.select_dtypes(include=["object", "category"]).columns:
            sample = df[col].dropna().astype(str).str.strip()
            sample = sample[sample != ""]
            if len(sample) > 0:
                converted = pd.to_numeric(sample, errors="coerce")
                if converted.notna().mean() > 0.90:
                    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

        col_map = {c.lower(): c for c in df.columns}

        # Telco Domain Rule: When tenure == 0, TotalCharges is blank because first bill hasn't settled
        if "totalcharges" in col_map and "tenure" in col_map:
            tc_col = col_map["totalcharges"]
            ten_col = col_map["tenure"]
            zero_tenure_mask = (df[ten_col] == 0) & (df[tc_col].isna())
            df.loc[zero_tenure_mask, tc_col] = 0.0

        # Domain Feature 1: Average Monthly Spend Ratio & Recent Price Hike Signal
        if "totalcharges" in col_map and "tenure" in col_map:
            tc_col = col_map["totalcharges"]
            ten_col = col_map["tenure"]
            df["avg_monthly_spend_historical"] = np.where(
                df[ten_col] > 0,
                df[tc_col] / df[ten_col].clip(lower=1),
                df.get(col_map.get("monthlycharges", ""), 0.0),
            )
            if "monthlycharges" in col_map:
                mc_col = col_map["monthlycharges"]
                df["monthly_to_historical_spend_ratio"] = df[mc_col] / (
                    df["avg_monthly_spend_historical"].replace(0, np.nan)
                )
                df["monthly_price_delta"] = df[mc_col] - df["avg_monthly_spend_historical"]

        # Domain Feature 2: Telco Service Bundle & Protection Counts
        addon_cols = [
            col_map[c]
            for c in [
                "onlinesecurity",
                "onlinebackup",
                "deviceprotection",
                "techsupport",
                "streamingtv",
                "streamingmovies",
            ]
            if c in col_map
        ]
        if addon_cols:
            df["total_addon_services"] = sum(
                (df[c].astype(str).str.strip().str.lower() == "yes").astype(int)
                for c in addon_cols
            )
            if "monthlycharges" in col_map:
                df["charge_per_service"] = df[col_map["monthlycharges"]] / (
                    1.0 + df["total_addon_services"]
                )

        protection_cols = [
            col_map[c]
            for c in ["onlinesecurity", "onlinebackup", "deviceprotection", "techsupport"]
            if c in col_map
        ]
        if protection_cols:
            df["protection_support_count"] = sum(
                (df[c].astype(str).str.strip().str.lower() == "yes").astype(int)
                for c in protection_cols
            )

        # Domain Feature 3: Unprotected Fiber Optic Risk Flag & AutoPay Flag
        if (
            "internetservice" in col_map
            and "onlinesecurity" in col_map
            and "techsupport" in col_map
        ):
            is_fiber = (
                df[col_map["internetservice"]].astype(str).str.strip().str.lower() == "fiber optic"
            )
            no_sec = df[col_map["onlinesecurity"]].astype(str).str.strip().str.lower() == "no"
            no_tech = df[col_map["techsupport"]].astype(str).str.strip().str.lower() == "no"
            df["unprotected_fiber_flag"] = (is_fiber & no_sec & no_tech).astype(int)

        if "paymentmethod" in col_map:
            df["is_autopay"] = (
                df[col_map["paymentmethod"]]
                .astype(str)
                .str.lower()
                .str.contains("automatic", na=False)
                .astype(int)
            )

        if "tenure" in col_map:
            ten_col = col_map["tenure"]
            df["tenure_cohort"] = pd.cut(
                df[ten_col],
                bins=[-1, 6, 12, 24, 48, 1000],
                labels=[
                    "0-6m_Onboarding",
                    "7-12m_Early",
                    "13-24m_Growing",
                    "25-48m_Established",
                    "49m+_Loyal",
                ],
            ).astype(str)

        # Domain Feature 4: Balance-to-Salary Ratio (Banking / SaaS churn datasets)
        if "balance" in col_map and "estimatedsalary" in col_map:
            bal_col = col_map["balance"]
            sal_col = col_map["estimatedsalary"]
            df["balance_to_salary_ratio"] = df[bal_col] / (df[sal_col].clip(lower=1.0))

        # Domain Feature 3: Tenure-to-Age Ratio
        if "tenure" in col_map and "age" in col_map:
            df["tenure_to_age_ratio"] = df[col_map["tenure"]] / (df[col_map["age"]].clip(lower=18))

        # Domain Feature 4: Support Calls per Tenure Month
        if "support_calls" in col_map and "tenure" in col_map:
            df["support_calls_per_month"] = df[col_map["support_calls"]] / (
                df[col_map["tenure"]].clip(lower=1)
            )

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Strict Featurization Ordering: Split raw data into train, validation, and test sets
        BEFORE fitting any imputation, scaling, or encoding pipelines.
        """
        df = df.dropna(subset=[self.target_column]).copy()
        df[self.target_column] = self.encode_target(df[self.target_column])

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[self.target_column],
            random_state=self.random_state,
        )

        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=train_val_df[self.target_column],
            random_state=self.random_state,
        )

        logger.info(
            f"Split dataset -> Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}"
        )
        return train_df, val_df, test_df

    def fit(self, train_df: pd.DataFrame) -> "ChurnFeaturePreprocessor":
        """Fit preprocessing transformers strictly on training data."""
        df = self.clean_and_engineer_domain_features(train_df)

        # Drop target and identifier columns
        exclude = {self.target_column} | set(self.id_columns)
        feature_cols = [c for c in df.columns if c not in exclude]

        self.numerical_cols = []
        self.categorical_cols = []
        self.dropped_cols = []

        for col in feature_cols:
            if df[col].nunique(dropna=True) <= 1:
                self.dropped_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                self.numerical_cols.append(col)
            else:
                cardinality = df[col].nunique(dropna=True)
                if cardinality <= self.max_categorical_cardinality:
                    self.categorical_cols.append(col)
                else:
                    logger.warning(
                        f"Dropping high-cardinality column '{col}' ({cardinality} unique values)"
                    )
                    self.dropped_cols.append(col)

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                ),
            ]
        )

        transformers = []
        if self.numerical_cols:
            transformers.append(("num", num_pipeline, self.numerical_cols))
        if self.categorical_cols:
            transformers.append(("cat", cat_pipeline, self.categorical_cols))

        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        self.pipeline.fit(df[self.numerical_cols + self.categorical_cols])
        self.feature_names_out_ = [
            name.split("__", 1)[-1] for name in self.pipeline.get_feature_names_out()
        ]

        logger.info(
            f"Preprocessor fitted on {len(self.numerical_cols)} numerical & "
            f"{len(self.categorical_cols)} categorical features -> "
            f"{len(self.feature_names_out_)} transformed features."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataset using the fitted preprocessing pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call .fit() first.")

        df_engineered = self.clean_and_engineer_domain_features(df)

        # Ensure all expected input columns exist
        for col in self.numerical_cols + self.categorical_cols:
            if col not in df_engineered.columns:
                df_engineered[col] = np.nan

        transformed_array = self.pipeline.transform(
            df_engineered[self.numerical_cols + self.categorical_cols]
        )
        return pd.DataFrame(
            transformed_array,
            columns=self.feature_names_out_,
            index=df.index,
        )

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed DataFrame."""
        return self.fit(train_df).transform(train_df)

    def save(self, path: str | Path) -> None:
        """Persist fitted preprocessor to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out)
        logger.info(f"Saved fitted preprocessor to {out}")

    @classmethod
    def load(cls, path: str | Path) -> "ChurnFeaturePreprocessor":
        """Load fitted preprocessor from disk."""
        return joblib.load(Path(path))

