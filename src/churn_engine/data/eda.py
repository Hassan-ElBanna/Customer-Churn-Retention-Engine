"""Comprehensive Exploratory Data Analysis (EDA) & MLOps Visualization Generator."""

import json
from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)

# Set clean visual style
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"
sns.set_theme(style="whitegrid", palette="deep")


class ChurnEDAAnalyzer:
    """Executes statistical EDA, anomaly detection, and generates visual plots."""

    def __init__(self, figures_dir: str | Path = "artifacts/reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_eda(self, raw_df: pd.DataFrame, target_col: str = "Churn") -> Dict[str, Any]:
        """Compute statistical insights and generate EDA figures."""
        df = raw_df.copy()
        df["Churn_Binary"] = (
            df[target_col].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
        )

        # Inspect TotalCharges whitespace anomalies
        raw_tc = df["TotalCharges"].astype(str).str.strip()
        blank_tc_mask = raw_tc == ""
        blank_tc_rows = df.loc[
            blank_tc_mask, ["customerID", "tenure", "MonthlyCharges", "Contract", target_col]
        ].to_dict(orient="records")

        df["TotalCharges_Num"] = pd.to_numeric(raw_tc, errors="coerce").fillna(0.0)

        # Engineered analytical columns
        addon_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        present_addons = [c for c in addon_cols if c in df.columns]
        df["Addon_Count"] = sum((df[c] == "Yes").astype(int) for c in present_addons)
        df["Tenure_Cohort"] = pd.cut(
            df["tenure"],
            bins=[-1, 6, 12, 24, 48, 100],
            labels=["0-6m (Onboarding)", "7-12m", "13-24m", "25-48m", "49-72m (Loyal)"],
        )

        total_customers = len(df)
        churned_customers = int(df["Churn_Binary"].sum())
        overall_churn_rate = round(float(df["Churn_Binary"].mean() * 100), 2)

        total_mrr = float(df["MonthlyCharges"].sum())
        churned_mrr = float(df.loc[df["Churn_Binary"] == 1, "MonthlyCharges"].sum())
        mrr_churn_pct = round(churned_mrr / max(total_mrr, 1.0) * 100, 2)

        # Segment breakdowns
        def _segment_stats(col: str) -> Dict[str, Any]:
            grp = (
                df.groupby(col, observed=False)
                .agg(
                    customers=("Churn_Binary", "count"),
                    churn_rate=("Churn_Binary", lambda s: round(float(s.mean() * 100), 2)),
                    avg_monthly_charges=("MonthlyCharges", lambda s: round(float(s.mean()), 2)),
                    mrr_at_risk=(
                        "MonthlyCharges",
                        lambda s: round(float(s[df.loc[s.index, "Churn_Binary"] == 1].sum()), 2),
                    ),
                )
                .reset_index()
            )
            return grp.to_dict(orient="records")

        unprotected_fiber = df[
            (df["InternetService"] == "Fiber optic")
            & (df["OnlineSecurity"] == "No")
            & (df["TechSupport"] == "No")
        ]
        unprotected_fiber_churn = round(float(unprotected_fiber["Churn_Binary"].mean() * 100), 2)

        insights: Dict[str, Any] = {
            "dataset_overview": {
                "total_rows": total_customers,
                "total_columns": int(raw_df.shape[1]),
                "churned_customers": churned_customers,
                "retained_customers": total_customers - churned_customers,
                "overall_churn_rate_pct": overall_churn_rate,
                "total_monthly_recurring_revenue": round(total_mrr, 2),
                "churned_monthly_recurring_revenue": round(churned_mrr, 2),
                "annualized_revenue_lost": round(churned_mrr * 12, 2),
                "mrr_lost_pct": mrr_churn_pct,
            },
            "anomalies_and_missing_values": {
                "blank_total_charges_count": int(blank_tc_mask.sum()),
                "all_blank_total_charges_have_zero_tenure": bool(
                    (df.loc[blank_tc_mask, "tenure"] == 0).all()
                ),
                "blank_total_charges_records": blank_tc_rows,
            },
            "numerical_summary_by_churn": {
                "retained": {
                    "median_tenure": float(df.loc[df["Churn_Binary"] == 0, "tenure"].median()),
                    "mean_tenure": round(
                        float(df.loc[df["Churn_Binary"] == 0, "tenure"].mean()), 2
                    ),
                    "median_monthly_charges": round(
                        float(df.loc[df["Churn_Binary"] == 0, "MonthlyCharges"].median()), 2
                    ),
                    "mean_monthly_charges": round(
                        float(df.loc[df["Churn_Binary"] == 0, "MonthlyCharges"].mean()), 2
                    ),
                },
                "churned": {
                    "median_tenure": float(df.loc[df["Churn_Binary"] == 1, "tenure"].median()),
                    "mean_tenure": round(
                        float(df.loc[df["Churn_Binary"] == 1, "tenure"].mean()), 2
                    ),
                    "median_monthly_charges": round(
                        float(df.loc[df["Churn_Binary"] == 1, "MonthlyCharges"].median()), 2
                    ),
                    "mean_monthly_charges": round(
                        float(df.loc[df["Churn_Binary"] == 1, "MonthlyCharges"].mean()), 2
                    ),
                },
            },
            "segment_breakdowns": {
                "by_contract": _segment_stats("Contract"),
                "by_internet_service": _segment_stats("InternetService"),
                "by_payment_method": _segment_stats("PaymentMethod"),
                "by_tenure_cohort": _segment_stats("Tenure_Cohort"),
                "by_senior_citizen": _segment_stats("SeniorCitizen"),
                "unprotected_fiber_optic": {
                    "customer_count": int(len(unprotected_fiber)),
                    "churn_rate_pct": unprotected_fiber_churn,
                },
            },
        }

        self._plot_target_and_revenue(df, insights)
        self._plot_contract_internet_payment(df)
        self._plot_addons_and_pricing(df)

        return insights

    def _plot_target_and_revenue(self, df: pd.DataFrame, insights: Dict[str, Any]) -> None:
        """Plot 1: Customer Churn Count, MRR Exposure, and Tenure Survival Distribution."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        ov = insights["dataset_overview"]

        # Subplot 1: Customer Count Split
        counts = [ov["retained_customers"], ov["churned_customers"]]
        labels = [
            f"Retained\n({counts[0]:,} | {100 - ov['overall_churn_rate_pct']:.1f}%)",
            f"Churned\n({counts[1]:,} | {ov['overall_churn_rate_pct']:.1f}%)",
        ]
        bars1 = axes[0].bar(labels, counts, color=["#2b8cbe", "#e34a33"], width=0.55)
        axes[0].set_title("Customer Base Split by Churn Status", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Number of Customers")
        for bar in bars1:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{int(bar.get_height()):,}",
                ha="center",
                fontweight="bold",
            )
        axes[0].set_ylim(0, max(counts) * 1.15)

        # Subplot 2: Monthly Recurring Revenue (MRR) Split
        retained_mrr = ov["total_monthly_recurring_revenue"] - ov["churned_monthly_recurring_revenue"]
        mrr_vals = [retained_mrr, ov["churned_monthly_recurring_revenue"]]
        mrr_labels = [
            f"Retained MRR\n(${retained_mrr:,.0f} | {100 - ov['mrr_lost_pct']:.1f}%)",
            f"Churned MRR\n(${mrr_vals[1]:,.0f} | {ov['mrr_lost_pct']:.1f}%)",
        ]
        bars2 = axes[1].bar(mrr_labels, mrr_vals, color=["#31a354", "#de2d26"], width=0.55)
        axes[1].set_title(
            "Monthly Recurring Revenue (MRR) Exposure", fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("Monthly Revenue ($ USD)")
        for bar in bars2:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5000,
                f"${bar.get_height():,.0f}/mo",
                ha="center",
                fontweight="bold",
            )
        axes[1].set_ylim(0, max(mrr_vals) * 1.15)

        # Subplot 3: Tenure Distribution by Churn Status
        sns.kdeplot(
            data=df,
            x="tenure",
            hue="Churn",
            palette={"No": "#2b8cbe", "Yes": "#e34a33"},
            fill=True,
            common_norm=False,
            alpha=0.35,
            linewidth=2,
            ax=axes[2],
        )
        axes[2].axvline(10, color="#e34a33", linestyle="--", linewidth=1.5, label="Churned Median (10m)")
        axes[2].axvline(38, color="#2b8cbe", linestyle="--", linewidth=1.5, label="Retained Median (38m)")
        axes[2].set_title("Customer Tenure Distribution by Churn", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Tenure (Months)")
        axes[2].set_ylabel("Density")
        axes[2].legend(loc="upper right")

        plt.suptitle(
            "Telco Customer Churn — Executive Volume, Revenue & Lifecycle Overview",
            fontsize=14,
            fontweight="bold",
            y=1.03,
        )
        plt.tight_layout()
        fig.savefig(self.figures_dir / "01_target_and_revenue_exposure.png")
        plt.close(fig)

    def _plot_contract_internet_payment(self, df: pd.DataFrame) -> None:
        """Plot 2: High-Risk Structural Drivers (Contract, Internet Service, Payment Method, Tenure Cohort)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        # 1. Contract Type Churn Rate
        c_grp = (
            df.groupby("Contract", observed=False)["Churn_Binary"]
            .agg(["mean", "count"])
            .reset_index()
        )
        c_grp["churn_pct"] = c_grp["mean"] * 100
        bars = axes[0, 0].bar(
            c_grp["Contract"],
            c_grp["churn_pct"],
            color=["#e34a33", "#fd8d3c", "#2b8cbe"],
            width=0.55,
        )
        axes[0, 0].set_title("Churn Rate by Contract Type", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Churn Rate (%)")
        axes[0, 0].set_ylim(0, 52)
        for bar, (_, row) in zip(bars, c_grp.iterrows()):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{row['churn_pct']:.1f}%\n(n={int(row['count']):,})",
                ha="center",
                fontsize=9.5,
                fontweight="bold",
            )

        # 2. Internet Service x Contract Heatmap
        piv = (
            df.pivot_table(
                index="InternetService",
                columns="Contract",
                values="Churn_Binary",
                aggfunc="mean",
            )
            * 100
        )
        sns.heatmap(
            piv,
            annot=False,
            cmap="YlOrRd",
            cbar_kws={"label": "Churn Rate (%)"},
            ax=axes[0, 1],
        )
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.iloc[i, j]
                axes[0, 1].text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if val > 35 else "black",
                    fontweight="bold",
                    fontsize=11,
                )
        axes[0, 1].set_title(
            "Interaction Heatmap: Internet Service vs. Contract (%)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, 1].set_xlabel("Contract Type")
        axes[0, 1].set_ylabel("Internet Service Tier")

        # 3. Payment Method Churn Rate
        p_grp = (
            df.groupby("PaymentMethod", observed=False)["Churn_Binary"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=True)
            .reset_index()
        )
        p_grp["churn_pct"] = p_grp["mean"] * 100
        bars_p = axes[1, 0].barh(
            p_grp["PaymentMethod"],
            p_grp["churn_pct"],
            color=["#2b8cbe", "#6baed6", "#fd8d3c", "#e34a33"],
            height=0.55,
        )
        axes[1, 0].set_title("Churn Rate by Payment Method", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Churn Rate (%)")
        axes[1, 0].set_xlim(0, 55)
        for bar, (_, row) in zip(bars_p, p_grp.iterrows()):
            axes[1, 0].text(
                bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{row['churn_pct']:.1f}% (n={int(row['count']):,})",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )

        # 4. Tenure Cohort Churn Trajectory
        t_grp = (
            df.groupby("Tenure_Cohort", observed=False)["Churn_Binary"]
            .agg(["mean", "count"])
            .reset_index()
        )
        t_grp["churn_pct"] = t_grp["mean"] * 100
        bars_t = axes[1, 1].bar(
            t_grp["Tenure_Cohort"].astype(str),
            t_grp["churn_pct"],
            color=["#de2d26", "#fb6a4a", "#fdae6b", "#6baed6", "#2b8cbe"],
            width=0.6,
        )
        axes[1, 1].set_title("Churn Rate Across Tenure Lifecycle Cohorts", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Churn Rate (%)")
        axes[1, 1].set_ylim(0, 62)
        axes[1, 1].tick_params(axis="x", rotation=15)
        for bar, (_, row) in zip(bars_t, t_grp.iterrows()):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{row['churn_pct']:.1f}%\n(n={int(row['count']):,})",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.suptitle(
            "Structural & Behavioral Churn Drivers — Contract, Fiber Optic, Payment & Onboarding Risk",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(self.figures_dir / "02_contract_internet_payment_matrix.png")
        plt.close(fig)

    def _plot_addons_and_pricing(self, df: pd.DataFrame) -> None:
        """Plot 3: Value-Added Services Stickiness & Monthly Pricing Distributions."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

        # 1. Churn by Individual Security/Support Service (among Internet customers)
        internet_df = df[df["InternetService"] != "No"].copy()
        svc_cols = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"]
        svc_records = []
        for col in svc_cols:
            for val in ["No", "Yes"]:
                sub = internet_df[internet_df[col] == val]
                svc_records.append(
                    {
                        "Service": col,
                        "Subscribed": f"Subscribed: {val}",
                        "Churn_Pct": float(sub["Churn_Binary"].mean() * 100),
                    }
                )
        svc_df = pd.DataFrame(svc_records)
        sns.barplot(
            data=svc_df,
            x="Service",
            y="Churn_Pct",
            hue="Subscribed",
            palette={"Subscribed: No": "#e34a33", "Subscribed: Yes": "#2b8cbe"},
            ax=axes[0],
        )
        axes[0].set_title(
            "Impact of Security & Support Add-Ons (Internet Users)",
            fontsize=11.5,
            fontweight="bold",
        )
        axes[0].set_ylabel("Churn Rate (%)")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].set_ylim(0, 50)

        # 2. Monthly Charges Distribution by Churn Status
        sns.boxplot(
            data=df,
            x="Churn",
            y="MonthlyCharges",
            hue="Churn",
            palette={"No": "#2b8cbe", "Yes": "#e34a33"},
            width=0.45,
            dodge=False,
            ax=axes[1],
        )
        axes[1].set_title("Monthly Charges Distribution by Churn Status", fontsize=11.5, fontweight="bold")
        axes[1].set_xlabel("Churned")
        axes[1].set_ylabel("Monthly Charges ($ USD)")

        # 3. Protection/Support Bundle Count vs Churn Rate
        prot_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        internet_df["Protection_Bundle_Count"] = sum(
            (internet_df[c] == "Yes").astype(int) for c in prot_cols
        )
        b_grp = (
            internet_df.groupby("Protection_Bundle_Count")["Churn_Binary"]
            .agg(["mean", "count"])
            .reset_index()
        )
        b_grp["churn_pct"] = b_grp["mean"] * 100
        bars_b = axes[2].bar(
            b_grp["Protection_Bundle_Count"].astype(str),
            b_grp["churn_pct"],
            color=["#de2d26", "#fb6a4a", "#fdae6b", "#6baed6", "#2b8cbe"],
            width=0.55,
        )
        axes[2].set_title(
            "Churn Rate vs. Active Protection/Support Count", fontsize=11.5, fontweight="bold"
        )
        axes[2].set_xlabel("Number of Active Protection/Support Add-Ons (0 to 4)")
        axes[2].set_ylabel("Churn Rate (%)")
        axes[2].set_ylim(0, 58)
        for bar, (_, row) in zip(bars_b, b_grp.iterrows()):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{row['churn_pct']:.1f}%\n(n={int(row['count']):,})",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.suptitle(
            "Service Ecosystem Stickiness & Pricing Sensitivity Analysis",
            fontsize=14,
            fontweight="bold",
            y=1.03,
        )
        plt.tight_layout()
        fig.savefig(self.figures_dir / "03_addon_services_and_pricing_dynamics.png")
        plt.close(fig)

    def plot_model_evaluation_and_retention(
        self,
        trained_models: Dict[str, Any],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_raw_df: pd.DataFrame,
        champion_name: str,
        champion_threshold: float,
        feature_importances: Dict[str, float],
        cost_fn: float = 500.0,
        cost_fp: float = 50.0,
    ) -> Dict[str, Any]:
        """Generate Plots 4 & 5: Model ROC/PR/Cost Curves, Feature Importances, and Retention Portfolio."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))

        colors = {
            "naive_baseline": "#969696",
            "logistic_regression": "#2b8cbe",
            "random_forest": "#31a354",
            "gradient_boosting": "#756bb1",
            "xgboost": "#e34a33",
            "voting_ensemble": "#d95f02",
        }

        # 1. Validation ROC Curves across all models
        for name, model in trained_models.items():
            probs = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, probs)
            from sklearn.metrics import roc_auc_score, average_precision_score

            auc_val = roc_auc_score(y_val, probs)
            axes[0, 0].plot(
                fpr,
                tpr,
                label=f"{name} (AUC = {auc_val:.4f})",
                color=colors.get(name, "#333333"),
                linewidth=2.2 if name == champion_name else 1.6,
            )
        axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0, 0].set_title("Validation ROC Curves Across Candidate Models", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate (Recall)")
        axes[0, 0].legend(loc="lower right")

        # 2. Validation Precision-Recall Curves
        for name, model in trained_models.items():
            probs = model.predict_proba(X_val)[:, 1]
            prec, rec, _ = precision_recall_curve(y_val, probs)
            from sklearn.metrics import average_precision_score

            ap_val = average_precision_score(y_val, probs)
            axes[0, 1].plot(
                rec,
                prec,
                label=f"{name} (PR-AUC = {ap_val:.4f})",
                color=colors.get(name, "#333333"),
                linewidth=2.2 if name == champion_name else 1.6,
            )
        axes[0, 1].set_title("Validation Precision-Recall Curves", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].legend(loc="upper right")

        # 3. Business Cost Optimization Curve on Test Set (Champion Model)
        champ_model = trained_models[champion_name]
        test_probs = champ_model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0.05, 0.90, 86)
        costs = []
        default_cost = 0.0
        opt_cost = 0.0
        for t in thresholds:
            preds = (test_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
            c = float(fn * cost_fn + fp * cost_fp)
            costs.append(c)
            if abs(t - 0.50) < 1e-4:
                default_cost = c
            if abs(t - champion_threshold) < 1e-4:
                opt_cost = c

        axes[1, 0].plot(thresholds, costs, color="#e34a33", linewidth=2.5)
        axes[1, 0].axvline(
            champion_threshold,
            color="#31a354",
            linestyle="--",
            linewidth=2,
            label=f"Optimal Thresh ({champion_threshold:.2f}) -> ${opt_cost:,.0f}",
        )
        axes[1, 0].axvline(
            0.50,
            color="#636363",
            linestyle=":",
            linewidth=2,
            label=f"Default Thresh (0.50) -> ${default_cost:,.0f}",
        )
        axes[1, 0].set_title(
            f"Expected Business Cost vs. Threshold (FN={cost_fn:.0f} USD, FP={cost_fp:.0f} USD)",
            fontsize=11.5,
            fontweight="bold",
        )
        axes[1, 0].set_xlabel("Decision Probability Threshold")
        axes[1, 0].set_ylabel("Total Expected Cost ($ USD)")
        axes[1, 0].legend(loc="upper left")

        # 4. Confusion Matrix at Optimal Threshold on Test Set
        test_preds = (test_probs >= champion_threshold).astype(int)
        cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=["Predicted Retained (0)", "Predicted Churn (1)"],
            yticklabels=["Actual Retained (0)", "Actual Churn (1)"],
            ax=axes[1, 1],
        )
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                axes[1, 1].text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:,}",
                    ha="center",
                    va="center",
                    color="white" if val > cm.max() / 2 else "black",
                    fontweight="bold",
                    fontsize=13,
                )
        axes[1, 1].set_title(
            f"Test Set Confusion Matrix ({champion_name} @ threshold={champion_threshold:.2f})",
            fontsize=11.5,
            fontweight="bold",
        )

        plt.suptitle(
            f"MLOps Model Evaluation & Business Cost Optimization (Champion: {champion_name})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(self.figures_dir / "04_model_comparison_and_business_roi.png")
        plt.close(fig)

        # Plot 5: Feature Importance & Retention Risk Tier Portfolio
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

        # Top Feature Importances
        if feature_importances:
            items = list(feature_importances.items())[:12][::-1]
            f_names = [k for k, _ in items]
            f_vals = [v for _, v in items]
            axes2[0].barh(f_names, f_vals, color="#2b8cbe", height=0.65)
            axes2[0].set_title(
                f"Top Predictive Features ({champion_name})", fontsize=12, fontweight="bold"
            )
            axes2[0].set_xlabel("Feature Importance / |Coefficient|")

        # Test Set Customer Retention Risk Tier Portfolio
        tier_df = test_raw_df.copy()
        tier_df["churn_prob"] = test_probs
        tier_df["annual_value"] = tier_df["MonthlyCharges"].astype(float) * 12.0
        tier_df["revenue_at_risk"] = tier_df["annual_value"] * tier_df["churn_prob"]
        tier_df["Risk_Tier"] = pd.cut(
            tier_df["churn_prob"],
            bins=[-0.01, 0.30, 0.60, 0.80, 1.01],
            labels=["Low (<30%)", "Medium (30-60%)", "High (60-80%)", "Critical (>=80%)"],
        )

        portfolio = (
            tier_df.groupby("Risk_Tier", observed=False)
            .agg(
                customers=("churn_prob", "count"),
                actual_churn_rate=("Churn", lambda s: round(float((s == "Yes").mean() * 100), 2)),
                total_annual_revenue=("annual_value", lambda s: round(float(s.sum()), 2)),
                revenue_at_risk=("revenue_at_risk", lambda s: round(float(s.sum()), 2)),
            )
            .reset_index()
        )

        bars_r = axes2[1].bar(
            portfolio["Risk_Tier"].astype(str),
            portfolio["revenue_at_risk"],
            color=["#2b8cbe", "#fdae6b", "#fb6a4a", "#de2d26"],
            width=0.55,
        )
        axes2[1].set_title(
            "Test Cohort Annual Revenue at Risk by Retention Tier",
            fontsize=12,
            fontweight="bold",
        )
        axes2[1].set_ylabel("Expected Annual Revenue at Risk ($ USD)")
        max_rev = max(portfolio["revenue_at_risk"].max(), 1.0)
        axes2[1].set_ylim(0, max_rev * 1.22)
        for bar, (_, row) in zip(bars_r, portfolio.iterrows()):
            axes2[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_rev * 0.02,
                f"${row['revenue_at_risk']:,.0f}\n(n={int(row['customers'])}, actual={row['actual_churn_rate']:.1f}%)",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.suptitle(
            "Explainability & Actionable Customer Retention Portfolio Segmentation",
            fontsize=14,
            fontweight="bold",
            y=1.03,
        )
        plt.tight_layout()
        fig2.savefig(self.figures_dir / "05_feature_importance_and_retention_tiers.png")
        plt.close(fig2)

        return {
            "business_cost_comparison": {
                "cost_at_default_050_threshold": round(default_cost, 2),
                "cost_at_optimal_threshold": round(opt_cost, 2),
                "net_cost_savings_on_test_set": round(default_cost - opt_cost, 2),
                "optimal_threshold": champion_threshold,
            },
            "retention_tier_portfolio": portfolio.to_dict(orient="records"),
        }

