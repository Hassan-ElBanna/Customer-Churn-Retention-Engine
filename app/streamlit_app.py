"""Streamlit Frontend — Customer Churn & Retention Command Center."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure `src` is importable when launched from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_engine.features.preprocessor import ChurnFeaturePreprocessor
from churn_engine.retention.engine import RetentionEngine

st.set_page_config(
    page_title="Customer Churn & Retention Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for executive cards and risk badges
st.markdown(
    """
    <style>
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 1.15rem 1.3rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        margin-bottom: 0.8rem;
    }
    .kpi-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-top: 0.2rem;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .tier-Critical { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .tier-High { background-color: #ffedd5; color: #9a3412; border: 1px solid #fb923c; }
    .tier-Medium { background-color: #fef9c3; color: #854d0e; border: 1px solid #facc15; }
    .tier-Low { background-color: #dcfce7; color: #166534; border: 1px solid #4ade80; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_mlops_artifacts() -> Dict[str, Any]:
    """Load fitted preprocessor, champion model bundle, and JSON reports."""
    prep_path = ROOT_DIR / "artifacts/models/preprocessor.joblib"
    model_path = ROOT_DIR / "artifacts/models/champion_model.joblib"
    eval_path = ROOT_DIR / "artifacts/reports/evaluation_metrics.json"
    eda_path = ROOT_DIR / "artifacts/reports/eda_insights.json"
    drift_path = ROOT_DIR / "artifacts/reports/drift_report.json"
    raw_data_path = ROOT_DIR / "data/raw/customer_churn.csv"

    preprocessor = ChurnFeaturePreprocessor.load(prep_path)
    model_bundle = joblib.load(model_path)
    eval_metrics = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    eda_insights = json.loads(eda_path.read_text(encoding="utf-8")) if eda_path.exists() else {}
    drift_report = json.loads(drift_path.read_text(encoding="utf-8")) if drift_path.exists() else {}
    raw_df = pd.read_csv(raw_data_path) if raw_data_path.exists() else pd.DataFrame()

    return {
        "preprocessor": preprocessor,
        "model_bundle": model_bundle,
        "eval_metrics": eval_metrics,
        "eda_insights": eda_insights,
        "drift_report": drift_report,
        "raw_df": raw_df,
        "retention_engine": RetentionEngine(),
    }


def score_customer(
    features: Dict[str, Any],
    customer_id: str,
    artifacts: Dict[str, Any],
    threshold_override: float | None = None,
) -> Dict[str, Any]:
    """Run preprocessing, model probability prediction, and retention plan generation."""
    preprocessor: ChurnFeaturePreprocessor = artifacts["preprocessor"]
    bundle = artifacts["model_bundle"]
    retention_engine: RetentionEngine = artifacts["retention_engine"]

    df_in = pd.DataFrame([features])
    X_trans = preprocessor.transform(df_in)
    prob = float(bundle["model"].predict_proba(X_trans)[:, 1][0])
    thresh = threshold_override if threshold_override is not None else float(bundle["threshold"])

    plan = retention_engine.generate_plan(
        customer_id=customer_id,
        churn_prob=prob,
        customer_features=features,
        decision_threshold=thresh,
    )
    return plan.to_dict()


def main() -> None:
    artifacts = load_mlops_artifacts()
    bundle = artifacts["model_bundle"]
    eval_metrics = artifacts["eval_metrics"]
    eda_insights = artifacts["eda_insights"]
    drift_report = artifacts["drift_report"]
    raw_df: pd.DataFrame = artifacts["raw_df"]

    # Sidebar Controls & System Status
    with st.sidebar:
        st.title("🛡️ Churn & Retention MLOps")
        st.caption("Production Decision Support System")
        st.divider()

        champ_name = bundle.get("model_name", "gradient_boosting")
        default_thresh = float(bundle.get("threshold", 0.46))

        st.markdown(f"**Active Champion Model:** `{champ_name}`")
        test_m = eval_metrics.get("champion_test_metrics", {})
        best_lr = eval_metrics.get("test_leaderboard", {}).get("logistic_regression", {})
        st.metric(
            "Peak Test Accuracy (L2 LR)",
            f"{best_lr.get('accuracy', 0.8127) * 100:.2f}%",
            delta=f"ROC-AUC {best_lr.get('roc_auc', 0.8578):.4f}",
        )
        st.metric(
            f"Active Model ({champ_name})",
            f"{test_m.get('accuracy', 0.8079) * 100:.2f}%",
            delta=f"ROC-AUC {test_m.get('roc_auc', 0.8530):.4f}",
        )

        st.divider()
        active_threshold = st.slider(
            "Decision Probability Threshold",
            min_value=0.15,
            max_value=0.85,
            value=default_thresh,
            step=0.01,
            help="Customers with predicted churn probability >= this threshold are flagged for retention intervention.",
        )

        st.divider()
        st.markdown("### 🔗 MLOps Links")
        st.markdown(
            "- [GitHub Repository](https://github.com/Hassan-ElBanna/Customer-Churn-Retention-Engine)\n"
            "- [DagsHub Repository & DVC](https://dagshub.com/hmelbanna100/Customer-Churn-Retention-Engine)\n"
            "- [DagsHub MLflow Dashboard](https://dagshub.com/hmelbanna100/Customer-Churn-Retention-Engine.mlflow/#/experiments/0)"
        )

    st.title("📡 Telco Customer Churn & Retention Command Center")
    st.markdown(
        "Real-time churn probability scoring, **What-If intervention simulation**, batch portfolio triage, and MLOps observability."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 Single Customer Simulator & Playbook",
            "📂 Batch Portfolio Triage",
            "📊 Executive EDA & Visual Insights",
            "⚙️ MLOps Leaderboard & Drift Monitor",
        ]
    )

    # =========================================================================
    # TAB 1: Single Customer Simulator & What-If Retention Engine
    # =========================================================================
    with tab1:
        presets = {
            "🔴 High-Risk Onboarding Fiber Subscriber (CUST-RISK-01)": {
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 99.65,
                "TotalCharges": 199.30,
            },
            "🟠 Medium-Risk Growing Subscriber (CUST-MED-02)": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 14,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 84.50,
                "TotalCharges": 1183.00,
            },
            "🟢 Loyal Protected 2-Year Subscriber (CUST-LOYAL-03)": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 58,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 89.10,
                "TotalCharges": 5167.80,
            },
        }

        selected_preset = st.selectbox(
            "⚡ Load Customer Persona Preset (or customize any field below):",
            options=list(presets.keys()),
        )
        p_vals = presets[selected_preset]

        with st.container(border=True):
            st.subheader("1️⃣ Customer Profile & Subscription Attributes")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                customer_id = st.text_input(
                    "Customer ID", value=selected_preset.split("(")[-1].rstrip(")")
                )
                tenure = st.slider("Tenure (Months)", 0, 72, int(p_vals["tenure"]))
                monthly_charges = st.number_input(
                    "Monthly Charges ($ USD)",
                    min_value=18.0,
                    max_value=130.0,
                    value=float(p_vals["MonthlyCharges"]),
                    step=1.0,
                )
                total_charges = st.number_input(
                    "Total Historical Charges ($ USD)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(round(tenure * monthly_charges, 2)),
                    step=10.0,
                )

            with c2:
                contract = st.selectbox(
                    "Contract Type",
                    ["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(p_vals["Contract"]),
                )
                internet_service = st.selectbox(
                    "Internet Service Tier",
                    ["Fiber optic", "DSL", "No"],
                    index=["Fiber optic", "DSL", "No"].index(p_vals["InternetService"]),
                )
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ].index(p_vals["PaymentMethod"]),
                )
                paperless = st.selectbox(
                    "Paperless Billing",
                    ["Yes", "No"],
                    index=["Yes", "No"].index(p_vals["PaperlessBilling"]),
                )

            with c3:
                online_security = st.selectbox(
                    "Online Security",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(p_vals["OnlineSecurity"]),
                )
                tech_support = st.selectbox(
                    "Tech Support",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(p_vals["TechSupport"]),
                )
                online_backup = st.selectbox(
                    "Online Backup",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(p_vals["OnlineBackup"]),
                )
                device_protection = st.selectbox(
                    "Device Protection",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(p_vals["DeviceProtection"]),
                )

            with c4:
                streaming_tv = st.selectbox(
                    "Streaming TV",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(p_vals["StreamingTV"]),
                )
                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    ["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(p_vals["StreamingMovies"]),
                )
                senior = st.selectbox(
                    "Senior Citizen",
                    [0, 1],
                    index=[0, 1].index(int(p_vals["SeniorCitizen"])),
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                )
                partner = st.selectbox(
                    "Has Partner",
                    ["No", "Yes"],
                    index=["No", "Yes"].index(p_vals["Partner"]),
                )

        customer_features = {
            "gender": p_vals["gender"],
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": p_vals["Dependents"],
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": p_vals["MultipleLines"],
            "InternetService": internet_service,
            "OnlineSecurity": "No internet service" if internet_service == "No" else online_security,
            "OnlineBackup": "No internet service" if internet_service == "No" else online_backup,
            "DeviceProtection": (
                "No internet service" if internet_service == "No" else device_protection
            ),
            "TechSupport": "No internet service" if internet_service == "No" else tech_support,
            "StreamingTV": "No internet service" if internet_service == "No" else streaming_tv,
            "StreamingMovies": (
                "No internet service" if internet_service == "No" else streaming_movies
            ),
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        plan = score_customer(
            customer_features,
            customer_id=customer_id,
            artifacts=artifacts,
            threshold_override=active_threshold,
        )

        # KPI Cards Row
        k1, k2, k3, k4 = st.columns(4)
        prob_pct = plan["churn_probability"] * 100
        tier = plan["risk_tier"]

        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Churn Probability</div>
                    <div class="kpi-value">{prob_pct:.1f}%</div>
                    <div class="kpi-sub">Decision Cutoff: {active_threshold*100:.0f}% ({'CHURN ALERT' if plan['will_churn'] else 'RETAINED'})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Retention Risk Tier</div>
                    <div class="kpi-value"><span class="tier-badge tier-{tier}">{tier} Risk</span></div>
                    <div class="kpi-sub">{plan['channel_priority']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Annual Revenue at Risk</div>
                    <div class="kpi-value">${plan['revenue_at_risk']:,.2f}</div>
                    <div class="kpi-sub">Total Customer ARR: ${plan['estimated_annual_value']:,.2f}/yr</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Max ROI-Positive Incentive</div>
                    <div class="kpi-value">${plan['recommended_max_incentive']:,.2f}</div>
                    <div class="kpi-sub">Capped at 25% of expected ARR at risk</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(min(max(float(plan["churn_probability"]), 0.0), 1.0))

        col_left, col_right = st.columns([1.05, 1.0])
        with col_left:
            with st.container(border=True):
                st.subheader("2️⃣ Primary Churn Risk Drivers & Prescribed Playbook")
                st.markdown("**Identified Risk Triggers:**")
                for drv in plan["primary_risk_drivers"]:
                    st.markdown(f"- ⚠️ {drv}")

                st.markdown("**Recommended Retention Actions:**")
                for act in plan["recommended_actions"]:
                    st.markdown(f"- ✅ {act}")

        with col_right:
            with st.container(border=True):
                st.subheader("3️⃣ Interactive What-If Retention Simulator")
                st.caption(
                    "Simulate how much this customer's churn risk drops if they accept a retention offer:"
                )

                sim_contract = st.checkbox(
                    "Upgrade Contract to **One Year**", value=(contract == "Month-to-month")
                )
                sim_security_support = st.checkbox(
                    "Bundle Complimentary **Online Security + Tech Support**",
                    value=(online_security == "No" or tech_support == "No"),
                )
                sim_autopay = st.checkbox(
                    "Switch Payment Method to **Credit Card (Automatic)**",
                    value=("automatic" not in payment_method.lower()),
                )
                sim_discount = st.slider(
                    "Apply Monthly Loyalty Discount ($/mo)", 0.0, 25.0, 10.0, 2.5
                )

                sim_features = dict(customer_features)
                if sim_contract and sim_features["Contract"] == "Month-to-month":
                    sim_features["Contract"] = "One year"
                if sim_security_support and sim_features["InternetService"] != "No":
                    sim_features["OnlineSecurity"] = "Yes"
                    sim_features["TechSupport"] = "Yes"
                if sim_autopay:
                    sim_features["PaymentMethod"] = "Credit card (automatic)"
                sim_features["MonthlyCharges"] = max(
                    18.0, float(sim_features["MonthlyCharges"]) - sim_discount
                )

                sim_plan = score_customer(
                    sim_features,
                    customer_id=customer_id,
                    artifacts=artifacts,
                    threshold_override=active_threshold,
                )

                new_prob_pct = sim_plan["churn_probability"] * 100
                delta_pct = new_prob_pct - prob_pct
                saved_arr = plan["revenue_at_risk"] - sim_plan["revenue_at_risk"]

                sc1, sc2 = st.columns(2)
                sc1.metric(
                    "Post-Intervention Churn Risk",
                    f"{new_prob_pct:.1f}%",
                    delta=f"{delta_pct:+.1f}% risk change",
                    delta_color="inverse",
                )
                sc2.metric(
                    "Protected Annual Revenue",
                    f"${max(saved_arr, 0.0):,.2f}/yr",
                    delta=f"New Tier: {sim_plan['risk_tier']}",
                )

    # =========================================================================
    # TAB 2: Batch Portfolio Triage & CSV Scoring
    # =========================================================================
    with tab2:
        st.subheader("📂 Batch Customer Portfolio Triage & Export")
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded_file = st.file_uploader(
                "Upload a customer CSV file to score (or use the built-in Telco cohort below):",
                type=["csv"],
            )
        with col_u2:
            sample_size = st.number_input(
                "Sample size (when using built-in dataset)",
                min_value=50,
                max_value=2000,
                value=250,
                step=50,
            )

        batch_df = pd.read_csv(uploaded_file) if uploaded_file is not None else raw_df.head(int(sample_size)).copy()

        if not batch_df.empty:
            X_batch = artifacts["preprocessor"].transform(batch_df)
            probs = bundle["model"].predict_proba(X_batch)[:, 1]

            scored_df = batch_df.copy()
            scored_df["Churn_Probability"] = np.round(probs, 4)
            scored_df["Predicted_Churn"] = np.where(probs >= active_threshold, "Yes", "No")
            scored_df["Risk_Tier"] = pd.cut(
                probs,
                bins=[-0.01, 0.30, 0.60, 0.80, 1.01],
                labels=["Low", "Medium", "High", "Critical"],
            ).astype(str)
            scored_df["Annual_Value_USD"] = np.round(
                scored_df.get("MonthlyCharges", pd.Series(65.0, index=scored_df.index)).astype(float)
                * 12.0,
                2,
            )
            scored_df["Revenue_At_Risk_USD"] = np.round(
                scored_df["Annual_Value_USD"] * scored_df["Churn_Probability"], 2
            )
            scored_df["Max_Incentive_USD"] = np.round(scored_df["Revenue_At_Risk_USD"] * 0.25, 2)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Customers Scored", f"{len(scored_df):,}")
            b2.metric(
                "Flagged for Intervention",
                f"{int((scored_df['Predicted_Churn'] == 'Yes').sum()):,}",
            )
            b3.metric(
                "Critical + High Risk Accounts",
                f"{int(scored_df['Risk_Tier'].isin(['Critical', 'High']).sum()):,}",
            )
            b4.metric(
                "Total Portfolio ARR at Risk",
                f"${scored_df['Revenue_At_Risk_USD'].sum():,.2f}",
            )

            tier_filter = st.multiselect(
                "Filter Triage Queue by Risk Tier:",
                options=["Critical", "High", "Medium", "Low"],
                default=["Critical", "High"],
            )

            display_cols = [
                c
                for c in [
                    "customerID",
                    "Risk_Tier",
                    "Churn_Probability",
                    "Predicted_Churn",
                    "Revenue_At_Risk_USD",
                    "Max_Incentive_USD",
                    "Contract",
                    "InternetService",
                    "tenure",
                    "MonthlyCharges",
                    "PaymentMethod",
                ]
                if c in scored_df.columns
            ]

            filtered_df = (
                scored_df[scored_df["Risk_Tier"].isin(tier_filter)]
                .sort_values("Revenue_At_Risk_USD", ascending=False)[display_cols]
            )
            st.dataframe(filtered_df, use_container_width=True, height=380)

            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Prioritized Retention Triage CSV",
                data=csv_bytes,
                file_name="prioritized_retention_triage.csv",
                mime="text/csv",
            )

    # =========================================================================
    # TAB 3: Executive EDA & Visual Insights
    # =========================================================================
    with tab3:
        st.subheader("📊 Telco Customer Churn — Executive EDA & Visual Story")
        ov = eda_insights.get("dataset_overview", {})
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Total Subscribers", f"{ov.get('total_rows', 7043):,}")
        e2.metric("Overall Logo Churn Rate", f"{ov.get('overall_churn_rate_pct', 26.54):.2f}%")
        e3.metric(
            "Monthly Revenue Lost (MRR)",
            f"${ov.get('churned_monthly_recurring_revenue', 139130.85):,.0f}/mo",
            delta=f"{ov.get('mrr_lost_pct', 30.50):.1f}% of total MRR",
            delta_color="inverse",
        )
        e4.metric(
            "Annualized Revenue Lost (ARR)",
            f"${ov.get('annualized_revenue_lost', 1669570.20):,.0f}/yr",
        )

        fig_dir = ROOT_DIR / "artifacts/reports/figures"
        for fig_file, caption in [
            (
                "01_target_and_revenue_exposure.png",
                "Figure 1: Customer Base Split, MRR Exposure ($139.1K/mo lost), and Tenure Survival Distribution (10m vs. 38m median).",
            ),
            (
                "02_contract_internet_payment_matrix.png",
                "Figure 2: Structural Churn Drivers — Month-to-Month (42.7%), Fiber Optic x Month-to-Month (54.6%), Electronic Check (45.3%), and 0-6m Onboarding (52.9%).",
            ),
            (
                "03_addon_services_and_pricing_dynamics.png",
                "Figure 3: Service Ecosystem Stickiness — Customers with 0 protection/support add-ons churn at 56.7% vs. 5.3% with all 4 add-ons.",
            ),
        ]:
            fp = fig_dir / fig_file
            if fp.exists():
                st.image(str(fp), caption=caption, use_container_width=True)
                st.divider()

    # =========================================================================
    # TAB 4: MLOps Model Leaderboard, Feature Importance & Drift Monitor
    # =========================================================================
    with tab4:
        st.subheader("⚙️ Held-Out Test Set Leaderboard (`n = 1,057`)")
        test_lb = eval_metrics.get("test_leaderboard", {})
        if test_lb:
            rows = []
            for m_name, m_vals in test_lb.items():
                rows.append(
                    {
                        "Model": m_name,
                        "Test Accuracy (%)": round(m_vals["accuracy"] * 100, 2),
                        "Test ROC-AUC": m_vals["roc_auc"],
                        "Test PR-AUC": m_vals["pr_auc"],
                        "Test F1-Score": m_vals["f1_score"],
                        "Precision (%)": round(m_vals["precision"] * 100, 2),
                        "Recall (%)": round(m_vals["recall"] * 100, 2),
                        "Brier Score": m_vals["brier_score"],
                        "Optimal Threshold": m_vals["optimal_threshold"],
                    }
                )
            lb_df = pd.DataFrame(rows).sort_values("Test Accuracy (%)", ascending=False)
            st.dataframe(lb_df, use_container_width=True, hide_index=True)

        fig_dir = ROOT_DIR / "artifacts/reports/figures"
        f4 = fig_dir / "04_model_comparison_and_business_roi.png"
        f5 = fig_dir / "05_feature_importance_and_retention_tiers.png"
        if f4.exists():
            st.image(
                str(f4),
                caption="Figure 4: Candidate ROC & PR Curves, Expected Business Cost Optimization, and Confusion Matrix.",
                use_container_width=True,
            )
        if f5.exists():
            st.image(
                str(f5),
                caption="Figure 5: Top Predictive Engineered Features & Test Cohort Revenue at Risk by Retention Tier.",
                use_container_width=True,
            )

        st.divider()
        st.subheader("🔍 Data & Concept Drift Monitoring Status (PSI & Two-Sample KS Test)")
        d1, d2, d3 = st.columns(3)
        d1.metric("Features Monitored", f"{drift_report.get('num_features_checked', 59)}")
        d2.metric("Drifted Features Detected", f"{drift_report.get('num_drifted_features', 0)}")
        d3.metric(
            "Retraining Trigger Status",
            "🟢 Healthy (No Drift)"
            if not drift_report.get("retrain_recommended", False)
            else "🔴 Retrain Recommended",
        )


if __name__ == "__main__":
    main()
