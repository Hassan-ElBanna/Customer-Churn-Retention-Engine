# Customer Churn & Retention Engine — End-to-End MLOps

A production-ready Machine Learning Operations (MLOps) system that predicts customer churn probability, identifies key risk drivers, optimizes decision thresholds for business ROI, and prescribes automated retention playbooks.

---

## Architecture & MLOps Phases

1. **Phase 1 — Data Ingestion & Validation (`src/churn_engine/data/`)**
   - Automated dataset ingestion from URLs (direct CSV/Parquet/XLSX, Google Drive, GitHub, or ZIP archives).
   - Automated schema, target distribution, duplicate, missingness, and zero-variance checks (`artifacts/reports/data_validation_report.json`).
2. **Phase 2 — Leakage-Free Feature Engineering (`src/churn_engine/features/preprocessor.py`)**
   - Strict stratified `Train / Validation / Test` split **before** fitting any imputers, encoders, or scalers.
   - Domain-engineered behavioral and financial ratios (`avg_monthly_spend_historical`, `monthly_to_historical_spend_ratio`, `balance_to_salary_ratio`, `tenure_to_age_ratio`).
3. **Phase 3 — Multi-Model Training, Evaluation & Champion Selection (`src/churn_engine/models/`)**
   - Compares Naive Baseline, Logistic Regression, Random Forest, and XGBoost.
   - Optimizes classification probability thresholds against business false-negative / false-positive costs.
   - Computes 95% bootstrapped ROC-AUC confidence intervals, PR-AUC, F1, and Brier calibration scores.
4. **Phase 4 — Retention & Next-Best-Action Engine (`src/churn_engine/retention/engine.py`)**
   - Segments customers into `Low`, `Medium`, `High`, and `Critical` risk tiers.
   - Estimates Customer Annual Recurring Value (ARR), calculates Revenue at Risk, caps maximum incentive budget, and recommends channel-specific interventions.
5. **Phase 5 — Evidently AI Monitoring, Drift Detection & Automated Actions (`src/churn_engine/monitoring/`)**
   - Runs **Evidently AI** (`evidently_monitor.py`) alongside custom PSI & Two-Sample KS checks (`drift.py`) to monitor feature drift, prediction drift, and classification SLAs (`accuracy > 0.75`, `roc_auc > 0.78`).
   - Exports interactive HTML reports (`artifacts/reports/evidently_drift_report.html`) and persistent action logs (`artifacts/reports/monitoring_actions.json`).
   - Triages alert severity (`OK`, `WARNING`, `CRITICAL`), optionally notifies `MONITORING_WEBHOOK_URL`, and automatically triggers model retraining (`scripts/monitor_and_act.py --auto-retrain`) when drift or degradation breaches thresholds.
6. **Phase 6 — FastAPI Serving, Prometheus Metrics & Grafana Stack (`src/churn_engine/api/main.py`, `monitoring/`)**
   - REST endpoints `/health`, `/predict`, `/predict/batch`, `/metrics` (Prometheus telemetry), `/monitoring/status`, and `/monitoring/check`.
   - Packaged with Docker Compose alongside **MLflow**, **Streamlit UI**, **Prometheus**, and an auto-provisioned **Grafana** observability dashboard (`monitoring/grafana/dashboards/churn_monitoring_dashboard.json`).

---

## Project Structure

```text
├── .github/workflows/mlops_ci.yml   # CI/CD linting & automated test workflow
├── app/
│   └── streamlit_app.py             # Interactive Streamlit Command Center & Evidently audit UI
├── configs/config.yaml              # Centralized hyperparameters, paths, monitoring & cost settings
├── data/
│   ├── raw/                         # Raw ingested dataset
│   └── processed/                   # Stratified train/val/test splits
├── artifacts/
│   ├── models/                      # Fitted preprocessor & champion model bundles
│   └── reports/                     # JSON validation, evaluation, Evidently HTML & action reports
├── monitoring/
│   ├── prometheus/prometheus.yml    # Prometheus scrape config targeting churn-api:8000/metrics
│   └── grafana/                     # Auto-provisioned datasource & Churn MLOps Grafana dashboard
├── pipelines/
│   └── training_pipeline.py         # End-to-end MLOps pipeline entrypoint
├── scripts/
│   ├── link_remotes.py              # GitHub, DVC & DagsHub MLflow linking helper
│   └── monitor_and_act.py           # CLI batch drift audit & automated retraining runner
├── src/churn_engine/
│   ├── api/                         # FastAPI serving app, /metrics & Pydantic schemas
│   ├── data/                        # DataIngestor, DataValidator & ChurnEDAAnalyzer
│   ├── features/                    # ChurnFeaturePreprocessor
│   ├── models/                      # ChurnModelTrainer & ModelEvaluator
│   ├── monitoring/                  # EvidentlyMonitor (reports/actions) & DriftDetector (PSI/KS)
│   ├── retention/                   # RetentionEngine & playbook generator
│   └── utils/                       # Structured logging & DagsHub/MLflow tracking
├── tests/
│   └── test_pipeline.py             # End-to-end unit & integration suite
├── Dockerfile                       # Production container image
├── docker-compose.yml               # FastAPI + Streamlit + MLflow + Prometheus + Grafana stack
├── dvc.yaml                         # DVC pipeline stages
├── pyproject.toml                   # Package metadata & tool configs
└── requirements.txt                 # Pinned dependencies
```

---

## Quickstart

### 1. Run Pipeline with Dataset URL
```bash
python -m pipelines.training_pipeline --url "<YOUR_DATASET_DOWNLOAD_LINK>"
```

### 2. Run Tests
```bash
pytest
```

### 3. Start the Interactive Streamlit Frontend or FastAPI Service
```bash
# Launch the Streamlit Customer Churn & Retention Command Center (http://localhost:8501)
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

# Or start the FastAPI REST API (http://localhost:8000/docs)
.\.venv\Scripts\python.exe -m uvicorn churn_engine.api.main:app --reload --port 8000
```

### 4. Link GitHub, DVC & MLflow with DagsHub

You can connect the project to your **GitHub** repository and **DagsHub** (which hosts both your **DVC remote storage** and your **MLflow tracking server**) in a single command using [link_remotes.py](file:///d:/Hassan/Projects/test%20projects/MLOps%20project/scripts/link_remotes.py):

```bash
.\.venv\Scripts\python.exe scripts/link_remotes.py \
  --github-url "https://github.com/<YOUR_GITHUB_USER>/<YOUR_REPO>.git" \
  --dagshub-owner "<YOUR_DAGSHUB_USER>" \
  --dagshub-repo "<YOUR_REPO>" \
  --dagshub-token "<YOUR_DAGSHUB_TOKEN>" \
  --push
```

Or copy [.env.example](file:///d:/Hassan/Projects/test%20projects/MLOps%20project/.env.example) to `.env`, fill in your credentials, and run:
```bash
.\.venv\Scripts\python.exe scripts/link_remotes.py --push
.\.venv\Scripts\python.exe -m pipelines.training_pipeline
```

### 5. Run Monitoring Audits, Automated Retraining & Grafana Stack

```bash
# Audit a production batch with Evidently AI and auto-retrain on CRITICAL drift/degradation
.\.venv\Scripts\python.exe scripts/monitor_and_act.py \
  --reference data/raw/customer_churn.csv \
  --current "<path_to_new_batch.csv>" \
  --auto-retrain

# Launch full containerized stack:
# FastAPI (:8000/docs & :8000/metrics), Streamlit (:8501), MLflow (:5000), Prometheus (:9090), Grafana (:3000, admin/admin)
docker compose up --build
```


