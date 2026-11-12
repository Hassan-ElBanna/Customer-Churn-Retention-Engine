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
5. **Phase 5 — Data & Concept Drift Monitoring (`src/churn_engine/monitoring/drift.py`)**
   - Computes Population Stability Index (PSI) and Two-Sample Kolmogorov-Smirnov (KS) tests to trigger automated retraining alerts.
6. **Phase 6 — FastAPI Real-Time & Batch Serving (`src/churn_engine/api/main.py`)**
   - REST endpoints `/health`, `/predict`, and `/predict/batch` packaged with Docker & Docker Compose alongside an MLflow Tracking Server.

---

## Project Structure

```text
├── .github/workflows/mlops_ci.yml   # CI/CD linting & automated test workflow
├── configs/config.yaml              # Centralized hyperparameters, paths & business cost settings
├── data/
│   ├── raw/                         # Raw ingested dataset
│   └── processed/                   # Stratified train/val/test splits
├── artifacts/
│   ├── models/                      # Fitted preprocessor & champion model bundles
│   └── reports/                     # JSON validation, evaluation & drift reports
├── pipelines/
│   └── training_pipeline.py         # End-to-end MLOps pipeline entrypoint
├── src/churn_engine/
│   ├── api/                         # FastAPI serving app & Pydantic schemas
│   ├── data/                        # DataIngestor & DataValidator
│   ├── features/                    # ChurnFeaturePreprocessor
│   ├── models/                      # ChurnModelTrainer & ModelEvaluator
│   ├── monitoring/                  # DriftDetector (PSI & KS-test)
│   ├── retention/                   # RetentionEngine & playbook generator
│   └── utils/                       # Structured logging
├── tests/
│   └── test_pipeline.py             # End-to-end unit & integration suite
├── Dockerfile                       # Production container image
├── docker-compose.yml               # FastAPI + MLflow stack
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

### 3. Start the Prediction & Retention API
```bash
uvicorn churn_engine.api.main:app --reload --port 8000
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


