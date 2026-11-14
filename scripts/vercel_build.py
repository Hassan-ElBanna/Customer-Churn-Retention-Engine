"""Build hook for Vercel serverless deployment.

Ensures runtime-compatible preprocessor and champion model artifacts exist
under ``artifacts/models/`` before Vercel packages the FastAPI function.
Uses only lightweight stdlib and core runtime dependencies.
"""

from __future__ import annotations

import base64
import configparser
import io
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_engine.config import load_config  # noqa: E402
from churn_engine.features.preprocessor import ChurnFeaturePreprocessor  # noqa: E402
from churn_engine.models.evaluator import ModelEvaluator  # noqa: E402
from churn_engine.utils.logger import get_logger  # noqa: E402

logger = get_logger("vercel_build")

MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "champion_model.joblib"
PREP_PATH = ROOT_DIR / "artifacts" / "models" / "preprocessor.joblib"
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
DVC_LOCK_PATH = ROOT_DIR / "dvc.lock"
DVC_CONFIG_PATH = ROOT_DIR / ".dvc" / "config"
EVAL_METRICS_PATH = ROOT_DIR / "artifacts" / "reports" / "evaluation_metrics.json"

FALLBACK_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

SMOKE_SAMPLE = {
    "customerID": "VERCEL-BUILD-SMOKE-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0,
    "TotalCharges": 900.0,
}


def _resolve_dataset_url(url: str) -> str:
    """Resolve known catalog links to direct raw CSV URLs."""
    cleaned = url.strip()
    if "kaggle.com/datasets/blastchar/telco-customer-churn" in cleaned.lower():
        return FALLBACK_DATASET_URL
    if "github.com" in cleaned and "/blob/" in cleaned:
        return cleaned.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return cleaned


def _verify_artifacts(model_path: Path, prep_path: Path) -> bool:
    """Verify both serialized artifacts exist, unpickle cleanly, and run inference."""
    if not model_path.exists() or not prep_path.exists():
        return False
    if model_path.stat().st_size == 0 or prep_path.stat().st_size == 0:
        return False

    try:
        bundle = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        transformed = preprocessor.transform(pd.DataFrame([SMOKE_SAMPLE]))
        proba = model.predict_proba(transformed)[:, 1]
        if proba.shape != (1,) or not (0.0 <= float(proba[0]) <= 1.0):
            raise ValueError(f"Unexpected smoke probability output: {proba}")
        logger.info(
            f"Verified compatible model artifacts ({model_path.name}, smoke prob={float(proba[0]):.4f})."
        )
        return True
    except Exception as exc:
        logger.warning(f"Existing model artifacts failed runtime verification: {exc}")
        return False


def _read_dvc_remote() -> tuple[str, str]:
    """Read DVC remote URL and optional username from .dvc/config."""
    import os

    remote_url = "https://dagshub.com/hmelbanna100/Customer-Churn-Retention-Engine.dvc"
    username = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_REPO_OWNER") or ""

    if DVC_CONFIG_PATH.exists():
        parser = configparser.ConfigParser()
        parser.read(DVC_CONFIG_PATH, encoding="utf-8")
        for section in parser.sections():
            if section.startswith("'remote ") or section.startswith("remote "):
                remote_url = parser.get(section, "url", fallback=remote_url).strip()
                if not username:
                    username = parser.get(section, "user", fallback="").strip()
                break

    return remote_url.rstrip("/"), username


def _try_download_from_dvc(target_rel_paths: list[str]) -> bool:
    """Attempt fetching DVC-tracked artifacts from the configured HTTP remote via MD5 hashes."""
    import os

    if not DVC_LOCK_PATH.exists():
        return False

    try:
        lock_data = yaml.safe_load(DVC_LOCK_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Unable to parse dvc.lock: {exc}")
        return False

    md5_by_path: dict[str, str] = {}
    for stage in (lock_data.get("stages") or {}).values():
        for out in stage.get("outs") or []:
            rel_path = out.get("path")
            md5 = out.get("md5")
            if rel_path in target_rel_paths and md5:
                md5_by_path[rel_path] = md5

    if len(md5_by_path) < len(target_rel_paths):
        return False

    remote_url, username = _read_dvc_remote()
    token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD") or ""

    headers = {"User-Agent": "churn-retention-engine-build/1.0"}
    if username and token:
        encoded = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"

    downloaded_all = True
    for rel_path, md5 in md5_by_path.items():
        cache_url = f"{remote_url}/files/md5/{md5[:2]}/{md5[2:]}"
        dest = ROOT_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(cache_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = resp.read()
            if payload.lstrip().startswith(b"<!DOCTYPE html"):
                raise ValueError("Remote returned HTML instead of binary artifact")
            dest.write_bytes(payload)
            logger.info(f"Fetched DVC artifact '{rel_path}' ({len(payload):,} bytes).")
        except (urllib.error.URLError, ValueError) as exc:
            logger.warning(f"Skipping DVC HTTP fetch for '{rel_path}': {exc}")
            downloaded_all = False

    return downloaded_all


def _load_training_dataframe(config: dict) -> pd.DataFrame:
    """Load Telco dataset from local raw path or public upstream CSV mirrors."""
    raw_rel = config.get("data", {}).get("raw_data_path", "data/raw/customer_churn.csv")
    local_raw = ROOT_DIR / raw_rel
    if local_raw.exists() and local_raw.stat().st_size > 0:
        logger.info(f"Loading local training dataset from {local_raw}")
        return pd.read_csv(local_raw)

    configured_url = _resolve_dataset_url(config.get("data", {}).get("dataset_url", ""))
    candidate_urls = []
    for url in (configured_url, FALLBACK_DATASET_URL):
        if url and url not in candidate_urls:
            candidate_urls.append(url)

    last_error: Exception | None = None
    for url in candidate_urls:
        try:
            logger.info(f"Downloading public Telco dataset from {url}")
            req = urllib.request.Request(
                url, headers={"User-Agent": "churn-retention-engine-build/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return pd.read_csv(io.BytesIO(resp.read()))
        except Exception as exc:
            last_error = exc
            logger.warning(f"Dataset fetch failed for {url}: {exc}")

    raise RuntimeError(f"Could not obtain dataset to build Vercel model artifacts: {last_error}")


def _build_compatible_artifacts(config: dict) -> None:
    """Train the tuned champion GradientBoostingClassifier using the active runtime."""
    data_cfg = config.get("data", {})
    feat_cfg = config.get("features", {})
    train_cfg = config.get("training", {})
    random_state = int(config.get("project", {}).get("random_state", 42))

    df = _load_training_dataframe(config)
    target_col = data_cfg.get("target_column", "Churn")

    preprocessor = ChurnFeaturePreprocessor(
        target_column=target_col,
        max_categorical_cardinality=int(feat_cfg.get("max_categorical_cardinality", 50)),
        random_state=random_state,
    )
    train_df, val_df, test_df = preprocessor.split_data(
        df,
        test_size=float(data_cfg.get("test_size", 0.15)),
        val_size=float(data_cfg.get("val_size", 0.15)),
    )

    X_train = preprocessor.fit_transform(train_df)
    y_train = preprocessor.encode_target(train_df[target_col]).values
    X_val = preprocessor.transform(val_df)
    y_val = preprocessor.encode_target(val_df[target_col]).values
    X_test = preprocessor.transform(test_df)
    y_test = preprocessor.encode_target(test_df[target_col]).values

    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=15,
        min_samples_leaf=10,
        subsample=0.85,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator(
        cost_fn=float(train_cfg.get("business_cost_fn", 300.0)),
        cost_fp=float(train_cfg.get("business_cost_fp", 75.0)),
        optimize_for=str(train_cfg.get("primary_metric", "accuracy")),
    )
    val_probs = model.predict_proba(X_val)[:, 1]
    opt_threshold, _ = evaluator.find_optimal_threshold(y_val, val_probs)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluator.evaluate(y_test, test_probs, threshold=opt_threshold)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(PREP_PATH)
    joblib.dump(
        {
            "model_name": "gradient_boosting",
            "model": model,
            "threshold": float(opt_threshold),
        },
        MODEL_PATH,
    )

    if not EVAL_METRICS_PATH.exists():
        EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_METRICS_PATH.write_text(
            json.dumps(
                {
                    "champion_model": "gradient_boosting",
                    "optimal_threshold": float(opt_threshold),
                    "models": {"gradient_boosting": test_metrics},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    logger.info(
        f"Built runtime-compatible artifacts (ROC-AUC={test_metrics['roc_auc']:.4f}, "
        f"threshold={opt_threshold:.3f})."
    )


def main() -> int:
    """Ensure compatible model and preprocessor artifacts are ready for Vercel bundling."""
    if _verify_artifacts(MODEL_PATH, PREP_PATH):
        return 0

    _try_download_from_dvc(
        [
            "artifacts/models/champion_model.joblib",
            "artifacts/models/preprocessor.joblib",
        ]
    )
    if _verify_artifacts(MODEL_PATH, PREP_PATH):
        return 0

    logger.info("Building runtime-compatible model artifacts for current scikit-learn version...")
    config = load_config(CONFIG_PATH)
    _build_compatible_artifacts(config)

    if not _verify_artifacts(MODEL_PATH, PREP_PATH):
        raise RuntimeError("Vercel build failed: generated model artifacts failed verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
