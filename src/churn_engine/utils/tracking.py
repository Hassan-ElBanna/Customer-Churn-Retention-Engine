"""MLflow & DagsHub Experiment Tracking Integration."""

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)

# Load environment variables from .env if present
load_dotenv()


class ExperimentTracker:
    """Manages MLflow tracking connected to DagsHub (or local MLflow fallback)."""

    def __init__(
        self,
        experiment_name: str = "customer_churn_retention",
        tracking_uri: Optional[str] = None,
        dagshub_owner: Optional[str] = None,
        dagshub_repo: Optional[str] = None,
        enabled: bool = True,
    ):
        self.experiment_name = experiment_name
        self.enabled = enabled
        self.mlflow: Optional[Any] = None
        self.active_uri: Optional[str] = None

        if not self.enabled:
            return

        try:
            self.mlflow = importlib.import_module("mlflow")
        except ImportError:
            logger.warning("MLflow not installed; experiment tracking disabled.")
            self.enabled = False
            return

        owner = dagshub_owner or os.getenv("DAGSHUB_REPO_OWNER", "").strip()
        repo = dagshub_repo or os.getenv("DAGSHUB_REPO_NAME", "").strip()
        token = (
            os.getenv("DAGSHUB_USER_TOKEN", "").strip()
            or os.getenv("DAGSHUB_TOKEN", "").strip()
            or os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
        )

        if owner and repo:
            dagshub_mlflow_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
            if token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = owner
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token
                os.environ["MLFLOW_TRACKING_URI"] = dagshub_mlflow_uri
                self.active_uri = dagshub_mlflow_uri
                logger.info(f"Configured DagsHub MLflow tracking via token: {dagshub_mlflow_uri}")
            else:
                try:
                    dagshub_mod = importlib.import_module("dagshub")
                    dagshub_mod.init(repo_owner=owner, repo_name=repo, mlflow=True)
                    self.active_uri = dagshub_mlflow_uri
                    logger.info(f"Initialized DagsHub MLflow session: {dagshub_mlflow_uri}")
                except Exception as exc:
                    logger.warning(
                        f"Could not authenticate with DagsHub ({exc}); falling back to local MLflow."
                    )

        if not self.active_uri:
            env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
            self.active_uri = env_uri or tracking_uri or "sqlite:///mlflow.db"

        try:
            self.mlflow.set_tracking_uri(self.active_uri)
            self.mlflow.set_experiment(self.experiment_name)
            logger.info(
                f"MLflow active -> URI: {self.active_uri} | Experiment: {self.experiment_name}"
            )
        except Exception as exc:
            logger.warning(f"MLflow setup failed ({exc}); continuing without remote logging.")
            self.enabled = False

    def log_candidate_run(
        self,
        model_name: str,
        model: Any,
        params: Dict[str, Any],
        val_metrics: Dict[str, Any],
    ) -> None:
        """Log an individual candidate model's hyperparameters and validation metrics."""
        if not self.enabled or self.mlflow is None:
            return

        try:
            with self.mlflow.start_run(run_name=f"candidate_{model_name}"):
                self.mlflow.set_tag("model_type", model_name)
                self.mlflow.set_tag("stage", "validation_candidate")

                clean_params = {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool)) or v is None
                }
                self.mlflow.log_params(clean_params)

                scalar_metrics = {
                    f"val_{k}": float(v)
                    for k, v in val_metrics.items()
                    if isinstance(v, (int, float))
                }
                if "confusion_matrix" in val_metrics:
                    for cm_k, cm_v in val_metrics["confusion_matrix"].items():
                        scalar_metrics[f"val_cm_{cm_k}"] = float(cm_v)
                self.mlflow.log_metrics(scalar_metrics)
        except Exception as exc:
            logger.warning(f"Failed to log candidate run '{model_name}' to MLflow: {exc}")

    def log_pipeline_summary(
        self,
        champion_name: str,
        champion_threshold: float,
        test_metrics: Dict[str, Any],
        feature_importances: Dict[str, float],
        artifact_paths: list[str | Path],
    ) -> None:
        """Log champion test metrics, feature importances, and report/figure artifacts."""
        if not self.enabled or self.mlflow is None:
            return

        try:
            with self.mlflow.start_run(run_name=f"champion_{champion_name}"):
                self.mlflow.set_tag("champion_model", champion_name)
                self.mlflow.log_param("champion_model", champion_name)
                self.mlflow.log_param("champion_threshold", champion_threshold)

                scalar_metrics = {
                    f"test_{k}": float(v)
                    for k, v in test_metrics.items()
                    if isinstance(v, (int, float))
                }
                if "confusion_matrix" in test_metrics:
                    for cm_k, cm_v in test_metrics["confusion_matrix"].items():
                        scalar_metrics[f"test_cm_{cm_k}"] = float(cm_v)
                self.mlflow.log_metrics(scalar_metrics)

                if feature_importances:
                    self.mlflow.log_dict(feature_importances, "feature_importances.json")

                for path in artifact_paths:
                    p = Path(path)
                    if p.exists():
                        if p.is_dir():
                            self.mlflow.log_artifacts(str(p), artifact_path=p.name)
                        else:
                            self.mlflow.log_artifact(str(p))

                logger.info(
                    f"Logged champion run '{champion_name}' and artifacts to MLflow ({self.active_uri})"
                )
        except Exception as exc:
            logger.warning(f"Failed to log champion summary to MLflow: {exc}")

