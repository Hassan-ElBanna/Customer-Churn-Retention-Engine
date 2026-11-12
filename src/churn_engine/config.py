"""Configuration loader and typed settings."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

