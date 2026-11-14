"""Vercel Python Serverless entrypoint wrapping the FastAPI inference & monitoring app."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from churn_engine.api.main import app  # noqa: E402

__all__ = ["app"]
