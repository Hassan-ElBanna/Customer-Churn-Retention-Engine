"""Phase 1: Data Ingestion from URL, Google Drive, Kaggle, or local path."""

from pathlib import Path
from urllib.parse import urlparse
import zipfile
import io
import pandas as pd
import requests

from churn_engine.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestor:
    """Handles dataset downloading, extraction, and loading."""

    def __init__(self, raw_data_path: str | Path = "data/raw/customer_churn.csv"):
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_download_url(url: str) -> str:
        """Convert Kaggle dataset page, Google Drive, or GitHub blob links into direct download links."""
        if "kaggle.com/datasets/blastchar/telco-customer-churn" in url.lower():
            return "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        if "kaggle.com/datasets/" in url:
            parts = url.split("kaggle.com/datasets/")[1].strip("/").split("?")[0].split("/")
            if len(parts) >= 2:
                owner, slug = parts[0], parts[1]
                return f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}"
        if "drive.google.com/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        if "github.com" in url and "/blob/" in url:
            return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return url

    def download_from_url(self, url: str, timeout: int = 120) -> Path:
        """Download dataset from a direct link (CSV, Parquet, Excel, or ZIP archive)."""
        direct_url = self._normalize_download_url(url.strip())
        logger.info(f"Downloading dataset from: {direct_url}")

        response = requests.get(direct_url, timeout=timeout, stream=True)
        response.raise_for_status()
        content = response.content

        # Check if downloaded payload is a ZIP archive
        if zipfile.is_zipfile(io.BytesIO(content)):
            logger.info("Detected ZIP archive. Extracting tabular dataset...")
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                tabular_files = [
                    f for f in zf.namelist()
                    if f.lower().endswith((".csv", ".parquet", ".xlsx"))
                    and not f.startswith("__MACOSX")
                ]
                if not tabular_files:
                    raise ValueError(f"No CSV/Parquet/XLSX found in ZIP archive: {zf.namelist()}")
                target_member = tabular_files[0]
                logger.info(f"Extracting '{target_member}' -> {self.raw_data_path}")
                with zf.open(target_member) as member_file:
                    raw_bytes = member_file.read()
                if target_member.lower().endswith(".csv"):
                    self.raw_data_path.write_bytes(raw_bytes)
                elif target_member.lower().endswith(".parquet"):
                    df = pd.read_parquet(io.BytesIO(raw_bytes))
                    df.to_csv(self.raw_data_path, index=False)
                elif target_member.lower().endswith(".xlsx"):
                    df = pd.read_excel(io.BytesIO(raw_bytes))
                    df.to_csv(self.raw_data_path, index=False)
        else:
            parsed_path = urlparse(direct_url).path.lower()
            if parsed_path.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content))
                df.to_csv(self.raw_data_path, index=False)
            elif parsed_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(content))
                df.to_csv(self.raw_data_path, index=False)
            else:
                self.raw_data_path.write_bytes(content)

        logger.info(f"Dataset saved to {self.raw_data_path} ({self.raw_data_path.stat().st_size:,} bytes)")
        return self.raw_data_path

    def load_data(self, url: str | None = None) -> pd.DataFrame:
        """Load dataset from disk or download from URL if provided."""
        if url:
            self.download_from_url(url)

        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.raw_data_path}. Please provide a dataset URL or place the CSV there."
            )

        df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded dataset with shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

