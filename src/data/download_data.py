"""
Automated data ingestion script for the IEEE-CIS Fraud Detection dataset.
Uses the Kaggle API to download and extract the data into data/raw/.

Usage:
    1. Ensure you have installed the kaggle package: pip install kaggle
    2. Ensure your Kaggle API token is placed at ~/.kaggle/kaggle.json
       (Go to https://www.kaggle.com/settings -> API -> Create New Token)
    3. Run: python download_data.py
"""

import zipfile
import shutil
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger(__name__)


# IEEE-CIS Kaggle Competition slug
COMPETITION_SLUG = "ieee-fraud-detection"


def download_and_extract_data(target_dir: Path) -> None:
    """Downloads dataset from Kaggle and extracts it to target_dir.

    Args:
        target_dir: The directory where raw CSVs should be placed.

    Raises:
        OSError: If the Kaggle API token is missing or invalid.
        RuntimeError: If the download fails for other reasons.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if data is already downloaded (Idempotency)
    train_file = target_dir / "train_transaction.csv"
    if train_file.exists():
        logger.info("Raw data already exists in data/raw/. Skipping download.")
        return

    logger.info(f"Downloading {COMPETITION_SLUG} dataset from Kaggle...")

    try:
        # Import kaggle inside the function so it doesn't crash the whole
        import kaggle
    except ImportError:
        raise ImportError(
            "The 'kaggle' package is required to download data. "
            "Please run: pip install kaggle"
        )

    try:
        # Download the competition files to a temporary zip path
        zip_path = target_dir / f"{COMPETITION_SLUG}.zip"
        kaggle.api.competition_download_files(COMPETITION_SLUG, path=str(target_dir))

        logger.info("Download complete. Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        # 3. Clean up the zip file to save disk space
        zip_path.unlink()
        logger.info(f"Extraction complete. Raw CSVs saved to {target_dir}")

    except Exception as e:
        # Provide a highly specific error message if Kaggle auth fails
        if "401" in str(e) or "Unauthorized" in str(e):
            logger.error("=" * 50)
            logger.error("KAGGLE AUTHENTICATION FAILED!")
            logger.error("Please place your API token at: ~/.kaggle/kaggle.json")
            logger.error("Go to: https://www.kaggle.com/settings -> API -> Create New Token")
            logger.error("=" * 50)
            raise OSError("Kaggle API Token missing or invalid.") from e
        else:
            logger.error(f"Failed to download data: {e}")
            raise


if __name__ == "__main__":
    # For standalone execution
    from src.utils.config import load_config

    cfg = load_config()
    raw_dir = cfg.get_path("data.raw_dir")
    download_and_extract_data(raw_dir)
