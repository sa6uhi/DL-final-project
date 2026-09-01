"""
Download the IEEE-CIS Fraud Detection raw CSVs into data/raw/.

Idempotent: skips download if train_transaction.csv already exists.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_REPO = "aliceczr/ieee-fraud-detection"
RESOLVE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"
FILES: List[str] = ["train_transaction.csv", "train_identity.csv"]
EXPECTED_SIZES = {
    "train_transaction.csv": 683_351_067,
    "train_identity.csv": 26_529_680,
}


def _download_file(url: str, dest: Path) -> None:
    """Stream a file from ``url`` to ``dest``.

    Args:
        url: Full remote URL to download.
        dest: Local destination path.

    Raises:
        OSError: If the download fails.
    """
    logger.info(f"Downloading {url} -> {dest}")
    request = Request(url, headers={"User-Agent": "dl-final-project/0.1"})
    with urlopen(request, timeout=120) as response, dest.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                logger.info(
                    "  %s: %d/%d MB (%.1f%%)",
                    dest.name,
                    downloaded // (1024 * 1024),
                    total // (1024 * 1024),
                    percent,
                )
    logger.info(f"Finished downloading {dest.name} ({downloaded} bytes)")


def download_and_extract_data(target_dir: Path) -> None:
    """Download the IEEE-CIS raw CSVs into ``target_dir``.

    Args:
        target_dir: Directory where the raw CSVs should be placed.

    Raises:
        OSError: If a download fails or a file fails the size check.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    if (target_dir / "train_transaction.csv").exists():
        logger.info("Raw data already exists in %s. Skipping download.", target_dir)
        return

    logger.info("Downloading IEEE-CIS fraud detection data from Hugging Face mirror...")

    for name in FILES:
        _download_file(f"{RESOLVE_URL}/{name}", target_dir / name)

    for name in FILES:
        dest = target_dir / name
        size = dest.stat().st_size
        expected = EXPECTED_SIZES[name]
        if size != expected:
            raise OSError(
                f"Downloaded {name} size mismatch: got {size} bytes, " f"expected {expected} bytes."
            )

    logger.info(f"Download complete. Raw CSVs saved to {target_dir}")


if __name__ == "__main__":
    from src.utils.config import load_config

    cfg = load_config()
    download_and_extract_data(cfg.get_path("data.raw_dir"))
