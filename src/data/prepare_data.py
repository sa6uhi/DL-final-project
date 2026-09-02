"""
Master data preparation script for the IEEE-CIS Fraud Detection pipeline.
Run this script AFTER running download_data.py

Usage:
    python -m src.data.prepare_data
"""

from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.data.temporal_split import load_and_merge_data, split_temporal
from src.data.preprocessor import FraudPreprocessor
from src.data.sequence_extractor import build_historical_sequences
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()

    processed_dir: Path = cfg.get_path("data.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving processed data to: %s", processed_dir)

    trans_path = cfg.get_path("data.raw_transaction_path")
    id_path = cfg.get_path("data.raw_identity_path")
    df_merged = load_and_merge_data(str(trans_path), str(id_path))

    train_df, val_df, test_df = split_temporal(df_merged, time_col=cfg.split.time_col)

    target_col = cfg.features.target_col
    id_cols = ["TransactionID", "TransactionDT"]
    seq_cols = cfg.sequence.feature_cols

    base_features = [c for c in train_df.columns if c not in id_cols + [target_col]]
    cont_cols = [c for c in base_features if train_df[c].dtype in ["float64", "int64"]]
    # pandas >= 3.0 reads text columns as the `str` dtype rather than `object`,
    cat_cols = [c for c in base_features if pd.api.types.is_string_dtype(train_df[c])]

    logger.info("Found %d continuous features.", len(cont_cols))
    logger.info("Found %d categorical features.", len(cat_cols))
    logger.info("Sequence features: %s", seq_cols)

    preprocessor = FraudPreprocessor(cont_cols=cont_cols, cat_cols=cat_cols)
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    preprocessor.save(str(processed_dir / "preprocessor.pkl"))

    train_df = build_historical_sequences(
        train_df, cfg.sequence.group_cols, seq_cols, cfg.sequence.k_window
    )
    val_df = build_historical_sequences(
        val_df, cfg.sequence.group_cols, seq_cols, cfg.sequence.k_window
    )
    test_df = build_historical_sequences(
        test_df, cfg.sequence.group_cols, seq_cols, cfg.sequence.k_window
    )

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    logger.info("Data preparation complete. Parquet files saved to %s", processed_dir)


if __name__ == "__main__":
    main()
