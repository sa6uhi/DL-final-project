"""
Master Data Preparation Script for the IEEE-CIS Fraud Detection Pipeline.
Run this script AFTER running download_data.py

Usage:
    python prepare_data.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.data.temporal_split import load_and_merge_data, split_temporal
from src.data.preprocessor import FraudPreprocessor
from src.data.sequence_extractor import build_historical_sequences

def main() -> None:
    # 1. Load Centralized Configuration
    cfg = load_config()
    
    # 2. Setup Output Directories
    processed_dir: Path = cfg.get_path("data.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed data to: {processed_dir}")
    
    # 3. Load and Merge Raw Data
    print("\n--- Loading Raw Data ---")
    trans_path = cfg.get_path("data.raw_transaction_path")
    id_path = cfg.get_path("data.raw_identity_path")
    df_merged = load_and_merge_data(str(trans_path), str(id_path))
    
    # 4. Strict Temporal Split
    print("\n--- Executing Temporal Split ---")
    train_df, val_df, test_df = split_temporal(df_merged, time_col=cfg.split.time_col)
    
    # 5. DYNAMICALLY find continuous and categorical columns
    print("\n--- Identifying Feature Types ---")
    target_col = cfg.features.target_col
    id_cols = ['TransactionID', 'TransactionDT']
    seq_cols = cfg.sequence.feature_cols
    
    # Get all columns except IDs, Target
    base_features = [c for c in train_df.columns if c not in id_cols + [target_col]]
    
    # Auto-detect continuous (numbers) and categorical (objects/strings)
    cont_cols = [c for c in base_features if train_df[c].dtype in ['float64', 'int64']]
    cat_cols = [c for c in base_features if train_df[c].dtype == 'object']
    
    # Remove seq_cols from cont_cols so they don't get processed twice
    cont_cols = [c for c in cont_cols if c not in seq_cols]
    
    print(f"Found {len(cont_cols)} continuous features.")
    print(f"Found {len(cat_cols)} categorical features.")
    print(f"Sequence features: {seq_cols}")
    
    # 6. Fit Preprocessor STRICTLY on Train
    print("\n--- Fitting Preprocessor (This takes a minute on real data) ---")
    preprocessor = FraudPreprocessor(cont_cols=cont_cols, cat_cols=cat_cols)
    
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)
    
    preprocessor.save(str(processed_dir / "preprocessor.pkl"))
    
    # 7. Extract Sequences (Run independently to prevent leakage!)
    print("\n--- Extracting Historical Sequences (This takes a few minutes on real data) ---")
    train_df = build_historical_sequences(train_df, cfg.sequence.group_cols, seq_cols, cfg.sequence.k_window)
    val_df = build_historical_sequences(val_df, cfg.sequence.group_cols, seq_cols, cfg.sequence.k_window)
    test_df = build_historical_sequences(test_df, cfg.sequence.group_cols, seq_cols, cfg.sequence.k_window)
    
    # 8. Save Final Artifacts
    print("\n--- Saving Parquet Files ---")
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    
    print("\n✅ SUCCESS! Real data preparation complete.")

if __name__ == "__main__":
    main()
