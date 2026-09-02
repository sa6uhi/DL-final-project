"""
Training script for Classical ML Baselines.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict

from src.utils.config import load_config
from src.models.baselines import get_baselines
from src.evaluation.metrics import evaluate_fraud_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    processed_dir: Path = cfg.get_path("data.processed_dir")
    checkpoint_dir: Path = cfg.get_path("paths.checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    target_col = cfg.features.target_col

    # 1. Load Data (Baselines don't use PyTorch DataLoaders, just Pandas)
    logger.info("Loading processed parquet files...")
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    # 2. Drop columns that baselines cannot process
    drop_cols = ["sequence_array", "TransactionID", "TransactionDT", target_col]
    # Only drop if they exist in the dataframe
    drop_cols = [c for c in drop_cols if c in train_df.columns]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df[target_col]

    logger.info(f"Training data shape: {X_train.shape}")

    # 3. Get Models
    models = get_baselines(y_train)

    # 4. Train and Evaluate Loop
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        logger.info(f"--- Training {name} ---")

        # Train
        model.fit(X_train, y_train)

        # Predict probabilities (predict_proba returns [prob_0, prob_1], we want prob_1)
        val_probs = model.predict_proba(X_val)[:, 1]

        # Evaluate
        metrics = evaluate_fraud_metrics(y_val.values, val_probs, name)
        results[name] = metrics

        # Save Model
        model_path = checkpoint_dir / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name} to {model_path}")

    logger.info("--- Baseline Training Complete ---")
    for name, scores in results.items():
        print(f"{name}: PR-AUC = {scores['PR-AUC']:.4f}")


if __name__ == "__main__":
    main()
