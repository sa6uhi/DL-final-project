"""Training script for Classical ML Baselines."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from src.evaluation.metrics_a import evaluate_fraud_metrics
from src.models.baselines import get_baselines
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: Optional[List[str]] = None) -> None:
    """Train classical baselines and persist models plus a metrics artifact.

    Args:
        argv: Command line arguments; uses ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(description="Train classical ML fraud baselines")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines",
        help="Directory for the baseline_metrics.json artifact.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
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

    # Drop any remaining string/object columns that baselines can't process
    X_train = X_train.select_dtypes(exclude=["object"])
    X_val = X_val.select_dtypes(exclude=["object"])

    # Force-fill any stray NaNs (LogReg is strict about missing values)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # Drop the *_is_nan indicator columns to save massive amounts of RAM
    is_nan_cols = [c for c in X_train.columns if c.endswith("_is_nan")]
    X_train = X_train.drop(columns=is_nan_cols)
    X_val = X_val.drop(columns=is_nan_cols)

    msg = f"Dropped {len(is_nan_cols)} NaN indicator columns. New shape: {X_train.shape}"
    logger.info(msg)

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
        logger.info(f"{name}: PR-AUC = {scores['PR-AUC']:.4f}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    metrics_path = results_dir / "baseline_metrics.json"
    serializable = {
        name: {metric: float(value) for metric, value in scores.items()}
        for name, scores in results.items()
    }
    payload = {
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "models": serializable,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote baseline metrics to {metrics_path}")


if __name__ == "__main__":
    main()
