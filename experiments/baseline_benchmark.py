"""
Experiment 1: Baseline Model Benchmark (PR-Curves).
Generates figures/baseline_pr_curves.png for the IEEE paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.utils.config import load_config
from src.evaluation.metrics_a import evaluate_fraud_metrics

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    processed_dir: Path = cfg.get_path("data.processed_dir")
    checkpoint_dir: Path = cfg.get_path("paths.checkpoints")
    figures_dir: Path = cfg.get_path("paths.figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_col = cfg.features.target_col

    # 1. Load Test Data
    logger.info("Loading test data for benchmarking...")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    drop_cols = ["sequence_array", "TransactionID", "TransactionDT", target_col]
    drop_cols = [c for c in drop_cols if c in test_df.columns]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col]

    X_test = X_test.select_dtypes(exclude=["object"]).fillna(0)
    is_nan_cols = [c for c in X_test.columns if c.endswith("_is_nan")]

    # 2. Load Models and Predict
    models_to_test = ["LogReg_Balanced.pkl", "RandomForest_Balanced.pkl", "LightGBM_Weighted.pkl"]

    plt.figure(figsize=(8, 6))

    for model_file in models_to_test:
        model_path = checkpoint_dir / model_file
        if not model_path.exists():
            logger.warning(f"Skipping {model_file}, not found.")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        name = model_file.replace(".pkl", "")

        # --- FORCE ALIGNMENT FIX ---
        # Only keep the columns the model was actually trained on
        if hasattr(model, "feature_names_in_"):
            X_test_subset = X_test[model.feature_names_in_]
        else:
            X_test_subset = X_test

        # Use X_test_subset here, NOT X_test!
        probs = model.predict_proba(X_test_subset)[:, 1]
        # -------------------------

        # Calculate PR Curve data
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)

        # Plot
        plt.plot(recall, precision, label=f"{name} (PR-AUC: {pr_auc:.3f})", lw=2)
        logger.info(f"{name} Test PR-AUC: {pr_auc:.4f}")

    # 3. Format and Save Plot
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Exp 1: Baseline Model Comparison (PR-Curves)", fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    save_path = figures_dir / "baseline_pr_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PR-Curves plot to {save_path}")


if __name__ == "__main__":
    main()
