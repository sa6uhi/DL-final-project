"""Generate SHAP explainability artifacts for the trained DAE anomaly detector.

This experiment explains the DAE anomaly-score component of the fraud system.
It does NOT claim to explain the final learned hybrid-gating decision.

Artifacts:
    figures/explainability/dae_shap_waterfall.png
    figures/explainability/dae_shap_global_importance.png
"""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.explainability.shap_explainer import compute_shap_values
from src.training.dae_features import (
    materialize_dae_features,
    resolve_dae_feature_columns,
)
from src.training.train_autoencoder import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_DATA = Path("data/processed/test.parquet")
DEFAULT_CHECKPOINT = Path("models/checkpoints/autoencoder.pt")
DEFAULT_OUTPUT_DIR = Path("figures/explainability")


def validate_binary_labels(labels: np.ndarray) -> None:
    """Validate fraud labels used for explainability sampling."""
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    if labels.size == 0:
        raise ValueError("labels must not be empty")

    unique = set(np.unique(labels).tolist())

    if not unique.issubset({0, 1}):
        raise ValueError("labels must contain only binary values 0 and 1")


def stratified_sample_indices(
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    """Return deterministic approximately class-balanced sample indices."""
    labels = np.asarray(labels)

    validate_binary_labels(labels)

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    if labels.size <= max_samples:
        return np.arange(labels.size, dtype=np.int64)

    rng = np.random.default_rng(random_state)

    legitimate = np.flatnonzero(labels == 0)
    fraud = np.flatnonzero(labels == 1)

    if legitimate.size == 0 or fraud.size == 0:
        return np.sort(
            rng.choice(
                labels.size,
                size=max_samples,
                replace=False,
            )
        )

    fraud_target = min(fraud.size, max_samples // 2)
    legit_target = min(legitimate.size, max_samples - fraud_target)

    remaining = max_samples - fraud_target - legit_target

    if remaining > 0:
        fraud_available = fraud.size - fraud_target
        fraud_extra = min(remaining, fraud_available)
        fraud_target += fraud_extra
        remaining -= fraud_extra

    if remaining > 0:
        legit_available = legitimate.size - legit_target
        legit_extra = min(remaining, legit_available)
        legit_target += legit_extra

    fraud_sample = rng.choice(
        fraud,
        size=fraud_target,
        replace=False,
    )

    legitimate_sample = rng.choice(
        legitimate,
        size=legit_target,
        replace=False,
    )

    selected = np.concatenate(
        [
            legitimate_sample,
            fraud_sample,
        ]
    )

    rng.shuffle(selected)

    return selected.astype(np.int64)


def select_legitimate_background(
    df: pd.DataFrame,
    n_background: int,
    random_state: int,
) -> pd.DataFrame:
    """Select legitimate transactions as the SHAP reference background."""
    if "isFraud" not in df.columns:
        raise KeyError("Evaluation data must contain an isFraud column")

    if n_background <= 0:
        raise ValueError("n_background must be positive")

    legitimate = df.loc[df["isFraud"] == 0]

    if legitimate.empty:
        raise ValueError("No legitimate transactions available for SHAP background")

    n_selected = min(n_background, len(legitimate))

    return legitimate.sample(
        n=n_selected,
        random_state=random_state,
        replace=False,
    )


def score_anomalies_batched(
    model: torch.nn.Module,
    features: torch.Tensor,
    l1_gamma: float,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Compute official DAE anomaly scores in deterministic batches."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional tensor")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if l1_gamma < 0:
        raise ValueError("l1_gamma must be non-negative")

    model = model.to(device)
    model.eval()

    scores: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])

            batch = features[start:stop].to(device)

            if not hasattr(model, "anomaly_score"):
                raise AttributeError("DAE model must provide an anomaly_score method")

            batch_scores = model.anomaly_score(
                batch,
                l1_gamma=l1_gamma,
                reduction="none",
            )

            scores.append(batch_scores.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def select_high_anomaly_fraud(
    df: pd.DataFrame,
    model: torch.nn.Module,
    feature_cols: list[str],
    l1_gamma: float,
    device: str,
    batch_size: int,
) -> tuple[pd.Series, float]:
    """Select the fraudulent test transaction with the highest DAE anomaly score."""
    if "isFraud" not in df.columns:
        raise KeyError("Evaluation data must contain an isFraud column")

    fraud_df = df.loc[df["isFraud"] == 1]

    if fraud_df.empty:
        raise ValueError("No fraudulent transactions available for local SHAP explanation")

    fraud_features = materialize_dae_features(
        fraud_df,
        feature_cols,
    )

    scores = score_anomalies_batched(
        model=model,
        features=fraud_features,
        l1_gamma=l1_gamma,
        device=device,
        batch_size=batch_size,
    )

    selected_position = int(np.argmax(scores))
    selected_score = float(scores[selected_position])

    selected_row = fraud_df.iloc[selected_position]

    return selected_row, selected_score


def plot_local_shap(
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    anomaly_score: float,
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Save a waterfall-style local SHAP explanation."""
    feature_values = np.asarray(feature_values, dtype=np.float64)
    shap_values = np.asarray(shap_values, dtype=np.float64)

    if feature_values.ndim != 1 or shap_values.ndim != 1:
        raise ValueError("feature_values and shap_values must be one-dimensional")

    if feature_values.shape != shap_values.shape:
        raise ValueError("feature_values and shap_values must have matching shapes")

    if len(feature_names) != feature_values.size:
        raise ValueError("feature_names must match feature count")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]

    selected_names = [feature_names[idx] for idx in top_indices]
    selected_values = feature_values[top_indices]
    selected_shap = shap_values[top_indices]

    order = np.arange(len(top_indices))[::-1]

    labels = [
        f"{name} = {value:.3g}"
        for name, value in zip(
            selected_names,
            selected_values,
        )
    ]

    bar_colors = ["#1F6B45" if value >= 0 else "#8FB69A" for value in selected_shap]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        order,
        selected_shap,
        color=bar_colors,
        alpha=0.9,
    )

    ax.set_yticks(order)
    ax.set_yticklabels(labels)
    ax.axvline(
        0.0,
        color="#4C4C4C",
        linewidth=1.0,
    )

    ax.set_xlabel("SHAP contribution to DAE anomaly score")
    ax.set_title(
        "DAE Anomaly Explanation — High-Anomaly Fraud Transaction\n"
        f"DAE anomaly score: {anomaly_score:.3f}"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Saved local DAE SHAP explanation to %s",
        output_path,
    )


def plot_global_shap_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Save global mean-absolute-SHAP feature importance."""
    values = np.asarray(shap_values, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("shap_values must be two-dimensional")

    if values.shape[1] != len(feature_names):
        raise ValueError("feature_names must match SHAP feature count")

    if values.shape[0] == 0:
        raise ValueError("shap_values must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    importance = np.mean(
        np.abs(values),
        axis=0,
    )

    top_indices = np.argsort(importance)[::-1][:top_k]
    top_indices = top_indices[::-1]

    selected_names = [feature_names[idx] for idx in top_indices]

    selected_importance = importance[top_indices]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        selected_names,
        selected_importance,
        color="#1F6B45",
        alpha=0.9,
    )

    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("DAE Anomaly Explanation — Global Feature Importance")

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Saved global DAE SHAP importance plot to %s",
        output_path,
    )


def run_explainability(
    config_path: Path,
    data_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    n_background: int = 64,
    n_global_samples: int = 64,
    global_top_k: int = 15,
    local_top_k: int = 10,
    random_state: int = 42,
    device: str = "cpu",
    batch_size: int = 2048,
) -> dict[str, object]:
    """Generate local and global DAE SHAP explainability artifacts."""
    if n_background <= 0:
        raise ValueError("n_background must be positive")

    if n_global_samples <= 0:
        raise ValueError("n_global_samples must be positive")

    config = load_config(config_path)

    seed_everything(random_state)

    logger.info(
        "Loading processed evaluation data from %s",
        data_path,
    )

    df = pd.read_parquet(data_path)

    if df.empty:
        raise ValueError("Evaluation data must not be empty")

    if "isFraud" not in df.columns:
        raise KeyError("Evaluation data must contain an isFraud column")

    labels = df["isFraud"].to_numpy()

    validate_binary_labels(labels)

    expected_dim = int(config.autoencoder.input_dim)

    l1_gamma = float(config.autoencoder.anomaly_score.l1_gamma)

    non_feature_cols = [
        "isFraud",
        "TransactionID",
        "TransactionDT",
        "sequence_array",
    ]

    feature_cols = resolve_dae_feature_columns(
        df=df,
        non_feature_cols=non_feature_cols,
        expected_dim=expected_dim,
    )

    logger.info(
        "Resolved %d DAE explainability features",
        len(feature_cols),
    )

    model = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    background_df = select_legitimate_background(
        df=df,
        n_background=n_background,
        random_state=random_state,
    )

    background_tensor = materialize_dae_features(
        background_df,
        feature_cols,
    )

    global_indices = stratified_sample_indices(
        labels=labels,
        max_samples=n_global_samples,
        random_state=random_state,
    )

    global_df = df.iloc[global_indices]

    global_tensor = materialize_dae_features(
        global_df,
        feature_cols,
    )

    logger.info(
        "Computing SHAP values for %d global transactions using %d "
        "legitimate background transactions",
        global_tensor.shape[0],
        background_tensor.shape[0],
    )

    global_shap = compute_shap_values(
        model=model,
        background=background_tensor.to(device),
        samples=global_tensor.to(device),
        l1_gamma=l1_gamma,
    )

    global_output = output_dir / "dae_shap_global_importance.png"

    plot_global_shap_importance(
        shap_values=global_shap,
        feature_names=feature_cols,
        output_path=global_output,
        top_k=global_top_k,
    )

    selected_row, anomaly_score = select_high_anomaly_fraud(
        df=df,
        model=model,
        feature_cols=feature_cols,
        l1_gamma=l1_gamma,
        device=device,
        batch_size=batch_size,
    )

    local_df = selected_row.to_frame().T

    local_tensor = materialize_dae_features(
        local_df,
        feature_cols,
    )

    logger.info(
        "Computing local SHAP explanation for selected fraud transaction "
        "with anomaly score %.4f",
        anomaly_score,
    )

    local_shap = compute_shap_values(
        model=model,
        background=background_tensor.to(device),
        samples=local_tensor.to(device),
        l1_gamma=l1_gamma,
    )

    local_output = output_dir / "dae_shap_waterfall.png"

    plot_local_shap(
        feature_values=local_tensor[0].cpu().numpy(),
        shap_values=local_shap[0],
        feature_names=feature_cols,
        anomaly_score=anomaly_score,
        output_path=local_output,
        top_k=local_top_k,
    )

    transaction_id: int | float | str | None = None

    if "TransactionID" in selected_row.index:
        transaction_id = selected_row["TransactionID"]

        if hasattr(transaction_id, "item"):
            transaction_id = transaction_id.item()

    logger.info(
        "DAE SHAP explainability complete | " "local=%s | global=%s",
        local_output,
        global_output,
    )

    return {
        "local_output": local_output,
        "global_output": global_output,
        "selected_transaction_id": transaction_id,
        "selected_anomaly_score": anomaly_score,
        "n_background": int(background_tensor.shape[0]),
        "n_global_samples": int(global_tensor.shape[0]),
        "n_features": len(feature_cols),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate local and global SHAP explanations " "for the trained DAE anomaly detector."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--global-samples",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--global-top-k",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--local-top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    return parser


def main() -> None:
    """Run the SHAP explainability experiment."""
    parser = build_parser()

    args = parser.parse_args()

    config = load_config(args.config)

    setup_logging(
        level=str(config.logging.level),
        log_file=str(config.get_path("logging.log_file")),
    )

    results = run_explainability(
        config_path=args.config,
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_background=args.background_size,
        n_global_samples=args.global_samples,
        global_top_k=args.global_top_k,
        local_top_k=args.local_top_k,
        random_state=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )

    logger.info(
        "Selected fraud transaction ID: %s",
        results["selected_transaction_id"],
    )

    logger.info(
        "Selected DAE anomaly score: %.4f",
        results["selected_anomaly_score"],
    )


if __name__ == "__main__":
    main()
