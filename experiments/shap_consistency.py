"""Evaluate stability of DAE SHAP feature importance across random seeds.

This experiment measures whether global SHAP explanations for the DAE anomaly
component remain similar when the legitimate background sample and stratified
evaluation sample change.

It does NOT explain or evaluate the final learned hybrid-gating decision.
"""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.shap_explainability import (
    select_legitimate_background,
    stratified_sample_indices,
    validate_binary_labels,
)
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
DEFAULT_FIGURE = Path("figures/explainability/dae_shap_consistency.png")
DEFAULT_RESULTS = Path("results/explainability/dae_shap_consistency.json")

DEFAULT_SEEDS = [42, 123, 2026]
DEFAULT_BACKGROUND_SIZE = 32
DEFAULT_GLOBAL_SAMPLES = 64
DEFAULT_TOP_K = 15


def mean_absolute_shap_importance(shap_values: np.ndarray) -> np.ndarray:
    """Compute global mean absolute SHAP importance for each feature."""
    values = np.asarray(shap_values, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("shap_values must be two-dimensional")

    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("shap_values must not be empty")

    if not np.isfinite(values).all():
        raise ValueError("shap_values must contain only finite values")

    return np.mean(np.abs(values), axis=0)


def top_feature_indices(
    importance: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Return indices of the most important features."""
    values = np.asarray(importance, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("importance must be one-dimensional")

    if values.size == 0:
        raise ValueError("importance must not be empty")

    if not np.isfinite(values).all():
        raise ValueError("importance must contain only finite values")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top_k = min(top_k, values.size)

    return np.argsort(values)[::-1][:top_k]


def top_k_jaccard(
    importance_a: np.ndarray,
    importance_b: np.ndarray,
    top_k: int,
) -> float:
    """Compute Jaccard similarity between two top-k feature sets."""
    first = set(top_feature_indices(importance_a, top_k).tolist())
    second = set(top_feature_indices(importance_b, top_k).tolist())

    union = first | second

    if not union:
        raise ValueError("top-k feature sets must not be empty")

    return float(len(first & second) / len(union))


def spearman_importance_correlation(
    importance_a: np.ndarray,
    importance_b: np.ndarray,
) -> float:
    """Compute Spearman correlation between global feature rankings."""
    first = np.asarray(importance_a, dtype=np.float64)
    second = np.asarray(importance_b, dtype=np.float64)

    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("importance arrays must be one-dimensional")

    if first.shape != second.shape:
        raise ValueError("importance arrays must have matching shapes")

    if first.size < 2:
        raise ValueError("importance arrays must contain at least two features")

    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("importance arrays must contain only finite values")

    result = spearmanr(first, second)

    correlation = float(result.statistic)

    if not np.isfinite(correlation):
        raise ValueError("Spearman correlation is undefined for the supplied importance arrays")

    return correlation


def compare_seed_importances(
    importances: dict[int, np.ndarray],
    top_k: int,
) -> list[dict[str, float | int]]:
    """Compute pairwise SHAP stability metrics across seeds."""
    if len(importances) < 2:
        raise ValueError("At least two seeds are required for consistency analysis")

    comparisons: list[dict[str, float | int]] = []

    for seed_a, seed_b in combinations(importances, 2):
        importance_a = importances[seed_a]
        importance_b = importances[seed_b]

        comparisons.append(
            {
                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "top_k_jaccard": top_k_jaccard(
                    importance_a,
                    importance_b,
                    top_k=top_k,
                ),
                "spearman_correlation": spearman_importance_correlation(
                    importance_a,
                    importance_b,
                ),
            }
        )

    return comparisons


def summarize_consistency(
    comparisons: list[dict[str, float | int]],
) -> dict[str, float]:
    """Summarize mean and minimum pairwise SHAP consistency."""
    if not comparisons:
        raise ValueError("comparisons must not be empty")

    jaccard = np.asarray(
        [row["top_k_jaccard"] for row in comparisons],
        dtype=np.float64,
    )

    spearman = np.asarray(
        [row["spearman_correlation"] for row in comparisons],
        dtype=np.float64,
    )

    return {
        "mean_top_k_jaccard": float(np.mean(jaccard)),
        "min_top_k_jaccard": float(np.min(jaccard)),
        "mean_spearman_correlation": float(np.mean(spearman)),
        "min_spearman_correlation": float(np.min(spearman)),
    }


def plot_consistency(
    comparisons: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Save pairwise SHAP consistency metrics as a green comparison plot."""
    if not comparisons:
        raise ValueError("comparisons must not be empty")

    labels = [f"{row['seed_a']} vs {row['seed_b']}" for row in comparisons]

    jaccard = [float(row["top_k_jaccard"]) for row in comparisons]

    spearman = [float(row["spearman_correlation"]) for row in comparisons]

    x = np.arange(len(comparisons))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        jaccard,
        width,
        label="Top-k Jaccard",
        color="#1F6B45",
        alpha=0.9,
    )

    ax.bar(
        x + width / 2,
        spearman,
        width,
        label="Spearman correlation",
        color="#8FB69A",
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(-1.0, 1.05)

    ax.axhline(
        0.0,
        color="#4C4C4C",
        linewidth=1.0,
    )

    ax.set_ylabel("Consistency score")
    ax.set_xlabel("Random-seed comparison")

    ax.set_title("DAE SHAP Explanation Consistency Across Random Seeds")

    ax.legend()

    ax.grid(
        axis="y",
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
        "Saved DAE SHAP consistency plot to %s",
        output_path,
    )


def export_consistency_results(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Export SHAP consistency diagnostics as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    logger.info(
        "Saved DAE SHAP consistency results to %s",
        output_path,
    )


def run_consistency(
    config_path: Path,
    data_path: Path,
    checkpoint_path: Path,
    figure_path: Path,
    results_path: Path,
    seeds: list[int],
    n_background: int = DEFAULT_BACKGROUND_SIZE,
    n_global_samples: int = DEFAULT_GLOBAL_SAMPLES,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run multi-seed DAE SHAP consistency evaluation."""
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required")

    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    if n_background <= 0:
        raise ValueError("n_background must be positive")

    if n_global_samples <= 0:
        raise ValueError("n_global_samples must be positive")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    config = load_config(config_path)

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

    model = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    importances: dict[int, np.ndarray] = {}
    seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        seed_everything(seed)

        logger.info(
            "Computing DAE SHAP consistency run for seed %d",
            seed,
        )

        background_df = select_legitimate_background(
            df=df,
            n_background=n_background,
            random_state=seed,
        )

        global_indices = stratified_sample_indices(
            labels=labels,
            max_samples=n_global_samples,
            random_state=seed,
        )

        global_df = df.iloc[global_indices]

        background_tensor = materialize_dae_features(
            background_df,
            feature_cols,
        )

        global_tensor = materialize_dae_features(
            global_df,
            feature_cols,
        )

        shap_values = compute_shap_values(
            model=model,
            background=background_tensor.to(device),
            samples=global_tensor.to(device),
            l1_gamma=l1_gamma,
        )

        importance = mean_absolute_shap_importance(
            shap_values,
        )

        importances[seed] = importance

        top_indices = top_feature_indices(
            importance,
            top_k=top_k,
        )

        seed_results.append(
            {
                "seed": int(seed),
                "n_background": int(background_tensor.shape[0]),
                "n_global_samples": int(global_tensor.shape[0]),
                "top_features": [
                    {
                        "feature_name": feature_cols[index],
                        "mean_abs_shap": float(importance[index]),
                    }
                    for index in top_indices
                ],
            }
        )

    comparisons = compare_seed_importances(
        importances=importances,
        top_k=top_k,
    )

    summary = summarize_consistency(
        comparisons,
    )

    result: dict[str, Any] = {
        "experiment": "dae_shap_consistency",
        "component": "DAE anomaly score",
        "seeds": [int(seed) for seed in seeds],
        "top_k": int(top_k),
        "n_features": len(feature_cols),
        "seed_results": seed_results,
        "pairwise_comparisons": comparisons,
        "summary": summary,
    }

    plot_consistency(
        comparisons=comparisons,
        output_path=figure_path,
    )

    export_consistency_results(
        payload=result,
        output_path=results_path,
    )

    logger.info(
        "DAE SHAP consistency complete | " "mean Jaccard=%.4f | mean Spearman=%.4f",
        summary["mean_top_k_jaccard"],
        summary["mean_spearman_correlation"],
    )

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for SHAP consistency evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DAE SHAP global feature-importance stability " "across random seeds."
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
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=DEFAULT_BACKGROUND_SIZE,
    )

    parser.add_argument(
        "--global-samples",
        type=int,
        default=DEFAULT_GLOBAL_SAMPLES,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser


def main() -> None:
    """Run DAE SHAP consistency evaluation."""
    parser = build_parser()

    args = parser.parse_args()

    config = load_config(args.config)

    setup_logging(
        level=str(config.logging.level),
        log_file=str(config.get_path("logging.log_file")),
    )

    results = run_consistency(
        config_path=args.config,
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        figure_path=args.figure,
        results_path=args.results,
        seeds=args.seeds,
        n_background=args.background_size,
        n_global_samples=args.global_samples,
        top_k=args.top_k,
        device=args.device,
    )

    logger.info(
        "Mean top-%d Jaccard similarity: %.4f",
        results["top_k"],
        results["summary"]["mean_top_k_jaccard"],
    )

    logger.info(
        "Mean Spearman importance correlation: %.4f",
        results["summary"]["mean_spearman_correlation"],
    )


if __name__ == "__main__":
    main()
