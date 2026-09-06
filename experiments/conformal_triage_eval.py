"""Experiment: conformal coverage versus analyst review workload.

Evaluates split-conformal fraud triage across a sweep of miscoverage
levels using held-out calibration and evaluation probabilities.
"""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.uncertainty.conformal_predictor import SplitConformalPredictor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Define the evaluation function for conformal triage
def evaluate_alpha(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    eval_probabilities: np.ndarray,
    eval_labels: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Evaluate conformal coverage and triage workload for one alpha."""
    calibration_probs_tensor = torch.as_tensor(
        calibration_probabilities,
        dtype=torch.float32,
    )
    calibration_labels_tensor = torch.as_tensor(
        calibration_labels,
        dtype=torch.long,
    )

    predictor = SplitConformalPredictor(alpha=alpha)
    predictor.fit(
        fraud_probabilities=calibration_probs_tensor,
        labels=calibration_labels_tensor,
    )

    prediction_sets = [
        predictor.predict_set(float(probability)) for probability in eval_probabilities
    ]

    covered = [
        int(label) in prediction
        for label, prediction in zip(eval_labels, prediction_sets, strict=True)
    ]

    decisions = [predictor.predict_triage(float(probability)) for probability in eval_probabilities]

    n_samples = len(prediction_sets)

    if n_samples == 0:
        raise ValueError("evaluation data must not be empty")

    eval_labels_array = np.asarray(eval_labels)
    covered_array = np.asarray(covered, dtype=bool)

    legitimate_mask = eval_labels_array == 0
    fraud_mask = eval_labels_array == 1

    legitimate_coverage = (
        float(np.mean(covered_array[legitimate_mask])) if legitimate_mask.any() else float("nan")
    )

    fraud_coverage = float(np.mean(covered_array[fraud_mask])) if fraud_mask.any() else float("nan")

    target_coverage = 1.0 - alpha
    empirical_coverage = float(np.mean(covered))
    coverage_gap = empirical_coverage - target_coverage

    return {
        "alpha": alpha,
        "target_coverage": target_coverage,
        "empirical_coverage": empirical_coverage,
        "coverage_gap": coverage_gap,
        "legitimate_coverage": legitimate_coverage,
        "fraud_coverage": fraud_coverage,
        "review_rate": decisions.count("human_review") / n_samples,
        "approve_rate": decisions.count("auto_approve") / n_samples,
        "block_rate": decisions.count("auto_block") / n_samples,
        "empty_set_rate": (
            sum(prediction == frozenset() for prediction in prediction_sets) / n_samples
        ),
        "average_set_size": float(np.mean([len(prediction) for prediction in prediction_sets])),
        "threshold": float(predictor.threshold),
    }


# Define the function to sweep across multiple alpha values for conformal triage evaluation
def sweep_alpha(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    eval_probabilities: np.ndarray,
    eval_labels: np.ndarray,
    alphas: list[float],
) -> list[dict[str, float]]:
    """Evaluate conformal triage across multiple miscoverage levels."""
    if not alphas:
        raise ValueError("alphas must not be empty")

    results = []

    for alpha in alphas:
        metrics = evaluate_alpha(
            calibration_probabilities=calibration_probabilities,
            calibration_labels=calibration_labels,
            eval_probabilities=eval_probabilities,
            eval_labels=eval_labels,
            alpha=alpha,
        )
        results.append(metrics)

        logger.info(
            "alpha=%.3f | coverage=%.4f | review_rate=%.4f | threshold=%.4f",
            alpha,
            metrics["empirical_coverage"],
            metrics["review_rate"],
            metrics["threshold"],
        )

    return results


# Define the function to plot coverage versus workload for conformal triage
def plot_coverage_vs_workload(
    results: list[dict[str, float]],
    output_path: str | Path,
    show_plot: bool = True,
) -> None:
    """Plot empirical coverage against human-review workload."""
    if not results:
        raise ValueError("results must not be empty")

    coverages = [result["empirical_coverage"] for result in results]
    review_rates = [result["review_rate"] for result in results]
    alphas = [result["alpha"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        review_rates,
        coverages,
        marker="o",
        linewidth=2,
        color="#3F7D5A",
    )

    for review_rate, coverage, alpha in zip(
        review_rates,
        coverages,
        alphas,
        strict=True,
    ):
        ax.annotate(
            f"α={alpha:g}",
            (review_rate, coverage),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Human Review Rate")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Conformal Triage: Coverage vs Analyst Workload")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")

    logger.info(
        "Saved conformal coverage-workload figure to %s",
        output,
    )

    if show_plot:
        plt.show()

    plt.close(fig)


# Define the main function to run the conformal triage evaluation from a probability archive
def main(argv: list[str] | None = None) -> None:
    """Run conformal triage evaluation from a probability archive."""
    parser = argparse.ArgumentParser(
        description="Evaluate conformal coverage versus analyst workload."
    )
    parser.add_argument(
        "--archive",
        required=True,
        help=(
            "NPZ archive containing calibration_probabilities, "
            "calibration_labels, eval_probabilities, and eval_labels."
        ),
    )
    args = parser.parse_args(argv)

    config = load_config()
    bundle = np.load(args.archive)

    required_keys = {
        "calibration_probabilities",
        "calibration_labels",
        "eval_probabilities",
        "eval_labels",
    }

    missing_keys = required_keys.difference(bundle.files)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"NPZ archive is missing required keys: {missing}")

    alphas = list(config.nested_get("evaluation.conformal.alpha_sweep"))

    results = sweep_alpha(
        calibration_probabilities=bundle["calibration_probabilities"],
        calibration_labels=bundle["calibration_labels"],
        eval_probabilities=bundle["eval_probabilities"],
        eval_labels=bundle["eval_labels"],
        alphas=alphas,
    )

    figure_path = config.get_path("paths.figures") / "conformal" / "coverage_vs_workload.png"

    plot_coverage_vs_workload(
        results=results,
        output_path=figure_path,
    )


if __name__ == "__main__":
    main()
