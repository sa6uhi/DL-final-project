"""Experiment: conformal coverage versus analyst review workload.

Evaluates split-conformal fraud triage across a sweep of miscoverage
levels using held-out calibration and evaluation probabilities.
"""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.uncertainty.conformal_predictor import SplitConformalPredictor
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_N_BOOTSTRAP = 500
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ECE_BINS = 10


def compute_probability_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> tuple[float, float]:
    """Compute Brier score and expected calibration error (ECE)."""
    probabilities_array = np.asarray(probabilities, dtype=float)
    labels_array = np.asarray(labels)

    if probabilities_array.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional array")

    if labels_array.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")

    if len(probabilities_array) == 0:
        raise ValueError("probabilities must not be empty")

    if len(probabilities_array) != len(labels_array):
        raise ValueError("probabilities and labels must have the same length")

    if not np.all(np.isfinite(probabilities_array)):
        raise ValueError("probabilities must contain only finite values")

    if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
        raise ValueError("probabilities must be between 0 and 1")

    if not np.all(np.isin(labels_array, [0, 1])):
        raise ValueError("labels must contain only binary values 0 and 1")

    if n_bins <= 0:
        raise ValueError("n_bins must be greater than zero")

    brier_score = float(np.mean((probabilities_array - labels_array.astype(float)) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == n_bins - 1:
            bin_mask = (probabilities_array >= lower) & (probabilities_array <= upper)
        else:
            bin_mask = (probabilities_array >= lower) & (probabilities_array < upper)

        if not bin_mask.any():
            continue

        bin_confidence = float(np.mean(probabilities_array[bin_mask]))
        bin_accuracy = float(np.mean(labels_array[bin_mask]))
        bin_weight = float(np.mean(bin_mask))

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return brier_score, float(ece)


def bootstrap_metric_intervals(
    eval_probabilities: np.ndarray,
    eval_labels: np.ndarray,
    covered: np.ndarray,
    decisions: list[str],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Estimate percentile bootstrap confidence intervals for evaluation metrics."""
    probabilities = np.asarray(eval_probabilities, dtype=float)
    labels = np.asarray(eval_labels)
    covered_array = np.asarray(covered, dtype=bool)
    decisions_array = np.asarray(decisions)

    n_samples = len(labels)

    if n_samples == 0:
        raise ValueError("evaluation data must not be empty")

    if not (len(probabilities) == len(covered_array) == len(decisions_array) == n_samples):
        raise ValueError("bootstrap inputs must have matching lengths")

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be greater than zero")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(seed)

    bootstrap_values: dict[str, list[float]] = {
        "empirical_coverage": [],
        "review_rate": [],
        "review_precision": [],
        "review_fraud_capture": [],
        "brier_score": [],
    }

    for _ in range(n_bootstrap):
        indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]
        sampled_covered = covered_array[indices]
        sampled_decisions = decisions_array[indices]

        review_mask = sampled_decisions == "human_review"
        fraud_mask = sampled_labels == 1

        bootstrap_values["empirical_coverage"].append(float(np.mean(sampled_covered)))

        bootstrap_values["review_rate"].append(float(np.mean(review_mask)))

        if review_mask.any():
            bootstrap_values["review_precision"].append(float(np.mean(fraud_mask[review_mask])))

        if fraud_mask.any():
            bootstrap_values["review_fraud_capture"].append(float(np.mean(review_mask[fraud_mask])))

        bootstrap_values["brier_score"].append(
            float(np.mean((sampled_probabilities - sampled_labels.astype(float)) ** 2))
        )

    tail_probability = (1.0 - confidence_level) / 2.0
    lower_quantile = tail_probability
    upper_quantile = 1.0 - tail_probability

    intervals: dict[str, tuple[float, float]] = {}

    for metric_name, values in bootstrap_values.items():
        if not values:
            intervals[metric_name] = (
                float("nan"),
                float("nan"),
            )
            continue

        intervals[metric_name] = (
            float(np.quantile(values, lower_quantile)),
            float(np.quantile(values, upper_quantile)),
        )

    return intervals


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

    review_mask = np.asarray(
        [decision == "human_review" for decision in decisions],
        dtype=bool,
    )

    reviewed_fraud_mask = review_mask & fraud_mask

    review_precision = (
        float(np.mean(fraud_mask[review_mask])) if review_mask.any() else float("nan")
    )

    review_fraud_capture = (
        float(np.sum(reviewed_fraud_mask) / np.sum(fraud_mask))
        if fraud_mask.any()
        else float("nan")
    )

    legitimate_coverage = (
        float(np.mean(covered_array[legitimate_mask])) if legitimate_mask.any() else float("nan")
    )

    fraud_coverage = float(np.mean(covered_array[fraud_mask])) if fraud_mask.any() else float("nan")

    brier_score, expected_calibration_error = compute_probability_calibration_metrics(
        probabilities=eval_probabilities,
        labels=eval_labels,
    )

    target_coverage = 1.0 - alpha
    empirical_coverage = float(np.mean(covered))
    coverage_gap = empirical_coverage - target_coverage

    bootstrap_intervals = bootstrap_metric_intervals(
        eval_probabilities=eval_probabilities,
        eval_labels=eval_labels_array,
        covered=covered_array,
        decisions=decisions,
    )

    return {
        "alpha": alpha,
        "target_coverage": target_coverage,
        "empirical_coverage": empirical_coverage,
        "empirical_coverage_ci_lower": bootstrap_intervals["empirical_coverage"][0],
        "empirical_coverage_ci_upper": bootstrap_intervals["empirical_coverage"][1],
        "coverage_gap": coverage_gap,
        "legitimate_coverage": legitimate_coverage,
        "fraud_coverage": fraud_coverage,
        "brier_score": brier_score,
        "brier_score_ci_lower": bootstrap_intervals["brier_score"][0],
        "brier_score_ci_upper": bootstrap_intervals["brier_score"][1],
        "expected_calibration_error": expected_calibration_error,
        "review_rate": decisions.count("human_review") / n_samples,
        "review_rate_ci_lower": bootstrap_intervals["review_rate"][0],
        "review_rate_ci_upper": bootstrap_intervals["review_rate"][1],
        "review_precision": review_precision,
        "review_precision_ci_lower": bootstrap_intervals["review_precision"][0],
        "review_precision_ci_upper": bootstrap_intervals["review_precision"][1],
        "review_fraud_capture": review_fraud_capture,
        "review_fraud_capture_ci_lower": bootstrap_intervals["review_fraud_capture"][0],
        "review_fraud_capture_ci_upper": bootstrap_intervals["review_fraud_capture"][1],
        "approve_rate": decisions.count("auto_approve") / n_samples,
        "block_rate": decisions.count("auto_block") / n_samples,
        "empty_set_rate": (
            sum(prediction == frozenset() for prediction in prediction_sets) / n_samples
        ),
        "singleton_rate": (sum(len(prediction) == 1 for prediction in prediction_sets) / n_samples),
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
    show_plot: bool = False,
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


def export_conformal_results(
    results: list[dict[str, float]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Export conformal evaluation metrics to CSV and JSON."""
    if not results:
        raise ValueError("results must not be empty")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "conformal_metrics.csv"
    json_path = output_directory / "conformal_metrics.json"

    fieldnames = list(results[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            results,
            json_file,
            indent=2,
            allow_nan=True,
        )

    logger.info("Saved conformal CSV metrics to %s", csv_path)
    logger.info("Saved conformal JSON metrics to %s", json_path)

    return csv_path, json_path


def get_git_commit() -> str | None:
    """Return the current Git commit hash when available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return completed.stdout.strip() or None


def export_experiment_metadata(
    output_dir: str | Path,
    archive_path: str | Path,
    alphas: list[float],
    seed: int,
    calibration_size: int,
    evaluation_size: int,
) -> Path:
    """Export reproducibility metadata for the conformal experiment."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": "conformal_triage_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(Path(archive_path)),
        "seed": int(seed),
        "alpha_sweep": [float(alpha) for alpha in alphas],
        "calibration_size": int(calibration_size),
        "evaluation_size": int(evaluation_size),
        "bootstrap_resamples": DEFAULT_N_BOOTSTRAP,
        "bootstrap_confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "ece_bins": DEFAULT_ECE_BINS,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "git_commit": get_git_commit(),
    }

    metadata_path = output_directory / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )

    logger.info(
        "Saved conformal experiment metadata to %s",
        metadata_path,
    )

    return metadata_path


def plot_threshold_sensitivity(
    results: list[dict[str, float]],
    output_path: str | Path,
    show_plot: bool = False,
) -> None:
    """Plot conformal threshold sensitivity across alpha values."""
    if not results:
        raise ValueError("results must not be empty")

    thresholds = [result["threshold"] for result in results]
    review_rates = [result["review_rate"] for result in results]
    coverages = [result["empirical_coverage"] for result in results]
    singleton_rates = [result["singleton_rate"] for result in results]
    alphas = [result["alpha"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        thresholds,
        review_rates,
        marker="o",
        linewidth=2,
        label="Human Review Rate",
        color="#1F6B45",
    )

    ax.plot(
        thresholds,
        coverages,
        marker="s",
        linewidth=2,
        label="Empirical Coverage",
        color="#3F7D5A",
    )

    ax.plot(
        thresholds,
        singleton_rates,
        marker="^",
        linewidth=2,
        label="Singleton Rate",
        color="#8FB69A",
    )

    for threshold, review_rate, alpha in zip(
        thresholds,
        review_rates,
        alphas,
        strict=True,
    ):
        ax.annotate(
            f"α={alpha:g}",
            (threshold, review_rate),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Conformal Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Conformal Threshold Sensitivity")
    ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    logger.info(
        "Saved conformal threshold-sensitivity figure to %s",
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

    threshold_figure_path = (
        config.get_path("paths.figures") / "conformal" / "threshold_sensitivity.png"
    )

    plot_threshold_sensitivity(
        results=results,
        output_path=threshold_figure_path,
    )

    results_dir = Path("results") / "conformal"

    export_conformal_results(
        results=results,
        output_dir=results_dir,
    )

    export_experiment_metadata(
        output_dir=results_dir,
        archive_path=args.archive,
        alphas=alphas,
        seed=int(config.nested_get("seed")),
        calibration_size=len(bundle["calibration_labels"]),
        evaluation_size=len(bundle["eval_labels"]),
    )


if __name__ == "__main__":
    main()
