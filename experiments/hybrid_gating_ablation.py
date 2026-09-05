"""Experiment: Fixed-alpha vs learned hybrid gating ablation.

Compares the fixed hybrid fusion baseline against the learned gating network
using held-out DAE anomaly scores and FT-CAT fraud probabilities.

The experiment expects an NPZ archive containing:
    calibration_scores
    eval_scores
    probabilities_ft
    labels
"""

# Import necessary libraries and modules
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluation.dae_anomaly_eval import load_npz
from src.evaluation.metrics import summarize
from src.models.hybrid_gating import HybridGate, PercentileNormalizer
from src.training.train_hybrid_gating import build_gate_features, load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Define the hybrid gating evaluation function
def evaluate_fixed_gate(
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
    probabilities_ft: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    percentile: float,
    max_fpr: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate the fixed-alpha hybrid gating baseline.

    Args:
        calibration_scores: DAE anomaly scores used only to fit normalization.
        eval_scores: DAE anomaly scores on the held-out evaluation split.
        probabilities_ft: FT-CAT fraud probabilities for evaluation samples.
        labels: Binary fraud labels for evaluation samples.
        alpha: Fixed weight assigned to the normalized anomaly score.
        percentile: Percentile used to normalize DAE anomaly scores.
        max_fpr: False-positive-rate operating point for TPR reporting.

    Returns:
        Tuple containing evaluation metrics and fused prediction scores.
    """
    normalizer = PercentileNormalizer(percentile=percentile).fit(
        torch.as_tensor(calibration_scores, dtype=torch.float32)
    )

    gate = HybridGate(alpha=alpha, normalizer=normalizer)

    with torch.no_grad():
        fused_scores = gate.fuse(
            torch.as_tensor(eval_scores, dtype=torch.float32),
            torch.as_tensor(probabilities_ft, dtype=torch.float32),
        )

    metrics = summarize(
        fused_scores.numpy(),
        labels,
        max_fpr=max_fpr,
    )

    logger.info("Fixed gate alpha=%.3f -> %s", alpha, metrics)

    return metrics, fused_scores.numpy()


def evaluate_learned_gate(
    eval_scores: np.ndarray,
    probabilities_ft: np.ndarray,
    labels: np.ndarray,
    checkpoint_path: str | Path,
    max_fpr: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate the trained learned hybrid gate.

    Args:
        eval_scores: DAE anomaly scores on the held-out evaluation split.
        probabilities_ft: FT-CAT fraud probabilities for evaluation samples.
        labels: Binary fraud labels for evaluation samples.
        checkpoint_path: Path to the trained learned-gate checkpoint.
        max_fpr: False-positive-rate operating point for TPR reporting.

    Returns:
        Tuple containing evaluation metrics and learned prediction scores.
    """
    model, normalizer = load_checkpoint(checkpoint_path, device="cpu")

    features = build_gate_features(
        anomaly_scores=torch.as_tensor(eval_scores, dtype=torch.float32),
        transformer_probabilities=torch.as_tensor(
            probabilities_ft,
            dtype=torch.float32,
        ),
        normalizer=normalizer,
        fit_normalizer=False,
    )

    with torch.no_grad():
        learned_scores = model(features).numpy()

    metrics = summarize(
        learned_scores,
        labels,
        max_fpr=max_fpr,
    )

    logger.info("Learned gate -> %s", metrics)

    return metrics, learned_scores


def analyze_gate_disagreement(
    fixed_scores: np.ndarray,
    learned_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Analyze where fixed and learned gates make different binary decisions."""
    fixed_scores = np.asarray(fixed_scores, dtype=float)
    learned_scores = np.asarray(learned_scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if fixed_scores.shape != learned_scores.shape:
        raise ValueError("fixed_scores and learned_scores must have the same shape")

    if fixed_scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")

    fixed_predictions = fixed_scores >= threshold
    learned_predictions = learned_scores >= threshold

    disagreement_mask = fixed_predictions != learned_predictions
    n_samples = len(labels)

    if n_samples == 0:
        raise ValueError("evaluation data must not be empty")

    disagreement_rate = float(np.mean(disagreement_mask))

    learned_correct_when_disagree = (
        np.mean(learned_predictions[disagreement_mask] == labels[disagreement_mask])
        if disagreement_mask.any()
        else float("nan")
    )

    fixed_correct_when_disagree = (
        np.mean(fixed_predictions[disagreement_mask] == labels[disagreement_mask])
        if disagreement_mask.any()
        else float("nan")
    )

    return {
        "disagreement_rate": disagreement_rate,
        "learned_correct_when_disagree": float(learned_correct_when_disagree),
        "fixed_correct_when_disagree": float(fixed_correct_when_disagree),
    }


def plot_gate_comparison(
    fixed_metrics: dict[str, float],
    learned_metrics: dict[str, float],
    output_path: str | Path,
    show_plot: bool = True,
) -> None:
    """Plot fixed-alpha versus learned-gate performance.

    Args:
        fixed_metrics: Evaluation metrics for the fixed-alpha gate.
        learned_metrics: Evaluation metrics for the learned gate.
        output_path: Destination path for the generated figure.
    """
    metric_keys = ["rocauc", "auprc", "tpr_at_fpr"]
    metric_labels = ["ROC-AUC", "PR-AUC", "TPR @ 1% FPR"]

    fixed_values = [fixed_metrics[key] for key in metric_keys]
    learned_values = [learned_metrics[key] for key in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        fixed_values,
        width,
        label="Fixed α Gate",
        color="#A8B5AE",
    )
    ax.bar(
        x + width / 2,
        learned_values,
        width,
        label="Learned Gate",
        color="#3F7D5A",
    )

    ax.set_ylabel("Score")
    ax.set_title("Hybrid Gating Ablation: Fixed vs Learned Fusion")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")

    logger.info("Saved hybrid gating ablation figure to %s", output)

    if show_plot:
        plt.show()

    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the fixed-alpha versus learned-gate ablation experiment."""
    parser = argparse.ArgumentParser(
        description="Compare fixed-alpha and learned hybrid fraud gating."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the project configuration file.",
    )
    parser.add_argument(
        "--archive",
        required=True,
        help=(
            "NPZ archive containing calibration_scores, eval_scores, "
            "probabilities_ft, and labels."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional learned-gate checkpoint path override.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    bundle = load_npz(args.archive)

    required_keys = {
        "calibration_scores",
        "eval_scores",
        "probabilities_ft",
        "labels",
    }
    missing_keys = required_keys.difference(bundle)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Archive is missing required arrays: {missing}")

    fixed_alpha = float(config.hybrid_gating.alpha)
    percentile = float(config.hybrid_gating.learned.normalize_percentile)
    max_fpr = float(config.evaluation.anomaly_eval.max_fpr)

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(config.hybrid_gating.learned.checkpoint_path)
    )

    logger.info("Evaluating fixed hybrid gate...")
    fixed_metrics, fixed_scores = evaluate_fixed_gate(
        calibration_scores=bundle["calibration_scores"],
        eval_scores=bundle["eval_scores"],
        probabilities_ft=bundle["probabilities_ft"],
        labels=bundle["labels"],
        alpha=fixed_alpha,
        percentile=percentile,
        max_fpr=max_fpr,
    )

    logger.info("Evaluating learned hybrid gate...")
    learned_metrics, learned_scores = evaluate_learned_gate(
        eval_scores=bundle["eval_scores"],
        probabilities_ft=bundle["probabilities_ft"],
        labels=bundle["labels"],
        checkpoint_path=checkpoint_path,
        max_fpr=max_fpr,
    )

    figure_path = config.get_path("paths.figures") / "hybrid_gating" / "hybrid_gating_ablation.png"

    plot_gate_comparison(
        fixed_metrics=fixed_metrics,
        learned_metrics=learned_metrics,
        output_path=figure_path,
    )

    disagreement_metrics = analyze_gate_disagreement(
        fixed_scores=fixed_scores,
        learned_scores=learned_scores,
        labels=bundle["labels"],
    )

    logger.info(
        "Gate disagreement rate: %.4f | "
        "Learned correct when disagree: %.4f | "
        "Fixed correct when disagree: %.4f",
        disagreement_metrics["disagreement_rate"],
        disagreement_metrics["learned_correct_when_disagree"],
        disagreement_metrics["fixed_correct_when_disagree"],
    )

    logger.info(
        "Gating ablation complete. Fixed PR-AUC: %.4f | Learned PR-AUC: %.4f",
        fixed_metrics["auprc"],
        learned_metrics["auprc"],
    )


if __name__ == "__main__":
    main()
