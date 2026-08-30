"""Exp-6: out-of-time anomaly detection evaluation of the trained DAE.

Evaluates reconstruction residuals on chronologically-later transactions:
ROC-AUC / AU-PRC / TPR@FPR=1% on raw residuals, percentile-normalized
residuals, and the alpha-swept hybrid gate (DAE residual vs. FT-Transformer
posterior) via :mod:`src.models.hybrid_gating`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.evaluation.metrics import summarize
from src.models.hybrid_gating import HybridGate, PercentileNormalizer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a flat NumPy ``.npz`` bundle into a plain dict.

    Args:
        path: Path to the ``.npz`` archive.

    Returns:
        Dict of arrays; ``npz`` objects are treated as plain dicts.

    Raises:
        FileNotFoundError: If the archive does not exist.
    """
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz archive not found: {npz_path}")
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def evaluate_split(
    scores: np.ndarray,
    labels: np.ndarray,
    max_fpr: float = 0.01,
) -> dict[str, float]:
    """Compute the three headline metrics on one labelled split.

    Args:
        scores: Per-sample anomaly scores.
        labels: Binary labels (0=legit, 1=fraud).
        max_fpr: Target FPR for the TPR operating point.

    Returns:
        Metric dict (rocauc / auprc / tpr_at_fpr / prevalence).
    """
    metrics = summarize(scores, labels, max_fpr=max_fpr)
    metrics["prevalence"] = round(float(np.asarray(labels).mean()), 6)
    return metrics


def sweep_alpha(
    calibration_scores: np.ndarray,
    eval_scores: np.ndarray,
    probabilities_ft: np.ndarray,
    labels: np.ndarray,
    alphas: Iterable[float],
    percentile: float = 99.9,
    max_fpr: float = 0.01,
) -> list[dict[str, float]]:
    """Sweep the hybrid-gate blend weight on the evaluation split.

    The percentile normalizer is fitted *only* on the calibration scores
    (temperature split, not test) to respect the anti-leakage rule.

    Args:
        calibration_scores: DAE residuals whose distribution defines the
            normalizer cap.
        eval_scores: DAE residuals on the evaluation split.
        probabilities_ft: Supervised fraud probabilities on the evaluation
            split.
        labels: Ground-truth labels on the evaluation split.
        alphas: Blend weights to sweep in ``[0, 1]``.
        percentile: Normalizer clip percentile in ``[0, 100]``.
        max_fpr: Target FPR for the reported TPR.

    Returns:
        One record per alpha: alpha, rocauc, auprc, tpr_at_fpr.
    """
    normalizer = PercentileNormalizer(percentile=percentile).fit(
        torch.as_tensor(calibration_scores, dtype=torch.float32)
    )
    records: list[dict[str, float]] = []
    for alpha in alphas:
        gate = HybridGate(alpha=float(alpha), normalizer=normalizer)
        fused = gate.fuse(
            torch.as_tensor(eval_scores, dtype=torch.float32),
            torch.as_tensor(probabilities_ft, dtype=torch.float32),
        ).numpy()
        metrics = summarize(fused, labels, max_fpr=max_fpr)
        records.append({"alpha": float(alpha), **metrics})
        logger.info("alpha=%s -> %s", alpha, metrics)
    return records


def write_eval_report(
    split_metrics: dict[str, float],
    alpha_sweep: list[dict[str, float]],
    output_dir: str | Path,
) -> Path:
    """Persist the evaluation summary as JSON + CSV.

    Args:
        split_metrics: Metrics for the raw anomaly evaluation.
        alpha_sweep: Per-alpha records from the gating sweep.
        output_dir: Destination directory (created if missing).

    Returns:
        Path to the written CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out.joinpath("anomaly_eval.json").write_text(
        json.dumps({"raw": split_metrics, "sweep": alpha_sweep}, indent=2)
    )
    csv_path = out / "alpha_sweep.csv"
    if alpha_sweep:
        fieldnames = list(alpha_sweep[0].keys())
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(alpha_sweep)
    logger.info("Anomaly report written to %s", out)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m src.evaluation.dae_anomaly_eval``."""
    parser = argparse.ArgumentParser(description="Exp-6 out-of-time DAE anomaly evaluation")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--archive",
        required=True,
        help="npz with keys: calibration_scores, eval_scores, probabilities_ft, labels",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the configured anomaly-eval output directory",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    eval_cfg = config["evaluation"]["anomaly_eval"]
    bundle = load_npz(args.archive)

    labels = bundle["labels"]
    split_metrics = evaluate_split(
        bundle["eval_scores"], labels, max_fpr=float(eval_cfg["max_fpr"])
    )
    sweep = sweep_alpha(
        calibration_scores=bundle["calibration_scores"],
        eval_scores=bundle["eval_scores"],
        probabilities_ft=bundle["probabilities_ft"],
        labels=labels,
        alphas=[float(a) for a in eval_cfg["alpha_sweep"]],
        percentile=float(config["autoencoder"]["anomaly_score"]["normalize_percentile"]),
        max_fpr=float(eval_cfg["max_fpr"]),
    )
    output_dir = args.output_dir or config["paths"]["experiments"] + "/anomaly_eval"
    write_eval_report(split_metrics, sweep, output_dir)


if __name__ == "__main__":
    main()
