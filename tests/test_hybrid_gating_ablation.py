"""Tests for the hybrid gating ablation experiment."""

# Import necessary modules and libraries
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.hybrid_gating_ablation import (
    analyze_gate_disagreement,
    evaluate_fixed_gate,
    evaluate_learned_gate,
    plot_gate_comparison,
    plot_gate_disagreement,
)

from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.training.train_hybrid_gating import save_checkpoint


# Define the test function for evaluating the fixed hybrid gate
def test_evaluate_fixed_gate_returns_expected_metrics() -> None:
    """Fixed hybrid gate returns valid evaluation metrics."""
    calibration_scores = np.array(
        [0.1, 0.2, 0.15, 0.3],
        dtype=np.float32,
    )
    eval_scores = np.array(
        [0.1, 0.2, 5.0, 6.0, 0.3, 7.0],
        dtype=np.float32,
    )
    probabilities_ft = np.array(
        [0.05, 0.06, 0.90, 0.95, 0.07, 0.92],
        dtype=np.float32,
    )
    labels = np.array(
        [0, 0, 1, 1, 0, 1],
        dtype=np.int64,
    )

    metrics, scores = evaluate_fixed_gate(
        calibration_scores=calibration_scores,
        eval_scores=eval_scores,
        probabilities_ft=probabilities_ft,
        labels=labels,
        alpha=0.5,
        percentile=99.0,
        max_fpr=0.5,
    )

    assert set(metrics) == {
        "rocauc",
        "auprc",
        "tpr_at_fpr",
    }

    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    assert scores.shape == eval_scores.shape
    assert np.isfinite(scores).all()


def test_evaluate_learned_gate_from_checkpoint(tmp_path: Path) -> None:
    """Learned gate can be restored and evaluated from a checkpoint."""
    anomaly_scores = torch.tensor(
        [0.1, 0.2, 5.0, 6.0, 0.3, 7.0],
        dtype=torch.float32,
    )
    probabilities_ft = np.array(
        [0.05, 0.06, 0.90, 0.95, 0.07, 0.92],
        dtype=np.float32,
    )
    labels = np.array(
        [0, 0, 1, 1, 0, 1],
        dtype=np.int64,
    )

    normalizer = PercentileNormalizer(percentile=99.0).fit(anomaly_scores)

    model = LearnedHybridGate(
        input_dim=2,
        hidden_dims=[4],
        dropout=0.0,
    )

    checkpoint_path = tmp_path / "hybrid_gating.pt"

    save_checkpoint(
        model=model,
        normalizer=normalizer,
        path=checkpoint_path,
    )

    metrics, scores = evaluate_learned_gate(
        eval_scores=anomaly_scores.numpy(),
        probabilities_ft=probabilities_ft,
        labels=labels,
        checkpoint_path=checkpoint_path,
        max_fpr=0.5,
    )

    assert set(metrics) == {
        "rocauc",
        "auprc",
        "tpr_at_fpr",
    }

    assert all(0.0 <= value <= 1.0 for value in metrics.values())

    assert scores.shape == anomaly_scores.numpy().shape
    assert np.isfinite(scores).all()


def test_evaluate_four_input_learned_gate_from_checkpoint(
    tmp_path: Path,
) -> None:
    """History-aware learned gate evaluates with velocity context."""
    anomaly_scores = torch.tensor(
        [0.1, 0.2, 5.0, 6.0, 0.3, 7.0],
        dtype=torch.float32,
    )

    probabilities_ft = np.array(
        [0.05, 0.06, 0.90, 0.95, 0.07, 0.92],
        dtype=np.float32,
    )

    velocity_features = np.array(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 2.0],
            [0.6, 3.0],
            [0.8, 4.0],
            [1.0, 5.0],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [0, 0, 1, 1, 0, 1],
        dtype=np.int64,
    )

    normalizer = PercentileNormalizer(percentile=99.0).fit(anomaly_scores)

    model = LearnedHybridGate(
        input_dim=4,
        hidden_dims=[4],
        dropout=0.0,
    )

    checkpoint_path = tmp_path / "hybrid_gating_4_input.pt"

    save_checkpoint(
        model=model,
        normalizer=normalizer,
        path=checkpoint_path,
    )

    metrics, scores = evaluate_learned_gate(
        eval_scores=anomaly_scores.numpy(),
        probabilities_ft=probabilities_ft,
        velocity_features=velocity_features,
        labels=labels,
        checkpoint_path=checkpoint_path,
        max_fpr=0.5,
    )

    assert set(metrics) == {
        "rocauc",
        "auprc",
        "tpr_at_fpr",
    }

    assert scores.shape == anomaly_scores.numpy().shape
    assert np.isfinite(scores).all()


def test_four_input_learned_gate_requires_velocity_features(
    tmp_path: Path,
) -> None:
    """Four-input checkpoints reject missing historical context."""
    anomaly_scores = torch.tensor(
        [0.1, 0.2, 5.0, 6.0],
        dtype=torch.float32,
    )

    normalizer = PercentileNormalizer(percentile=99.0).fit(anomaly_scores)

    model = LearnedHybridGate(
        input_dim=4,
        hidden_dims=[4],
        dropout=0.0,
    )

    checkpoint_path = tmp_path / "hybrid_gating_4_input.pt"

    save_checkpoint(
        model=model,
        normalizer=normalizer,
        path=checkpoint_path,
    )

    with pytest.raises(ValueError, match="velocity_features"):
        evaluate_learned_gate(
            eval_scores=anomaly_scores.numpy(),
            probabilities_ft=np.array(
                [0.1, 0.2, 0.8, 0.9],
                dtype=np.float32,
            ),
            labels=np.array(
                [0, 0, 1, 1],
                dtype=np.int64,
            ),
            checkpoint_path=checkpoint_path,
            max_fpr=0.5,
        )


def test_plot_gate_comparison_saves_figure(tmp_path: Path) -> None:
    """Ablation comparison figure is saved to the requested subfolder."""
    fixed_metrics = {
        "rocauc": 0.80,
        "auprc": 0.70,
        "tpr_at_fpr": 0.60,
    }
    learned_metrics = {
        "rocauc": 0.90,
        "auprc": 0.85,
        "tpr_at_fpr": 0.75,
    }

    output_path = tmp_path / "figures" / "hybrid_gating" / "hybrid_gating_ablation.png"

    plot_gate_comparison(
        fixed_metrics=fixed_metrics,
        learned_metrics=learned_metrics,
        output_path=output_path,
        show_plot=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_analyze_gate_disagreement_reports_expected_rates() -> None:
    fixed_scores = np.array([0.2, 0.7, 0.8, 0.3])
    learned_scores = np.array([0.2, 0.4, 0.9, 0.6])
    labels = np.array([0, 0, 1, 1])

    metrics = analyze_gate_disagreement(
        fixed_scores=fixed_scores,
        learned_scores=learned_scores,
        labels=labels,
        threshold=0.5,
    )

    assert np.isclose(metrics["disagreement_rate"], 0.5)
    assert np.isclose(metrics["learned_correct_when_disagree"], 1.0)
    assert np.isclose(metrics["fixed_correct_when_disagree"], 0.0)


def test_plot_gate_disagreement_saves_figure(tmp_path: Path) -> None:
    fixed_scores = np.array([0.1, 0.7, 0.8, 0.3])
    learned_scores = np.array([0.2, 0.4, 0.9, 0.6])
    labels = np.array([0, 0, 1, 1])

    output_path = tmp_path / "figures" / "hybrid_gating" / "gate_disagreement.png"

    plot_gate_disagreement(
        fixed_scores=fixed_scores,
        learned_scores=learned_scores,
        labels=labels,
        output_path=output_path,
        show_plot=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
