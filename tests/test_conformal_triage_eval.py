# Import necessary libraries and modules
from pathlib import Path

import numpy as np

from experiments.conformal_triage_eval import (
    evaluate_alpha,
    plot_coverage_vs_workload,
    sweep_alpha,
)


# Define the test function for evaluating conformal triage
def test_evaluate_alpha_returns_expected_metrics() -> None:
    calibration_probabilities = np.array([0.05, 0.10, 0.90, 0.95])
    calibration_labels = np.array([0, 0, 1, 1])

    eval_probabilities = np.array([0.02, 0.20, 0.80, 0.98])
    eval_labels = np.array([0, 0, 1, 1])

    metrics = evaluate_alpha(
        calibration_probabilities=calibration_probabilities,
        calibration_labels=calibration_labels,
        eval_probabilities=eval_probabilities,
        eval_labels=eval_labels,
        alpha=0.10,
    )

    expected_keys = {
        "alpha",
        "target_coverage",
        "empirical_coverage",
        "coverage_gap",
        "legitimate_coverage",
        "fraud_coverage",
        "review_rate",
        "approve_rate",
        "block_rate",
        "empty_set_rate",
        "average_set_size",
        "threshold",
    }

    assert set(metrics) == expected_keys
    assert 0.0 <= metrics["empirical_coverage"] <= 1.0

    assert np.isclose(
        metrics["coverage_gap"],
        metrics["empirical_coverage"] - metrics["target_coverage"],
    )

    assert 0.0 <= metrics["review_rate"] <= 1.0
    assert 0.0 <= metrics["approve_rate"] <= 1.0
    assert 0.0 <= metrics["block_rate"] <= 1.0
    assert 0.0 <= metrics["empty_set_rate"] <= 1.0
    assert 0.0 <= metrics["threshold"] <= 1.0
    assert 0.0 <= metrics["legitimate_coverage"] <= 1.0
    assert 0.0 <= metrics["fraud_coverage"] <= 1.0


def test_sweep_alpha_returns_one_result_per_alpha() -> None:
    calibration_probabilities = np.array([0.05, 0.10, 0.90, 0.95])
    calibration_labels = np.array([0, 0, 1, 1])

    eval_probabilities = np.array([0.02, 0.20, 0.80, 0.98])
    eval_labels = np.array([0, 0, 1, 1])

    alphas = [0.01, 0.05, 0.10]

    results = sweep_alpha(
        calibration_probabilities=calibration_probabilities,
        calibration_labels=calibration_labels,
        eval_probabilities=eval_probabilities,
        eval_labels=eval_labels,
        alphas=alphas,
    )

    assert len(results) == len(alphas)
    assert [result["alpha"] for result in results] == alphas


def test_plot_coverage_vs_workload_saves_figure(tmp_path: Path) -> None:
    results = [
        {
            "alpha": 0.01,
            "empirical_coverage": 0.99,
            "review_rate": 0.40,
        },
        {
            "alpha": 0.05,
            "empirical_coverage": 0.96,
            "review_rate": 0.25,
        },
        {
            "alpha": 0.10,
            "empirical_coverage": 0.91,
            "review_rate": 0.15,
        },
    ]

    output_path = tmp_path / "coverage_vs_workload.png"

    plot_coverage_vs_workload(
        results=results,
        output_path=output_path,
        show_plot=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_main_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "conformal_inputs.npz"

    np.savez(
        archive_path,
        calibration_probabilities=np.array([0.05, 0.10, 0.90, 0.95]),
        calibration_labels=np.array([0, 0, 1, 1]),
        eval_probabilities=np.array([0.02, 0.20, 0.80, 0.98]),
        eval_labels=np.array([0, 0, 1, 1]),
    )

    monkeypatch.setattr(
        "experiments.conformal_triage_eval.plot_coverage_vs_workload",
        lambda results, output_path: None,
    )

    from experiments.conformal_triage_eval import main

    main(["--archive", str(archive_path)])
