# Import necessary libraries and modules
from pathlib import Path

import numpy as np

from experiments import conformal_triage_eval
from experiments.conformal_triage_eval import (
    bootstrap_metric_intervals,
    compute_probability_calibration_metrics,
    evaluate_alpha,
    export_conformal_results,
    export_experiment_metadata,
    plot_coverage_vs_workload,
    plot_threshold_sensitivity,
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
        "empirical_coverage_ci_lower",
        "empirical_coverage_ci_upper",
        "coverage_gap",
        "legitimate_coverage",
        "fraud_coverage",
        "brier_score",
        "brier_score_ci_lower",
        "brier_score_ci_upper",
        "expected_calibration_error",
        "review_rate",
        "review_rate_ci_lower",
        "review_rate_ci_upper",
        "review_precision",
        "review_precision_ci_lower",
        "review_precision_ci_upper",
        "review_fraud_capture",
        "review_fraud_capture_ci_lower",
        "review_fraud_capture_ci_upper",
        "approve_rate",
        "block_rate",
        "empty_set_rate",
        "singleton_rate",
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
    assert 0.0 <= metrics["review_precision"] <= 1.0
    assert 0.0 <= metrics["review_fraud_capture"] <= 1.0
    assert 0.0 <= metrics["approve_rate"] <= 1.0
    assert 0.0 <= metrics["block_rate"] <= 1.0
    assert 0.0 <= metrics["empty_set_rate"] <= 1.0
    assert 0.0 <= metrics["singleton_rate"] <= 1.0
    assert 0.0 <= metrics["threshold"] <= 1.0
    assert 0.0 <= metrics["legitimate_coverage"] <= 1.0
    assert 0.0 <= metrics["fraud_coverage"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0

    assert (
        metrics["empirical_coverage_ci_lower"]
        <= metrics["empirical_coverage"]
        <= metrics["empirical_coverage_ci_upper"]
    )

    assert (
        metrics["review_rate_ci_lower"] <= metrics["review_rate"] <= metrics["review_rate_ci_upper"]
    )

    assert (
        metrics["brier_score_ci_lower"] <= metrics["brier_score"] <= metrics["brier_score_ci_upper"]
    )

    derived_singleton_rate = 2.0 - 2.0 * metrics["empty_set_rate"] - metrics["average_set_size"]

    assert np.isclose(
        metrics["singleton_rate"],
        derived_singleton_rate,
    )


def test_bootstrap_metric_intervals_are_deterministic() -> None:
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])
    labels = np.array([0, 0, 1, 1])
    covered = np.array([True, True, True, True])
    decisions = [
        "auto_approve",
        "human_review",
        "human_review",
        "auto_block",
    ]

    first = bootstrap_metric_intervals(
        eval_probabilities=probabilities,
        eval_labels=labels,
        covered=covered,
        decisions=decisions,
        n_bootstrap=100,
        seed=42,
    )

    second = bootstrap_metric_intervals(
        eval_probabilities=probabilities,
        eval_labels=labels,
        covered=covered,
        decisions=decisions,
        n_bootstrap=100,
        seed=42,
    )

    assert first == second


def test_bootstrap_metric_intervals_reject_invalid_bootstrap_count() -> None:
    with np.testing.assert_raises(ValueError):
        bootstrap_metric_intervals(
            eval_probabilities=np.array([0.1, 0.9]),
            eval_labels=np.array([0, 1]),
            covered=np.array([True, True]),
            decisions=["auto_approve", "auto_block"],
            n_bootstrap=0,
        )


def test_probability_calibration_metrics_perfect_predictions() -> None:
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])

    brier_score, ece = compute_probability_calibration_metrics(
        probabilities=probabilities,
        labels=labels,
        n_bins=10,
    )

    assert np.isclose(brier_score, 0.0)
    assert np.isclose(ece, 0.0)


def test_probability_calibration_metrics_known_brier_score() -> None:
    probabilities = np.array([0.1, 0.4, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    brier_score, ece = compute_probability_calibration_metrics(
        probabilities=probabilities,
        labels=labels,
        n_bins=2,
    )

    expected_brier = np.mean((probabilities - labels.astype(float)) ** 2)

    assert np.isclose(brier_score, expected_brier)
    assert 0.0 <= ece <= 1.0


def test_probability_calibration_metrics_reject_invalid_probabilities() -> None:
    probabilities = np.array([0.1, 1.2])
    labels = np.array([0, 1])

    with np.testing.assert_raises(ValueError):
        compute_probability_calibration_metrics(
            probabilities=probabilities,
            labels=labels,
        )


def test_probability_calibration_metrics_reject_nonbinary_labels() -> None:
    probabilities = np.array([0.1, 0.9])
    labels = np.array([0, 2])

    with np.testing.assert_raises(ValueError):
        compute_probability_calibration_metrics(
            probabilities=probabilities,
            labels=labels,
        )


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


def test_export_conformal_results_writes_csv_and_json(tmp_path) -> None:
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
    ]

    csv_path, json_path = export_conformal_results(
        results=results,
        output_dir=tmp_path,
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert csv_path.stat().st_size > 0
    assert json_path.stat().st_size > 0


def test_plot_threshold_sensitivity_saves_figure(tmp_path) -> None:
    results = [
        {
            "alpha": 0.01,
            "threshold": 0.98,
            "review_rate": 0.40,
            "empirical_coverage": 0.99,
            "singleton_rate": 0.60,
        },
        {
            "alpha": 0.05,
            "threshold": 0.90,
            "review_rate": 0.25,
            "empirical_coverage": 0.96,
            "singleton_rate": 0.75,
        },
    ]

    output_path = tmp_path / "threshold_sensitivity.png"

    plot_threshold_sensitivity(
        results=results,
        output_path=output_path,
        show_plot=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_experiment_metadata_writes_manifest(tmp_path) -> None:
    metadata_path = export_experiment_metadata(
        output_dir=tmp_path,
        archive_path="example.npz",
        alphas=[0.01, 0.05],
        seed=42,
        calibration_size=100,
        evaluation_size=200,
    )

    assert metadata_path.exists()
    assert metadata_path.stat().st_size > 0


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
        conformal_triage_eval,
        "plot_coverage_vs_workload",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        conformal_triage_eval,
        "plot_threshold_sensitivity",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        conformal_triage_eval,
        "export_conformal_results",
        lambda **kwargs: (
            tmp_path / "conformal_metrics.csv",
            tmp_path / "conformal_metrics.json",
        ),
    )

    monkeypatch.setattr(
        conformal_triage_eval,
        "export_experiment_metadata",
        lambda **kwargs: tmp_path / "metadata.json",
    )

    from experiments.conformal_triage_eval import main

    main(["--archive", str(archive_path)])
