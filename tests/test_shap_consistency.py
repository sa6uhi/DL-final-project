"""Tests for DAE SHAP explanation consistency utilities."""

# Import necessary modules and libraries
from pathlib import Path

import numpy as np
import pytest

from experiments.shap_consistency import (
    compare_seed_importances,
    export_consistency_results,
    mean_absolute_shap_importance,
    plot_consistency,
    spearman_importance_correlation,
    summarize_consistency,
    top_feature_indices,
    top_k_jaccard,
)


def test_mean_absolute_shap_importance() -> None:
    shap_values = np.array(
        [
            [1.0, -2.0, 3.0],
            [-3.0, 4.0, -1.0],
        ]
    )

    importance = mean_absolute_shap_importance(shap_values)

    assert np.allclose(
        importance,
        np.array([2.0, 3.0, 2.0]),
    )


def test_mean_absolute_shap_importance_rejects_nonfinite() -> None:
    shap_values = np.array(
        [
            [1.0, np.nan],
            [2.0, 3.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        mean_absolute_shap_importance(shap_values)


def test_top_feature_indices_returns_largest_features() -> None:
    importance = np.array([0.1, 0.8, 0.4, 0.7])

    indices = top_feature_indices(
        importance,
        top_k=2,
    )

    assert indices.tolist() == [1, 3]


def test_top_k_jaccard_identical_sets() -> None:
    first = np.array([0.9, 0.8, 0.1, 0.0])

    second = np.array([0.8, 0.9, 0.2, 0.1])

    similarity = top_k_jaccard(
        first,
        second,
        top_k=2,
    )

    assert similarity == pytest.approx(1.0)


def test_top_k_jaccard_partial_overlap() -> None:
    first = np.array([0.9, 0.8, 0.2, 0.1])

    second = np.array([0.9, 0.1, 0.8, 0.2])

    similarity = top_k_jaccard(
        first,
        second,
        top_k=2,
    )

    assert similarity == pytest.approx(1.0 / 3.0)


def test_spearman_importance_correlation_identical_ranking() -> None:
    first = np.array([0.1, 0.2, 0.3, 0.4])

    second = np.array([1.0, 2.0, 3.0, 4.0])

    correlation = spearman_importance_correlation(
        first,
        second,
    )

    assert correlation == pytest.approx(1.0)


def test_spearman_importance_correlation_reversed_ranking() -> None:
    first = np.array([0.1, 0.2, 0.3, 0.4])

    second = np.array([4.0, 3.0, 2.0, 1.0])

    correlation = spearman_importance_correlation(
        first,
        second,
    )

    assert correlation == pytest.approx(-1.0)


def test_compare_seed_importances_returns_all_pairs() -> None:
    importances = {
        42: np.array([0.9, 0.8, 0.2]),
        123: np.array([0.8, 0.7, 0.3]),
        2026: np.array([0.7, 0.9, 0.1]),
    }

    comparisons = compare_seed_importances(
        importances,
        top_k=2,
    )

    assert len(comparisons) == 3

    pairs = {
        (
            row["seed_a"],
            row["seed_b"],
        )
        for row in comparisons
    }

    assert pairs == {
        (42, 123),
        (42, 2026),
        (123, 2026),
    }


def test_compare_seed_importances_requires_two_seeds() -> None:
    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        compare_seed_importances(
            {42: np.array([0.1, 0.2, 0.3])},
            top_k=2,
        )


def test_summarize_consistency() -> None:
    comparisons = [
        {
            "seed_a": 42,
            "seed_b": 123,
            "top_k_jaccard": 0.8,
            "spearman_correlation": 0.9,
        },
        {
            "seed_a": 42,
            "seed_b": 2026,
            "top_k_jaccard": 0.6,
            "spearman_correlation": 0.7,
        },
    ]

    summary = summarize_consistency(comparisons)

    assert summary["mean_top_k_jaccard"] == pytest.approx(0.7)

    assert summary["min_top_k_jaccard"] == pytest.approx(0.6)

    assert summary["mean_spearman_correlation"] == pytest.approx(0.8)

    assert summary["min_spearman_correlation"] == pytest.approx(0.7)


def test_plot_consistency_saves_figure(
    tmp_path: Path,
) -> None:
    comparisons = [
        {
            "seed_a": 42,
            "seed_b": 123,
            "top_k_jaccard": 0.8,
            "spearman_correlation": 0.9,
        },
        {
            "seed_a": 42,
            "seed_b": 2026,
            "top_k_jaccard": 0.6,
            "spearman_correlation": 0.75,
        },
    ]

    output = tmp_path / "consistency.png"

    plot_consistency(
        comparisons,
        output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_export_consistency_results_writes_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "consistency.json"

    payload = {
        "experiment": "dae_shap_consistency",
        "summary": {
            "mean_top_k_jaccard": 0.75,
            "mean_spearman_correlation": 0.9,
        },
    }

    export_consistency_results(
        payload,
        output,
    )

    assert output.is_file()

    content = output.read_text(encoding="utf-8")

    assert "dae_shap_consistency" in content
    assert "mean_top_k_jaccard" in content
