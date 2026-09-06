"""Tests for the DAE SHAP explainability experiment."""

# Import necessary libraries and modules
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.shap_explainability import (
    plot_global_shap_importance,
    plot_local_shap,
    score_anomalies_batched,
    select_high_anomaly_fraud,
    select_legitimate_background,
    stratified_sample_indices,
    validate_binary_labels,
)


def test_validate_binary_labels_accepts_binary_labels() -> None:
    validate_binary_labels(np.array([0, 1, 0, 1]))


def test_validate_binary_labels_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_binary_labels(np.array([]))


def test_validate_binary_labels_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        validate_binary_labels(np.array([0, 1, 2]))


def test_validate_binary_labels_rejects_two_dimensional_labels() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_binary_labels(np.array([[0, 1]]))


def test_stratified_sample_indices_is_deterministic() -> None:
    labels = np.array([0] * 20 + [1] * 20)

    first = stratified_sample_indices(
        labels=labels,
        max_samples=10,
        random_state=42,
    )

    second = stratified_sample_indices(
        labels=labels,
        max_samples=10,
        random_state=42,
    )

    assert np.array_equal(first, second)
    assert len(first) == 10
    assert len(np.unique(first)) == 10


def test_stratified_sample_indices_contains_both_classes() -> None:
    labels = np.array([0] * 20 + [1] * 20)

    indices = stratified_sample_indices(
        labels=labels,
        max_samples=10,
        random_state=42,
    )

    selected_labels = labels[indices]

    assert 0 in selected_labels
    assert 1 in selected_labels


def test_stratified_sample_indices_returns_all_when_below_limit() -> None:
    labels = np.array([0, 1, 0, 1])

    indices = stratified_sample_indices(
        labels=labels,
        max_samples=10,
        random_state=42,
    )

    assert np.array_equal(
        indices,
        np.arange(len(labels)),
    )


def test_stratified_sample_indices_rejects_invalid_limit() -> None:
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="positive"):
        stratified_sample_indices(
            labels=labels,
            max_samples=0,
            random_state=42,
        )


def test_select_legitimate_background_only_returns_legitimate() -> None:
    df = pd.DataFrame(
        {
            "feature": np.arange(10, dtype=float),
            "isFraud": [0, 1] * 5,
        }
    )

    background = select_legitimate_background(
        df=df,
        n_background=3,
        random_state=42,
    )

    assert len(background) == 3
    assert (background["isFraud"] == 0).all()


def test_select_legitimate_background_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "feature": np.arange(20, dtype=float),
            "isFraud": [0] * 20,
        }
    )

    first = select_legitimate_background(
        df=df,
        n_background=5,
        random_state=42,
    )

    second = select_legitimate_background(
        df=df,
        n_background=5,
        random_state=42,
    )

    assert first.index.tolist() == second.index.tolist()


def test_select_legitimate_background_rejects_missing_labels() -> None:
    df = pd.DataFrame({"feature": [1.0, 2.0]})

    with pytest.raises(KeyError, match="isFraud"):
        select_legitimate_background(
            df=df,
            n_background=1,
            random_state=42,
        )


def test_select_legitimate_background_rejects_no_legitimate_rows() -> None:
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
            "isFraud": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="No legitimate"):
        select_legitimate_background(
            df=df,
            n_background=1,
            random_state=42,
        )


class DummyAnomalyModel(torch.nn.Module):
    """Small deterministic model exposing the DAE anomaly-score interface."""

    def anomaly_score(
        self,
        x: torch.Tensor,
        l1_gamma: float = 0.4,
        reduction: str = "none",
    ) -> torch.Tensor:
        del l1_gamma

        scores = x.pow(2).sum(dim=-1)

        if reduction == "mean":
            return scores.mean()

        if reduction == "sum":
            return scores.sum()

        return scores


def test_score_anomalies_batched_returns_expected_scores() -> None:
    model = DummyAnomalyModel()

    features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 2.0],
        ]
    )

    scores = score_anomalies_batched(
        model=model,
        features=features,
        l1_gamma=0.4,
        batch_size=2,
    )

    expected = np.array(
        [
            5.0,
            25.0,
            8.0,
        ]
    )

    assert np.allclose(scores, expected)


def test_score_anomalies_batched_rejects_empty_features() -> None:
    model = DummyAnomalyModel()

    with pytest.raises(ValueError, match="non-empty"):
        score_anomalies_batched(
            model=model,
            features=torch.empty((0, 2)),
            l1_gamma=0.4,
        )


def test_score_anomalies_batched_rejects_negative_gamma() -> None:
    model = DummyAnomalyModel()

    with pytest.raises(ValueError, match="non-negative"):
        score_anomalies_batched(
            model=model,
            features=torch.ones((2, 2)),
            l1_gamma=-0.1,
        )


def test_select_high_anomaly_fraud_selects_largest_score() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 10.0],
            "feature_b": [1.0, 2.0, 10.0],
            "isFraud": [0, 1, 1],
            "TransactionID": [100, 101, 102],
        }
    )

    model = DummyAnomalyModel()

    row, score = select_high_anomaly_fraud(
        df=df,
        model=model,
        feature_cols=["feature_a", "feature_b"],
        l1_gamma=0.4,
        device="cpu",
        batch_size=2,
    )

    assert row["TransactionID"] == 102
    assert score == pytest.approx(200.0)


def test_select_high_anomaly_fraud_rejects_no_fraud() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "isFraud": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="No fraudulent"):
        select_high_anomaly_fraud(
            df=df,
            model=DummyAnomalyModel(),
            feature_cols=["feature_a"],
            l1_gamma=0.4,
            device="cpu",
            batch_size=2,
        )


def test_plot_local_shap_saves_figure(tmp_path: Path) -> None:
    output = tmp_path / "local.png"

    plot_local_shap(
        feature_values=np.array([1.0, 2.0, 3.0]),
        shap_values=np.array([0.2, -0.8, 0.4]),
        feature_names=["A", "B", "C"],
        anomaly_score=12.5,
        output_path=output,
        top_k=3,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_global_shap_importance_saves_figure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "global.png"

    shap_values = np.array(
        [
            [0.1, -0.8, 0.3],
            [0.2, -0.4, 0.6],
            [0.3, -0.2, 0.9],
        ]
    )

    plot_global_shap_importance(
        shap_values=shap_values,
        feature_names=["A", "B", "C"],
        output_path=output,
        top_k=3,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
