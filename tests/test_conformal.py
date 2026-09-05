"""Tests for split-conformal fraud prediction utilities."""

# Import standard libraries
import pytest
import torch

from src.uncertainty.conformal_predictor import (
    SplitConformalPredictor,
    binary_nonconformity_scores,
    conformal_quantile,
    prediction_set,
    triage_decision,
)


def test_binary_nonconformity_scores() -> None:
    probabilities = torch.tensor([0.1, 0.8, 0.3, 0.9])
    labels = torch.tensor([0, 1, 1, 0])

    scores = binary_nonconformity_scores(probabilities, labels)

    expected = torch.tensor([0.1, 0.2, 0.7, 0.9])

    assert torch.allclose(scores, expected)


def test_binary_nonconformity_rejects_mismatched_shapes() -> None:
    probabilities = torch.tensor([0.1, 0.8])
    labels = torch.tensor([0])

    with pytest.raises(ValueError, match="matching shapes"):
        binary_nonconformity_scores(probabilities, labels)


def test_binary_nonconformity_rejects_invalid_probabilities() -> None:
    probabilities = torch.tensor([0.1, 1.2])
    labels = torch.tensor([0, 1])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_nonconformity_scores(probabilities, labels)


def test_binary_nonconformity_rejects_invalid_labels() -> None:
    probabilities = torch.tensor([0.1, 0.8])
    labels = torch.tensor([0, 2])

    with pytest.raises(ValueError, match="only 0 or 1"):
        binary_nonconformity_scores(probabilities, labels)


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    threshold = conformal_quantile(scores, alpha=0.2)

    assert threshold == pytest.approx(0.5)


def test_conformal_quantile_rejects_invalid_alpha() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile(scores, alpha=0.0)


def test_conformal_quantile_rejects_empty_scores() -> None:
    scores = torch.tensor([])

    with pytest.raises(ValueError, match="must not be empty"):
        conformal_quantile(scores, alpha=0.1)


def test_prediction_set_auto_approve() -> None:
    result = prediction_set(
        fraud_probability=0.10,
        threshold=0.30,
    )

    assert result == frozenset({0})
    assert triage_decision(result) == "auto_approve"


def test_prediction_set_auto_block() -> None:
    result = prediction_set(
        fraud_probability=0.90,
        threshold=0.30,
    )

    assert result == frozenset({1})
    assert triage_decision(result) == "auto_block"


def test_prediction_set_human_review() -> None:
    result = prediction_set(
        fraud_probability=0.50,
        threshold=0.60,
    )

    assert result == frozenset({0, 1})
    assert triage_decision(result) == "human_review"


def test_prediction_set_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="fraud_probability"):
        prediction_set(
            fraud_probability=1.2,
            threshold=0.5,
        )


def test_split_conformal_predictor_fit_and_predict() -> None:
    predictor = SplitConformalPredictor(alpha=0.2)

    calibration_probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9])
    calibration_labels = torch.tensor([0, 0, 1, 1])

    predictor.fit(
        fraud_probabilities=calibration_probabilities,
        labels=calibration_labels,
    )

    assert predictor.is_fitted
    assert predictor.threshold is not None

    result = predictor.predict_set(0.1)

    assert result == frozenset({0})
    assert predictor.predict_triage(0.1) == "auto_approve"


def test_split_conformal_predictor_requires_fit() -> None:
    predictor = SplitConformalPredictor(alpha=0.1)

    with pytest.raises(RuntimeError, match="must be fitted"):
        predictor.predict_set(0.5)


def test_split_conformal_predictor_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        SplitConformalPredictor(alpha=1.0)


def test_triage_decision_empty_set_routes_to_human_review() -> None:
    assert triage_decision(frozenset()) == "human_review"


def test_conformal_quantile_returns_one_when_rank_exceeds_sample_size() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3])

    threshold = conformal_quantile(scores, alpha=0.01)

    assert threshold == 1.0


def test_conformal_quantile_rejects_scores_outside_unit_interval() -> None:
    scores = torch.tensor([0.1, 1.2])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        conformal_quantile(scores, alpha=0.1)
