"""Unit tests for :mod:`src.evaluation.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    average_precision,
    roc_auc,
    summarize,
    tpr_at_fpr,
    validate_scores_labels,
)


def test_roc_auc_perfect_separation() -> None:
    """Perfectly separated ranks give AUC = 1.0."""
    scores = np.array([0.1, 0.2, 0.3, 5.0, 6.0])
    labels = np.array([0, 0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_random_is_half() -> None:
    """Uninformative scores with balanced labels approach AUC = 0.5."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=4000)
    labels = rng.integers(0, 2, size=4000)
    assert roc_auc(scores, labels) == pytest.approx(0.5, abs=0.02)


def test_roc_auc_inverted_is_zero() -> None:
    """Inverted ranking gives AUC = 0.0."""
    scores = np.array([5.0, 6.0, 0.1, 0.2, 0.3])
    labels = np.array([0, 0, 1, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_rejects_single_class() -> None:
    """AUC raises when only one class is present."""
    with pytest.raises(ValueError):
        roc_auc(np.array([1.0, 2.0]), np.array([1, 1]))


def test_roc_auc_handles_ties() -> None:
    """All-zero scores with mixed labels produce the random baseline 0.5."""
    assert roc_auc(np.zeros(10), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])) == pytest.approx(0.5)


def test_average_precision_perfect() -> None:
    """All positives ranked first give AP = 1.0."""
    scores = np.array([9.0, 8.0, 0.1, 0.2, 0.3])
    labels = np.array([1, 1, 0, 0, 0])
    assert average_precision(scores, labels) == pytest.approx(1.0)


def test_average_precision_rejects_no_positives() -> None:
    """AP raises with zero positive samples."""
    with pytest.raises(ValueError):
        average_precision(np.array([1.0, 2.0]), np.array([0, 0]))


def test_tpr_at_fpr_selects_operating_point() -> None:
    """Scores with 1 fraud at the top achieve TPR=1.0 at FPR<=0.01."""
    scores = np.array([500.0, *np.arange(200, dtype=float)])
    labels = np.array([1, *np.zeros(200, dtype=int)])
    assert tpr_at_fpr(scores, labels, max_fpr=0.01) == pytest.approx(1.0)


def test_tpr_at_fpr_bad_target_raises() -> None:
    """A target FPR outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        tpr_at_fpr(np.ones(5), np.array([0, 1, 0, 1, 0]), max_fpr=1.5)


def test_validate_rejects_mismatched_lengths() -> None:
    """Length mismatch is rejected."""
    with pytest.raises(ValueError):
        validate_scores_labels(np.array([1.0, 2.0]), np.array([0, 1, 0]))


def test_validate_rejects_empty() -> None:
    """Empty evaluation sets are rejected."""
    with pytest.raises(ValueError):
        validate_scores_labels(np.array([]), np.array([]))


def test_validate_rejects_nan() -> None:
    """NaN scores are rejected."""
    with pytest.raises(ValueError):
        validate_scores_labels(np.array([1.0, np.nan]), np.array([0, 1]))


def test_validate_rejects_non_binary_labels() -> None:
    """Non-binary labels are rejected."""
    with pytest.raises(ValueError):
        validate_scores_labels(np.array([1.0, 2.0]), np.array([0, 2]))


def test_summarize_reports_all_metrics() -> None:
    """Summary contains rocauc / auprc / tpr_at_fpr for a separable toy case."""
    summary = summarize(np.array([0.1, 0.2, 0.3, 5.0, 6.0]), np.array([0, 0, 0, 1, 1]))
    assert summary["rocauc"] == pytest.approx(1.0)
    assert set(summary) == {"rocauc", "auprc", "tpr_at_fpr"}
