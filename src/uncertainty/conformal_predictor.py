"""Split-conformal prediction utilities for fraud triage."""

# Import necessary libraries
from __future__ import annotations

import torch


# Define the function to compute nonconformity scores for binary fraud labels
def binary_nonconformity_scores(
    fraud_probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute nonconformity scores for observed binary fraud labels.

    The score is one minus the probability assigned to the observed
    class. Lower scores therefore indicate predictions that conform
    more closely to the observed label.

    Args:
        fraud_probabilities: Fraud-class probabilities with shape
            ``(n_samples,)`` and values in ``[0, 1]``.
        labels: Binary labels with shape ``(n_samples,)``.

    Returns:
        One nonconformity score per sample.

    Raises:
        ValueError: If inputs have invalid shapes, lengths, values,
            or labels.
    """
    if fraud_probabilities.ndim != 1 or labels.ndim != 1:
        raise ValueError("fraud_probabilities and labels must be one-dimensional")

    if fraud_probabilities.shape != labels.shape:
        raise ValueError("fraud_probabilities and labels must have matching shapes")

    if fraud_probabilities.numel() == 0:
        raise ValueError("fraud_probabilities and labels must not be empty")

    if not torch.isfinite(fraud_probabilities).all():
        raise ValueError("fraud_probabilities must contain only finite values")

    if ((fraud_probabilities < 0.0) | (fraud_probabilities > 1.0)).any():
        raise ValueError("fraud_probabilities must be in [0, 1]")

    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must contain only 0 or 1")

    true_class_probabilities = torch.where(
        labels == 1,
        fraud_probabilities,
        1.0 - fraud_probabilities,
    )

    return 1.0 - true_class_probabilities


def conformal_quantile(
    calibration_scores: torch.Tensor,
    alpha: float,
) -> float:
    """Compute the split-conformal finite-sample calibration threshold.

    Args:
        calibration_scores: One-dimensional nonconformity scores from
            a held-out calibration split.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Finite-sample split-conformal threshold.

    Raises:
        ValueError: If the calibration scores are invalid or alpha is
            outside ``(0, 1)``.
    """
    if calibration_scores.ndim != 1:
        raise ValueError("calibration_scores must be one-dimensional")

    if calibration_scores.numel() == 0:
        raise ValueError("calibration_scores must not be empty")

    if not torch.isfinite(calibration_scores).all():
        raise ValueError("calibration_scores must contain only finite values")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    n = calibration_scores.numel()

    rank = int(torch.ceil(torch.tensor((n + 1) * (1.0 - alpha))).item())
    rank = min(rank, n)

    sorted_scores = torch.sort(calibration_scores).values

    return float(sorted_scores[rank - 1].item())


def prediction_set(
    fraud_probability: float,
    threshold: float,
) -> frozenset[int]:
    """Build a binary conformal prediction set for one fraud probability.

    Args:
        fraud_probability: Predicted probability of the fraud class.
        threshold: Calibrated nonconformity threshold.

    Returns:
        A prediction set containing class ``0``, class ``1``, or both.

    Raises:
        ValueError: If probability or threshold is outside ``[0, 1]``.
    """
    if not 0.0 <= fraud_probability <= 1.0:
        raise ValueError("fraud_probability must be in [0, 1]")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    class_zero_score = fraud_probability
    class_one_score = 1.0 - fraud_probability

    labels: set[int] = set()

    if class_zero_score <= threshold:
        labels.add(0)

    if class_one_score <= threshold:
        labels.add(1)

    return frozenset(labels)


def triage_decision(prediction: frozenset[int]) -> str:
    """Map a conformal prediction set to an operational fraud decision."""
    if prediction == frozenset({0}):
        return "auto_approve"

    if prediction == frozenset({1}):
        return "auto_block"

    if prediction == frozenset({0, 1}):
        return "human_review"

    raise ValueError("prediction set must be {0}, {1}, or {0, 1}")


class SplitConformalPredictor:
    """Binary split-conformal predictor for fraud triage."""

    def __init__(self, alpha: float = 0.01) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        self.alpha = alpha
        self.threshold: float | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether calibration has been completed."""
        return self.threshold is not None

    def fit(
        self,
        fraud_probabilities: torch.Tensor,
        labels: torch.Tensor,
    ) -> "SplitConformalPredictor":
        """Calibrate the predictor on a held-out calibration split."""
        scores = binary_nonconformity_scores(
            fraud_probabilities=fraud_probabilities,
            labels=labels,
        )

        self.threshold = conformal_quantile(
            calibration_scores=scores,
            alpha=self.alpha,
        )

        return self

    def predict_set(
        self,
        fraud_probability: float,
    ) -> frozenset[int]:
        """Return the conformal prediction set for one transaction."""
        if self.threshold is None:
            raise RuntimeError("SplitConformalPredictor must be fitted before prediction")

        return prediction_set(
            fraud_probability=fraud_probability,
            threshold=self.threshold,
        )

    def predict_triage(
        self,
        fraud_probability: float,
    ) -> str:
        """Return the operational triage decision for one transaction."""
        result = self.predict_set(fraud_probability)

        return triage_decision(result)
