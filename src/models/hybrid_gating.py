"""Hybrid gating scorer fusing DAE residuals with FT-Transformer posteriors.

The gating layer combines two independent fraud signals into a single
calibrated risk score:

.. math::

    Score_{final} = \\alpha \\cdot \\mathrm{Normalize}(S_{anomaly})
                    + (1 - \\alpha) \\cdot P_{FT}

where :math:`S_{anomaly}` is the autoencoder reconstruction residual and
:math:`P_{FT}` is the supervised transformer fraud probability. Percentile
normalization makes the two signals scale-comparable before fusion.
"""

from __future__ import annotations

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PercentileNormalizer:
    """Clips scores at a configurable percentile and scales to ``[0, 1]``.

    The normalizer is fit on a calibration score distribution (e.g. DAE
    residuals over the validation split) and can be serialized so serving
    uses the exact same transformation as evaluation.

    Args:
        percentile: Upper percentile (in ``[0, 100]``) used as the clip cap.
        epsilon: Additive floor to avoid division by zero.
    """

    def __init__(self, percentile: float = 99.9, epsilon: float = 1.0e-8) -> None:
        """Initialize the normalizer."""
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.percentile = float(percentile)
        self.epsilon = float(epsilon)
        self.cap: float | None = None

    def fit(self, scores: torch.Tensor) -> "PercentileNormalizer":
        """Estimate the clip cap from a calibration score batch.

        Args:
            scores: 1D tensor of calibration scores.

        Returns:
            The fitted normalizer (self).

        Raises:
            ValueError: If scores are empty or contain NaN values.
        """
        if scores.numel() == 0:
            raise ValueError("Cannot fit normalizer on an empty score batch")
        if torch.isnan(scores).any():
            raise ValueError("NaN scores cannot be used for calibration")
        self.cap = float(torch.quantile(scores, self.percentile / 100.0).clamp_min(self.epsilon))
        logger.info("Fitted percentile normalizer: cap=%.6f (p%s)", self.cap, self.percentile)
        return self

    def transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Clip and scale scores into ``[0, 1]``.

        Args:
            scores: 1D or 2D tensor of raw scores.

        Returns:
            Tensor of the same shape in ``[0, 1]``.

        Raises:
            ValueError: If the normalizer has not been fitted or NaN input.
        """
        if self.cap is None:
            raise ValueError("PercentileNormalizer must be fitted before transform")
        if torch.isnan(scores).any():
            raise ValueError("NaN scores cannot be transformed")
        return (scores.clamp(max=self.cap) / self.cap).clamp(0.0, 1.0)

    def fit_transform(self, scores: torch.Tensor) -> torch.Tensor:
        """Fit on ``scores`` and return the transformed values.

        Args:
            scores: 1D tensor of calibration scores.

        Returns:
            Normalized scores of the same shape.
        """
        return self.fit(scores).transform(scores)

    def state_dict(self) -> dict[str, float]:
        """Serializable state for checkpoint/restore.

        Returns:
            Dict mapping parameter names to values.
        """
        return {"percentile": self.percentile, "epsilon": self.epsilon, "cap": self.cap if self.cap is not None else -1.0}

    @classmethod
    def from_state_dict(cls, state: dict[str, float]) -> "PercentileNormalizer":
        """Reconstruct a normalizer from its state.

        Args:
            state: Dictionary produced by :meth:`state_dict`.

        Returns:
            The restored normalizer (fitted).
        """
        normalizer = cls(percentile=float(state["percentile"]), epsilon=float(state["epsilon"]))
        cap = float(state["cap"])
        normalizer.cap = None if cap < 0.0 else cap
        return normalizer


class HybridGate:
    """Weighted fusion of anomaly residuals and supervised probabilities.

    Args:
        alpha: Weight assigned to the normalized anomaly signal in
            ``[0, 1]``; the remaining mass goes to the transformer score.
        normalizer: Fitted :class:`PercentileNormalizer` applied to the
            anomaly signal before fusion.
    """

    def __init__(self, alpha: float, normalizer: PercentileNormalizer | None = None) -> None:
        """Initialize the gating scorer."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if normalizer is not None and normalizer.cap is None:
            raise ValueError("normalizer must be fitted before use in HybridGate")
        self.alpha = float(alpha)
        self.normalizer = normalizer if normalizer is not None else PercentileNormalizer()

    def fuse(
        self, anomaly_scores: torch.Tensor, probability_ft: torch.Tensor
    ) -> torch.Tensor:
        """Compute the blended final risk score.

        Args:
            anomaly_scores: DAE reconstruction residuals per sample.
            probability_ft: FT-Transformer fraud probabilities per sample.

        Returns:
            Blended scores of the same flat shape as the inputs.

        Raises:
            ValueError: If inputs are not shape-compatible or NaN.
            ValueError: If probabilities fall outside ``[0, 1]``.
        """
        scores = anomaly_scores.reshape(-1)
        probs = probability_ft.reshape(-1)
        if scores.shape != probs.shape:
            raise ValueError(
                f"Shape mismatch: anomaly {tuple(anomaly_scores.shape)} vs "
                f"prob {tuple(probability_ft.shape)}"
            )
        if torch.isnan(scores).any() or torch.isnan(probs).any():
            raise ValueError("NaN inputs cannot be fused")
        if ((probs < 0.0) | (probs > 1.0)).any():
            raise ValueError("probability_ft must be within [0, 1]")
        if self.normalizer.cap is None:
            raise ValueError("HybridGate normalizer must be fitted before fusion")
        normalized = self.normalizer.transform(scores)
        fused = self.alpha * normalized + (1.0 - self.alpha) * probs
        return fused.clamp(0.0, 1.0)

    def __call__(self, anomaly_scores: torch.Tensor, probability_ft: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`fuse` so the gate is callable."""
        return self.fuse(anomaly_scores, probability_ft)
