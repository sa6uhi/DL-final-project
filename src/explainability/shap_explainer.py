"""SHAP-based explanation utilities for fraud predictions."""

# Import necessary libraries
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class DAEAnomalyScoreWrapper(nn.Module):
    """Differentiable wrapper exposing a DAE anomaly score for SHAP."""

    def __init__(self, model: nn.Module, l1_gamma: float = 0.4) -> None:
        super().__init__()

        if l1_gamma < 0:
            raise ValueError("l1_gamma must be non-negative")

        self.model = model
        self.l1_gamma = l1_gamma
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return differentiable per-sample DAE anomaly scores."""
        reconstruction = self.model(x)
        residual = x - reconstruction

        scores = residual.pow(2).sum(dim=-1)
        scores = scores + self.l1_gamma * residual.abs().sum(dim=-1)

        return scores.unsqueeze(-1)


def compute_shap_values(
    model: nn.Module,
    background: torch.Tensor,
    samples: torch.Tensor,
    l1_gamma: float = 0.4,
) -> np.ndarray:
    """Compute SHAP values for DAE anomaly scores.

    Args:
        model: Trained denoising autoencoder.
        background: Background feature tensor with shape
            ``(n_background, n_features)``.
        samples: Samples to explain with shape
            ``(n_samples, n_features)``.
        l1_gamma: L1 contribution to the anomaly score.

    Returns:
        SHAP attribution matrix with shape
        ``(n_samples, n_features)``.

    Raises:
        ValueError: If background or samples have invalid shapes.
    """
    if background.ndim != 2 or samples.ndim != 2:
        raise ValueError("background and samples must be two-dimensional")

    if background.shape[1] != samples.shape[1]:
        raise ValueError("background and samples must have the same feature count")

    if background.shape[0] == 0 or samples.shape[0] == 0:
        raise ValueError("background and samples must not be empty")

    if not torch.isfinite(background).all() or not torch.isfinite(samples).all():
        raise ValueError("background and samples must contain only finite values")

    import shap

    wrapped_model = DAEAnomalyScoreWrapper(
        model=model,
        l1_gamma=l1_gamma,
    )

    explainer = shap.GradientExplainer(
        wrapped_model,
        background,
    )

    shap_values = explainer.shap_values(samples)

    values = np.asarray(shap_values)

    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]

    return values


def rank_feature_drivers(
    features: list[float],
    shap_values: np.ndarray,
    top_k: int = 5,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank the strongest SHAP feature drivers by absolute attribution.

    Args:
        features: Original transaction feature values.
        shap_values: One-dimensional SHAP attribution vector.
        top_k: Maximum number of drivers to return.
        feature_names: Optional names corresponding to the feature vector.

    Returns:
        Driver dictionaries ordered by descending absolute SHAP magnitude.

    Raises:
        ValueError: If inputs are empty, incompatible, or invalid.
    """
    values = np.asarray(features, dtype=np.float64)
    attributions = np.asarray(shap_values, dtype=np.float64)

    if values.ndim != 1 or attributions.ndim != 1:
        raise ValueError("features and shap_values must be one-dimensional")

    if values.size == 0:
        raise ValueError("features must not be empty")

    if values.shape != attributions.shape:
        raise ValueError("features and shap_values must have matching shapes")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if not np.isfinite(values).all() or not np.isfinite(attributions).all():
        raise ValueError("features and shap_values must contain only finite values")

    if feature_names is None:
        names = [f"feature_{idx}" for idx in range(values.size)]
    else:
        if len(feature_names) != values.size:
            raise ValueError("feature_names must match the feature count")
        names = feature_names

    top_indices = np.argsort(np.abs(attributions))[::-1][:top_k]

    return [
        {
            "feature_name": names[idx],
            "attribution": float(attributions[idx]),
            "value": float(values[idx]),
        }
        for idx in top_indices
    ]


# Define the function to explain a transaction using SHAP
def explain_transaction(
    features: list[float],
    model: nn.Module | None = None,
    background: torch.Tensor | None = None,
    top_k: int = 5,
    ft_probability: float | None = None,
    l1_gamma: float = 0.4,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the top feature drivers for one transaction.

    Args:
        features: Transaction feature vector.
        top_k: Maximum number of feature drivers to return.
        ft_probability: Optional FT-Transformer fraud probability.

    Returns:
        Ranked feature attribution dictionaries.

    Raises:
        ValueError: If inputs are invalid.
        ImportError: If SHAP is unavailable.
    """
    if not features:
        raise ValueError("features must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    values = np.asarray(features, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("features must be one-dimensional")

    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values")

    if ft_probability is not None and not 0.0 <= ft_probability <= 1.0:
        raise ValueError("ft_probability must be in [0, 1]")

    try:
        import shap  # noqa: F401
    except ImportError as exc:
        raise ImportError("SHAP is required for SHAP explanations") from exc

    if model is None:
        raise ValueError("model is required for SHAP explanations")

    if background is None:
        raise ValueError("background is required for SHAP explanations")

    sample = torch.as_tensor(
        values.reshape(1, -1),
        dtype=torch.float32,
    )

    shap_values = compute_shap_values(
        model=model,
        background=background,
        samples=sample,
        l1_gamma=l1_gamma,
    )

    return rank_feature_drivers(
        features=features,
        shap_values=shap_values[0],
        top_k=top_k,
        feature_names=feature_names,
    )
