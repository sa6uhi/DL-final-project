"""Training utilities for the learned hybrid fraud gate."""

# Importing necessary libraries
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT = Path("models/checkpoints/hybrid_gating.pt")


@dataclass(frozen=True)
class GateData:
    """Raw upstream fraud signals and labels for one data split."""

    anomaly_scores: torch.Tensor
    ft_probabilities: torch.Tensor
    labels: torch.Tensor


def validate_gate_data(data: GateData, name: str = "gate_data") -> None:
    """Validate upstream signals and labels used to train the learned gate.

    Args:
        data: Gate inputs and binary fraud labels.
        name: Human-readable split name used in error messages.

    Raises:
        ValueError: If tensors are empty, not one-dimensional, have different
            lengths, contain non-finite values, or contain invalid labels or
            probabilities.
    """
    tensors = {
        "anomaly_scores": data.anomaly_scores,
        "ft_probabilities": data.ft_probabilities,
        "labels": data.labels,
    }

    for tensor_name, tensor in tensors.items():
        if tensor.ndim != 1:
            raise ValueError(f"{name}.{tensor_name} must be one-dimensional")

        if tensor.numel() == 0:
            raise ValueError(f"{name}.{tensor_name} must not be empty")

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}.{tensor_name} must contain only finite values")

    if not (data.anomaly_scores.shape == data.ft_probabilities.shape == data.labels.shape):
        raise ValueError(f"{name} tensors must have matching shapes")

    if ((data.ft_probabilities < 0.0) | (data.ft_probabilities > 1.0)).any():
        raise ValueError(f"{name}.ft_probabilities must be in [0, 1]")

    if not ((data.labels == 0) | (data.labels == 1)).all():
        raise ValueError(f"{name}.labels must contain only 0 or 1")


def build_gate_features(
    anomaly_scores: torch.Tensor,
    transformer_probabilities: torch.Tensor,
    normalizer: PercentileNormalizer,
    fit_normalizer: bool = False,
) -> torch.Tensor:
    """Build the two-feature input matrix for the learned gate.

    Args:
        anomaly_scores: Raw DAE anomaly scores of shape ``(n_samples,)``.
        transformer_probabilities: FT-CAT fraud probabilities of shape
            ``(n_samples,)``.
        normalizer: Percentile normalizer for anomaly scores.
        fit_normalizer: Whether to fit the normalizer before transforming.

    Returns:
        Tensor of shape ``(n_samples, 2)`` containing normalized anomaly
        scores and Transformer fraud probabilities.

    Raises:
        ValueError: If inputs have invalid shapes, lengths, values, or
            probability ranges.
    """
    if anomaly_scores.dim() != 1 or transformer_probabilities.dim() != 1:
        raise ValueError("Gate inputs must be one-dimensional tensors")

    if anomaly_scores.shape != transformer_probabilities.shape:
        raise ValueError("Anomaly scores and Transformer probabilities must have matching shapes")

    if anomaly_scores.numel() == 0:
        raise ValueError("Gate inputs must not be empty")

    if not torch.isfinite(anomaly_scores).all():
        raise ValueError("Anomaly scores must contain only finite values")

    if not torch.isfinite(transformer_probabilities).all():
        raise ValueError("Transformer probabilities must contain only finite values")

    if ((transformer_probabilities < 0.0) | (transformer_probabilities > 1.0)).any():
        raise ValueError("Transformer probabilities must be in [0, 1]")

    if fit_normalizer:
        normalized_anomaly = normalizer.fit_transform(anomaly_scores)
    else:
        normalized_anomaly = normalizer.transform(anomaly_scores)

    return torch.stack(
        (normalized_anomaly, transformer_probabilities),
        dim=1,
    )


def make_gate_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for learned-gate training or validation.

    Args:
        features: Gate input features with shape ``(n_samples, n_features)``.
        labels: Binary fraud labels with shape ``(n_samples,)``.
        batch_size: Number of samples per mini-batch.
        shuffle: Whether to shuffle samples before each epoch.

    Returns:
        DataLoader yielding ``(features, labels)`` mini-batches.

    Raises:
        ValueError: If shapes are invalid, lengths differ, inputs are empty,
            or batch_size is not positive.
    """
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional tensor")

    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional tensor")

    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must contain the same number of samples")

    if features.shape[0] == 0:
        raise ValueError("features and labels must not be empty")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset = TensorDataset(features, labels.float())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_gate_loss(
    model: LearnedHybridGate,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> float:
    """Calculate average loss on held-out gate data.

    Args:
        model: Learned hybrid gate being evaluated.
        data_loader: Validation data loader.
        loss_fn: Binary classification loss function.
        device: Device on which evaluation is performed.

    Returns:
        Average loss across all validation samples.

    Raises:
        ValueError: If the data loader contains no samples.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            predictions = model(features)
            loss = loss_fn(predictions, labels)

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation data loader must not be empty")

    return total_loss / total_samples


def save_checkpoint(
    model: LearnedHybridGate,
    normalizer: PercentileNormalizer,
    path: str | Path,
) -> None:
    """Save learned gate weights, architecture metadata, and normalizer state.

    Args:
        model: Trained learned hybrid gate.
        normalizer: Fitted anomaly-score normalizer.
        path: Destination checkpoint path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "meta": {
            "input_dim": model.input_dim,
            "hidden_dims": model.hidden_dims,
            "dropout": model.dropout,
        },
        "normalizer": normalizer.state_dict(),
    }

    torch.save(payload, out)
    logger.info("Saved learned hybrid gate checkpoint to %s", out)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[LearnedHybridGate, PercentileNormalizer]:
    """Restore a learned hybrid gate and its fitted normalizer.

    Args:
        path: Checkpoint file created by :func:`save_checkpoint`.
        device: Target device for the restored model.

    Returns:
        Tuple containing the restored model and fitted normalizer.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        KeyError: If required checkpoint metadata is missing.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload: dict[str, Any] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    meta = payload.get("meta")
    normalizer_state = payload.get("normalizer")

    if meta is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has no architecture metadata")
    if normalizer_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} has no normalizer state")

    model = LearnedHybridGate(
        input_dim=int(meta["input_dim"]),
        hidden_dims=[int(dim) for dim in meta["hidden_dims"]],
        dropout=float(meta["dropout"]),
    )

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    normalizer = PercentileNormalizer.from_state_dict(normalizer_state)

    return model, normalizer
