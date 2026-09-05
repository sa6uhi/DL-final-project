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
from src.training.trainer_utils import EarlyStopping, resolve_device
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT = Path("models/checkpoints/hybrid_gating.pt")


@dataclass(frozen=True)
class GateData:
    """Raw upstream fraud signals and labels for one data split."""

    anomaly_scores: torch.Tensor
    ft_probabilities: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class GateTrainingHistory:
    """Loss history recorded during learned-gate training."""

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


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
    history: GateTrainingHistory | None = None,
) -> None:
    """Save learned gate weights, architecture metadata, and normalizer state.

    Args:
        model: Trained learned hybrid gate.
        normalizer: Fitted anomaly-score normalizer.
        path: Destination checkpoint path.
        history: Optional training-loss history to store in the checkpoint.
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
    if history is not None:
        payload["training_history"] = {
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
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


def train_gate(
    train_data: GateData,
    val_data: GateData,
    config: Config,
    device: str | None = None,
) -> tuple[LearnedHybridGate, PercentileNormalizer]:
    """Train the learned hybrid gate end-to-end and save its checkpoint.

    Args:
        train_data: Training split signals and fraud labels.
        val_data: Validation split signals and fraud labels, used for
            early stopping and picking the best model weights.
        config: Central project configuration.
        device: Device to train on; auto-detected when ``None``.

    Returns:
        Tuple of the trained gate (best weights restored) and the
        normalizer fitted on the training split.

    Raises:
        ValueError: If the input data fails validation, or if the built
            gate features do not match the configured ``input_dim``.
    """
    validate_gate_data(train_data, name="train_data")
    validate_gate_data(val_data, name="val_data")

    training_config = config.hybrid_gating.learned.training

    seed = int(config.seed)

    epochs = int(training_config.epochs)
    batch_size = int(training_config.batch_size)
    learning_rate = float(training_config.lr)
    weight_decay = float(training_config.weight_decay)
    patience = int(training_config.early_stopping_patience)
    min_delta = float(training_config.min_delta)

    if epochs <= 0:
        raise ValueError("Training epochs must be positive")

    if batch_size <= 0:
        raise ValueError("Training batch_size must be positive")

    if learning_rate <= 0.0:
        raise ValueError("Training learning rate must be positive")

    if weight_decay < 0.0:
        raise ValueError("Training weight_decay must be non-negative")

    if patience <= 0:
        raise ValueError("Early stopping patience must be positive")

    if min_delta < 0.0:
        raise ValueError("Early stopping min_delta must be non-negative")

    torch.manual_seed(seed)

    resolved_device = resolve_device(device)

    normalizer = PercentileNormalizer(
        percentile=float(config.hybrid_gating.learned.normalize_percentile)
    )

    train_features = build_gate_features(
        anomaly_scores=train_data.anomaly_scores,
        transformer_probabilities=train_data.ft_probabilities,
        normalizer=normalizer,
        fit_normalizer=True,
    )

    val_features = build_gate_features(
        anomaly_scores=val_data.anomaly_scores,
        transformer_probabilities=val_data.ft_probabilities,
        normalizer=normalizer,
        fit_normalizer=False,
    )

    model = LearnedHybridGate(
        input_dim=int(config.hybrid_gating.learned.input_dim),
        hidden_dims=[int(dim) for dim in config.hybrid_gating.learned.hidden_dims],
        dropout=float(config.hybrid_gating.learned.dropout),
    ).to(resolved_device)

    if train_features.shape[1] != model.input_dim:
        raise ValueError(
            "Generated training gate feature count does not match "
            "configured input_dim: "
            f"{train_features.shape[1]} != {model.input_dim}"
        )

    if val_features.shape[1] != model.input_dim:
        raise ValueError(
            "Generated validation gate feature count does not match "
            "configured input_dim: "
            f"{val_features.shape[1]} != {model.input_dim}"
        )

    train_loader = make_gate_loader(
        features=train_features,
        labels=train_data.labels,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_gate_loader(
        features=val_features,
        labels=val_data.labels,
        batch_size=batch_size,
        shuffle=False,
    )

    loss_fn = nn.BCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_epoch = 0

    # Training loop
    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for features, labels in train_loader:
            features = features.to(resolved_device)
            labels = labels.to(resolved_device)

            optimizer.zero_grad()

            predictions = model(features)

            loss = loss_fn(
                predictions,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_samples = features.shape[0]
            total_train_loss += loss.item() * batch_samples
            total_train_samples += batch_samples

        train_loss = total_train_loss / total_train_samples

        # Validation
        val_loss = evaluate_gate_loss(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=resolved_device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopping.update(
            val_loss,
            model,
        )

        if early_stopping.best_score == val_loss:
            best_epoch = epoch + 1

        logger.info(
            "Gate epoch %d/%d - train loss: %.6f - validation loss: %.6f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
        )

        if early_stopping.should_stop:
            logger.info(
                "Early stopping triggered at epoch %d",
                epoch + 1,
            )
            break

    # Restore the best model weights
    early_stopping.restore(model)

    history = GateTrainingHistory(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=float(early_stopping.best_score),
    )

    model.eval()

    # Save trained gate checkpoint
    checkpoint_path = Path(config.hybrid_gating.learned.checkpoint_path)

    save_checkpoint(
        model=model,
        normalizer=normalizer,
        path=checkpoint_path,
        history=history,
    )

    logger.info(
        "Finished learned gate training. " "Best validation loss: %.6f",
        early_stopping.best_score,
    )

    return model, normalizer
