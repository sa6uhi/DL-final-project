"""Training utilities for the learned hybrid fraud gate."""

# Importing necessary libraries
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT = Path("models/checkpoints/hybrid_gating.pt")

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