"""Tests for learned hybrid gate training utilities."""

# Importing necessary libraries
from __future__ import annotations
from pathlib import Path

import torch

from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.training.train_hybrid_gating import load_checkpoint, save_checkpoint


# Test functions
def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Saved and restored gate should produce identical predictions."""
    model = LearnedHybridGate(
        input_dim=2,
        hidden_dims=[16, 8],
        dropout=0.1,
    )
    model.eval()

    normalizer = PercentileNormalizer(percentile=99.0).fit(
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    )

    checkpoint_path = tmp_path / "hybrid_gating.pt"

    features = torch.tensor(
        [
            [0.2, 0.8],
            [0.6, 0.3],
        ],
        dtype=torch.float32,
    )

    expected = model(features)

    save_checkpoint(model, normalizer, checkpoint_path)
    restored_model, restored_normalizer = load_checkpoint(checkpoint_path)

    actual = restored_model(features)

    assert torch.allclose(expected, actual)
    assert restored_normalizer.state_dict() == normalizer.state_dict()
