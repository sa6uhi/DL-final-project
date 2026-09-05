"""Tests for learned hybrid gate training utilities."""

# Importing necessary libraries
from __future__ import annotations
from pathlib import Path

import torch

from src.utils.config import Config
from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.training.train_hybrid_gating import (
    GateData,
    build_gate_features,
    evaluate_gate_loss,
    load_checkpoint,
    make_gate_loader,
    save_checkpoint,
    validate_gate_data,
    train_gate,
)


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


def test_validate_gate_data_accepts_valid_input() -> None:
    """Valid gate data should pass validation without errors."""
    data = GateData(
        anomaly_scores=torch.tensor([0.1, 0.2, 0.3]),
        ft_probabilities=torch.tensor([0.2, 0.5, 0.8]),
        labels=torch.tensor([0, 1, 0]),
    )

    validate_gate_data(data)


def test_validate_gate_data_rejects_mismatched_shapes() -> None:
    """Gate data tensors must contain the same number of samples."""
    data = GateData(
        anomaly_scores=torch.tensor([0.1, 0.2]),
        ft_probabilities=torch.tensor([0.3]),
        labels=torch.tensor([0, 1]),
    )

    try:
        validate_gate_data(data)
    except ValueError as exc:
        assert "matching shapes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")


def test_validate_gate_data_rejects_invalid_probability() -> None:
    """Transformer probabilities must stay inside the [0, 1] range."""
    data = GateData(
        anomaly_scores=torch.tensor([0.1, 0.2]),
        ft_probabilities=torch.tensor([0.5, 1.2]),
        labels=torch.tensor([0, 1]),
    )

    try:
        validate_gate_data(data)
    except ValueError as exc:
        assert "must be in [0, 1]" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid probability")


def test_validate_gate_data_rejects_invalid_label() -> None:
    """Fraud labels must be binary."""
    data = GateData(
        anomaly_scores=torch.tensor([0.1, 0.2]),
        ft_probabilities=torch.tensor([0.4, 0.8]),
        labels=torch.tensor([0, 2]),
    )

    try:
        validate_gate_data(data)
    except ValueError as exc:
        assert "only 0 or 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid label")


def test_build_gate_features_returns_expected_shape() -> None:
    """Feature builder should return one row per sample and two columns."""
    anomaly_scores = torch.tensor([0.1, 0.2, 0.3])
    ft_probabilities = torch.tensor([0.2, 0.5, 0.8])

    normalizer = PercentileNormalizer(percentile=100.0)

    features = build_gate_features(
        anomaly_scores,
        ft_probabilities,
        normalizer,
        fit_normalizer=True,
    )

    assert features.shape == (3, 2)
    assert torch.allclose(features[:, 1], ft_probabilities)
    assert torch.all(features[:, 0] >= 0.0)
    assert torch.all(features[:, 0] <= 1.0)


def test_evaluate_gate_loss_returns_finite_value() -> None:
    """Validation loss should be a finite non-negative number."""
    model = LearnedHybridGate(
        input_dim=2,
        hidden_dims=[8],
        dropout=0.0,
    )

    features = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.7, 0.8],
            [0.9, 0.95],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    loader = make_gate_loader(
        features=features,
        labels=labels,
        batch_size=2,
        shuffle=False,
    )

    loss_fn = torch.nn.BCELoss()

    loss = evaluate_gate_loss(
        model=model,
        data_loader=loader,
        loss_fn=loss_fn,
        device="cpu",
    )

    assert loss >= 0.0
    assert torch.isfinite(torch.tensor(loss))


def test_train_gate_trains_and_saves_checkpoint(tmp_path: Path) -> None:
    """Training should return an eval-mode gate and save its checkpoint."""
    config = Config(
        {
            "seed": 42,
            "hybrid_gating": {
                "learned": {
                    "input_dim": 2,
                    "hidden_dims": [8],
                    "dropout": 0.0,
                    "normalize_percentile": 99.0,
                    "training": {
                        "lr": 1.0e-2,
                        "weight_decay": 0.0,
                        "epochs": 5,
                        "batch_size": 4,
                        "early_stopping_patience": 3,
                        "min_delta": 0.0,
                    },
                    "checkpoint_path": str(tmp_path / "hybrid_gating.pt"),
                }
            },
        }
    )

    train_data = GateData(
        anomaly_scores=torch.tensor([0.1, 0.2, 0.3, 0.4, 2.0, 2.2, 2.4, 2.6]),
        ft_probabilities=torch.tensor([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95]),
        labels=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )

    val_data = GateData(
        anomaly_scores=torch.tensor([0.15, 0.35, 2.1, 2.5]),
        ft_probabilities=torch.tensor([0.10, 0.20, 0.82, 0.92]),
        labels=torch.tensor([0, 0, 1, 1]),
    )

    model, normalizer = train_gate(
        train_data=train_data,
        val_data=val_data,
        config=config,
        device="cpu",
    )

    checkpoint_path = Path(config.hybrid_gating.learned.checkpoint_path)

    assert isinstance(model, LearnedHybridGate)
    assert isinstance(normalizer, PercentileNormalizer)
    assert not model.training
    assert checkpoint_path.is_file()
