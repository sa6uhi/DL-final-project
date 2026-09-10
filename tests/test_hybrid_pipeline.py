"""Tests for leakage-safe hybrid gating and conformal orchestration."""

# Import necessary modules and libraries
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.training.hybrid_pipeline import (
    autoencoder_anomaly_scores,
    create_hybrid_data_split,
    learned_gate_probabilities,
    make_gate_data,
    train_hybrid_calibration_pipeline,
    transformer_probabilities,
    velocity_features_from_frame,
)
from src.training.train_hybrid_gating import GateData
from src.utils.config import Config


# Define dummy models for testing hybrid gating and conformal orchestration
class DummyTransformer(nn.Module):
    """Minimal Transformer-compatible model returning fixed logits."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        del x_cont, x_cat, sequence
        return self._logits.clone()


class DummyTupleTransformer(nn.Module):
    """Transformer-compatible model returning logits inside a tuple."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        del x_cont, x_cat, sequence
        return self._logits.clone(), object()


class DummyAutoencoder(nn.Module):
    """Minimal autoencoder exposing the production anomaly-score API."""

    def anomaly_score(
        self,
        features: torch.Tensor,
        l1_gamma: float,
        reduction: str,
    ) -> torch.Tensor:
        assert l1_gamma == pytest.approx(0.4)
        assert reduction == "none"
        return features.sum(dim=1)


class InvalidAutoencoder(nn.Module):
    """Model without the required anomaly-score method."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def build_config() -> Config:
    """Construct a minimal configuration required by hybrid splitting."""
    return Config(
        {
            "split": {
                "time_col": "TransactionDT",
            },
            "hybrid_gating": {
                "split": {
                    "gate_fraction": 0.5,
                    "gate_train_fraction": 0.8,
                }
            },
        }
    )


def test_create_hybrid_data_split_is_strictly_chronological() -> None:
    """Assert gate train, gate validation, and conformal data never overlap."""
    validation_df = pd.DataFrame(
        {
            "TransactionDT": np.arange(100) * 10,
            "isFraud": np.zeros(100),
        }
    )

    split = create_hybrid_data_split(
        validation_df=validation_df,
        config=build_config(),
    )

    assert len(split.gate_train) == 40
    assert len(split.gate_val) == 10
    assert len(split.conformal_calibration) == 50

    assert split.gate_train["TransactionDT"].max() < split.gate_val["TransactionDT"].min()

    assert (
        split.gate_val["TransactionDT"].max() < split.conformal_calibration["TransactionDT"].min()
    )

    gate_train_times = set(split.gate_train["TransactionDT"])
    gate_val_times = set(split.gate_val["TransactionDT"])
    conformal_times = set(split.conformal_calibration["TransactionDT"])

    assert gate_train_times.isdisjoint(gate_val_times)
    assert gate_train_times.isdisjoint(conformal_times)
    assert gate_val_times.isdisjoint(conformal_times)


def test_transformer_probabilities_applies_sigmoid() -> None:
    """Assert raw FT-CAT logits are converted into fraud probabilities."""
    logits = torch.tensor([-2.0, 0.0, 2.0])

    transformer = DummyTransformer(logits)

    x_cont = torch.zeros(3, 2)
    x_cat = torch.zeros(3, 1, dtype=torch.long)
    sequence = torch.zeros(3, 5, 2)

    probabilities = transformer_probabilities(
        transformer=transformer,
        x_cont=x_cont,
        x_cat=x_cat,
        sequence=sequence,
    )

    expected = torch.sigmoid(logits)

    assert torch.allclose(probabilities, expected)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_transformer_probabilities_accepts_tuple_output() -> None:
    """Assert tuple model outputs use their first item as logits."""
    logits = torch.tensor([-1.0, 1.0])

    transformer = DummyTupleTransformer(logits)

    probabilities = transformer_probabilities(
        transformer=transformer,
        x_cont=torch.zeros(2, 1),
        x_cat=torch.zeros(2, 1, dtype=torch.long),
        sequence=torch.zeros(2, 5, 1),
    )

    assert torch.allclose(
        probabilities,
        torch.sigmoid(logits),
    )


def test_transformer_probabilities_restores_training_mode() -> None:
    """Assert inference does not permanently change Transformer mode."""
    transformer = DummyTransformer(torch.tensor([0.0, 1.0]))
    transformer.train()

    transformer_probabilities(
        transformer=transformer,
        x_cont=torch.zeros(2, 1),
        x_cat=torch.zeros(2, 1, dtype=torch.long),
        sequence=torch.zeros(2, 5, 1),
    )

    assert transformer.training


def test_transformer_probabilities_rejects_non_vector_logits() -> None:
    """Assert malformed Transformer output shapes are rejected."""
    transformer = DummyTransformer(torch.zeros(2, 1))

    with pytest.raises(
        ValueError,
        match="Transformer logits must be 1D",
    ):
        transformer_probabilities(
            transformer=transformer,
            x_cont=torch.zeros(2, 1),
            x_cat=torch.zeros(2, 1, dtype=torch.long),
            sequence=torch.zeros(2, 5, 1),
        )


def test_transformer_probabilities_rejects_non_finite_logits() -> None:
    """Assert non-finite FT-CAT logits are rejected."""
    transformer = DummyTransformer(torch.tensor([0.0, float("nan")]))

    with pytest.raises(
        ValueError,
        match="Transformer logits must contain only finite values",
    ):
        transformer_probabilities(
            transformer=transformer,
            x_cont=torch.zeros(2, 1),
            x_cat=torch.zeros(2, 1, dtype=torch.long),
            sequence=torch.zeros(2, 5, 1),
        )


def test_autoencoder_anomaly_scores_uses_model_api() -> None:
    """Assert DAE scores are obtained from anomaly_score with no reduction."""
    autoencoder = DummyAutoencoder()

    features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    scores = autoencoder_anomaly_scores(
        autoencoder=autoencoder,
        features=features,
        l1_gamma=0.4,
    )

    assert torch.equal(
        scores,
        torch.tensor([3.0, 7.0]),
    )


def test_autoencoder_anomaly_scores_requires_anomaly_method() -> None:
    """Assert incompatible autoencoders are rejected clearly."""
    with pytest.raises(
        AttributeError,
        match="Autoencoder must expose a callable anomaly_score method",
    ):
        autoencoder_anomaly_scores(
            autoencoder=InvalidAutoencoder(),
            features=torch.zeros(2, 3),
            l1_gamma=0.4,
        )


def test_autoencoder_anomaly_scores_rejects_non_finite_values() -> None:
    """Assert invalid anomaly-score outputs are rejected."""

    class NonFiniteAutoencoder(nn.Module):
        def anomaly_score(
            self,
            features: torch.Tensor,
            l1_gamma: float,
            reduction: str,
        ) -> torch.Tensor:
            del features, l1_gamma, reduction
            return torch.tensor([1.0, float("inf")])

    with pytest.raises(
        ValueError,
        match="Autoencoder anomaly scores must contain only finite values",
    ):
        autoencoder_anomaly_scores(
            autoencoder=NonFiniteAutoencoder(),
            features=torch.zeros(2, 3),
            l1_gamma=0.4,
        )


def test_make_gate_data_moves_tensors_to_cpu() -> None:
    """Assert gate data construction standardizes tensor dtypes and device."""
    data = make_gate_data(
        anomaly_scores=torch.tensor([1.0, 2.0], dtype=torch.float64),
        ft_probabilities=torch.tensor([0.2, 0.8], dtype=torch.float64),
        velocity_features=torch.tensor(
            [[0.2, 1.0], [0.8, 2.0]],
            dtype=torch.float64,
        ),
        labels=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )

    assert isinstance(data, GateData)

    assert data.anomaly_scores.device.type == "cpu"
    assert data.ft_probabilities.device.type == "cpu"
    assert data.velocity_features.device.type == "cpu"
    assert data.labels.device.type == "cpu"

    assert data.anomaly_scores.dtype == torch.float32
    assert data.ft_probabilities.dtype == torch.float32
    assert data.velocity_features.dtype == torch.float32
    assert data.labels.dtype == torch.int64


def test_velocity_features_from_frame_extracts_sequence_context() -> None:
    """Assert processed sequence history becomes learned-gate context."""
    frame = pd.DataFrame(
        {
            "sequence_array": [
                np.array(
                    [
                        [100.0, 1.0],
                        [50.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        [20.0, 1.0],
                        [40.0, 2.0],
                        [60.0, 3.0],
                        [80.0, 4.0],
                        [100.0, 5.0],
                    ],
                    dtype=np.float32,
                ),
            ]
        }
    )

    features = velocity_features_from_frame(frame)

    expected = torch.tensor(
        [
            [0.4, np.log1p(75.0)],
            [1.0, np.log1p(60.0)],
        ],
        dtype=torch.float32,
    )

    assert features.shape == (2, 2)
    assert torch.allclose(features, expected)


def test_velocity_features_from_frame_rejects_missing_sequence() -> None:
    """Assert velocity extraction requires historical sequence data."""
    frame = pd.DataFrame({"isFraud": [0, 1]})

    with pytest.raises(KeyError, match="sequence_array"):
        velocity_features_from_frame(frame)


def test_velocity_features_from_frame_rejects_empty_frame() -> None:
    """Assert velocity extraction rejects an empty transaction split."""
    frame = pd.DataFrame(columns=["sequence_array"])

    with pytest.raises(ValueError, match="empty frame"):
        velocity_features_from_frame(frame)


def test_learned_gate_probabilities_reuses_existing_normalizer() -> None:
    """Assert inference transforms scores without refitting normalization."""
    anomaly_train = torch.tensor([1.0, 2.0, 3.0, 4.0])

    normalizer = PercentileNormalizer(percentile=99.9)
    normalizer.fit(anomaly_train)

    original_state = normalizer.state_dict().copy()

    gate = LearnedHybridGate(
        input_dim=4,
        hidden_dims=[4],
        dropout=0.0,
    )

    probabilities = learned_gate_probabilities(
        gate=gate,
        normalizer=normalizer,
        anomaly_scores=torch.tensor([10.0, 20.0]),
        ft_probabilities=torch.tensor([0.25, 0.75]),
        velocity_features=torch.tensor(
            [
                [0.4, 1.5],
                [0.8, 2.5],
            ]
        ),
    )

    assert probabilities.shape == (2,)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)

    assert normalizer.state_dict() == original_state


def test_learned_gate_probabilities_restores_training_mode() -> None:
    """Assert gate inference does not permanently modify training mode."""
    normalizer = PercentileNormalizer(percentile=99.9)
    normalizer.fit(torch.tensor([1.0, 2.0, 3.0]))

    gate = LearnedHybridGate(
        input_dim=4,
        hidden_dims=[4],
        dropout=0.0,
    )
    gate.train()

    learned_gate_probabilities(
        gate=gate,
        normalizer=normalizer,
        anomaly_scores=torch.tensor([1.5, 2.5]),
        ft_probabilities=torch.tensor([0.2, 0.8]),
        velocity_features=torch.tensor(
            [
                [0.4, 1.5],
                [0.8, 2.5],
            ]
        ),
    )

    assert gate.training


def test_train_hybrid_calibration_pipeline_uses_conformal_data_after_gate_training() -> None:
    """Assert conformal labels are used only after learned-gate training."""
    config = Config(
        {
            "seed": 42,
            "hybrid_gating": {
                "learned": {
                    "input_dim": 4,
                    "hidden_dims": [4],
                    "dropout": 0.0,
                    "normalize_percentile": 99.9,
                    "training": {
                        "lr": 1.0e-3,
                        "weight_decay": 0.0,
                        "epochs": 1,
                        "batch_size": 2,
                        "early_stopping_patience": 1,
                        "min_delta": 0.0,
                    },
                    "checkpoint_path": "models/checkpoints/test_hybrid_gate.pt",
                }
            },
            "evaluation": {
                "conformal": {
                    "alpha": 0.01,
                }
            },
        }
    )

    gate_train_data = GateData(
        anomaly_scores=torch.tensor([1.0, 2.0]),
        ft_probabilities=torch.tensor([0.1, 0.9]),
        velocity_features=torch.tensor(
            [
                [0.2, 1.0],
                [0.8, 2.0],
            ]
        ),
        labels=torch.tensor([0, 1]),
    )

    gate_val_data = GateData(
        anomaly_scores=torch.tensor([1.5, 2.5]),
        ft_probabilities=torch.tensor([0.2, 0.8]),
        velocity_features=torch.tensor(
            [
                [0.4, 1.5],
                [1.0, 2.5],
            ]
        ),
        labels=torch.tensor([0, 1]),
    )

    conformal_anomaly_scores = torch.tensor([3.0, 4.0])
    conformal_ft_probabilities = torch.tensor([0.3, 0.7])
    conformal_velocity_features = torch.tensor(
        [
            [0.6, 2.0],
            [0.8, 3.0],
        ]
    )
    conformal_labels = torch.tensor([0, 1])

    gate = LearnedHybridGate(
        input_dim=4,
        hidden_dims=[4],
        dropout=0.0,
    )

    normalizer = PercentileNormalizer(percentile=99.9)
    normalizer.fit(gate_train_data.anomaly_scores)

    fused_probabilities = torch.tensor([0.25, 0.75])

    with (
        patch(
            "src.training.hybrid_pipeline.train_gate",
            return_value=(gate, normalizer),
        ) as mock_train_gate,
        patch(
            "src.training.hybrid_pipeline.learned_gate_probabilities",
            return_value=fused_probabilities,
        ) as mock_gate_probabilities,
    ):
        artifacts = train_hybrid_calibration_pipeline(
            gate_train_data=gate_train_data,
            gate_val_data=gate_val_data,
            conformal_anomaly_scores=conformal_anomaly_scores,
            conformal_ft_probabilities=conformal_ft_probabilities,
            conformal_velocity_features=conformal_velocity_features,
            conformal_labels=conformal_labels,
            config=config,
            device="cpu",
        )

    mock_train_gate.assert_called_once_with(
        train_data=gate_train_data,
        val_data=gate_val_data,
        config=config,
        device="cpu",
    )

    mock_gate_probabilities.assert_called_once_with(
        gate=gate,
        normalizer=normalizer,
        anomaly_scores=conformal_anomaly_scores,
        ft_probabilities=conformal_ft_probabilities,
        velocity_features=conformal_velocity_features,
    )

    assert artifacts.gate is gate
    assert artifacts.normalizer is normalizer
    assert artifacts.conformal_predictor.threshold is not None
