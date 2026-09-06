# Import necessary libraries and modules
import numpy as np
import pytest
import torch

from src.training.gate_velocity import extract_velocity_features


# Test cases for the extract_velocity_features function
def test_extract_velocity_features_returns_expected_values() -> None:
    sequences = [
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

    features = extract_velocity_features(sequences)

    expected = torch.tensor(
        [
            [0.4, np.log1p(75.0)],
            [1.0, np.log1p(60.0)],
        ],
        dtype=torch.float32,
    )

    assert features.shape == (2, 2)
    assert torch.allclose(features, expected)


def test_extract_velocity_features_handles_empty_history() -> None:
    sequences = [
        np.zeros((5, 3), dtype=np.float32),
    ]

    features = extract_velocity_features(sequences)

    expected = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    assert torch.equal(features, expected)


def test_extract_velocity_features_uses_configured_amount_index() -> None:
    sequences = [
        np.array(
            [
                [1.0, 10.0],
                [2.0, 30.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    ]

    features = extract_velocity_features(
        sequences,
        transaction_amount_index=1,
    )

    assert torch.allclose(
        features,
        torch.tensor([[0.4, np.log1p(20.0)]], dtype=torch.float32),
    )


def test_extract_velocity_features_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        extract_velocity_features([])


def test_extract_velocity_features_rejects_one_dimensional_sequence() -> None:
    sequences = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    ]

    with pytest.raises(ValueError, match="must be two-dimensional"):
        extract_velocity_features(sequences)


def test_extract_velocity_features_rejects_inconsistent_shapes() -> None:
    sequences = [
        np.zeros((5, 2), dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
    ]

    with pytest.raises(ValueError, match="same shape"):
        extract_velocity_features(sequences)


def test_extract_velocity_features_rejects_invalid_amount_index() -> None:
    sequences = [
        np.zeros((5, 2), dtype=np.float32),
    ]

    with pytest.raises(
        ValueError,
        match="outside the sequence feature range",
    ):
        extract_velocity_features(
            sequences,
            transaction_amount_index=2,
        )


def test_extract_velocity_features_rejects_nonfinite_values() -> None:
    sequence = np.zeros((5, 2), dtype=np.float32)
    sequence[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        extract_velocity_features([sequence])
