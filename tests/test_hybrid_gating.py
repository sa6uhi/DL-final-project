"""Unit tests for :mod:`src.models.hybrid_gating`."""

from __future__ import annotations

import pytest
import torch

from src.models.hybrid_gating import (
    HybridGate,
    LearnedHybridGate,
    PercentileNormalizer,
)


@pytest.fixture()
def scores() -> torch.Tensor:
    """Calibration and query scores with a clear outlier."""
    return torch.tensor([0.1, 0.2, 0.3, 0.4, 5.0], dtype=torch.float32)


@pytest.fixture()
def fitted_normalizer(scores: torch.Tensor) -> PercentileNormalizer:
    """Normalizer fitted on the calibration scores."""
    return PercentileNormalizer(percentile=99.0).fit(scores)


@pytest.fixture()
def normalized(scores: torch.Tensor) -> torch.Tensor:
    """Reference normalization of calibration scores."""
    return (scores.clamp(max=scores.max()) / scores.max()).clamp(0.0, 1.0)


def test_normalizer_fit_transform_bounds(
    fitted_normalizer: PercentileNormalizer, normalized: torch.Tensor
) -> None:
    """Fitted transform maps scores into [0, 1]."""
    out = fitted_normalizer.transform(normalized)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_normalizer_caps_extreme_values(fitted_normalizer: PercentileNormalizer) -> None:
    """Values above the percentile cap are clipped to the capped value."""
    assert float(fitted_normalizer.cap) >= 0.0
    transformed = fitted_normalizer.transform(torch.tensor([10.0, 100.0]))
    assert (transformed <= 1.0 + 1e-6).all()


def test_normalizer_not_fitted_raises() -> None:
    """transform before fit raises ValueError."""
    normalizer = PercentileNormalizer()
    with pytest.raises(ValueError):
        normalizer.transform(torch.tensor([1.0, 2.0]))


def test_normalizer_empty_fit_raises() -> None:
    """Fitting on an empty tensor raises ValueError."""
    with pytest.raises(ValueError):
        PercentileNormalizer().fit(torch.tensor([]))


def test_normalizer_nan_fit_raises() -> None:
    """Fitting on NaN scores raises ValueError."""
    with pytest.raises(ValueError):
        PercentileNormalizer().fit(torch.tensor([1.0, float("nan")]))


def test_normalizer_bad_percentile_raises() -> None:
    """Percentile outside [0, 100] raises ValueError."""
    with pytest.raises(ValueError):
        PercentileNormalizer(percentile=101.0)


def test_normalizer_state_roundtrip(
    fitted_normalizer: PercentileNormalizer, normalized: torch.Tensor
) -> None:
    """Serialized normalizer restores identical transforms."""
    restored = PercentileNormalizer.from_state_dict(fitted_normalizer.state_dict())
    assert restored.transform(normalized).equal(fitted_normalizer.transform(normalized))


def test_hybrid_gate_fusion_value(
    fitted_normalizer: PercentileNormalizer, scores: torch.Tensor
) -> None:
    """Fusion equals alpha*norm + (1-alpha)*prob elementwise."""
    alpha = 0.5
    gate = HybridGate(alpha=alpha, normalizer=fitted_normalizer)
    probs = torch.full((5,), 0.2)
    fused = gate.fuse(scores, probs)
    expected = alpha * fitted_normalizer.transform(scores) + (1 - alpha) * probs
    assert torch.allclose(fused, expected)
    assert (fused >= 0.0).all() and (fused <= 1.0).all()


def test_hybrid_gate_is_callable(
    fitted_normalizer: PercentileNormalizer, scores: torch.Tensor
) -> None:
    """Calling the gate directly equals fuse()."""
    gate = HybridGate(alpha=0.7, normalizer=fitted_normalizer)
    probs = torch.full((5,), 0.9)
    assert torch.allclose(gate(scores, probs), gate.fuse(scores, probs))


def test_hybrid_gate_shape_mismatch_raises(
    fitted_normalizer: PercentileNormalizer,
) -> None:
    """Mismatched anomaly/probability shapes raise ValueError."""
    gate = HybridGate(alpha=0.5, normalizer=fitted_normalizer)
    with pytest.raises(ValueError):
        gate.fuse(torch.ones(3), torch.ones(4))


def test_hybrid_gate_probability_range_raises(
    fitted_normalizer: PercentileNormalizer, scores: torch.Tensor
) -> None:
    """Probabilities outside [0, 1] raise ValueError."""
    gate = HybridGate(alpha=0.5, normalizer=fitted_normalizer)
    with pytest.raises(ValueError):
        gate.fuse(scores, torch.full((5,), 1.5))


def test_hybrid_gate_nan_input_raises(
    fitted_normalizer: PercentileNormalizer, scores: torch.Tensor
) -> None:
    """NaN inputs raise ValueError."""
    gate = HybridGate(alpha=0.5, normalizer=fitted_normalizer)
    with pytest.raises(ValueError):
        gate.fuse(scores, torch.tensor([float("nan"), 0.1, 0.2, 0.3, 0.4]))


def test_hybrid_gate_bad_alpha_raises() -> None:
    """Alpha outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        HybridGate(alpha=-0.1)
    with pytest.raises(ValueError):
        HybridGate(alpha=1.1)


def test_hybrid_gate_unfitted_normalizer_raises() -> None:
    """An unfitted normalizer is rejected at construction."""
    with pytest.raises(ValueError):
        HybridGate(alpha=0.5, normalizer=PercentileNormalizer())


def test_hybrid_gate_fuses_empty_shapes(
    fitted_normalizer: PercentileNormalizer,
) -> None:
    """Empty inputs raise a clear ValueError about shape mismatch."""
    gate = HybridGate(alpha=0.5, normalizer=fitted_normalizer)
    with pytest.raises(ValueError):
        gate.fuse(torch.tensor([]), torch.tensor([0.1]))


@pytest.fixture()
def learned_gate() -> LearnedHybridGate:
    """Create a small learned gate for unit tests."""
    return LearnedHybridGate(
        input_dim=2,
        hidden_dims=[16, 8],
        dropout=0.1,
    )


@pytest.fixture()
def feature_batch() -> torch.Tensor:
    """Create a batch of learned-gate input features."""
    return torch.rand(5, 2)


def test_learned_gate_output_shape(
    learned_gate: LearnedHybridGate,
    feature_batch: torch.Tensor,
) -> None:
    """Return one fraud probability per transaction."""
    output = learned_gate(feature_batch)

    assert output.shape == (5,)


def test_learned_gate_output_range(
    learned_gate: LearnedHybridGate,
    feature_batch: torch.Tensor,
) -> None:
    """Return probabilities bounded between zero and one."""
    output = learned_gate(feature_batch)

    assert (output >= 0.0).all()
    assert (output <= 1.0).all()


def test_learned_gate_gradient_flows(
    learned_gate: LearnedHybridGate,
    feature_batch: torch.Tensor,
) -> None:
    """Allow gradients to reach every trainable parameter."""
    output = learned_gate(feature_batch)
    output.sum().backward()

    assert all(
        parameter.grad is not None
        for parameter in learned_gate.parameters()
    )


def test_learned_gate_wrong_ndim_raises(
    learned_gate: LearnedHybridGate,
) -> None:
    """Reject one-dimensional input tensors."""
    with pytest.raises(ValueError):
        learned_gate(torch.rand(2))


def test_learned_gate_wrong_feature_count_raises(
    learned_gate: LearnedHybridGate,
) -> None:
    """Reject inputs with the wrong number of features."""
    with pytest.raises(ValueError):
        learned_gate(torch.rand(5, 3))


def test_learned_gate_bad_input_dim_raises() -> None:
    """Reject non-positive input dimensions."""
    with pytest.raises(ValueError):
        LearnedHybridGate(
            input_dim=0,
            hidden_dims=[16],
            dropout=0.1,
        )


def test_learned_gate_bad_hidden_dims_raises() -> None:
    """Reject empty or non-positive hidden dimensions."""
    with pytest.raises(ValueError):
        LearnedHybridGate(
            input_dim=2,
            hidden_dims=[],
            dropout=0.1,
        )

    with pytest.raises(ValueError):
        LearnedHybridGate(
            input_dim=2,
            hidden_dims=[16, 0],
            dropout=0.1,
        )


def test_learned_gate_bad_dropout_raises() -> None:
    """Reject dropout values outside the valid range."""
    with pytest.raises(ValueError):
        LearnedHybridGate(
            input_dim=2,
            hidden_dims=[16],
            dropout=1.0,
        )


def test_learned_gate_state_dict_roundtrip(
    learned_gate: LearnedHybridGate,
    feature_batch: torch.Tensor,
) -> None:
    """Restore identical predictions from a saved state dictionary."""
    learned_gate.eval()

    restored = LearnedHybridGate(
        input_dim=2,
        hidden_dims=[16, 8],
        dropout=0.1,
    )
    restored.load_state_dict(learned_gate.state_dict())
    restored.eval()

    assert torch.allclose(
        learned_gate(feature_batch),
        restored(feature_batch),
    )


def test_learned_gate_non_finite_input_raises(
    learned_gate: LearnedHybridGate,
) -> None:
    """Reject NaN and infinite input values."""
    with pytest.raises(ValueError):
        learned_gate(
            torch.tensor(
                [
                    [0.2, float("nan")],
                    [0.4, 0.6],
                ]
            )
        )

    with pytest.raises(ValueError):
        learned_gate(
            torch.tensor(
                [
                    [0.2, float("inf")],
                    [0.4, 0.6],
                ]
            )
        )


def test_learned_gate_eval_mode_is_deterministic(
    learned_gate: LearnedHybridGate,
    feature_batch: torch.Tensor,
) -> None:
    """Return identical predictions in evaluation mode."""
    learned_gate.eval()

    first_output = learned_gate(feature_batch)
    second_output = learned_gate(feature_batch)

    assert torch.allclose(first_output, second_output)

    