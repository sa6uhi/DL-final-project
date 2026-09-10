# Import necessary modules and libraries
from __future__ import annotations

import pytest
import torch

from src.utils.tensor_validation import (
    validate_probability_tensor,
    validate_same_batch_size,
    validate_same_device,
    validate_same_shape,
    validate_tensor,
)


# Define test functions for tensor validation
def test_validate_tensor_returns_original_tensor() -> None:
    tensor = torch.tensor([[1.0, 2.0]])

    result = validate_tensor(tensor)

    assert result is tensor


def test_validate_tensor_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        validate_tensor([1.0, 2.0], name="features")  # type: ignore[arg-type]


def test_validate_tensor_checks_ndim() -> None:
    tensor = torch.ones(2, 3)

    validate_tensor(tensor, ndim=2)

    with pytest.raises(ValueError, match="must be 1D"):
        validate_tensor(tensor, name="scores", ndim=1)


def test_validate_tensor_rejects_negative_ndim_contract() -> None:
    with pytest.raises(ValueError, match="ndim must be non-negative"):
        validate_tensor(torch.ones(2), ndim=-1)


def test_validate_tensor_checks_exact_shape() -> None:
    tensor = torch.ones(3, 4)

    validate_tensor(tensor, shape=(3, 4))

    with pytest.raises(ValueError, match="invalid size at dimension 1"):
        validate_tensor(tensor, shape=(3, 5))


def test_validate_tensor_supports_shape_wildcards() -> None:
    tensor = torch.ones(8, 4)

    validate_tensor(tensor, shape=(None, 4))


def test_validate_tensor_rejects_shape_rank_mismatch() -> None:
    tensor = torch.ones(2, 3)

    with pytest.raises(
        ValueError,
        match="must have 3 dimensions for shape validation, got 2",
    ):
        validate_tensor(tensor, shape=(2, 3, 4))


def test_validate_tensor_checks_feature_dimension() -> None:
    tensor = torch.ones(8, 4)

    validate_tensor(tensor, feature_dim=4)

    with pytest.raises(ValueError, match="must have 5 features"):
        validate_tensor(tensor, name="features", feature_dim=5)


def test_validate_tensor_rejects_invalid_feature_dimension_contract() -> None:
    with pytest.raises(ValueError, match="feature_dim must be positive"):
        validate_tensor(torch.ones(2, 3), feature_dim=0)


def test_validate_tensor_rejects_feature_dimension_for_scalar() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        validate_tensor(torch.tensor(1.0), feature_dim=1)


def test_validate_tensor_rejects_empty_tensor_by_default() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_tensor(torch.empty(0), name="scores")


def test_validate_tensor_can_allow_empty_tensor() -> None:
    tensor = torch.empty(0)

    assert validate_tensor(tensor, allow_empty=True) is tensor


@pytest.mark.parametrize(
    "bad_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_tensor_rejects_non_finite_values(bad_value: float) -> None:
    tensor = torch.tensor([1.0, bad_value])

    with pytest.raises(ValueError, match="only finite values"):
        validate_tensor(tensor, name="scores")


def test_validate_tensor_can_disable_finite_check() -> None:
    tensor = torch.tensor([float("nan")])

    assert validate_tensor(tensor, require_finite=False) is tensor


def test_validate_tensor_accepts_allowed_dtype() -> None:
    tensor = torch.ones(2, dtype=torch.float32)

    validate_tensor(
        tensor,
        dtypes=(torch.float32, torch.float64),
    )


def test_validate_tensor_rejects_disallowed_dtype() -> None:
    tensor = torch.ones(2, dtype=torch.int64)

    with pytest.raises(TypeError, match="following dtypes"):
        validate_tensor(
            tensor,
            name="features",
            dtypes=(torch.float32, torch.float64),
        )


def test_validate_tensor_rejects_empty_dtype_contract() -> None:
    with pytest.raises(ValueError, match="at least one dtype"):
        validate_tensor(torch.ones(2), dtypes=())


def test_validate_tensor_checks_minimum_value() -> None:
    validate_tensor(torch.tensor([0.0, 1.0]), min_value=0.0)

    with pytest.raises(ValueError, match=r"values >= 0.0"):
        validate_tensor(torch.tensor([-0.1, 0.5]), min_value=0.0)


def test_validate_tensor_checks_maximum_value() -> None:
    validate_tensor(torch.tensor([0.0, 1.0]), max_value=1.0)

    with pytest.raises(ValueError, match=r"values <= 1.0"):
        validate_tensor(torch.tensor([0.5, 1.1]), max_value=1.0)


def test_validate_tensor_rejects_invalid_bounds() -> None:
    with pytest.raises(
        ValueError,
        match="min_value must not exceed max_value",
    ):
        validate_tensor(
            torch.tensor([0.5]),
            min_value=1.0,
            max_value=0.0,
        )


def test_validate_probability_tensor_accepts_boundaries() -> None:
    probabilities = torch.tensor([0.0, 0.5, 1.0])

    assert validate_probability_tensor(probabilities, ndim=1) is probabilities


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([-0.01, 0.5]),
        torch.tensor([0.5, 1.01]),
        torch.tensor([float("nan"), 0.5]),
        torch.tensor([float("inf"), 0.5]),
    ],
)
def test_validate_probability_tensor_rejects_invalid_values(
    probabilities: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        validate_probability_tensor(probabilities)


def test_validate_same_batch_size_accepts_matching_batches() -> None:
    first = torch.ones(8, 4)
    second = torch.ones(8)
    third = torch.ones(8, 2)

    validate_same_batch_size(
        first,
        second,
        third,
        names=("features", "scores", "history"),
    )


def test_validate_same_batch_size_rejects_mismatch() -> None:
    first = torch.ones(8, 4)
    second = torch.ones(7)

    with pytest.raises(ValueError, match="batch sizes must match"):
        validate_same_batch_size(
            first,
            second,
            names=("features", "scores"),
        )


def test_validate_same_batch_size_rejects_scalar() -> None:
    with pytest.raises(ValueError, match="batch dimension"):
        validate_same_batch_size(
            torch.tensor(1.0),
            torch.ones(1),
        )


def test_validate_same_batch_size_requires_two_tensors() -> None:
    with pytest.raises(ValueError, match="At least two tensors"):
        validate_same_batch_size(torch.ones(2))


def test_validate_same_shape_accepts_matching_shapes() -> None:
    first = torch.ones(3, 4)
    second = torch.zeros(3, 4)

    validate_same_shape(first, second)


def test_validate_same_shape_rejects_mismatch() -> None:
    first = torch.ones(3, 4)
    second = torch.ones(3, 5)

    with pytest.raises(ValueError, match="Tensor shapes must match"):
        validate_same_shape(
            first,
            second,
            names=("scores", "labels"),
        )


def test_validate_same_device_accepts_matching_devices() -> None:
    first = torch.ones(3)
    second = torch.zeros(3)

    validate_same_device(first, second)


def test_validation_collection_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        validate_same_shape(
            torch.ones(2),
            [1.0, 2.0],  # type: ignore[arg-type]
        )


def test_validation_collection_checks_name_count() -> None:
    with pytest.raises(ValueError, match="one name for each tensor"):
        validate_same_shape(
            torch.ones(2),
            torch.ones(2),
            names=("only_one_name",),
        )


def test_validation_collection_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_same_shape(
            torch.ones(2),
            torch.ones(2),
            names=("first", ""),
        )


def test_validate_tensor_accepts_scalar_without_feature_contract() -> None:
    tensor = torch.tensor(1.0)

    assert validate_tensor(tensor) is tensor


def test_validate_tensor_accepts_zero_dim_contract() -> None:
    tensor = torch.tensor(1.0)

    assert validate_tensor(tensor, ndim=0) is tensor


def test_validate_tensor_rejects_empty_multidimensional_tensor() -> None:
    tensor = torch.empty(0, 4)

    with pytest.raises(ValueError, match="must not be empty"):
        validate_tensor(tensor, name="features", ndim=2)


def test_validate_tensor_allows_empty_multidimensional_tensor() -> None:
    tensor = torch.empty(0, 4)

    assert (
        validate_tensor(
            tensor,
            name="features",
            ndim=2,
            feature_dim=4,
            allow_empty=True,
        )
        is tensor
    )


def test_validate_tensor_rejects_nan_in_matrix() -> None:
    tensor = torch.tensor([[1.0, 2.0], [float("nan"), 4.0]])

    with pytest.raises(ValueError, match="only finite values"):
        validate_tensor(tensor, name="features", ndim=2)


def test_validate_tensor_rejects_positive_infinity_in_matrix() -> None:
    tensor = torch.tensor([[1.0, float("inf")]])

    with pytest.raises(ValueError, match="only finite values"):
        validate_tensor(tensor, name="features", ndim=2)


def test_validate_tensor_rejects_negative_infinity_in_matrix() -> None:
    tensor = torch.tensor([[1.0, float("-inf")]])

    with pytest.raises(ValueError, match="only finite values"):
        validate_tensor(tensor, name="features", ndim=2)


def test_validate_tensor_bounds_are_inclusive() -> None:
    tensor = torch.tensor([0.0, 0.5, 1.0])

    assert (
        validate_tensor(
            tensor,
            min_value=0.0,
            max_value=1.0,
        )
        is tensor
    )


def test_validate_tensor_rejects_value_just_below_minimum() -> None:
    tensor = torch.tensor([-1e-6, 0.5])

    with pytest.raises(ValueError, match=r"values >= 0.0"):
        validate_tensor(tensor, min_value=0.0)


def test_validate_tensor_rejects_value_just_above_maximum() -> None:
    tensor = torch.tensor([0.5, 1.000001])

    with pytest.raises(ValueError, match=r"values <= 1.0"):
        validate_tensor(tensor, max_value=1.0)


def test_validate_tensor_supports_float64() -> None:
    tensor = torch.ones(4, dtype=torch.float64)

    assert (
        validate_tensor(
            tensor,
            dtypes=(torch.float32, torch.float64),
        )
        is tensor
    )


def test_validate_tensor_supports_boolean_dtype_when_allowed() -> None:
    tensor = torch.tensor([True, False])

    assert validate_tensor(tensor, dtypes=(torch.bool,)) is tensor


def test_validate_tensor_feature_dimension_uses_last_axis() -> None:
    tensor = torch.ones(2, 3, 4)

    assert validate_tensor(tensor, feature_dim=4) is tensor


def test_validate_tensor_feature_dimension_rejects_last_axis_mismatch() -> None:
    tensor = torch.ones(2, 3, 4)

    with pytest.raises(ValueError, match="must have 5 features"):
        validate_tensor(tensor, feature_dim=5)


def test_validate_tensor_combined_contract() -> None:
    tensor = torch.rand(16, 4, dtype=torch.float32)

    assert (
        validate_tensor(
            tensor,
            name="gate_features",
            ndim=2,
            shape=(None, 4),
            feature_dim=4,
            dtypes=(torch.float32,),
            min_value=0.0,
            max_value=1.0,
        )
        is tensor
    )


def test_validate_probability_tensor_rejects_empty_by_default() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_probability_tensor(torch.empty(0))


def test_validate_probability_tensor_can_allow_empty() -> None:
    tensor = torch.empty(0)

    assert (
        validate_probability_tensor(
            tensor,
            allow_empty=True,
        )
        is tensor
    )


def test_validate_probability_tensor_rejects_wrong_rank() -> None:
    probabilities = torch.rand(4, 1)

    with pytest.raises(ValueError, match="must be 1D"):
        validate_probability_tensor(
            probabilities,
            name="probabilities",
            ndim=1,
        )


def test_validate_probability_tensor_accepts_single_probability() -> None:
    probabilities = torch.tensor([0.5])

    assert (
        validate_probability_tensor(
            probabilities,
            ndim=1,
        )
        is probabilities
    )


def test_validate_same_batch_size_works_with_different_ranks() -> None:
    first = torch.ones(5)
    second = torch.ones(5, 4)
    third = torch.ones(5, 3, 2)

    validate_same_batch_size(first, second, third)


def test_validate_same_batch_size_reports_named_tensor() -> None:
    scores = torch.ones(8)
    probabilities = torch.ones(7)

    with pytest.raises(
        ValueError,
        match=r"scores has 8, probabilities has 7",
    ):
        validate_same_batch_size(
            scores,
            probabilities,
            names=("scores", "probabilities"),
        )


def test_validate_same_batch_size_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="history must be a torch.Tensor"):
        validate_same_batch_size(
            torch.ones(2),
            [1.0, 2.0],  # type: ignore[arg-type]
            names=("features", "history"),
        )


def test_validate_same_shape_accepts_scalar_tensors() -> None:
    first = torch.tensor(1.0)
    second = torch.tensor(2.0)

    validate_same_shape(first, second)


def test_validate_same_shape_detects_rank_difference() -> None:
    first = torch.ones(4)
    second = torch.ones(4, 1)

    with pytest.raises(ValueError, match="Tensor shapes must match"):
        validate_same_shape(first, second)


def test_validate_same_shape_accepts_different_dtypes() -> None:
    first = torch.ones(3, dtype=torch.float32)
    second = torch.ones(3, dtype=torch.float64)

    validate_same_shape(first, second)


def test_validate_same_device_requires_two_tensors() -> None:
    with pytest.raises(ValueError, match="At least two tensors"):
        validate_same_device(torch.ones(2))


def test_validate_same_device_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="second must be a torch.Tensor"):
        validate_same_device(
            torch.ones(2),
            [1.0, 2.0],  # type: ignore[arg-type]
            names=("first", "second"),
        )


def test_validate_same_device_accepts_three_cpu_tensors() -> None:
    first = torch.ones(2)
    second = torch.zeros(2)
    third = torch.rand(2)

    validate_same_device(
        first,
        second,
        third,
        names=("first", "second", "third"),
    )


def test_validate_same_shape_requires_two_tensors() -> None:
    with pytest.raises(ValueError, match="At least two tensors"):
        validate_same_shape(torch.ones(2))


def test_validate_same_batch_size_checks_name_count() -> None:
    with pytest.raises(ValueError, match="one name for each tensor"):
        validate_same_batch_size(
            torch.ones(2),
            torch.ones(2),
            names=("first",),
        )


def test_validate_same_device_checks_name_count() -> None:
    with pytest.raises(ValueError, match="one name for each tensor"):
        validate_same_device(
            torch.ones(2),
            torch.ones(2),
            names=("first",),
        )
