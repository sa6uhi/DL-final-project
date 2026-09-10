# Import necessary modules and libraries
from __future__ import annotations

from collections.abc import Sequence

import torch


# Define a function to validate tensor properties
def validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
    ndim: int | None = None,
    shape: Sequence[int | None] | None = None,
    feature_dim: int | None = None,
    allow_empty: bool = False,
    require_finite: bool = True,
    dtypes: Sequence[torch.dtype] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> torch.Tensor:
    """Validate common tensor invariants and return the original tensor.

    ``None`` entries in ``shape`` act as wildcards. For example,
    ``shape=(None, 4)`` accepts any batch size with exactly four features.

    Args:
        tensor: Tensor to validate.
        name: Human-readable tensor name used in error messages.
        ndim: Required number of dimensions.
        shape: Required shape. ``None`` dimensions are treated as wildcards.
        feature_dim: Required size of the final dimension.
        allow_empty: Whether tensors containing zero elements are allowed.
        require_finite: Whether NaN and infinite values are rejected.
        dtypes: Optional collection of accepted PyTorch dtypes.
        min_value: Optional inclusive lower bound.
        max_value: Optional inclusive upper bound.

    Returns:
        The original tensor, unchanged.

    Raises:
        TypeError: If ``tensor`` is not a torch.Tensor or has an invalid dtype.
        ValueError: If any requested tensor invariant is violated.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    if ndim is not None:
        if ndim < 0:
            raise ValueError("ndim must be non-negative")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")

    if shape is not None:
        expected_shape = tuple(shape)

        if len(expected_shape) != tensor.ndim:
            raise ValueError(
                f"{name} must have {len(expected_shape)} dimensions for "
                f"shape validation, got {tensor.ndim}"
            )

        for dim_index, (actual, expected) in enumerate(
            zip(tensor.shape, expected_shape, strict=True)
        ):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} has invalid size at dimension {dim_index}: "
                    f"expected {expected}, got {actual}"
                )

    if feature_dim is not None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if tensor.ndim == 0:
            raise ValueError(
                f"{name} must have at least one dimension when " "feature_dim is specified"
            )
        if tensor.shape[-1] != feature_dim:
            raise ValueError(f"{name} must have {feature_dim} features, " f"got {tensor.shape[-1]}")

    if not allow_empty and tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")

    if dtypes is not None:
        allowed_dtypes = tuple(dtypes)
        if not allowed_dtypes:
            raise ValueError("dtypes must contain at least one dtype")
        if tensor.dtype not in allowed_dtypes:
            expected = ", ".join(str(dtype) for dtype in allowed_dtypes)
            raise TypeError(
                f"{name} must have one of the following dtypes: " f"{expected}; got {tensor.dtype}"
            )

    if require_finite and tensor.numel() > 0 and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")

    if min_value is not None and max_value is not None:
        if min_value > max_value:
            raise ValueError("min_value must not exceed max_value")

    if tensor.numel() > 0:
        if min_value is not None and torch.any(tensor < min_value):
            raise ValueError(f"{name} must contain values >= {min_value}")

        if max_value is not None and torch.any(tensor > max_value):
            raise ValueError(f"{name} must contain values <= {max_value}")

    return tensor


def validate_probability_tensor(
    tensor: torch.Tensor,
    *,
    name: str = "probabilities",
    ndim: int | None = None,
    allow_empty: bool = False,
) -> torch.Tensor:
    """Validate a finite probability tensor with values in [0, 1]."""
    return validate_tensor(
        tensor,
        name=name,
        ndim=ndim,
        allow_empty=allow_empty,
        require_finite=True,
        min_value=0.0,
        max_value=1.0,
    )


def validate_same_batch_size(
    *tensors: torch.Tensor,
    names: Sequence[str] | None = None,
) -> None:
    """Require tensors to have identical first-dimension batch sizes."""
    if len(tensors) < 2:
        raise ValueError("At least two tensors are required")

    resolved_names = _resolve_names(tensors, names)

    for tensor, name in zip(tensors, resolved_names, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim == 0:
            raise ValueError(f"{name} must have a batch dimension")

    expected = tensors[0].shape[0]

    for tensor, name in zip(
        tensors[1:],
        resolved_names[1:],
        strict=True,
    ):
        if tensor.shape[0] != expected:
            raise ValueError(
                "Tensor batch sizes must match: "
                f"{resolved_names[0]} has {expected}, "
                f"{name} has {tensor.shape[0]}"
            )


def validate_same_shape(
    *tensors: torch.Tensor,
    names: Sequence[str] | None = None,
) -> None:
    """Require tensors to have identical shapes."""
    if len(tensors) < 2:
        raise ValueError("At least two tensors are required")

    resolved_names = _resolve_names(tensors, names)

    for tensor, name in zip(tensors, resolved_names, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    expected = tensors[0].shape

    for tensor, name in zip(
        tensors[1:],
        resolved_names[1:],
        strict=True,
    ):
        if tensor.shape != expected:
            raise ValueError(
                "Tensor shapes must match: "
                f"{resolved_names[0]} has {tuple(expected)}, "
                f"{name} has {tuple(tensor.shape)}"
            )


def validate_same_device(
    *tensors: torch.Tensor,
    names: Sequence[str] | None = None,
) -> None:
    """Require tensors to reside on the same device."""
    if len(tensors) < 2:
        raise ValueError("At least two tensors are required")

    resolved_names = _resolve_names(tensors, names)

    for tensor, name in zip(tensors, resolved_names, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    expected = tensors[0].device

    for tensor, name in zip(
        tensors[1:],
        resolved_names[1:],
        strict=True,
    ):
        if tensor.device != expected:
            raise ValueError(
                "Tensor devices must match: "
                f"{resolved_names[0]} is on {expected}, "
                f"{name} is on {tensor.device}"
            )


def _resolve_names(
    tensors: Sequence[torch.Tensor],
    names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return validated display names for a tensor collection."""
    if names is None:
        return tuple(f"tensor_{index}" for index in range(len(tensors)))

    resolved = tuple(names)

    if len(resolved) != len(tensors):
        raise ValueError("names must contain exactly one name for each tensor")

    if any(not name for name in resolved):
        raise ValueError("Tensor names must not be empty")

    return resolved
