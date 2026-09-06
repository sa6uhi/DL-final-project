# Import necessary modules and libraries
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Define a function to extract velocity features from historical sequences
def extract_velocity_features(
    sequence_arrays: Sequence[object],
    *,
    transaction_amount_index: int = 0,
) -> torch.Tensor:
    """Extract causal transaction-velocity features from historical sequences.

    Each sequence is expected to have shape ``(K, D)`` and contain only
    transactions that occurred before the current transaction. Zero-padded
    rows represent unavailable history.

    The returned features are:

    1. History density: fraction of the K historical rows that contain data.
    2. Historical amount intensity: log1p of the mean absolute
       TransactionAmt across available historical rows.

    Args:
        sequence_arrays: Historical sequence arrays for aligned transactions.
        transaction_amount_index: Column index of TransactionAmt inside each
            historical sequence. The project sequence contract places
            TransactionAmt at index 0.

    Returns:
        Float tensor with shape ``(n_samples, 2)``.

    Raises:
        ValueError: If sequences are empty, malformed, inconsistent,
            non-finite, or transaction_amount_index is invalid.
    """
    if len(sequence_arrays) == 0:
        raise ValueError("sequence_arrays must not be empty")

    rows: list[list[float]] = []
    expected_shape: tuple[int, int] | None = None

    for row_index, sequence in enumerate(sequence_arrays):
        array = np.asarray(sequence, dtype=np.float32)

        if array.ndim != 2:
            raise ValueError(f"sequence_arrays[{row_index}] must be two-dimensional")

        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"sequence_arrays[{row_index}] must not have empty dimensions")

        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError("All historical sequences must have the same shape")

        if not 0 <= transaction_amount_index < array.shape[1]:
            raise ValueError("transaction_amount_index is outside the sequence feature range")

        if not np.isfinite(array).all():
            raise ValueError(f"sequence_arrays[{row_index}] contains non-finite values")

        active_rows = np.any(array != 0.0, axis=1)
        history_count = int(active_rows.sum())

        history_density = history_count / float(array.shape[0])

        if history_count == 0:
            history_amount_intensity = 0.0
        else:
            historical_amounts = array[
                active_rows,
                transaction_amount_index,
            ]
            history_amount_mean = float(np.mean(np.abs(historical_amounts)))
            history_amount_intensity = float(np.log1p(history_amount_mean))

        rows.append(
            [
                float(history_density),
                history_amount_intensity,
            ]
        )

    features = torch.tensor(rows, dtype=torch.float32)

    logger.debug(
        "Extracted gate velocity features with shape %s",
        tuple(features.shape),
    )

    return features
