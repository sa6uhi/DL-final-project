"""Deterministic feature contract for DAE training and inference."""

# Import necessary modules and libraries
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype

from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_dae_feature_columns(
    df: pd.DataFrame,
    non_feature_cols: Sequence[str],
    expected_dim: int | None = None,
) -> list[str]:
    """Resolve the ordered numeric feature columns consumed by the DAE.

    Column order is preserved exactly as it appears in the processed
    DataFrame. The same resolver should therefore be used during DAE training
    and later hybrid inference.

    Args:
        df: Processed transaction DataFrame.
        non_feature_cols: Columns excluded from model inputs.
        expected_dim: Optional expected DAE input width.

    Returns:
        Ordered list of numeric DAE feature columns.

    Raises:
        ValueError: If no numeric features are available or the resolved
            feature count does not match ``expected_dim``.
    """
    excluded = set(non_feature_cols)

    feature_cols = [
        col for col in df.columns if col not in excluded and is_numeric_dtype(df[col].dtype)
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for DAE")

    if expected_dim is not None and len(feature_cols) != expected_dim:
        raise ValueError(
            "DAE feature count does not match configured input dimension: "
            f"{len(feature_cols)} != {expected_dim}"
        )

    return feature_cols


def materialize_dae_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> torch.Tensor:
    """Materialize processed DAE columns as a finite float32 tensor.

    Args:
        df: Processed transaction DataFrame.
        feature_cols: Ordered DAE feature contract.

    Returns:
        Tensor of shape ``(n_rows, n_features)``.

    Raises:
        ValueError: If the frame is empty, the feature contract is empty,
            or resulting values are non-finite.
        KeyError: If a required feature column is absent.
    """
    if df.empty:
        raise ValueError("DAE source DataFrame must not be empty")

    columns = list(feature_cols)

    if not columns:
        raise ValueError("DAE feature contract must not be empty")

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise KeyError(f"DAE source DataFrame is missing feature columns: {missing[:5]}")

    matrix = df.loc[:, columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if matrix.ndim != 2:
        raise ValueError("DAE feature matrix must be two-dimensional")

    if not np.isfinite(matrix).all():
        raise ValueError("DAE feature matrix must contain only finite values")

    tensor = torch.from_numpy(matrix)

    logger.info(
        "Materialized %d DAE rows with %d features",
        tensor.shape[0],
        tensor.shape[1],
    )

    return tensor
