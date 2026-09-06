"""Tests for the deterministic DAE feature contract."""

# Import necessary modules and libraries
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.dae_features import (
    materialize_dae_features,
    resolve_dae_feature_columns,
)


# Define unit tests for DAE feature contract functions
def test_resolve_dae_feature_columns_preserves_dataframe_order() -> None:
    """Assert numeric DAE features retain processed DataFrame ordering."""
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "feature_b": [1.0, 2.0],
            "feature_a": [3.0, 4.0],
            "category": ["x", "y"],
            "feature_a_is_nan": [0.0, 1.0],
            "isFraud": [0, 1],
        }
    )

    columns = resolve_dae_feature_columns(
        df,
        non_feature_cols=["TransactionID", "isFraud"],
    )

    assert columns == [
        "feature_b",
        "feature_a",
        "feature_a_is_nan",
    ]


def test_resolve_dae_feature_columns_excludes_numeric_metadata() -> None:
    """Assert numeric identifiers and labels are excluded explicitly."""
    df = pd.DataFrame(
        {
            "TransactionID": [10, 20],
            "TransactionDT": [100, 200],
            "feature": [1.0, 2.0],
            "isFraud": [0, 1],
        }
    )

    columns = resolve_dae_feature_columns(
        df,
        non_feature_cols=[
            "TransactionID",
            "TransactionDT",
            "isFraud",
        ],
    )

    assert columns == ["feature"]


def test_resolve_dae_feature_columns_validates_expected_dimension() -> None:
    """Assert resolved feature width matches the DAE architecture."""
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="DAE feature count does not match configured input dimension",
    ):
        resolve_dae_feature_columns(
            df,
            non_feature_cols=[],
            expected_dim=3,
        )


def test_resolve_dae_feature_columns_rejects_no_numeric_features() -> None:
    """Assert a frame without usable numeric features is rejected."""
    df = pd.DataFrame(
        {
            "category": ["a", "b"],
        }
    )

    with pytest.raises(
        ValueError,
        match="No numeric feature columns found for DAE",
    ):
        resolve_dae_feature_columns(
            df,
            non_feature_cols=[],
        )


def test_materialize_dae_features_uses_requested_column_order() -> None:
    """Assert materialization follows the supplied feature contract."""
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [10.0, 20.0],
        }
    )

    tensor = materialize_dae_features(
        df,
        feature_cols=["feature_b", "feature_a"],
    )

    expected = torch.tensor(
        [
            [10.0, 1.0],
            [20.0, 2.0],
        ],
        dtype=torch.float32,
    )

    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, expected)


def test_materialize_dae_features_rejects_missing_columns() -> None:
    """Assert missing contract columns fail before model inference."""
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
        }
    )

    with pytest.raises(
        KeyError,
        match="missing feature columns",
    ):
        materialize_dae_features(
            df,
            feature_cols=["feature_a", "feature_b"],
        )


def test_materialize_dae_features_rejects_non_finite_values() -> None:
    """Assert NaN or infinite processed values are rejected."""
    df = pd.DataFrame(
        {
            "feature_a": [1.0, np.nan],
        }
    )

    with pytest.raises(
        ValueError,
        match="must contain only finite values",
    ):
        materialize_dae_features(
            df,
            feature_cols=["feature_a"],
        )


def test_materialize_dae_features_rejects_empty_frame() -> None:
    """Assert empty inference frames are rejected."""
    df = pd.DataFrame(
        {
            "feature_a": pd.Series(dtype=float),
        }
    )

    with pytest.raises(
        ValueError,
        match="DAE source DataFrame must not be empty",
    ):
        materialize_dae_features(
            df,
            feature_cols=["feature_a"],
        )


def test_resolver_matches_autoencoder_numeric_feature_contract() -> None:
    """Assert DAE resolver excludes metadata while preserving numeric order."""
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "feature_b": [1.0, 2.0],
            "feature_a": [3.0, 4.0],
            "isFraud": [0, 1],
        }
    )

    columns = resolve_dae_feature_columns(
        df,
        non_feature_cols=["TransactionID", "isFraud"],
    )

    features = df[df["isFraud"] == 0][columns].to_numpy(dtype=np.float32)

    assert columns == ["feature_b", "feature_a"]
    assert features.shape == (1, 2)
    assert np.array_equal(
        features,
        np.array([[1.0, 3.0]], dtype=np.float32),
    )
