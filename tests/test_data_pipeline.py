"""Unit tests for Data Pipeline (Temporal Split & Preprocessor)."""

# Import necessary libraries and modules
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import FraudPreprocessor
from src.data.temporal_split import (
    split_temporal,
    split_validation_for_gate_and_conformal,
)


# Define unit tests for temporal splitting and preprocessing functions
def test_temporal_split_zero_leakage() -> None:
    """Assert strict chronological separation across train, val, and test."""
    dummy_df = pd.DataFrame(
        {
            "TransactionDT": np.arange(100) * 100,
            "isFraud": np.zeros(100),
        }
    )

    train_df, val_df, test_df = split_temporal(
        dummy_df,
        time_col="TransactionDT",
    )

    assert train_df["TransactionDT"].max() < val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() < test_df["TransactionDT"].min()

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15


def test_temporal_split_sorts_unsorted_input() -> None:
    """Assert temporal splitting sorts input before creating boundaries."""
    dummy_df = pd.DataFrame(
        {
            "TransactionDT": [500, 100, 300, 200, 400, 600, 700, 800, 900, 1000],
            "isFraud": np.zeros(10),
        }
    )

    train_df, val_df, test_df = split_temporal(
        dummy_df,
        time_col="TransactionDT",
    )

    assert train_df["TransactionDT"].is_monotonic_increasing
    assert val_df["TransactionDT"].is_monotonic_increasing
    assert test_df["TransactionDT"].is_monotonic_increasing

    assert train_df["TransactionDT"].max() < val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() < test_df["TransactionDT"].min()


def test_temporal_split_rejects_empty_input() -> None:
    """Assert an empty dataset cannot be temporally split."""
    dummy_df = pd.DataFrame(columns=["TransactionDT", "isFraud"])

    with pytest.raises(
        ValueError,
        match="Input DataFrame must not be empty",
    ):
        split_temporal(
            dummy_df,
            time_col="TransactionDT",
        )


def test_temporal_split_rejects_missing_time_column() -> None:
    """Assert the requested temporal column must exist."""
    dummy_df = pd.DataFrame(
        {
            "isFraud": [0, 1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing temporal column",
    ):
        split_temporal(
            dummy_df,
            time_col="TransactionDT",
        )


def test_validation_split_is_chronological_and_disjoint() -> None:
    """Assert gate and conformal subsets are separate chronological periods."""
    val_df = pd.DataFrame(
        {
            "TransactionDT": np.arange(20) * 100,
            "isFraud": np.zeros(20),
        }
    )

    gate_df, conformal_df = split_validation_for_gate_and_conformal(
        val_df,
        time_col="TransactionDT",
        gate_fraction=0.5,
    )

    assert len(gate_df) == 10
    assert len(conformal_df) == 10

    assert gate_df["TransactionDT"].is_monotonic_increasing
    assert conformal_df["TransactionDT"].is_monotonic_increasing

    assert gate_df["TransactionDT"].max() < conformal_df["TransactionDT"].min()

    gate_times = set(gate_df["TransactionDT"])
    conformal_times = set(conformal_df["TransactionDT"])

    assert gate_times.isdisjoint(conformal_times)


def test_validation_split_sorts_unsorted_input() -> None:
    """Assert validation subdivision sorts observations chronologically."""
    val_df = pd.DataFrame(
        {
            "TransactionDT": [
                800,
                100,
                500,
                200,
                700,
                300,
                600,
                400,
            ],
            "isFraud": np.zeros(8),
        }
    )

    gate_df, conformal_df = split_validation_for_gate_and_conformal(
        val_df,
        time_col="TransactionDT",
    )

    assert gate_df["TransactionDT"].is_monotonic_increasing
    assert conformal_df["TransactionDT"].is_monotonic_increasing

    assert gate_df["TransactionDT"].max() < conformal_df["TransactionDT"].min()


def test_validation_split_keeps_duplicate_boundary_timestamp_together() -> None:
    """Assert duplicate boundary timestamps are not split across subsets."""
    val_df = pd.DataFrame(
        {
            "TransactionDT": [
                100,
                200,
                300,
                400,
                500,
                500,
                500,
                600,
                700,
                800,
            ],
            "isFraud": np.zeros(10),
        }
    )

    gate_df, conformal_df = split_validation_for_gate_and_conformal(
        val_df,
        time_col="TransactionDT",
        gate_fraction=0.5,
    )

    assert 500 not in set(gate_df["TransactionDT"])
    assert 500 in set(conformal_df["TransactionDT"])

    assert gate_df["TransactionDT"].max() < conformal_df["TransactionDT"].min()


@pytest.mark.parametrize(
    "gate_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_validation_split_rejects_invalid_gate_fraction(
    gate_fraction: float,
) -> None:
    """Assert gate_fraction must lie strictly between zero and one."""
    val_df = pd.DataFrame(
        {
            "TransactionDT": np.arange(10),
            "isFraud": np.zeros(10),
        }
    )

    with pytest.raises(
        ValueError,
        match="gate_fraction must be strictly between 0 and 1",
    ):
        split_validation_for_gate_and_conformal(
            val_df,
            gate_fraction=gate_fraction,
        )


def test_validation_split_rejects_empty_input() -> None:
    """Assert validation subdivision rejects empty input."""
    val_df = pd.DataFrame(columns=["TransactionDT", "isFraud"])

    with pytest.raises(
        ValueError,
        match="Validation DataFrame must not be empty",
    ):
        split_validation_for_gate_and_conformal(val_df)


def test_validation_split_rejects_missing_time_column() -> None:
    """Assert validation subdivision requires the temporal column."""
    val_df = pd.DataFrame(
        {
            "isFraud": [0, 1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing temporal column",
    ):
        split_validation_for_gate_and_conformal(
            val_df,
            time_col="TransactionDT",
        )


def test_validation_split_rejects_single_observation() -> None:
    """Assert at least two validation observations are required."""
    val_df = pd.DataFrame(
        {
            "TransactionDT": [100],
            "isFraud": [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two observations",
    ):
        split_validation_for_gate_and_conformal(val_df)


def test_validation_split_rejects_identical_timestamps() -> None:
    """Assert strict separation is impossible when every timestamp is equal."""
    val_df = pd.DataFrame(
        {
            "TransactionDT": [100, 100, 100, 100],
            "isFraud": [0, 1, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Unable to create a non-empty gate subset",
    ):
        split_validation_for_gate_and_conformal(val_df)


def test_preprocessor_creates_nan_indicators() -> None:
    """Assert the preprocessor adds missingness indicators and scales data."""
    dummy_df = pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "isFraud": np.random.choice(
                [0, 1],
                size=100,
                p=[0.9, 0.1],
            ),
        }
    )

    cont_cols = ["feature_1", "feature_2"]

    preprocessor = FraudPreprocessor(
        cont_cols=cont_cols,
        cat_cols=[],
    )

    dummy_df.loc[0, "feature_1"] = np.nan

    result_df = preprocessor.fit_transform(dummy_df)

    assert "feature_1_is_nan" in result_df.columns
    assert result_df.loc[0, "feature_1_is_nan"] == 1
    assert result_df.loc[1, "feature_1_is_nan"] == 0
