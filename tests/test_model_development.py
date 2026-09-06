"""Tests for leakage-safe upstream model-development splitting."""

# Import necessary modules and libraries
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.training.model_development import create_model_development_split


# Define unit tests for leakage-safe upstream model-development splitting
def make_config(
    *,
    time_col: str = "TransactionDT",
    train_fraction: float = 0.9,
) -> SimpleNamespace:
    """Create the minimal config shape required by the split helper."""
    return SimpleNamespace(
        split=SimpleNamespace(
            time_col=time_col,
            model_train_fraction=train_fraction,
        )
    )


def test_create_model_development_split_is_chronological() -> None:
    """Upstream training must remain earlier than early stopping."""
    df = pd.DataFrame(
        {
            "TransactionDT": list(range(10)),
            "isFraud": [0, 1] * 5,
        }
    )

    model_train, model_val = create_model_development_split(
        df,
        make_config(train_fraction=0.8),
    )

    assert len(model_train) == 8
    assert len(model_val) == 2
    assert model_train["TransactionDT"].max() < model_val["TransactionDT"].min()


def test_create_model_development_split_uses_configured_time_column() -> None:
    """The helper must respect the configured chronological field."""
    df = pd.DataFrame(
        {
            "event_time": [10, 20, 30, 40],
            "isFraud": [0, 0, 1, 0],
        }
    )

    model_train, model_val = create_model_development_split(
        df,
        make_config(
            time_col="event_time",
            train_fraction=0.5,
        ),
    )

    assert model_train["event_time"].tolist() == [10, 20]
    assert model_val["event_time"].tolist() == [30, 40]


def test_create_model_development_split_keeps_boundary_duplicates_together() -> None:
    """Equal boundary timestamps must remain entirely on validation side."""
    df = pd.DataFrame(
        {
            "TransactionDT": [1, 2, 3, 4, 4, 4, 5, 6],
            "isFraud": [0, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    model_train, model_val = create_model_development_split(
        df,
        make_config(train_fraction=0.5),
    )

    assert model_train["TransactionDT"].max() == 3
    assert model_val["TransactionDT"].min() == 4
    assert (model_val["TransactionDT"] == 4).sum() == 3
    assert not (model_train["TransactionDT"] == 4).any()


@pytest.mark.parametrize(
    "fraction",
    [0.0, 1.0, -0.1, 1.1],
)
def test_create_model_development_split_rejects_invalid_fraction(
    fraction: float,
) -> None:
    """Invalid configured fractions must fail clearly."""
    df = pd.DataFrame(
        {
            "TransactionDT": [1, 2, 3],
            "isFraud": [0, 1, 0],
        }
    )

    with pytest.raises(
        ValueError,
        match="train_fraction must be strictly between 0 and 1",
    ):
        create_model_development_split(
            df,
            make_config(train_fraction=fraction),
        )
