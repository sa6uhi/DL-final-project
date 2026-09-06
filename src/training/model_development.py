"""Leakage-safe preparation of upstream model-development data."""

# Import necessary modules and libraries
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.temporal_split import split_model_development
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIME_COL = "TransactionDT"
DEFAULT_MODEL_TRAIN_FRACTION = 0.9


# Read split settings from both the project Config class and lightweight
# test objects exposing config.split.<key> attributes.
def _get_split_setting(
    config: Config | Any,
    key: str,
    default: Any,
) -> Any:
    """Read one split setting from Config-like objects.

    Supports the project's Config class as well as lightweight test objects
    exposing ``config.split.<key>`` attributes.

    Args:
        config: Project configuration or compatible object.
        key: Name of the setting inside the ``split`` section.
        default: Value used when the setting or section is absent.

    Returns:
        The configured value, or ``default`` when unavailable.
    """
    nested_get = getattr(config, "nested_get", None)
    if callable(nested_get):
        return nested_get(f"split.{key}", default)

    split_cfg = getattr(config, "split", None)
    if split_cfg is None:
        return default

    return getattr(split_cfg, key, default)


# Define the main function to create a leakage-safe upstream model-development split.
def create_model_development_split(
    train_df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subdivide the original training block for upstream model development.

    The original chronological training split is divided into an earlier
    model-training subset and a later early-stopping subset. The project's
    original validation split can therefore remain untouched for learned-gate
    development and conformal calibration.

    Small standalone test configs may omit the ``split`` section. In that
    case, the project defaults are used.

    Args:
        train_df: Original processed training DataFrame.
        config: Project configuration containing optional temporal split settings.

    Returns:
        Tuple containing ``model_train_df`` and ``model_val_df``.
    """
    time_col = str(
        _get_split_setting(
            config,
            "time_col",
            DEFAULT_TIME_COL,
        )
    )

    train_fraction = float(
        _get_split_setting(
            config,
            "model_train_fraction",
            DEFAULT_MODEL_TRAIN_FRACTION,
        )
    )

    model_train_df, model_val_df = split_model_development(
        train_df,
        time_col=time_col,
        train_fraction=train_fraction,
    )

    logger.info(
        "Prepared leakage-safe upstream split: %d model-train / %d model-val rows",
        len(model_train_df),
        len(model_val_df),
    )

    return model_train_df, model_val_df
