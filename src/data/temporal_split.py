"""Temporal dataset splitting utilities."""

# Import necessary libraries and modules
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Define functions for loading and splitting temporal datasets
def load_and_merge_data(transaction_path: str, identity_path: str) -> pd.DataFrame:
    """Load and merge transaction and identity datasets.

    Args:
        transaction_path: Path to train_transaction.csv.
        identity_path: Path to train_identity.csv.

    Returns:
        A merged Pandas DataFrame joined on TransactionID.
    """
    logger.info("Loading transactions from %s...", transaction_path)
    df_trans = pd.read_csv(transaction_path)

    logger.info("Loading identity from %s...", identity_path)
    df_id = pd.read_csv(identity_path)

    # Outer join to keep all transactions, even if identity is missing.
    df_merged = df_trans.merge(df_id, on="TransactionID", how="left")

    logger.info("Merged dataset shape: %s", df_merged.shape)

    return df_merged


def split_temporal(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a strict chronological 70/15/15 train/validation/test split.

    The returned splits preserve temporal order:

        train -> validation -> test

    No future observations are allowed to appear in an earlier split.

    Args:
        df: Input transaction DataFrame.
        time_col: Column representing chronological transaction time.

    Returns:
        Tuple containing ``train_df``, ``val_df``, and ``test_df``.

    Raises:
        ValueError: If the input is empty, the time column is missing,
            a split would be empty, or temporal leakage is detected.
    """
    if df.empty:
        raise ValueError("Input DataFrame must not be empty")

    if time_col not in df.columns:
        raise ValueError(f"Missing temporal column: {time_col}")

    # Ensure strict chronological ordering before calculating split boundaries.
    df = df.sort_values(by=time_col).reset_index(drop=True)

    n_total = len(df)

    idx_train_end = int(n_total * 0.70)
    idx_val_end = int(n_total * 0.85)

    if idx_train_end <= 0:
        raise ValueError("Training split would be empty")

    if idx_val_end <= idx_train_end:
        raise ValueError("Validation split would be empty")

    if idx_val_end >= n_total:
        raise ValueError("Test split would be empty")

    train_df = df.iloc[:idx_train_end].reset_index(drop=True)
    val_df = df.iloc[idx_train_end:idx_val_end].reset_index(drop=True)
    test_df = df.iloc[idx_val_end:].reset_index(drop=True)

    max_train_time = train_df[time_col].max()
    min_val_time = val_df[time_col].min()
    max_val_time = val_df[time_col].max()
    min_test_time = test_df[time_col].min()

    if max_train_time >= min_val_time:
        raise ValueError("TEMPORAL LEAKAGE: Train overlaps with validation")

    if max_val_time >= min_test_time:
        raise ValueError("TEMPORAL LEAKAGE: Validation overlaps with test")

    logger.info(
        "Temporal split complete - Train: %d, Val: %d, Test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    logger.info(
        "Time boundaries - Train_End: %s, Val_Start: %s, " "Val_End: %s, Test_Start: %s",
        max_train_time,
        min_val_time,
        max_val_time,
        min_test_time,
    )

    return train_df, val_df, test_df


def split_validation_for_gate_and_conformal(
    val_df: pd.DataFrame,
    time_col: str = "TransactionDT",
    gate_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation data into disjoint gate and conformal subsets.

    This function creates two chronologically ordered subsets from the
    validation period:

        earlier validation data -> learned gate development
        later validation data   -> conformal calibration

    Keeping these subsets disjoint prevents the labels used to develop the
    learned hybrid gate from also being reused to calibrate the conformal
    predictor.

    The split is approximately controlled by ``gate_fraction``. Because strict
    chronological separation is required, all observations sharing the
    boundary timestamp are assigned to the conformal subset.

    Args:
        val_df: Chronologically held-out validation DataFrame.
        time_col: Column representing chronological transaction time.
        gate_fraction: Approximate fraction of validation observations assigned
            to gate development. Must be strictly between 0 and 1.

    Returns:
        Tuple containing ``gate_df`` and ``conformal_df``.

    Raises:
        ValueError: If inputs are invalid, either resulting subset is empty,
            or strict temporal separation cannot be established.
    """
    if val_df.empty:
        raise ValueError("Validation DataFrame must not be empty")

    if time_col not in val_df.columns:
        raise ValueError(f"Missing temporal column: {time_col}")

    if not 0.0 < gate_fraction < 1.0:
        raise ValueError("gate_fraction must be strictly between 0 and 1")

    if len(val_df) < 2:
        raise ValueError("Validation DataFrame must contain at least two observations")

    sorted_val = val_df.sort_values(by=time_col).reset_index(drop=True)

    target_index = int(len(sorted_val) * gate_fraction)

    # Protect against an index of zero caused by very small datasets.
    target_index = max(1, min(target_index, len(sorted_val) - 1))

    boundary_time = sorted_val.iloc[target_index][time_col]

    # Everything strictly before the boundary belongs to gate development.
    # All rows at the boundary timestamp move to conformal calibration.
    # This avoids sharing one timestamp across the two subsets.
    gate_df = sorted_val[sorted_val[time_col] < boundary_time].reset_index(drop=True)

    conformal_df = sorted_val[sorted_val[time_col] >= boundary_time].reset_index(drop=True)

    if gate_df.empty:
        raise ValueError(
            "Unable to create a non-empty gate subset with strict " "temporal separation"
        )

    if conformal_df.empty:
        raise ValueError("Unable to create a non-empty conformal calibration subset")

    max_gate_time = gate_df[time_col].max()
    min_conformal_time = conformal_df[time_col].min()

    if max_gate_time >= min_conformal_time:
        raise ValueError(
            "TEMPORAL LEAKAGE: Gate-development data overlaps with " "conformal calibration data"
        )

    logger.info(
        "Validation subdivision complete - Gate: %d, Conformal: %d",
        len(gate_df),
        len(conformal_df),
    )

    logger.info(
        "Validation subdivision boundaries - Gate_End: %s, " "Conformal_Start: %s",
        max_gate_time,
        min_conformal_time,
    )

    return gate_df, conformal_df


def split_gate_development(
    gate_df: pd.DataFrame,
    time_col: str = "TransactionDT",
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split gate-development data into chronological train and validation sets.

    The learned hybrid gate requires a training subset and a separate
    early-stopping validation subset. Both must remain earlier than the
    conformal-calibration period.

    The returned ordering is:

        gate training -> gate validation

    All observations sharing the split-boundary timestamp are assigned to the
    gate-validation subset so the same timestamp cannot appear on both sides
    of the temporal boundary.

    Args:
        gate_df: Gate-development portion of the original validation period.
        time_col: Column representing chronological transaction time.
        train_fraction: Approximate fraction assigned to learned-gate training.
            Must be strictly between 0 and 1.

    Returns:
        Tuple containing ``gate_train_df`` and ``gate_val_df``.

    Raises:
        ValueError: If inputs are invalid, either resulting subset is empty,
            or strict temporal separation cannot be established.
    """
    if gate_df.empty:
        raise ValueError("Gate-development DataFrame must not be empty")

    if time_col not in gate_df.columns:
        raise ValueError(f"Missing temporal column: {time_col}")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    if len(gate_df) < 2:
        raise ValueError("Gate-development DataFrame must contain at least two observations")

    sorted_gate = gate_df.sort_values(by=time_col).reset_index(drop=True)

    target_index = int(len(sorted_gate) * train_fraction)
    target_index = max(1, min(target_index, len(sorted_gate) - 1))

    boundary_time = sorted_gate.iloc[target_index][time_col]

    gate_train_df = sorted_gate[sorted_gate[time_col] < boundary_time].reset_index(drop=True)

    gate_val_df = sorted_gate[sorted_gate[time_col] >= boundary_time].reset_index(drop=True)

    if gate_train_df.empty:
        raise ValueError(
            "Unable to create a non-empty gate-training subset with strict " "temporal separation"
        )

    if gate_val_df.empty:
        raise ValueError("Unable to create a non-empty gate-validation subset")

    max_train_time = gate_train_df[time_col].max()
    min_val_time = gate_val_df[time_col].min()

    if max_train_time >= min_val_time:
        raise ValueError(
            "TEMPORAL LEAKAGE: Gate-training data overlaps with " "gate-validation data"
        )

    logger.info(
        "Gate-development split complete - Train: %d, Validation: %d",
        len(gate_train_df),
        len(gate_val_df),
    )

    logger.info(
        "Gate-development boundaries - Train_End: %s, Validation_Start: %s",
        max_train_time,
        min_val_time,
    )

    return gate_train_df, gate_val_df


def split_model_development(
    train_df: pd.DataFrame,
    time_col: str = "TransactionDT",
    train_fraction: float = 0.9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split base training data for upstream model fitting and early stopping.

    The original 70% chronological training block is subdivided into:

        earlier data -> upstream model training
        later data   -> upstream early-stopping validation

    This keeps the original validation period independent of FT-CAT and DAE
    model fitting so it can later be reserved for learned-gate development
    and conformal calibration.

    All observations sharing the boundary timestamp are assigned to the
    model-validation subset to preserve strict chronological separation.

    Args:
        train_df: Original chronological training DataFrame.
        time_col: Column representing chronological transaction time.
        train_fraction: Approximate fraction assigned to upstream training.
            Must be strictly between 0 and 1.

    Returns:
        Tuple containing ``model_train_df`` and ``model_val_df``.

    Raises:
        ValueError: If inputs are invalid, either subset would be empty,
            or strict temporal separation cannot be established.
    """
    if train_df.empty:
        raise ValueError("Training DataFrame must not be empty")

    if time_col not in train_df.columns:
        raise ValueError(f"Missing temporal column: {time_col}")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    if len(train_df) < 2:
        raise ValueError("Training DataFrame must contain at least two observations")

    sorted_train = train_df.sort_values(by=time_col).reset_index(drop=True)

    target_index = int(len(sorted_train) * train_fraction)
    target_index = max(1, min(target_index, len(sorted_train) - 1))

    boundary_time = sorted_train.iloc[target_index][time_col]

    model_train_df = sorted_train[sorted_train[time_col] < boundary_time].reset_index(drop=True)

    model_val_df = sorted_train[sorted_train[time_col] >= boundary_time].reset_index(drop=True)

    if model_train_df.empty:
        raise ValueError(
            "Unable to create a non-empty model-training subset with strict " "temporal separation"
        )

    if model_val_df.empty:
        raise ValueError("Unable to create a non-empty model-validation subset")

    max_train_time = model_train_df[time_col].max()
    min_val_time = model_val_df[time_col].min()

    if max_train_time >= min_val_time:
        raise ValueError(
            "TEMPORAL LEAKAGE: Model-training data overlaps with " "model-validation data"
        )

    logger.info(
        "Model-development split complete - Train: %d, Validation: %d",
        len(model_train_df),
        len(model_val_df),
    )

    logger.info(
        "Model-development boundaries - Train_End: %s, Validation_Start: %s",
        max_train_time,
        min_val_time,
    )

    return model_train_df, model_val_df


def main(argv: Optional[List[str]] = None) -> None:
    """Merge raw CSVs, apply the strict temporal split, and write parquets.

    Args:
        argv: Command line arguments; uses ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(
        description="Temporal 70/15/15 split of IEEE-CIS style raw data"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--time-col", type=str, default="TransactionDT")
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_merged = load_and_merge_data(
        str(input_dir / "train_transaction.csv"),
        str(input_dir / "train_identity.csv"),
    )
    train_df, val_df, test_df = split_temporal(df_merged, time_col=args.time_col)

    train_df.to_parquet(output_dir / "train_raw.parquet", index=False)
    val_df.to_parquet(output_dir / "val_raw.parquet", index=False)
    test_df.to_parquet(output_dir / "test_raw.parquet", index=False)
    logger.info("Raw temporal splits written to %s", output_dir)


if __name__ == "__main__":
    main()
