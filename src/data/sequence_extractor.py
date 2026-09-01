import pandas as pd
import numpy as np
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_historical_sequences(
    df: pd.DataFrame, group_cols: List[str], seq_feature_cols: List[str], k: int = 5
) -> pd.DataFrame:
    """Constructs rolling historical transaction sequences per cardholder.

    For each row, this function looks back at the previous K transactions for the
    same cardholder and stacks them into a 2D tensor. If fewer than K historical
    transactions exist, the sequence is zero-padded.

    WARNING: To prevent temporal data leakage, this function MUST be called
    independently on the Train, Validation, and Test splits AFTER temporal splitting.

    Args:
        df: A strictly chronologically sorted Pandas DataFrame (e.g., train split only).
        group_cols: Columns to group by to identify a unique cardholder
                    (e.g., ['card1', 'addr1']).
        seq_feature_cols: The specific numeric columns to include in the historical
                          sequence (e.g., ['TransactionAmt', 'D1', 'D15']).
                          Do NOT include IDs or the target label `isFraud`.
        k: The number of historical transactions to look back (window size).

    Returns:
        The input DataFrame with an added column 'sequence_array', where each row
        contains a np.ndarray of shape (K, len(seq_feature_cols)).
    """
    if k <= 0:
        raise ValueError("Window size must be greater than 0")

    logger.info("Building historical sequences (K=%d) for %d records...", k, len(df))
    df_out = df.reset_index(drop=True).copy()
    d_features = len(seq_feature_cols)

    lagged_dfs = []
    for i in range(1, k + 1):
        shifted = df_out.groupby(group_cols)[seq_feature_cols].shift(i)
        shifted.columns = [f"{col}_lag_{i}" for col in seq_feature_cols]
        lagged_dfs.append(shifted)

    lagged_df = pd.concat(lagged_dfs, axis=1)
    lagged_values = lagged_df.values
    sequences = lagged_values.reshape(len(df_out), k, d_features, order="C")
    sequences = np.nan_to_num(sequences, nan=0.0)

    df_out["sequence_array"] = sequences.tolist()

    logger.info("Successfully generated sequences. Final shape per row: (%d, %d)", k, d_features)
    return df_out


def get_sequence_feature_columns(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    """Select numeric columns suitable for the sequence tensor.

    Args:
        df: The processed DataFrame.
        exclude_cols: Columns to exclude (e.g., ['isFraud', 'TransactionID', 'TransactionDT']).

    Returns:
        A list of numeric column names for seq_feature_cols.
    """
    potential_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    seq_cols = [col for col in potential_cols if col not in exclude_cols]
    logger.warning(
        "Identified %d sequence features. Consider manually curating this list "
        "to avoid overly large sequence tensors.",
        len(seq_cols),
    )
    return seq_cols
