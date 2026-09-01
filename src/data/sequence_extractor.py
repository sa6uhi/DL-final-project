import pandas as pd
import numpy as np
from typing import List
from src.utils.logger import get_logger
logger = get_logger(__name__)

def build_historical_sequences(
    df: pd.DataFrame,
    group_cols: List[str],
    seq_feature_cols: List[str],
    k: int=5
)->pd.DataFrame:
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
    if k<=0:
        raise ValueError("Window size must be greater than 0")

    logger.info(f"Building historical sequences (K={k}) for {len(df)} records...")
    df_out = df.reset_index(drop=True)
    d_features=len(seq_feature_cols)

    # 1. Create K lagged versions of the sequence features
    # We shift the data backwards within each cardholder group
    lagged_dfs=[]
    for i in range(1,k+1):
        # Shift by 'i' periods. The first 'i' rows for each card will be NaN
        shifted=df_out.groupby(group_cols)[seq_feature_cols].shift(i)
        # Rename columns to avoid collisions during concatenation
        shifted.columns = [f"{col}_lag_{i}" for col in seq_feature_cols]
        lagged_dfs.append(shifted)

    # 2. Concatenate all lags horizontally
    # Resulting DataFrame has K * d_features columns ordered chronologically backward
    lagged_df=pd.concat(lagged_dfs,axis=1)

    # 3. Vectorized Reshaping (Fast Path)
    # Convert to numpy array: shape (N, K*D)
    lagged_values=lagged_df.values

    # Reshape to (N,K,D)
    # Order='C' means it reads row-by-row, matching our lag_1, lag_2... column order
    sequences=lagged_values.reshape(len(df_out),k,d_features,order='C')

    # 4. Handle Missing History (Padding)
    # Where shift() produced NaNs (not enough history), pad with 0.0
    # Since the preprocessor standardizes data, 0.0 represents the mean/center
    sequences=np.nan_to_num(sequences, nan=0.0)

    # 5. Assign the list of numpy arrays back to the DataFrame
    # We convert the 3D numpy array to a list of 2D arrays so Pandas can store it in a single column
    df_out['sequence_array']=list(sequences)

    logger.info(f"Successfully generated sequences. Final shape per row: ({k}, {d_features})")
    return df_out

def get_sequence_feature_columns(df: pd.DataFrame,exclude_cols: List[str])->List[str]:
    """Helper to intelligently select columns for the sequence tensor.

    We shouldn't put ALL 400+ IEEE-CIS features into the sequence tensor—it will
    make the Transformer too heavy. This helper grabs a sensible subset (e.g.,
    Transaction amounts, time deltas, and distances).

    Args:
        df: The processed DataFrame.
        exclude_cols: Columns to definitely exclude (e.g., ['isFraud', 'TransactionID', 'TransactionDT']).

    Returns:
        A list of column names suitable for seq_feature_cols.
    """
    # Example logic: Grab numeric columns, exclude IDs, targets, and categorical one-hots
    potential_cols=df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter out exclusions
    seq_cols=[col for col in potential_cols if col not in exclude_cols]

    # PRO-TIP for IEEE-CIS: The 'V' features (V1-V339) are mostly static PCA components and don't change much transaction-to-transaction.

    logger.warning(f"Identified {len(seq_cols)} sequence features. Consider manually curating this list to avoid overly large sequence tensors.")
    return seq_cols
