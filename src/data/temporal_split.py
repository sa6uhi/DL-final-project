import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.logger import get_logger
logger = get_logger(__name__)

def load_and_merge_data(transaction_path:str,identity_path:str)->pd.DataFrame:
    """Loads and merges transaction and identity datasets.

    Args:
        transaction_path: Path to train_transaction.csv.
        identity_path: Path to train_identity.csv.

    Returns:
        A merged Pandas DataFrame on TransactionID.
    """
    logger.info(f"Loading transactions from {transaction_path}...")
    df_trans=pd.read_csv(transaction_path)

    logger.info(f"Loading identity from {identity_path}...")
    df_id=pd.read_csv(identity_path)

    # Outer join to keep all transactions, even if identity is missing
    df_merged=df_trans.merge(df_id, on="TransactionID",how="left")
    logger.info(f"Merged dataset shape: {df_merged.shape}")
    return df_merged

def split_temporal(df:pd.DataFrame,time_col:str="TransactionDT")->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Enforces strict chronological 70/15/15 split with zero leakage assertion.

    Args:
        df: The merged input DataFrame.
        time_col: The column representing chronological time.

    Returns:
        A tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If temporal leakage is detected between splits.
    """
    # Ensure data is strictly sorted by time
    df=df.sort_values(by=time_col).reset_index(drop=True)

    # Calculate split indices
    n_total=len(df)
    idx_train_end=int(n_total*0.7)
    idx_val_end=int(n_total*0.85)

    train_df=df.iloc[:idx_train_end].copy()
    val_df=df.iloc[idx_train_end:idx_val_end].copy()
    test_df=df.iloc[idx_val_end:].copy()

    # ZERO LEAKAGE ASSERTION (Crucial for IEEE Paper)
    max_train_time=train_df[time_col].max()
    min_val_time=val_df[time_col].min()
    min_test_time=test_df[time_col].min()

    assert max_train_time < min_val_time, "TEMPORAL LEAKAGE: Train overlaps with Validation!"
    assert min_val_time < min_test_time, "TEMPORAL LEAKAGE: Validation overlaps with Test!"

    logger.info(f"Temporal Split Complete - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Time Boundaries - Train_End: {max_train_time}, Val_Start: {min_val_time}, Test_Start: {min_test_time}")
    
    return train_df, val_df, test_df
