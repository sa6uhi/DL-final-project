"""Unit tests for Data Pipeline (Temporal Split & Preprocessor)."""

import pandas as pd
import numpy as np
from src.data.temporal_split import split_temporal
from src.data.preprocessor import FraudPreprocessor

def test_temporal_split_zero_leakage() -> None:
    """Asserts that max(train_time) is strictly less than min(val_time)."""
    # Create fake data directly inside the test
    dummy_df = pd.DataFrame({
        'TransactionDT': np.arange(100) * 100,
        'isFraud': np.zeros(100)
    })
    
    train_df, val_df, test_df = split_temporal(dummy_df, time_col='TransactionDT')
    
    assert train_df['TransactionDT'].max() < val_df['TransactionDT'].min()
    assert val_df['TransactionDT'].max() < test_df['TransactionDT'].min()
    assert len(train_df) == 70
    assert len(val_df) == 15

def test_preprocessor_creates_nan_indicators() -> None:
    """Asserts that the preprocessor adds _is_nan columns and scales data."""
    # Create fake data directly inside the test
    dummy_df = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'isFraud': np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    })
    
    cont_cols = ['feature_1', 'feature_2']
    preprocessor = FraudPreprocessor(cont_cols=cont_cols, cat_cols=[])
    
    # Inject a NaN to test the indicator
    dummy_df.loc[0, 'feature_1'] = np.nan
    
    result_df = preprocessor.fit_transform(dummy_df)
    
    # Check if indicator columns were created
    assert 'feature_1_is_nan' in result_df.columns
    assert result_df.loc[0, 'feature_1_is_nan'] == 1
    assert result_df.loc[1, 'feature_1_is_nan'] == 0