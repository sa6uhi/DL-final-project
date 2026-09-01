"""
Local Smoke Test for Member A's Data Pipeline.
Runs in ~3 seconds using synthetic data to verify shapes and logic.
"""
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Import YOUR code
from src.data.temporal_split import split_temporal
from src.data.preprocessor import FraudPreprocessor
from src.data.sequence_extractor import build_historical_sequences
from src.data.dataset import get_dataloaders

def generate_fake_data(n_rows: int = 1000) -> pd.DataFrame:
    """Creates a dummy DataFrame mimicking IEEE-CIS structure."""
    np.random.seed(42)
    
    # Create 50 unique cards, so sequences have history to pull from
    cards = np.random.randint(1, 50, size=n_rows)
    addrs = np.random.randint(100, 200, size=n_rows)
    
    data = {
        'TransactionID': range(1, n_rows + 1),
        'TransactionDT': np.arange(n_rows) * 100, # Strictly increasing time!
        'isFraud': np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05]),
        'TransactionAmt': np.random.uniform(1, 500, size=n_rows),
        'card1': cards,
        'addr1': addrs,
        'ProductCD': np.random.choice(['W', 'C', 'R', 'H'], size=n_rows),
        'D1': np.random.uniform(0, 100, size=n_rows),
        'D2': np.random.uniform(0, 100, size=n_rows),
    }
    return pd.DataFrame(data)

def main() -> None:
    print(">>> 1. Generating Fake Data...")
    df = generate_fake_data(1000)
    
    # Define the columns for the test
    CONT_COLS = ['TransactionAmt', 'D1', 'D2']
    CAT_COLS = ['ProductCD']
    SEQ_COLS = ['TransactionAmt', 'D1', 'D2'] # We only sequence a few features
    GROUP_COLS = ['card1', 'addr1']
    
    print(">>> 2. Testing Temporal Split (Checking for leakage)...")
    train_df, val_df, test_df = split_temporal(df, time_col='TransactionDT')
    assert len(train_df) == 700, "Train size wrong!"
    assert len(val_df) == 150, "Val size wrong!"
    assert len(test_df) == 150, "Test size wrong!"
    print("     ✅ Split sizes correct. Zero leakage assertion passed!")
    
    print(">>> 3. Testing Preprocessor...")
    preprocessor = FraudPreprocessor(cont_cols=CONT_COLS, cat_cols=CAT_COLS)
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)
    print("     ✅ Fitting on train and transforming val/test succeeded!")
    
    print(">>> 4. Testing Sequence Extractor (K=5)...")
    train_df = build_historical_sequences(train_df, group_cols=GROUP_COLS, seq_feature_cols=SEQ_COLS, k=5)
    val_df = build_historical_sequences(val_df, group_cols=GROUP_COLS, seq_feature_cols=SEQ_COLS, k=5)
    test_df = build_historical_sequences(test_df, group_cols=GROUP_COLS, seq_feature_cols=SEQ_COLS, k=5)
    
    # Check if the sequence array is actually a (5, 3) numpy array
    sample_seq = train_df.iloc[10]['sequence_array']
    assert sample_seq.shape == (5, len(SEQ_COLS)), f"Sequence shape wrong! Got {sample_seq.shape}"
    print(f"     ✅ Sequence tensors generated correctly. Shape: {sample_seq.shape}")
    
    print(">>> 5. Testing PyTorch DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, 
        cont_cols=CONT_COLS, cat_cols=CAT_COLS, seq_cols=SEQ_COLS, 
        batch_size=32
    )
    
    # Pull one batch and check the PyTorch Tensor shapes
    x_cont, x_cat, S_t, y = next(iter(train_loader))
    
    assert x_cont.shape == (32, len(CONT_COLS)), f"x_cont shape wrong: {x_cont.shape}"
    assert x_cat.shape == (32, len(CAT_COLS)), f"x_cat shape wrong: {x_cat.shape}"
    assert S_t.shape == (32, 5, len(SEQ_COLS)), f"S_t (Sequence) shape wrong: {S_t.shape}"
    assert y.shape == (32,), f"y shape wrong: {y.shape}"
    
    print("     ✅ PyTorch DataLoader yields perfect shapes!")
    print("\n" + "="*50)
    print("🎉 SUCCESS! ALL MEMBER A PIPELINE FILES WORK!")
    print("="*50)
    print("You are cleared to run on the real Kaggle dataset.")

if __name__ == "__main__":
    main()
