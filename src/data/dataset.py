import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
logger = logging.getLogger(__name__)

class TransactionSequenceDataset(Dataset):
    """PyTorch Dataset yielding current features, historical sequences, and labels."""
    def __init__(self, df: pd.DataFrame,cont_cols: list,cat_cols: list,seq_cols:list,label_col:str='isFraud') -> None:
        """
        Args:
            df: Processed dataframe containing a column of numpy arrays for sequences.
            cont_cols: List of continuous feature column names.
            cat_cols: List of categorical feature column names.
            seq_cols: List of feature names to extract from the historical sequence.
            label_col: Name of the target column.
        """
        self.df = df.reset_index(drop=True)
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.seq_cols = seq_cols
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self,idx:int)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        row=self.df.iloc[idx]

        # 1. Current Continuous Features (x_t)
        x_cont=torch.tensor(row[self.cont_cols].values,dtype=torch.float32)

        # 2. Current Categorical Features (x_t)
        x_cat=torch.tensor(row[self.cat_cols].values,dtype=torch.long)

        # 3. Historical Sequence Tensor S_t in R^{K x D}
        seq_array=np.vstack(row['sequence_array']) # Shape: (5, num_features)

        # Extract only the specific continuous/categorical features needed for the sequence
        S_t=torch.tensor(seq_array,dtype=torch.float32)

        # 4. Label
        y=torch.tensor(row[self.label_col],dtype=torch.float32)

        return x_cont,x_cat,S_t,y

def get_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    cont_cols: list, cat_cols: list, seq_cols: list, batch_size: int = 1024) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates DataLoaders for train, val, and test sets."""

    train_dataset=TransactionSequenceDataset(train_df,cont_cols,cat_cols,seq_cols)
    val_dataset=TransactionSequenceDataset(val_df,cont_cols,cat_cols,seq_cols)
    test_dataset=TransactionSequenceDataset(test_df,cont_cols,cat_cols,seq_cols)
    
    # Shuffle only training data
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)

    logger.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader, test_loader
