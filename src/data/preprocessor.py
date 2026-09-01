import pandas as pd
import numpy as np
import pickle
from typing import Tuple,Dict,Any
from sklearn.preprocessing import RobustScaler
from src.utils.config import load_config
from src.utils.logger import get_logger
logger = get_logger(__name__)

class FraudPreprocessor:
    """Handles scaling, missingness indicators, and categorical encoding."""

    def __init__(self, cont_cols: list, cat_cols: list) -> None:
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.scaler = RobustScaler()
        self.cat_vocabularies: Dict[str, Dict[Any, int]] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits scalers and vocabularies strictly on the training set."""
        logger.info("Fitting preprocessor on training data...")
        df_processed = df.copy()
        
        # 1. Continuous Features: Handle NaNs and Log transform
        # Create all missingness indicators AT ONCE (fixes the fragmentation warning)
        nan_indicators = df_processed[self.cont_cols].isna().astype(int)
        nan_indicators.columns = [f"{col}_is_nan" for col in self.cont_cols]
        
        # Fill NaNs with 0 for math operations
        df_processed[self.cont_cols] = df_processed[self.cont_cols].fillna(0)
        
        # Log transform. Clip negatives to 0 to avoid the "invalid value in log1p" warning
        df_processed[self.cont_cols] = np.log1p(df_processed[self.cont_cols].clip(lower=0))
        
        # Fit and transform scaler
        scaled_data = self.scaler.fit_transform(df_processed[self.cont_cols])
        df_processed[self.cont_cols] = scaled_data
        
        # Add all 390 NaN indicator columns in ONE fast operation
        df_processed = pd.concat([df_processed, nan_indicators], axis=1)
        
        # 2. Categorical Features: Build vocab & map to ints
        for col in self.cat_cols:
            df_processed[col] = df_processed[col].fillna("MISSING")
            unique_cats = df_processed[col].unique()
            self.cat_vocabularies[col] = {cat: idx + 1 for idx, cat in enumerate(unique_cats)}
            df_processed[col] = df_processed[col].map(self.cat_vocabularies[col])
            
        return df_processed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms validation/test data. Maps unseen categories to <UNK> (0)."""
        logger.info("Transforming data using fitted preprocessor...")
        df_processed = df.copy()
        
        # Fast vectorized continuous transform
        nan_indicators = df_processed[self.cont_cols].isna().astype(int)
        nan_indicators.columns = [f"{col}_is_nan" for col in self.cont_cols]
        
        df_processed[self.cont_cols] = df_processed[self.cont_cols].fillna(0)
        df_processed[self.cont_cols] = np.log1p(df_processed[self.cont_cols].clip(lower=0))
        
        scaled_data = self.scaler.transform(df_processed[self.cont_cols])
        df_processed[self.cont_cols] = scaled_data
        
        df_processed = pd.concat([df_processed, nan_indicators], axis=1)
        
        # Categorical transform
        for col in self.cat_cols:
            df_processed[col] = df_processed[col].fillna("MISSING")
            df_processed[col] = df_processed[col].map(self.cat_vocabularies[col]).fillna(0).astype(int)
            
        return df_processed

    def save(self,path:str)->None:
        with open(path,'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")
