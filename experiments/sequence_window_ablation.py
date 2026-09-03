"""
Experiment 2: Sequence Window Ablation Study.
Tests K in {1, 3, 5, 10} to prove K=5 is optimal.
Generates figures/sequence_window_study.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src.utils.config import load_config
from src.data.sequence_extractor import build_historical_sequences
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main() -> None:
    cfg = load_config()
    processed_dir: Path = cfg.get_path("data.processed_dir")
    figures_dir: Path = cfg.get_path("paths.figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    seq_cols=cfg.sequence.feature_cols
    group_cols=cfg.sequence.group_cols
    target_col=cfg.features.target_col

    # Use a 10% SAMPLE to make this ablation run in shorter time
    SAMPLE_FRAC=0.1

    logger.info("Loading data for sequence ablation (using 10% sample)...")
    train_df=pd.read_parquet(processed_dir/"train.parquet").sample(frac=SAMPLE_FRAC, random_state=42)
    val_df=pd.read_parquet(processed_dir/"val.parquet").sample(frac=SAMPLE_FRAC,random_state=42)

    k_values=[1,3,5,10]
    pr_aucs=[]

    for k in k_values:
        logger.info(f"--- Evaluating K={k} ---")

        # 1. Extract sequences for this specific K
        train_seq=build_historical_sequences(train_df,group_cols,seq_cols,k)
        val_seq=build_historical_sequences(val_df,group_cols,seq_cols,k)

        # 2. Flatten the sequence arrays into 1D features for LightGBM
        def flatten_sequences(df: pd.DataFrame)->pd.DataFrame:
            seq_arrays=np.stack(df['sequence_array'].values)
            n_features=seq_arrays.shape[2]
            flattened_cols = [f"{col}_lag{i+1}" for i in range(k) for col in seq_cols]
            seq_df=pd.DataFrame(seq_arrays.reshape(len(df),-1),columns=flattened_cols,index=df.index)
            return pd.concat([df.drop(columns=['sequence_array']),seq_df],axis=1)

        train_flat=flatten_sequences(train_seq)
        val_flat=flatten_sequences(val_seq)

        # 3. Prepare X/y (Select only numeric columns, drop objects/strings)
        X_train = train_flat.select_dtypes(exclude=['object'])
        X_val = val_flat.select_dtypes(exclude=['object'])
        
        # --- Explicitly drop IDs and the TARGET COLUMN ---
        leaky_cols = ['TransactionID', 'TransactionDT', target_col]
        X_train = X_train.drop(columns=leaky_cols)
        X_val = X_val.drop(columns=leaky_cols)
        
        y_train = train_flat[target_col]
        y_val = val_flat[target_col]

        # 4. Train a fast LightGBM
        scale_weight=(y_train==0).sum()/(y_train==1).sum()
        lgbm=LGBMClassifier(n_estimators=50,scale_pos_weight=scale_weight,verbose=-1,random_state=42)
        lgbm.fit(X_train,y_train)

        # 5. Evaluate
        val_probs=lgbm.predict_proba(X_val)[:,1]
        pr_auc=average_precision_score(y_val,val_probs)
        pr_aucs.append(pr_auc)
        logger.info(f"K={k} -> PR-AUC: {pr_auc:.4f}")

    # 6. Plot Results
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, pr_aucs, marker='o', linewidth=2, color='b')
    plt.xlabel('Sequence Window Size (K)', fontsize=12)
    plt.ylabel('Validation PR-AUC', fontsize=12)
    plt.title('Exp 2: Impact of Historical Window Size (K)', fontsize=14)
    plt.xticks(k_values)
    plt.grid(alpha=0.3)

    save_path = figures_dir / "sequence_window_study.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sequence ablation plot to {save_path}")

if __name__ == "__main__":
    main()
