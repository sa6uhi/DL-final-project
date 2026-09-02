"""Classical ML Baseline Models for Fraud Detection.

Implements Class-Weighted Logistic Regression, Balanced Random Forest,
and LightGBM with dynamic scale_pos_weight calculation.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_scale_pos_weight(y: Any) -> float:
    """Calculates scale_pos_weight to handle imbalanced data in tree models.

    Args:
        y: Array-like of target labels (0 for legitimate, 1 for fraud).

    Returns:
        Float ratio of negative counts to positive counts.
    """
    neg_counts = (y == 0).sum()
    pos_counts = (y == 1).sum()
    return neg_counts / pos_counts if pos_counts > 0 else 1.0


def get_baselines(y_train: Any) -> Dict[str, Any]:
    """Instantiates the baseline models with fraud-specific hyperparameters.

    Args:
        y_train: Training labels used to calculate imbalance ratio.

    Returns:
        Dictionary mapping model names to instantiated sklearn/LGBM objects.
    """
    scale_weight = get_scale_pos_weight(y_train)
    logger.info(f"Calculated scale_pos_weight for imbalanced data: {scale_weight:.2f}")

    models = {
        "LogReg_Balanced": LogisticRegression(
            class_weight="balanced", max_iter=1000, n_jobs=-1, random_state=42
        ),
        "RandomForest_Balanced": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        ),
        "LightGBM_Weighted": LGBMClassifier(
            n_estimators=200,
            scale_pos_weight=scale_weight,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
    }
    return models
