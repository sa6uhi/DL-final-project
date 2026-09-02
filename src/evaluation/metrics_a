"""Evaluation metrics for fraud detection."""

from typing import Dict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_fraud_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, model_name: str
) -> Dict[str, float]:
    """Calculates PR-AUC and ROC-AUC.

    Args:
        y_true: Ground truth binary labels.
        y_probs: Predicted probabilities for the positive class (fraud).
        model_name: Name of the model for logging.

    Returns:
        Dictionary containing PR-AUC and ROC-AUC scores.
    """
    pr_auc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    logger.info(f"{model_name} -> PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
    return {"PR-AUC": pr_auc, "ROC-AUC": roc_auc}
