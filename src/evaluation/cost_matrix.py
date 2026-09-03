"""Financial Cost Matrix for Fraud Detection.

Moves evaluation beyond standard ML metrics (PR-AUC) to actual business impact.
Formula: Total Cost = (Audit Cost * False Positives) + (Transaction Amount * False Negatives)
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_financial_cost(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    transaction_amounts: np.ndarray | pd.Series,
    audit_cost: float = 5.0,
) -> Dict[str, float]:
    """Calculates the financial impact of the model's predictions.

    Args:
        y_true: Ground truth labels (0 = legitimate, 1 = fraud).
        y_pred: Predicted labels (0 = legitimate, 1 = fraud).
        transaction_amounts: The dollar amount of each transaction.
        audit_cost: The operational cost of manually reviewing a False Positive.

    Returns:
        Dictionary containing cost breakdown and total savings.
    """
    # Ensure numpy arrays for fast calculation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amounts = np.asarray(transaction_amounts)

    # Identify specific error types
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)

    # Calculate costs
    fp_cost = np.sum(false_positives) * audit_cost
    fn_cost = np.sum(amounts[false_negatives])
    total_model_cost = fp_cost + fn_cost

    # Calculate baseline cost
    total_fraud_loss = np.sum(amounts[y_true == 1])

    # Calculate savings
    total_savings = total_fraud_loss - total_model_cost

    results = {
        "total_model_cost": total_model_cost,
        "false_positive_audit_cost": fp_cost,
        "false_negative_fraud_loss": fn_cost,
        "total_fraud_loss_if_no_model": total_fraud_loss,
        "net_savings_vs_no_model": total_savings,
        "savings_percentage": (
            (total_savings / total_fraud_loss * 100)
            if total_fraud_loss > 0
            else 0.0
        ),
    }

    logger.info(
        f"Financial Evaluation -> Model Cost: ${total_model_cost:,.2f} "
        f"| Net Savings: ${total_savings:,.2f} ({results['savings_percentage']:.1f}%)"
    )
    return results
