"""Unit tests for Baseline Models and Metrics."""

import numpy as np
from src.models.baselines import get_baselines, get_scale_pos_weight
from src.evaluation.metrics_a import evaluate_fraud_metrics

def test_get_baselines_returns_three_models() -> None:
    """Asserts that the baseline factory returns exactly 3 models."""
    y_dummy = np.array([0]*90 + [1]*10)
    models = get_baselines(y_dummy)
    
    assert isinstance(models, dict)
    assert len(models) == 3
    assert "LightGBM_Weighted" in models

def test_scale_pos_weight_calculation() -> None:
    """Asserts the imbalance ratio is calculated correctly."""
    y_dummy = np.array([0]*90 + [1]*10) # 9:1 ratio
    weight = get_scale_pos_weight(y_dummy)
    assert weight == 9.0

def test_evaluate_metrics_returns_pr_auc() -> None:
    """Asserts the metrics function returns a dictionary with PR-AUC."""
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.4, 0.8, 0.9]) # Perfect predictions
    
    metrics = evaluate_fraud_metrics(y_true, y_probs, "TestModel")
    
    assert isinstance(metrics, dict)
    assert "PR-AUC" in metrics
    assert metrics["PR-AUC"] == 1.0  # Perfect predictions = 1.0 PR-AUC