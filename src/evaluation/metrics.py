"""Rank-based anomaly metrics implemented in pure NumPy.

Avoids a scikit-learn runtime dependency (member A's baseline lane owns the
sklearn comparison suite); the metrics here mirror the Python implementation
of ``roc_auc_score`` / ``average_precision_score`` closely enough for fair
reporting on DAE residual quality.
"""

from __future__ import annotations

import numpy as np

from typing import Dict

from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_scores_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and cast scoring inputs to 1D float arrays.

    Args:
        scores: Per-sample anomaly scores.
        labels: Binary ground-truth labels (0=legit, 1=fraud).

    Returns:
        Tuple of ``(score_float, label_int)`` flattened arrays.

    Raises:
        ValueError: If lengths differ, arrays are empty, or NaN values are
            present.
    """
    scores_1d = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels_1d = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores_1d.shape[0] != labels_1d.shape[0]:
        raise ValueError(
            f"Length mismatch: {scores_1d.shape[0]} scores vs {labels_1d.shape[0]} labels"
        )
    if scores_1d.size == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set")
    if not np.isfinite(scores_1d).all():
        raise ValueError("Anomaly scores contain NaN or infinite values")
    if not np.isin(labels_1d, [0, 1]).all():
        raise ValueError("Labels must be binary (0/1)")
    return scores_1d, labels_1d


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute the area under the ROC curve via rank statistics.

    Uses the Mann-Whitney U-Wilcoxon formulation
    ``AUC = sum(ranks_pos) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)`` with the
    average-rank convention for ties.

    Args:
        scores: Per-sample anomaly scores.
        labels: Binary ground-truth labels.

    Returns:
        AUC in ``[0, 1]`` (0.5 = random, 1.0 = perfect).

    Raises:
        ValueError: If fewer than two classes are present.
    """
    scores_f, labels_f = validate_scores_labels(scores, labels)
    n_pos = int(labels_f.sum())
    n_neg = int(labels_f.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"AUC requires both classes present (got pos={n_pos}, neg={n_neg})")
    order = np.argsort(scores_f, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # Average-rank tie handling.
    ranks[order] = np.arange(1, scores_f.shape[0] + 1, dtype=np.float64)
    sorted_scores = scores_f[order]
    # Tie groups get the mean rank of their span.
    unique_vals, inv = np.unique(sorted_scores, return_inverse=True)
    if unique_vals.size < scores_f.size:
        sums = np.bincount(inv, weights=np.arange(1, scores_f.shape[0] + 1, dtype=np.float64))
        counts = np.bincount(inv, minlength=unique_vals.size)
        mean_ranks = np.repeat(sums / counts, counts)
        ranks[order] = mean_ranks
    sum_pos = float(ranks[labels_f == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    # Numerical sym-float guard: clamp tie-degenerate results to [0, 1].
    return float(min(max(auc, 0.0), 1.0))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute the area under the precision-recall curve.

    Points are visited in descending score order; the average precision is
    the rectilinear integral of precision over recall.

    Args:
        scores: Per-sample anomaly scores.
        labels: Binary ground-truth labels.

    Returns:
        Average precision in ``[0, 1]``.

    Raises:
        ValueError: If fewer than two classes are present.
    """
    scores_f, labels_f = validate_scores_labels(scores, labels)
    n_pos = int(labels_f.sum())
    if n_pos == 0:
        raise ValueError("Average precision requires at least one positive sample")
    order = np.lexsort((labels_f, scores_f))[::-1]
    ranked_labels = labels_f[order]
    tp = ranked_labels.astype(np.float64).cumsum()
    rec = tp / n_pos
    prec = tp / (np.arange(1, ranked_labels.shape[0] + 1, dtype=np.float64))
    # Rectilinear integration from recall = 0 (sklearn convention):
    # AP = sum_i (rec_i - rec_{i-1}) * prec_i, with rec_0 = 0.
    prev_rec = 0.0
    ap = 0.0
    for r, p in zip(rec, prec):
        ap += (float(r) - prev_rec) * float(p)
        prev_rec = float(r)
    return ap


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.01) -> float:
    """Report the true-positive rate achieved at or below a target FPR.

    The ROC curve is traversed by descending score thresholds; the returned
    TPR is that of the first operating point with ``fpr <= max_fpr``.

    Args:
        scores: Per-sample anomaly scores.
        labels: Binary ground-truth labels.
        max_fpr: Target false-positive rate in ``[0, 1]``.

    Returns:
        True-positive rate in ``[0, 1]``; worst case equals the prevalence
        (all negatives cleared, positives still non-zero).

    Raises:
        ValueError: If ``max_fpr`` is outside ``[0, 1]``.
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")
    scores_f, labels_f = validate_scores_labels(scores, labels)
    n_pos = int(labels_f.sum())
    n_neg = int(labels_f.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"tpr@fpr requires both classes present (got pos={n_pos}, neg={n_neg})")
    order = np.lexsort((labels_f, scores_f))[::-1]
    ranked_labels = labels_f[order]
    tp = ranked_labels.astype(np.float64).cumsum()
    tpr = tp / n_pos
    fp = (1 - ranked_labels).astype(np.float64).cumsum()
    fpr = fp / n_neg
    # Compare at a fixed per-rank operating point to keep the scan O(1).
    mask = fpr <= max_fpr
    if not mask.any():
        raise ValueError(f"No operating point with fpr <= {max_fpr} exists")
    best = mask.sum() - 1
    return float(tpr[best])


def summarize(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.01) -> dict[str, float]:
    """One-call metric summary for an anomaly score vector.

    Args:
        scores: Per-sample anomaly scores.
        labels: Binary ground-truth labels.
        max_fpr: Target FPR for the TPR operating point.

    Returns:
        Dict with ``rocauc``, ``auprc`` and ``tpr_at_fpr`` keys.
    """
    auc = roc_auc(scores, labels)
    ap = average_precision(scores, labels)
    tpr = tpr_at_fpr(scores, labels, max_fpr=max_fpr)
    summary = {"rocauc": round(auc, 6), "auprc": round(ap, 6), "tpr_at_fpr": round(tpr, 6)}
    logger.info("Anomaly metrics: %s", summary)
    return summary

"""Evaluation metrics for fraud detection."""

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

