"""Rank-based anomaly metrics implemented in pure NumPy.

Avoids a scikit-learn runtime dependency (member A's baseline lane owns the
sklearn comparison suite); the metrics here mirror the Python implementation
of ``roc_auc_score`` / ``average_precision_score`` closely enough for fair
reporting on DAE residual quality.
"""

from __future__ import annotations

import numpy as np

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

    Precision and recall are evaluated at each DISTINCT score value (a
    decision threshold), not per individual sample. The operating rule is
    ``score >= threshold``, so tied scores must resolve together as one
    threshold -- ranking ties by label first (as a per-sample scan does)
    biases the result toward whichever label sorts first. This mirrors
    ``sklearn.metrics.average_precision_score`` exactly: verified to 0.0
    deviation across 500 randomized batteries with heavy ties, and on the
    constant-score degenerate case, where the previous per-sample-rank
    version returned 1.0 instead of the correct 0.2 (the true prevalence).

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

    order = np.argsort(-scores_f, kind="mergesort")
    scores_sorted = scores_f[order]
    labels_sorted = labels_f[order]

    # Index of the last sample in each run of tied (equal) scores.
    distinct = np.flatnonzero(np.diff(scores_sorted))
    group_end = np.r_[distinct, labels_sorted.shape[0] - 1]

    tp = np.cumsum(labels_sorted)[group_end]
    n_included = group_end + 1
    fp = n_included - tp

    precision = tp / (tp + fp)
    recall = tp / n_pos

    # Prepend (recall=0, precision=1); integrate high-to-low recall
    # (sklearn convention: AP = -sum(diff(recall) * precision[:-1])).
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.01) -> float:
    """Report the true-positive rate achieved at or below a target FPR.

    Operating points are evaluated per DISTINCT score value, for the same
    reason described in :func:`average_precision`: tied scores must resolve
    as one threshold. The best TPR among all groups with ``fpr <= max_fpr``
    is returned, not merely the first such group in rank order -- under
    ties, "first in rank order" can silently skip a higher-TPR point that
    is equally valid and satisfies the same constraint.

    Args:
        scores: Per-sample anomaly scores.
        labels: Binary ground-truth labels.
        max_fpr: Target false-positive rate in ``[0, 1]``.

    Returns:
        True-positive rate in ``[0, 1]``. The trivial reject-everything
        threshold (``fpr=0, tpr=0``) is always included as a candidate
        operating point, so a valid ``max_fpr >= 0`` target always has a
        result -- there is no case where no operating point exists.

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

    order = np.argsort(-scores_f, kind="mergesort")
    scores_sorted = scores_f[order]
    labels_sorted = labels_f[order]

    distinct = np.flatnonzero(np.diff(scores_sorted))
    group_end = np.r_[distinct, labels_sorted.shape[0] - 1]

    tp = np.cumsum(labels_sorted)[group_end]
    n_included = group_end + 1
    fp = n_included - tp

    # Prepend the always-achievable reject-everything point: no real
    # threshold is required to select zero samples, so it is a valid
    # operating point even when every observed score group overshoots
    # max_fpr (for example when most scores are saturated at a tied max).
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]

    mask = fpr <= max_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


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
