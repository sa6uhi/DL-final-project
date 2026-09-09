"""Cross-attention extraction and heatmap rendering for the FT-CAT transformer.

:class:`~src.models.ft_transformer.FTCATransformer` already exposes its raw
temporal cross-attention weights through ``forward(..., return_attention=True)``.
That tensor is a per-head map of shape ``(batch, n_heads, n_query_tokens,
seq_len)`` describing, for every feature token of the *current* transaction,
how much probability mass the model placed on each of the ``K`` prior
cardholder transactions.

This module turns those raw weights into the artefacts the paper, the slide
deck, and the analyst dashboard consume:

* :func:`collect_attention` runs a batched, gradient-free forward pass and
  captures the weights alongside the scores and labels of the same rows.
* :func:`cls_history_attention` isolates the ``[CLS]`` query row -- the only
  token the classification head actually reads -- which is the honest answer to
  "which past transaction drove this decision?".
* :func:`mean_history_attention` and :func:`top_attended_features` aggregate a
  capture for reporting.
* :func:`plot_cross_attention_heatmap` and :func:`plot_case_study` render the
  figures.

Only the ``ft_cat`` variant has cross-attention at all; the ``ft_self_only``
and ``mlp`` ablation arms deliberately never look at history, so a capture is
undefined for them and is rejected with an explicit error rather than a silent
``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.models.ft_transformer import AttentionMaps, HistoryEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Query index of the learnable [CLS] token. FeatureTokenizer.forward prepends
# it before the continuous and categorical feature tokens, so it is always 0.
CLS_QUERY_INDEX: int = 0

DEFAULT_BATCH_SIZE: int = 512
DEFAULT_MAX_ROWS: int = 4096


@dataclass(frozen=True)
class AttentionCapture:
    """Cross-attention weights captured over a set of scored transactions.

    Attributes:
        weights: Per-head attention of shape ``(n_rows, n_heads, n_queries,
            seq_len)``. Row ``i``, head ``h``, query ``q`` is a distribution
            over the ``K`` history slots.
        scores: Fraud probabilities for the same rows, shape ``(n_rows,)``.
        labels: Binary ground-truth labels for the same rows, ``(n_rows,)``.
        padding: Boolean mask of shape ``(n_rows, seq_len)`` flagging history
            slots that are zero padding rather than real prior transactions.
    """

    weights: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    padding: np.ndarray

    def __post_init__(self) -> None:
        """Validate that every array describes the same rows.

        Raises:
            ValueError: If the weights are not 4D, or if the scores, labels,
                and padding mask disagree with the weights on row count or on
                the history length.
        """
        if self.weights.ndim != 4:
            raise ValueError(
                f"weights must be 4D (rows, heads, queries, seq_len), "
                f"got {tuple(self.weights.shape)}"
            )
        n_rows, _, _, seq_len = self.weights.shape
        for name, array in (("scores", self.scores), ("labels", self.labels)):
            if array.shape != (n_rows,):
                raise ValueError(f"{name} must have shape ({n_rows},), got {tuple(array.shape)}")
        if self.padding.shape != (n_rows, seq_len):
            raise ValueError(
                f"padding must have shape ({n_rows}, {seq_len}), got {tuple(self.padding.shape)}"
            )

    def __len__(self) -> int:
        """Return the number of captured rows."""
        return int(self.weights.shape[0])

    @property
    def n_heads(self) -> int:
        """Number of attention heads."""
        return int(self.weights.shape[1])

    @property
    def n_queries(self) -> int:
        """Number of query tokens, including ``[CLS]``."""
        return int(self.weights.shape[2])

    @property
    def seq_len(self) -> int:
        """History window length ``K``."""
        return int(self.weights.shape[3])

    def fraud_mask(self) -> np.ndarray:
        """Return a boolean mask selecting the fraudulent rows."""
        return self.labels == 1

    def legitimate_mask(self) -> np.ndarray:
        """Return a boolean mask selecting the legitimate rows."""
        return self.labels == 0


def collect_attention(
    model: nn.Module,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    seq: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rows: int | None = DEFAULT_MAX_ROWS,
) -> AttentionCapture:
    """Run a gradient-free forward pass and capture the cross-attention maps.

    The full weight tensor holds ``n_rows * n_heads * n_queries * seq_len``
    floats, which for the production feature contract is several kilobytes per
    row. ``max_rows`` therefore caps the capture at a size that stays inside
    the project RAM envelope; it truncates from the front of the supplied
    rows, so callers wanting a specific subset should slice before calling.

    Args:
        model: A trained ``ft_cat`` model exposing ``return_attention``.
        x_cont: Continuous features, ``(n_rows, n_continuous)``.
        x_cat: Categorical codes, ``(n_rows, n_categorical)``.
        seq: History windows, ``(n_rows, seq_len, seq_dim)``.
        labels: Binary fraud labels, ``(n_rows,)``.
        device: Device to run the forward pass on.
        batch_size: Rows scored per forward pass.
        max_rows: Optional cap on the number of rows captured; ``None`` or a
            non-positive value captures every row.

    Returns:
        An :class:`AttentionCapture` for the captured rows.

    Raises:
        ValueError: If ``batch_size`` is not positive, if the inputs disagree
            on row count, or if the model returns no cross-attention -- which
            is the case for the ``ft_self_only`` and ``mlp`` ablation arms.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    rows = {
        "x_cont": int(x_cont.shape[0]),
        "x_cat": int(x_cat.shape[0]),
        "seq": int(seq.shape[0]),
        "labels": int(labels.shape[0]),
    }
    if len(set(rows.values())) != 1:
        raise ValueError(f"inputs disagree on row count: {rows}")

    n_rows = int(x_cont.shape[0])
    if n_rows == 0:
        raise ValueError("cannot capture attention over an empty batch")
    if max_rows is not None and 0 < max_rows < n_rows:
        logger.info("Capping attention capture at %d of %d rows", max_rows, n_rows)
        n_rows = int(max_rows)

    model = model.to(device)
    was_training = model.training
    model.eval()

    weight_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    try:
        with torch.no_grad():
            for start in range(0, n_rows, batch_size):
                stop = min(start + batch_size, n_rows)
                output = model(
                    x_cont[start:stop].to(device),
                    x_cat[start:stop].to(device),
                    seq[start:stop].to(device),
                    return_attention=True,
                )
                if not isinstance(output, tuple):
                    raise ValueError(
                        "model did not honour return_attention=True; expected a "
                        "(logits, AttentionMaps) tuple"
                    )
                logits, maps = output
                if not isinstance(maps, AttentionMaps) or maps.cross_attn is None:
                    raise ValueError(
                        "model returned no cross-attention weights; only the 'ft_cat' "
                        "variant carries a temporal cross-attention layer"
                    )
                weight_chunks.append(maps.cross_attn.detach().float().cpu().numpy())
                score_chunks.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
    finally:
        model.train(was_training)

    capture = AttentionCapture(
        weights=np.concatenate(weight_chunks, axis=0),
        scores=np.concatenate(score_chunks, axis=0),
        labels=labels[:n_rows].detach().cpu().numpy().astype(np.int64),
        padding=HistoryEncoder.padding_mask(seq[:n_rows]).detach().cpu().numpy(),
    )
    logger.info(
        "Captured cross-attention for %d rows: %d heads, %d query tokens, K=%d",
        len(capture),
        capture.n_heads,
        capture.n_queries,
        capture.seq_len,
    )
    return capture


def cls_history_attention(capture: AttentionCapture) -> np.ndarray:
    """Extract the attention the ``[CLS]`` token pays to history.

    The classification head reads only the ``[CLS]`` embedding, so this row of
    the attention matrix is the one that genuinely explains the decision; the
    other query rows describe how individual feature tokens were contextualised
    on the way there.

    Args:
        capture: A capture produced by :func:`collect_attention`.

    Returns:
        Array of shape ``(n_rows, n_heads, seq_len)``.
    """
    return capture.weights[:, :, CLS_QUERY_INDEX, :]


def mean_history_attention(capture: AttentionCapture, mask: np.ndarray | None = None) -> np.ndarray:
    """Average the ``[CLS]`` history attention over a subset of rows.

    Args:
        capture: A capture produced by :func:`collect_attention`.
        mask: Optional boolean row mask, e.g. ``capture.fraud_mask()``. When
            omitted, every row is averaged.

    Returns:
        Array of shape ``(n_heads, seq_len)``.

    Raises:
        ValueError: If ``mask`` has the wrong length or selects no rows.
    """
    attention = cls_history_attention(capture)
    if mask is None:
        return attention.mean(axis=0)
    if mask.shape != (len(capture),):
        raise ValueError(f"mask must have shape ({len(capture)},), got {tuple(mask.shape)}")
    if not bool(mask.any()):
        raise ValueError("mask selects no rows; cannot average an empty subset")
    return attention[mask].mean(axis=0)


def top_attended_features(
    capture: AttentionCapture,
    feature_names: Sequence[str],
    top_n: int = 5,
    mask: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """Rank feature tokens by how sharply they lean on transaction history.

    Every query token emits a distribution over the ``K`` history slots, so
    total mass is constant at 1 and cannot discriminate. What varies -- and
    what marks a token as genuinely history-driven -- is how *concentrated*
    that distribution is: a token that pins one specific prior transaction is
    reacting to velocity, whereas a token spread flat across the window is
    effectively ignoring history. Concentration is measured as the peak
    attention weight, averaged over heads and rows.

    Args:
        capture: A capture produced by :func:`collect_attention`.
        feature_names: Names of the non-``[CLS]`` query tokens, in tokenizer
            order (continuous columns first, then categorical).
        top_n: Number of tokens to return; ``<= 0`` returns all of them.
        mask: Optional boolean row mask restricting the average.

    Returns:
        ``(name, peak_attention)`` pairs sorted by descending concentration.

    Raises:
        ValueError: If ``feature_names`` does not cover every non-``[CLS]``
            query token, or if ``mask`` selects no rows.
    """
    expected = capture.n_queries - 1
    if len(feature_names) != expected:
        raise ValueError(
            f"feature_names must name the {expected} non-CLS query tokens, "
            f"got {len(feature_names)}"
        )

    weights = capture.weights
    if mask is not None:
        if mask.shape != (len(capture),):
            raise ValueError(f"mask must have shape ({len(capture)},), got {tuple(mask.shape)}")
        if not bool(mask.any()):
            raise ValueError("mask selects no rows; cannot rank an empty subset")
        weights = weights[mask]

    # (rows, heads, queries, K) -> peak over K, then mean over rows and heads.
    peak = weights.max(axis=3).mean(axis=(0, 1))[CLS_QUERY_INDEX + 1 :]
    order = sorted(range(expected), key=lambda i: (-float(peak[i]), str(feature_names[i])))
    keep = expected if top_n <= 0 else min(top_n, expected)
    return [(str(feature_names[i]), float(peak[i])) for i in order[:keep]]


def _history_labels(seq_len: int) -> list[str]:
    """Return axis labels for the history window.

    ``build_historical_sequences`` writes the most recent prior transaction at
    index 0, so slot ``i`` is lag ``i + 1``.

    Args:
        seq_len: History window length ``K``.

    Returns:
        Labels such as ``["t-1", "t-2", ...]``.
    """
    return [f"t-{i + 1}" for i in range(seq_len)]


def _save_figure(fig: "plt.Figure", output_path: str | Path, show_plot: bool) -> Path:
    """Write a figure to disk, creating parent directories as needed."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A figure that already owns a layout engine (constrained layout, used
    # wherever a colorbar spans several axes) rejects tight_layout with a
    # warning and slightly wrong geometry, so only apply it when absent.
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    logger.info("Saved attention figure to %s", output)
    if show_plot:
        plt.show()
    plt.close(fig)
    return output


def plot_cross_attention_heatmap(
    capture: AttentionCapture,
    output_path: str | Path,
    show_plot: bool = False,
) -> Path:
    """Plot mean per-head history attention for fraud versus legitimate rows.

    Both panels share one colour scale so the comparison is honest: a visibly
    hotter fraud panel at recent lags is the velocity signal the cross-attention
    layer exists to capture.
    """
    fraud_mask = capture.fraud_mask()
    legit_mask = capture.legitimate_mask()
    if not fraud_mask.any() or not legit_mask.any():
        raise ValueError(
            f"heatmap needs both classes (fraud={int(fraud_mask.sum())}, "
            f"legitimate={int(legit_mask.sum())})"
        )

    panels = [
        (f"Fraudulent (n={int(fraud_mask.sum())})", mean_history_attention(capture, fraud_mask)),
        (f"Legitimate (n={int(legit_mask.sum())})", mean_history_attention(capture, legit_mask)),
    ]
    vmin = min(float(matrix.min()) for _, matrix in panels)
    vmax = max(float(matrix.max()) for _, matrix in panels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    lags = _history_labels(capture.seq_len)
    image = None
    for ax, (title, matrix) in zip(axes, panels):
        image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("History slot")
        ax.set_xticks(range(capture.seq_len))
        ax.set_xticklabels(lags)
        ax.set_yticks(range(capture.n_heads))
        ax.set_yticklabels([f"head {h}" for h in range(capture.n_heads)])
        for head in range(capture.n_heads):
            for lag in range(capture.seq_len):
                ax.text(
                    lag,
                    head,
                    f"{matrix[head, lag]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[head, lag] < (vmin + vmax) / 2 else "black",
                )

    axes[0].set_ylabel("Attention head")
    fig.colorbar(image, ax=axes, shrink=0.85, label="Mean [CLS] attention weight")
    fig.suptitle("FT-CAT temporal cross-attention: [CLS] over cardholder history")
    return _save_figure(fig, output_path, show_plot)


def plot_case_study(
    capture: AttentionCapture,
    row_index: int,
    feature_names: Sequence[str],
    output_path: str | Path,
    top_n: int = 12,
    show_plot: bool = False,
) -> Path:
    """Plot one transaction feature-token by history-lag attention map.

    Heads are averaged so the panel answers a single question -- which feature
    of this transaction looked at which prior transaction -- for the slide-deck
    case study. Only the ``top_n`` most sharply attending tokens are drawn,
    since the production contract carries over a hundred of them.
    """
    if not 0 <= row_index < len(capture):
        raise IndexError(f"row_index {row_index} out of range for {len(capture)} captured rows")

    expected = capture.n_queries - 1
    if len(feature_names) != expected:
        raise ValueError(
            f"feature_names must name the {expected} non-CLS query tokens, "
            f"got {len(feature_names)}"
        )

    # Average heads, drop the [CLS] query row, keep the sharpest tokens.
    row = capture.weights[row_index].mean(axis=0)[CLS_QUERY_INDEX + 1 :]
    keep = expected if top_n <= 0 else min(top_n, expected)
    order = sorted(range(expected), key=lambda i: (-float(row[i].max()), str(feature_names[i])))
    chosen = order[:keep]
    matrix = row[chosen]

    fig, ax = plt.subplots(figsize=(7.6, 0.34 * keep + 2.6))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(range(capture.seq_len))
    ax.set_xticklabels(_history_labels(capture.seq_len))
    ax.set_yticks(range(keep))
    ax.set_yticklabels([str(feature_names[i]) for i in chosen], fontsize=8)
    ax.set_xlabel("History slot")
    ax.set_ylabel("Feature token")

    padded = capture.padding[row_index]
    for lag in range(capture.seq_len):
        if not bool(padded[lag]):
            continue
        ax.axvspan(lag - 0.5, lag + 0.5, facecolor="none", edgecolor="#7FA98F", hatch="//", lw=0.0)
        ax.text(
            lag,
            keep / 2.0 - 0.5,
            "no history\n(masked)",
            ha="center",
            va="center",
            fontsize=7,
            rotation=90,
            color="#DDDDDD",
        )

    n_real = int(capture.seq_len - padded.sum())
    verdict = "fraud" if capture.labels[row_index] == 1 else "legitimate"
    ax.set_title(
        f"Cross-attention case study (row {row_index}, "
        f"score {capture.scores[row_index]:.3f}, label {verdict})\n"
        f"{n_real} of {capture.seq_len} history slots hold a real prior transaction",
        fontsize=10,
    )
    fig.colorbar(image, ax=ax, shrink=0.85, label="Head-averaged attention weight")
    return _save_figure(fig, output_path, show_plot)


def full_history_mask(capture: AttentionCapture) -> np.ndarray:
    """Flag rows whose ``K`` history slots all hold real prior transactions.

    Attention over a partly padded window is degenerate -- the mask forces the
    whole distribution onto the few surviving slots -- so aggregate statistics
    and case studies are only interpretable on rows with a full window.
    """
    return ~capture.padding.any(axis=1)
