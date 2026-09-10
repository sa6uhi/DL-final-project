"""Unit tests for the FT-CAT cross-attention extraction module (Member B).

Covers the capture dataclass guards, the batched forward pass and its refusal
to pretend the history-blind ablation arms have cross-attention, the ``[CLS]``
row extraction, the aggregation helpers, and both figure writers. Everything
runs on tiny randomly initialised models pinned to the CPU, so the suite stays
fast and machine-independent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.attention_maps import (
    CLS_QUERY_INDEX,
    AttentionCapture,
    cls_history_attention,
    collect_attention,
    full_history_mask,
    mean_history_attention,
    plot_case_study,
    plot_cross_attention_heatmap,
    top_attended_features,
)
from src.models.ft_transformer import FTCATransformer, TabularMLP

N_ROWS = 24
N_CONT = 3
CARDS = [3, 4]
SEQ_LEN = 5
SEQ_DIM = 2
D_MODEL = 8
N_HEADS = 2
FEATURE_NAMES = [f"c{i}" for i in range(N_CONT)] + [f"k{i}" for i in range(len(CARDS))]


def make_model(use_cross_attention: bool = True) -> FTCATransformer:
    """Build a tiny FT-CAT model matching the fixture shapes.

    Args:
        use_cross_attention: Whether to include the temporal cross-attention
            layer, i.e. ``ft_cat`` versus the ``ft_self_only`` arm.

    Returns:
        An untrained model in eval mode.
    """
    torch.manual_seed(0)
    model = FTCATransformer(
        n_continuous=N_CONT,
        categorical_cardinalities=CARDS,
        seq_len=SEQ_LEN,
        seq_dim=SEQ_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dim_feedforward=16,
        n_layers=1,
        dropout=0.0,
        use_cross_attention=use_cross_attention,
    )
    return model.eval()


def make_inputs(
    n_rows: int = N_ROWS, seed: int = 0, pad_first: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build synthetic model inputs with both classes present.

    Args:
        n_rows: Number of rows to generate.
        seed: Seed for the generator.
        pad_first: Zero out the first row history so the padding mask has
            something to flag.

    Returns:
        Tuple of ``(x_cont, x_cat, seq, labels)``.
    """
    generator = torch.Generator().manual_seed(seed)
    x_cont = torch.randn(n_rows, N_CONT, generator=generator)
    x_cat = torch.stack(
        [torch.randint(0, card, (n_rows,), generator=generator) for card in CARDS], dim=1
    )
    seq = torch.randn(n_rows, SEQ_LEN, SEQ_DIM, generator=generator)
    if pad_first:
        seq[0, 0] = 0.0
    # Alternating labels guarantee both classes without relying on randomness.
    labels = torch.arange(n_rows, dtype=torch.float32) % 2
    return x_cont, x_cat, seq, labels


@pytest.fixture()
def capture() -> AttentionCapture:
    """A capture from a randomly initialised ``ft_cat`` model."""
    return collect_attention(make_model(), *make_inputs(), device="cpu")


def test_collect_attention_returns_expected_shapes(capture: AttentionCapture) -> None:
    """The capture is 4D and its axes match the model configuration."""
    assert capture.weights.shape == (N_ROWS, N_HEADS, 1 + N_CONT + len(CARDS), SEQ_LEN)
    assert capture.scores.shape == (N_ROWS,)
    assert capture.labels.shape == (N_ROWS,)
    assert capture.padding.shape == (N_ROWS, SEQ_LEN)
    assert len(capture) == N_ROWS
    assert capture.n_heads == N_HEADS
    assert capture.n_queries == 1 + N_CONT + len(CARDS)
    assert capture.seq_len == SEQ_LEN


def test_attention_rows_are_distributions(capture: AttentionCapture) -> None:
    """Each query row is a softmax over history and therefore sums to one."""
    sums = capture.weights.sum(axis=3)
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert (capture.weights >= 0.0).all()


def test_scores_are_probabilities(capture: AttentionCapture) -> None:
    """Captured scores are sigmoid outputs, not raw logits."""
    assert ((capture.scores >= 0.0) & (capture.scores <= 1.0)).all()


def test_class_masks_partition_the_capture(capture: AttentionCapture) -> None:
    """The fraud and legitimate masks are complementary."""
    assert (capture.fraud_mask() | capture.legitimate_mask()).all()
    assert not (capture.fraud_mask() & capture.legitimate_mask()).any()


def test_batching_does_not_change_the_capture() -> None:
    """Splitting the forward pass into batches is numerically transparent."""
    inputs = make_inputs()
    single = collect_attention(make_model(), *inputs, batch_size=N_ROWS)
    batched = collect_attention(make_model(), *inputs, batch_size=5)
    assert np.allclose(single.weights, batched.weights, atol=1e-6)
    assert np.allclose(single.scores, batched.scores, atol=1e-6)


def test_max_rows_truncates_the_capture() -> None:
    """``max_rows`` caps the capture from the front of the supplied rows."""
    capped = collect_attention(make_model(), *make_inputs(), max_rows=8)
    assert len(capped) == 8


def test_max_rows_none_captures_everything() -> None:
    """Passing ``None`` disables the cap."""
    assert len(collect_attention(make_model(), *make_inputs(), max_rows=None)) == N_ROWS


def test_padding_mask_flags_zeroed_history_slots() -> None:
    """Zero-filled history slots are reported as padding, real ones are not."""
    capture = collect_attention(make_model(), *make_inputs(pad_first=True))
    assert bool(capture.padding[0, 0])
    assert not bool(capture.padding[1].any())


def test_collect_attention_rejects_self_only_variant() -> None:
    """The history-blind arm has no cross-attention and must say so."""
    with pytest.raises(ValueError, match="no cross-attention"):
        collect_attention(make_model(use_cross_attention=False), *make_inputs())


def test_collect_attention_rejects_mlp_control() -> None:
    """The MLP control likewise carries no attention weights."""
    model = TabularMLP(
        n_continuous=N_CONT,
        categorical_cardinalities=CARDS,
        seq_len=SEQ_LEN,
        seq_dim=SEQ_DIM,
        d_model=D_MODEL,
    ).eval()
    with pytest.raises(ValueError, match="no cross-attention"):
        collect_attention(model, *make_inputs())


def test_collect_attention_rejects_non_positive_batch_size() -> None:
    """A non-positive batch size is a programming error, not a silent no-op."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        collect_attention(make_model(), *make_inputs(), batch_size=0)


def test_collect_attention_rejects_mismatched_rows() -> None:
    """Inputs describing different row counts are refused."""
    x_cont, x_cat, seq, labels = make_inputs()
    with pytest.raises(ValueError, match="disagree on row count"):
        collect_attention(make_model(), x_cont[:4], x_cat, seq, labels)


def test_collect_attention_rejects_empty_batch() -> None:
    """An empty batch has no attention to capture."""
    x_cont, x_cat, seq, labels = make_inputs()
    with pytest.raises(ValueError, match="empty batch"):
        collect_attention(make_model(), x_cont[:0], x_cat[:0], seq[:0], labels[:0])


def test_collect_attention_restores_training_mode() -> None:
    """Capturing must not leave a training model stuck in eval mode."""
    model = make_model().train()
    collect_attention(model, *make_inputs())
    assert model.training


def test_capture_rejects_non_4d_weights() -> None:
    """The dataclass guards its own core invariant."""
    with pytest.raises(ValueError, match="weights must be 4D"):
        AttentionCapture(
            weights=np.zeros((2, 2, 2)),
            scores=np.zeros(2),
            labels=np.zeros(2),
            padding=np.zeros((2, 2), dtype=bool),
        )


@pytest.mark.parametrize("field", ["scores", "labels"])
def test_capture_rejects_misaligned_row_vectors(field: str) -> None:
    """Scores and labels must describe exactly the captured rows."""
    payload = {
        "weights": np.zeros((4, 2, 3, SEQ_LEN)),
        "scores": np.zeros(4),
        "labels": np.zeros(4),
        "padding": np.zeros((4, SEQ_LEN), dtype=bool),
    }
    payload[field] = np.zeros(3)
    with pytest.raises(ValueError, match=f"{field} must have shape"):
        AttentionCapture(**payload)


def test_capture_rejects_misshaped_padding() -> None:
    """The padding mask must cover every row and every history slot."""
    with pytest.raises(ValueError, match="padding must have shape"):
        AttentionCapture(
            weights=np.zeros((4, 2, 3, SEQ_LEN)),
            scores=np.zeros(4),
            labels=np.zeros(4),
            padding=np.zeros((4, SEQ_LEN + 1), dtype=bool),
        )


def test_cls_history_attention_selects_the_cls_row(capture: AttentionCapture) -> None:
    """The helper returns exactly query row zero."""
    attention = cls_history_attention(capture)
    assert attention.shape == (N_ROWS, N_HEADS, SEQ_LEN)
    assert np.array_equal(attention, capture.weights[:, :, CLS_QUERY_INDEX, :])


def test_mean_history_attention_without_mask_averages_all(capture: AttentionCapture) -> None:
    """An omitted mask averages every captured row."""
    mean = mean_history_attention(capture)
    assert mean.shape == (N_HEADS, SEQ_LEN)
    assert np.allclose(mean, cls_history_attention(capture).mean(axis=0))


def test_mean_history_attention_honours_the_mask(capture: AttentionCapture) -> None:
    """A mask restricts the average to the selected rows."""
    mask = capture.fraud_mask()
    expected = cls_history_attention(capture)[mask].mean(axis=0)
    assert np.allclose(mean_history_attention(capture, mask), expected)


def test_mean_history_attention_rejects_empty_mask(capture: AttentionCapture) -> None:
    """Averaging nothing is an error rather than a NaN."""
    with pytest.raises(ValueError, match="selects no rows"):
        mean_history_attention(capture, np.zeros(len(capture), dtype=bool))


def test_mean_history_attention_rejects_wrong_mask_length(capture: AttentionCapture) -> None:
    """A mask of the wrong length cannot be aligned with the capture."""
    with pytest.raises(ValueError, match="mask must have shape"):
        mean_history_attention(capture, np.ones(3, dtype=bool))


def test_top_attended_features_ranks_by_concentration(capture: AttentionCapture) -> None:
    """The ranking is descending and limited to ``top_n`` named tokens."""
    ranked = top_attended_features(capture, FEATURE_NAMES, top_n=3)
    assert len(ranked) == 3
    assert [name for name, _ in ranked] == sorted(
        [name for name, _ in ranked],
        key=lambda name: -dict(ranked)[name],
    )
    assert all(name in FEATURE_NAMES for name, _ in ranked)
    assert all(0.0 <= value <= 1.0 for _, value in ranked)


def test_top_attended_features_returns_all_when_top_n_non_positive(
    capture: AttentionCapture,
) -> None:
    """A non-positive ``top_n`` returns every feature token."""
    assert len(top_attended_features(capture, FEATURE_NAMES, top_n=0)) == len(FEATURE_NAMES)


def test_top_attended_features_honours_the_mask(capture: AttentionCapture) -> None:
    """Restricting to fraud rows is allowed and still returns named tokens."""
    ranked = top_attended_features(capture, FEATURE_NAMES, top_n=2, mask=capture.fraud_mask())
    assert len(ranked) == 2


def test_top_attended_features_rejects_wrong_name_count(capture: AttentionCapture) -> None:
    """Names must cover every non-``[CLS]`` query token."""
    with pytest.raises(ValueError, match="must name the"):
        top_attended_features(capture, FEATURE_NAMES[:-1])


def test_top_attended_features_rejects_empty_mask(capture: AttentionCapture) -> None:
    """Ranking over no rows is an error."""
    with pytest.raises(ValueError, match="selects no rows"):
        top_attended_features(capture, FEATURE_NAMES, mask=np.zeros(len(capture), dtype=bool))


def test_top_attended_features_rejects_wrong_mask_length(capture: AttentionCapture) -> None:
    """A mask of the wrong length cannot be aligned with the capture."""
    with pytest.raises(ValueError, match="mask must have shape"):
        top_attended_features(capture, FEATURE_NAMES, mask=np.ones(3, dtype=bool))


def test_plot_cross_attention_heatmap_writes_a_png(
    capture: AttentionCapture, tmp_path: Path
) -> None:
    """The heatmap is written to disk with its parent directory created."""
    output = tmp_path / "nested" / "attention_cross_heatmap.png"
    written = plot_cross_attention_heatmap(capture, output)
    assert written == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_cross_attention_heatmap_requires_both_classes() -> None:
    """A single-class capture has no contrast to draw."""
    x_cont, x_cat, seq, labels = make_inputs()
    single_class = collect_attention(make_model(), x_cont, x_cat, seq, torch.zeros_like(labels))
    with pytest.raises(ValueError, match="needs both classes"):
        plot_cross_attention_heatmap(single_class, "unused.png")


def test_plot_case_study_writes_a_png(capture: AttentionCapture, tmp_path: Path) -> None:
    """The case study figure is written for a valid row."""
    output = tmp_path / "attention_case_study.png"
    assert plot_case_study(capture, 0, FEATURE_NAMES, output) == output
    assert output.stat().st_size > 0


def test_plot_case_study_accepts_non_positive_top_n(
    capture: AttentionCapture, tmp_path: Path
) -> None:
    """A non-positive ``top_n`` draws every feature token."""
    output = tmp_path / "all_tokens.png"
    plot_case_study(capture, 1, FEATURE_NAMES, output, top_n=0)
    assert output.is_file()


def test_plot_case_study_rejects_out_of_range_row(
    capture: AttentionCapture, tmp_path: Path
) -> None:
    """An out-of-range row index is an ``IndexError``."""
    with pytest.raises(IndexError, match="out of range"):
        plot_case_study(capture, len(capture), FEATURE_NAMES, tmp_path / "x.png")


def test_plot_case_study_rejects_wrong_name_count(
    capture: AttentionCapture, tmp_path: Path
) -> None:
    """Names must cover every non-``[CLS]`` query token."""
    with pytest.raises(ValueError, match="must name the"):
        plot_case_study(capture, 0, FEATURE_NAMES[:2], tmp_path / "x.png")


def test_full_history_mask_flags_unpadded_rows() -> None:
    """Rows whose every history slot is real come back True."""
    capture = collect_attention(make_model(), *make_inputs(pad_first=True))
    mask = full_history_mask(capture)
    assert mask.shape == (len(capture),)
    # make_inputs(pad_first=True) zeroes one slot of row 0 only.
    assert not bool(mask[0])
    assert bool(mask[1:].all())


def test_full_history_mask_is_all_true_without_padding(capture: AttentionCapture) -> None:
    """Fully populated windows are all flagged as usable."""
    assert bool(full_history_mask(capture).all())


def test_plot_case_study_marks_padded_slots(tmp_path: Path) -> None:
    """A partly padded window still renders, with the masked slots marked."""
    capture = collect_attention(make_model(), *make_inputs(pad_first=True))
    output = tmp_path / "padded_case.png"
    plot_case_study(capture, 0, FEATURE_NAMES, output)
    assert output.stat().st_size > 0
