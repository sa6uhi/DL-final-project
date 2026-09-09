"""Unit tests for the attention heatmap experiment CLI (Member B).

Covers the balanced row sampler, the capture summary the paper quotes, and the
CLI end to end against a checkpoint written by the project's own
``save_checkpoint`` envelope, including its two failure modes: a missing
checkpoint and a checkpoint written without a feature contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.attention_heatmaps import (
    balanced_row_index,
    main,
    summarize_capture,
)
from src.evaluation.attention_maps import collect_attention
from src.models.ft_transformer import FTCATransformer
from src.training.trainer_utils import save_checkpoint

from tests.test_attention_maps import FEATURE_NAMES, make_inputs, make_model
from tests.test_transformer_cat_ablation import (
    CARDS,
    CAT_COLS,
    CONT_COLS,
    NON_FEATURE_COLS,
    SEQ_COLS,
    SEQ_LEN,
    make_frame,
)


@pytest.fixture()
def capture() -> object:
    """A capture from a randomly initialised ``ft_cat`` model."""
    return collect_attention(make_model(), *make_inputs(), device="cpu")


def test_balanced_row_index_draws_both_classes() -> None:
    """The sampler returns an equal quota from each class."""
    labels = np.array([0] * 50 + [1] * 20, dtype=np.int64)
    index = balanced_row_index(labels, per_class=10, seed=42)
    assert index.size == 20
    assert int((labels[index] == 1).sum()) == 10
    assert int((labels[index] == 0).sum()) == 10


def test_balanced_row_index_caps_at_availability() -> None:
    """A quota larger than the minority class is capped, not padded."""
    labels = np.array([0] * 50 + [1] * 3, dtype=np.int64)
    index = balanced_row_index(labels, per_class=25, seed=42)
    assert int((labels[index] == 1).sum()) == 3
    assert int((labels[index] == 0).sum()) == 25


def test_balanced_row_index_is_sorted_and_unique() -> None:
    """Indices come back sorted and without duplicates."""
    labels = np.array([0, 1] * 40, dtype=np.int64)
    index = balanced_row_index(labels, per_class=10, seed=7)
    assert np.array_equal(index, np.sort(index))
    assert index.size == np.unique(index).size


def test_balanced_row_index_is_deterministic() -> None:
    """The same seed draws the same rows."""
    labels = np.array([0, 1] * 40, dtype=np.int64)
    assert np.array_equal(
        balanced_row_index(labels, 10, seed=7), balanced_row_index(labels, 10, seed=7)
    )


def test_balanced_row_index_requires_both_classes() -> None:
    """A single-class split cannot produce a fraud/legitimate contrast."""
    with pytest.raises(ValueError, match="need both classes"):
        balanced_row_index(np.zeros(10, dtype=np.int64), per_class=2, seed=42)


def test_summarize_capture_reports_the_contrast(capture: object) -> None:
    """The summary is JSON-serialisable and carries both class attentions."""
    summary = summarize_capture(capture, FEATURE_NAMES, top_n=2)
    assert summary["n_rows"] == len(capture)
    assert summary["n_fraud"] + summary["n_legitimate"] == summary["n_rows"]
    assert summary["seq_len"] == SEQ_LEN
    assert len(summary["fraud_cls_attention"]) == summary["n_heads"]
    assert len(summary["top_history_driven_features"]) == 2
    assert 0.0 <= summary["fraud_most_recent_slot"] <= 1.0
    assert 0.0 <= summary["padded_slot_rate"] <= 1.0
    # Must survive a round trip: the CLI writes this straight to JSON.
    assert json.loads(json.dumps(summary))["n_rows"] == len(capture)


def _write_checkpoint(path: Path, with_spec: bool = True) -> Path:
    """Write an FT-CAT checkpoint in the project's standard envelope.

    Args:
        path: Destination checkpoint path.
        with_spec: Whether to embed the ``feature_spec`` payload.

    Returns:
        The written checkpoint path.
    """
    model = FTCATransformer(
        n_continuous=len(CONT_COLS),
        categorical_cardinalities=CARDS,
        seq_len=SEQ_LEN,
        seq_dim=len(SEQ_COLS),
        d_model=8,
        n_heads=2,
        dim_feedforward=16,
        n_layers=1,
        dropout=0.0,
    )
    extra = {}
    if with_spec:
        extra["feature_spec"] = {
            "continuous_cols": list(CONT_COLS),
            "categorical_cols": list(CAT_COLS),
            "categorical_cardinalities": list(CARDS),
            "sequence_cols": list(SEQ_COLS),
            "seq_len": SEQ_LEN,
        }
    return save_checkpoint(model, path, extra=extra)


def _write_cli_fixtures(tmp_path: Path, with_spec: bool = True) -> tuple[Path, Path]:
    """Write a test parquet, a checkpoint, and a config for the CLI tests.

    Args:
        tmp_path: Directory to populate.
        with_spec: Whether the checkpoint embeds a feature contract.

    Returns:
        Tuple of ``(config_path, checkpoint_path)``.
    """
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    make_frame(n_rows=96, seed=2).to_parquet(processed / "test.parquet", index=False)

    checkpoint = _write_checkpoint(tmp_path / "checkpoints" / "ft_transformer.pt", with_spec)

    payload = {
        "seed": 42,
        "logging": {"level": "INFO", "log_file": str(tmp_path / "system.log")},
        "data": {
            "test_data_path": str(processed / "test.parquet"),
            "non_feature_cols": list(NON_FEATURE_COLS),
        },
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return config_path, checkpoint


def test_main_writes_both_figures_and_a_summary(tmp_path: Path) -> None:
    """The CLI renders the heatmap, the case study, and the JSON summary."""
    config_path, checkpoint = _write_cli_fixtures(tmp_path)
    heatmap = tmp_path / "attention_cross_heatmap.png"
    case_study = tmp_path / "attention_case_study.png"
    summary_path = tmp_path / "attention_summary.json"

    main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--heatmap",
            str(heatmap),
            "--case-study",
            str(case_study),
            "--out",
            str(summary_path),
            "--per-class",
            "8",
            "--top-features",
            "3",
        ]
    )

    assert heatmap.stat().st_size > 0
    assert case_study.stat().st_size > 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_rows"] == 16
    assert summary["checkpoint"] == str(checkpoint)
    assert len(summary["top_history_driven_features"]) == 3
    # The case study must be drawn from a fraudulent row.
    assert 0 <= summary["case_study_row"] < summary["n_rows"]
    assert 0.0 <= summary["case_study_score"] <= 1.0


def test_main_reports_a_missing_checkpoint(tmp_path: Path) -> None:
    """A missing checkpoint points the user at the training command."""
    config_path, _ = _write_cli_fixtures(tmp_path)
    with pytest.raises(FileNotFoundError, match="Train FT-CAT first"):
        main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(tmp_path / "absent.pt"),
            ]
        )


def test_main_rejects_a_checkpoint_without_a_feature_spec(tmp_path: Path) -> None:
    """A checkpoint from another writer cannot define the tensor layout."""
    config_path, checkpoint = _write_cli_fixtures(tmp_path, with_spec=False)
    with pytest.raises(KeyError, match="feature_spec"):
        main(["--config", str(config_path), "--checkpoint", str(checkpoint)])


def test_main_reports_a_missing_test_split(tmp_path: Path) -> None:
    """A missing parquet points at the data preparation step."""
    config_path, checkpoint = _write_cli_fixtures(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--test-data",
                str(tmp_path / "absent.parquet"),
            ]
        )


def test_main_case_study_row_is_a_fraudulent_transaction(tmp_path: Path) -> None:
    """The chosen case study is a fraud row -- the one the policy auto-blocks."""
    config_path, checkpoint = _write_cli_fixtures(tmp_path)
    summary_path = tmp_path / "summary.json"
    main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--heatmap",
            str(tmp_path / "h.png"),
            "--case-study",
            str(tmp_path / "c.png"),
            "--out",
            str(summary_path),
            "--per-class",
            "8",
        ]
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["case_study_label"] == 1
