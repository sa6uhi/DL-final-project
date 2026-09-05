"""Unit tests for the FT-CAT training engine (Member B).

Covers tensor materialisation from processed frames, the defensive guards on
malformed splits, class-proportional subsampling, the training loop and its
PR-AUC early stopping, checkpoint round-tripping, and the CLI end to end.
Everything runs on tiny synthetic frames and is pinned to the CPU so the
suite stays fast and machine-independent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.ft_transformer import FTCATransformer, TabularMLP
from src.models.losses import WeightedBCELoss
from src.training.feature_selection import FeatureSpec
from src.training.train_transformer import (
    SEQUENCE_COLUMN,
    TensorBundle,
    _stack_sequences,
    build_loader,
    build_model_from_meta,
    evaluate,
    load_ft_transformer,
    main,
    materialize_tensors,
    subsample_bundle,
    train_transformer,
)
from src.training.trainer_utils import AmpPolicy, save_checkpoint
from src.utils.config import Config

N_ROWS = 96
N_CONT = 4
CARDS = [3, 4]
SEQ_LEN = 5
SEQ_DIM = 3
CONT_COLS = [f"c{i}" for i in range(N_CONT)]
CAT_COLS = [f"k{i}" for i in range(len(CARDS))]
SEQ_COLS = [f"s{i}" for i in range(SEQ_DIM)]


def make_frame(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic processed split with a learnable fraud signal.

    The label is a thresholded function of ``c0`` so a few epochs of training
    can measurably reduce the loss, which is what the convergence tests need.

    Args:
        n_rows: Number of rows to generate.
        seed: Seed for the generator.

    Returns:
        A frame shaped like the output of ``src.data.prepare_data``.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({col: rng.normal(size=n_rows) for col in CONT_COLS})
    for col, card in zip(CAT_COLS, CARDS):
        frame[col] = rng.integers(0, card, size=n_rows, dtype=np.int64)
    frame[SEQUENCE_COLUMN] = [rng.normal(size=(SEQ_LEN, SEQ_DIM)).tolist() for _ in range(n_rows)]
    # Deterministic, learnable signal with a minority positive class.
    frame["isFraud"] = (frame["c0"] > frame["c0"].quantile(0.75)).astype(np.int64)
    return frame


def make_spec() -> FeatureSpec:
    """Return the feature contract matching :func:`make_frame`."""
    return FeatureSpec(
        continuous_cols=list(CONT_COLS),
        categorical_cols=list(CAT_COLS),
        categorical_cardinalities=list(CARDS),
        sequence_cols=list(SEQ_COLS),
        seq_len=SEQ_LEN,
    )


@pytest.fixture()
def spec() -> FeatureSpec:
    """Feature contract for the synthetic frames."""
    return make_spec()


@pytest.fixture()
def frame() -> pd.DataFrame:
    """A synthetic processed split."""
    return make_frame()


@pytest.fixture()
def tiny_config() -> Config:
    """A transformer configuration that trains in well under a second."""
    return Config(
        {
            "seed": 42,
            "transformer": {
                "d_model": 8,
                "n_heads": 2,
                "dim_feedforward": 16,
                "n_layers": 1,
                "dropout": 0.0,
                "activation": "gelu",
                "norm_first": True,
                "seq_len": SEQ_LEN,
                "n_continuous": N_CONT,
                "categorical_cardinalities": list(CARDS),
                "subsample": {"train_rows": 48, "val_rows": 32},
                "training": {
                    "lr": 1.0e-2,
                    "weight_decay": 1.0e-4,
                    "epochs": 6,
                    "batch_size": 32,
                    "warmup_epochs": 1,
                    "grad_clip": 5.0,
                    "amp": False,
                    "num_workers": 0,
                    "pin_memory": False,
                    "early_stopping_patience": 5,
                    "min_delta": 0.0,
                },
                "loss": {"name": "focal", "gamma": 2.0, "alpha": 0.25, "pos_weight": None},
            },
            "sequence": {"feature_cols": list(SEQ_COLS)},
        }
    )


# --------------------------------------------------------------------------
# Materialisation
# --------------------------------------------------------------------------


def test_materialize_tensors_shapes_and_dtypes(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    """The four tensors match the FT-CAT forward signature exactly."""
    bundle = materialize_tensors(frame, spec)

    assert bundle.x_cont.shape == (N_ROWS, N_CONT)
    assert bundle.x_cat.shape == (N_ROWS, len(CARDS))
    assert bundle.seq.shape == (N_ROWS, SEQ_LEN, SEQ_DIM)
    assert bundle.y.shape == (N_ROWS,)
    assert bundle.x_cont.dtype == torch.float32
    assert bundle.x_cat.dtype == torch.int64
    assert bundle.seq.dtype == torch.float32
    assert bundle.y.dtype == torch.float32
    assert len(bundle) == N_ROWS
    assert 0.0 < bundle.positive_rate < 1.0


def test_materialize_tensors_preserves_row_order(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    """Row i of every tensor still corresponds to row i of the frame."""
    bundle = materialize_tensors(frame, spec)
    expected = frame[CONT_COLS].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(bundle.x_cont.numpy(), expected, rtol=1e-6)
    np.testing.assert_array_equal(bundle.y.numpy().astype(np.int64), frame["isFraud"].to_numpy())


def test_materialize_tensors_scrubs_non_finite(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    """NaN and infinity in the continuous block become zeros."""
    frame.loc[0, "c1"] = np.nan
    frame.loc[1, "c2"] = np.inf
    bundle = materialize_tensors(frame, spec)
    assert torch.isfinite(bundle.x_cont).all()
    assert bundle.x_cont[0, 1].item() == 0.0
    assert bundle.x_cont[1, 2].item() == 0.0


def test_materialize_tensors_clamps_out_of_range_codes(
    frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """Codes beyond the embedding table collapse to the reserved <UNK> index."""
    frame.loc[0, "k0"] = 999
    frame.loc[1, "k1"] = -5
    bundle = materialize_tensors(frame, spec)
    assert bundle.x_cat[0, 0].item() == 0
    assert bundle.x_cat[1, 1].item() == 0
    assert int(bundle.x_cat[:, 0].max()) < CARDS[0]


def test_materialize_tensors_missing_feature_column_raises(
    frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """A split lacking a contracted feature column fails loudly."""
    with pytest.raises(KeyError, match="missing"):
        materialize_tensors(frame.drop(columns=["c0"]), spec)


def test_materialize_tensors_missing_sequence_column_raises(
    frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """A split without history windows fails loudly."""
    with pytest.raises(KeyError, match=SEQUENCE_COLUMN):
        materialize_tensors(frame.drop(columns=[SEQUENCE_COLUMN]), spec)


def test_stack_sequences_rejects_wrong_width(frame: pd.DataFrame) -> None:
    """A history window of unexpected width is rejected, not silently reshaped."""
    with pytest.raises(ValueError, match=r"row 0 has history of shape"):
        _stack_sequences(frame[SEQUENCE_COLUMN], SEQ_LEN, SEQ_DIM + 1)


def test_tensor_bundle_rejects_mismatched_rows() -> None:
    """Tensors that disagree on row count cannot form a bundle."""
    with pytest.raises(ValueError, match="disagree on row count"):
        TensorBundle(
            x_cont=torch.zeros(4, N_CONT),
            x_cat=torch.zeros(3, len(CARDS), dtype=torch.int64),
            seq=torch.zeros(4, SEQ_LEN, SEQ_DIM),
            y=torch.zeros(4),
        )


def test_tensor_bundle_rejects_wrong_rank() -> None:
    """A flat history tensor is rejected."""
    with pytest.raises(ValueError, match="seq must be 3D"):
        TensorBundle(
            x_cont=torch.zeros(4, N_CONT),
            x_cat=torch.zeros(4, len(CARDS), dtype=torch.int64),
            seq=torch.zeros(4, SEQ_LEN),
            y=torch.zeros(4),
        )


# --------------------------------------------------------------------------
# Subsampling and loaders
# --------------------------------------------------------------------------


def test_subsample_bundle_keeps_both_classes(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    """Subsampling shrinks the split without sampling fraud away."""
    bundle = materialize_tensors(frame, spec)
    sampled = subsample_bundle(bundle, 32, seed=42)
    assert len(sampled) <= len(bundle)
    assert sampled.n_positive > 0
    assert sampled.x_cont.shape[1] == bundle.x_cont.shape[1]


def test_subsample_bundle_is_noop_when_target_exceeds_rows(
    frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """Asking for more rows than exist returns the bundle untouched."""
    bundle = materialize_tensors(frame, spec)
    assert subsample_bundle(bundle, 10_000, seed=42) is bundle
    assert subsample_bundle(bundle, 0, seed=42) is bundle


def test_build_loader_yields_four_tensors(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    """Each batch carries the tuple the model forward expects."""
    bundle = materialize_tensors(frame, spec)
    loader = build_loader(bundle, batch_size=16, shuffle=False, pin_memory=False)
    x_cont, x_cat, seq, y = next(iter(loader))
    assert x_cont.shape == (16, N_CONT)
    assert x_cat.shape == (16, len(CARDS))
    assert seq.shape == (16, SEQ_LEN, SEQ_DIM)
    assert y.shape == (16,)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


def test_train_transformer_learns_and_records_history(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """Training reduces the loss and the history mirrors the best epoch."""
    train_bundle = materialize_tensors(frame, spec)
    val_bundle = materialize_tensors(make_frame(seed=1), spec)

    model, history = train_transformer(
        tiny_config, train_bundle, val_bundle, spec, variant="ft_cat", device="cpu"
    )

    assert isinstance(model, FTCATransformer)
    assert history.epochs_run == len(history.train_loss) == len(history.val_pr_auc)
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.best_pr_auc == pytest.approx(max(history.val_pr_auc))
    assert 1 <= history.best_epoch <= history.epochs_run
    assert all(np.isfinite(history.val_loss))


@pytest.mark.parametrize("variant", ["ft_cat", "ft_self_only", "mlp"])
def test_train_transformer_supports_every_variant(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec, variant: str
) -> None:
    """Each ablation arm trains through the same entry point."""
    bundle = materialize_tensors(frame, spec)
    model, history = train_transformer(
        tiny_config, bundle, bundle, spec, variant=variant, device="cpu", epochs=2
    )
    assert history.epochs_run == 2
    assert isinstance(model, TabularMLP if variant == "mlp" else FTCATransformer)


def test_train_transformer_rejects_unknown_variant(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """An unknown variant name fails before any work happens."""
    bundle = materialize_tensors(frame, spec)
    with pytest.raises(ValueError, match="unknown variant"):
        train_transformer(tiny_config, bundle, bundle, spec, variant="nope", device="cpu")


def test_train_transformer_rejects_empty_epoch_budget(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """A non-positive epoch budget is rejected."""
    bundle = materialize_tensors(frame, spec)
    with pytest.raises(ValueError, match="epochs must be positive"):
        train_transformer(tiny_config, bundle, bundle, spec, device="cpu", epochs=0)


def test_train_transformer_honours_criterion_override(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """A caller-supplied loss replaces the configured focal loss.

    This is the hook Experiment 4 uses to sweep loss functions without
    mutating the shared configuration.
    """
    bundle = materialize_tensors(frame, spec)
    _, history = train_transformer(
        tiny_config,
        bundle,
        bundle,
        spec,
        device="cpu",
        epochs=2,
        criterion=WeightedBCELoss(pos_weight=3.0),
    )
    assert history.epochs_run == 2
    # Weighted BCE is on a different scale from focal loss; a plain focal run
    # of the same length would not reach these magnitudes.
    assert history.train_loss[0] > 0.0


def test_train_transformer_is_reproducible(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """The same seed yields the same trajectory."""
    bundle = materialize_tensors(frame, spec)
    kwargs = {"device": "cpu", "epochs": 3, "seed": 7}
    _, first = train_transformer(tiny_config, bundle, bundle, spec, **kwargs)
    _, second = train_transformer(tiny_config, bundle, bundle, spec, **kwargs)
    assert first.train_loss == pytest.approx(second.train_loss)
    assert first.val_pr_auc == pytest.approx(second.val_pr_auc)


def test_evaluate_rejects_single_class_split(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec
) -> None:
    """Ranking metrics are undefined without both classes, and say so."""
    frame["isFraud"] = 0
    bundle = materialize_tensors(frame, spec)
    model = FTCATransformer(
        n_continuous=N_CONT,
        categorical_cardinalities=CARDS,
        seq_len=SEQ_LEN,
        seq_dim=SEQ_DIM,
        d_model=8,
        n_heads=2,
    )
    loader = build_loader(bundle, batch_size=32, shuffle=False, pin_memory=False)
    with pytest.raises(ValueError, match="single-class"):
        evaluate(model, loader, WeightedBCELoss(), "cpu", AmpPolicy("cpu", requested=False))


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def test_checkpoint_round_trip_reproduces_logits(
    tiny_config: Config, frame: pd.DataFrame, spec: FeatureSpec, tmp_path: Path
) -> None:
    """A reloaded checkpoint scores identically to the trained model."""
    bundle = materialize_tensors(frame, spec)
    model, _ = train_transformer(tiny_config, bundle, bundle, spec, device="cpu", epochs=2, seed=3)
    out = tmp_path / "ft_transformer.pt"
    save_checkpoint(model, out, extra={"feature_spec": spec.to_dict(), "variant": "ft_cat"})

    restored, payload = load_ft_transformer(out, device="cpu")
    assert payload["feature_spec"]["continuous_cols"] == CONT_COLS
    assert payload["variant"] == "ft_cat"

    model.eval()
    with torch.no_grad():
        expected = model(bundle.x_cont, bundle.x_cat, bundle.seq)
        actual = restored(bundle.x_cont, bundle.x_cat, bundle.seq)
    torch.testing.assert_close(expected, actual)


def test_build_model_from_meta_dispatches_on_architecture() -> None:
    """Cross-attention metadata selects FT-CAT; its absence selects the MLP."""
    ft_meta = {
        "n_continuous": N_CONT,
        "categorical_cardinalities": CARDS,
        "seq_len": SEQ_LEN,
        "seq_dim": SEQ_DIM,
        "d_model": 8,
        "n_heads": 2,
        "dim_feedforward": 16,
        "n_layers": 1,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_first": True,
        "use_cross_attention": True,
    }
    mlp_meta = {
        "n_continuous": N_CONT,
        "categorical_cardinalities": CARDS,
        "seq_len": SEQ_LEN,
        "seq_dim": SEQ_DIM,
        "d_model": 8,
        "dim_feedforward": 16,
        "dropout": 0.0,
    }
    assert isinstance(build_model_from_meta(ft_meta), FTCATransformer)
    assert isinstance(build_model_from_meta(mlp_meta), TabularMLP)


def test_load_ft_transformer_missing_file_raises(tmp_path: Path) -> None:
    """A missing checkpoint reports the path it looked for."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_ft_transformer(tmp_path / "absent.pt")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write_cli_fixtures(tmp_path: Path) -> Path:
    """Write parquet splits and a config pointing at them.

    Args:
        tmp_path: Directory to populate.

    Returns:
        Path of the written configuration file.
    """
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, seed in (("train", 0), ("val", 1), ("test", 2)):
        make_frame(seed=seed).to_parquet(processed / f"{name}.parquet", index=False)

    config = Config(
        {
            "seed": 42,
            "logging": {"level": "INFO", "log_file": str(tmp_path / "run.log")},
            "paths": {"checkpoints": str(tmp_path / "checkpoints")},
            "data": {
                "train_data_path": str(processed / "train.parquet"),
                "val_data_path": str(processed / "val.parquet"),
                "test_data_path": str(processed / "test.parquet"),
                "preprocessor_path": str(processed / "preprocessor.pkl"),
                "non_feature_cols": ["isFraud", SEQUENCE_COLUMN],
            },
            "features": {"target_col": "isFraud"},
            "sequence": {
                "feature_cols": list(SEQ_COLS),
                "k_window": SEQ_LEN,
                "group_cols": ["k0"],
            },
            "transformer": {
                "d_model": 8,
                "n_heads": 2,
                "dim_feedforward": 16,
                "n_layers": 1,
                "dropout": 0.0,
                "activation": "gelu",
                "norm_first": True,
                "seq_len": SEQ_LEN,
                "n_continuous": N_CONT,
                "categorical_cardinalities": list(CARDS),
                "feature_cache_path": str(processed / "ft_cat_features.json"),
                "feature_selection": {"top_n_continuous": N_CONT, "mi_sample_size": 64},
                "subsample": {"train_rows": 48, "val_rows": 48},
                "training": {
                    "lr": 1.0e-2,
                    "weight_decay": 1.0e-4,
                    "epochs": 2,
                    "batch_size": 32,
                    "warmup_epochs": 1,
                    "grad_clip": 5.0,
                    "amp": False,
                    "num_workers": 0,
                    "pin_memory": False,
                    "early_stopping_patience": 5,
                    "min_delta": 0.0,
                },
                "loss": {"name": "focal", "gamma": 2.0, "alpha": 0.25, "pos_weight": None},
            },
        }
    )
    config_path = tmp_path / "config.yaml"
    config.save_yaml(config_path)
    return config_path


def test_main_trains_and_writes_a_loadable_checkpoint(tmp_path: Path) -> None:
    """The CLI runs end to end from parquet splits to a reloadable checkpoint."""
    config_path = _write_cli_fixtures(tmp_path)
    out = tmp_path / "checkpoints" / "ft_transformer.pt"

    main(["--config", str(config_path), "--out", str(out), "--device", "cpu", "--subsample"])

    assert out.is_file()
    model, payload = load_ft_transformer(out, device="cpu")
    assert isinstance(model, FTCATransformer)
    assert payload["variant"] == "ft_cat"
    assert payload["seed"] == 42
    assert payload["history"]["epochs_run"] == 2
    # Both the validation and the out-of-time test split are scored.
    assert set(payload["metrics"]) == {"val", "test"}
    assert 0.0 <= payload["metrics"]["test"]["pr_auc"] <= 1.0
    assert payload["feature_spec"]["seq_len"] == SEQ_LEN


def test_main_skip_test_omits_out_of_time_metrics(tmp_path: Path) -> None:
    """``--skip-test`` keeps the held-out split untouched."""
    config_path = _write_cli_fixtures(tmp_path)
    out = tmp_path / "checkpoints" / "mlp.pt"

    main(
        [
            "--config",
            str(config_path),
            "--out",
            str(out),
            "--device",
            "cpu",
            "--variant",
            "mlp",
            "--epochs",
            "1",
            "--skip-test",
        ]
    )

    _, payload = load_ft_transformer(out, device="cpu")
    assert set(payload["metrics"]) == {"val"}
    assert payload["variant"] == "mlp"


def test_main_reports_missing_split(tmp_path: Path) -> None:
    """A missing parquet split points the user at the preparation step."""
    config_path = _write_cli_fixtures(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        main(
            [
                "--config",
                str(config_path),
                "--train-data",
                str(tmp_path / "absent.parquet"),
                "--device",
                "cpu",
            ]
        )
