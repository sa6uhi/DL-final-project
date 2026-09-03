"""Unit tests for :mod:`src.models.autoencoder` and the DAE trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.autoencoder import DenoisingAutoencoder
from src.training.train_autoencoder import (
    load_checkpoint,
    save_checkpoint,
    train_autoencoder,
)
from src.utils.config import Config

torch.backends.cudnn.deterministic = True


@pytest.fixture()
def ae(config) -> DenoisingAutoencoder:
    """Small deterministic autoencoder for shape tests."""
    torch.manual_seed(42)
    model = DenoisingAutoencoder(
        input_dim=int(config.autoencoder.input_dim),
        encoder_hidden_dims=[int(h) for h in config.autoencoder.encoder_hidden_dims],
        latent_dim=int(config.autoencoder.latent_dim),
    )
    model.eval()
    return model


@pytest.fixture()
def config_small(tmp_path):
    """Minimal config overriding the DAE to tiny architecture and 2 epochs."""
    cfg = Config(
        {
            "autoencoder": {
                "input_dim": 8,
                "encoder_hidden_dims": [4],
                "latent_dim": 2,
                "dropout": 0.2,
                "noise_std": 0.1,
                "feature_dropout_prob": 0.1,
                "activation_slope": 0.2,
                "training": {
                    "lr": 1.0e-3,
                    "weight_decay": 1.0e-5,
                    "epochs": 3,
                    "batch_size": 16,
                    "num_workers": 0,
                    "pin_memory": False,
                    "early_stopping_patience": 5,
                    "min_delta": 1.0e-6,
                },
                "loss": {"mse_weight": 0.5, "bce_weight": 0.5},
            },
            "seed": 42,
        },
        base_dir=tmp_path,
    )
    return cfg


def test_forward_shape_matches_input(ae: DenoisingAutoencoder, fake_matrix: np.ndarray) -> None:
    """Forward output has the same shape as the input batch."""
    x = torch.from_numpy(fake_matrix)
    out = ae(x)
    assert tuple(out.shape) == tuple(x.shape)


def test_forward_wrong_dim_raises(ae: DenoisingAutoencoder) -> None:
    """Feeding a 3D tensor raises ValueError."""
    x = torch.randn(4, 5, 6)
    with pytest.raises(ValueError):
        ae(x)


def test_forward_feature_mismatch_raises(ae: DenoisingAutoencoder) -> None:
    """Feeding the wrong feature dimension raises ValueError."""
    x = torch.randn(4, 7)
    with pytest.raises(ValueError):
        ae(x)


def test_encode_decode_shapes(ae: DenoisingAutoencoder, fake_matrix: np.ndarray) -> None:
    """Latent embeddings have the configured latent dimension."""
    x = torch.from_numpy(fake_matrix)
    z = ae.encode(x)
    assert tuple(z.shape) == (x.size(0), ae.latent_dim)
    recon = ae.decode(z)
    assert tuple(recon.shape) == tuple(x.shape)


def test_decoder_expands_symmetrically_through_reversed_hiddens() -> None:
    """Decoder reconstructs input_dim from the latent via reversed hidden dims."""
    model = DenoisingAutoencoder(input_dim=100, encoder_hidden_dims=[256, 128], latent_dim=32)
    decoder_ins = []
    decoder_outs = []
    for name, mod in model.decoder.named_children():
        if isinstance(mod, torch.nn.Linear):
            decoder_ins.append(mod.in_features)
            decoder_outs.append(mod.out_features)
    # Encoder narrows D -> 256 -> 128 -> 32; decoder must widen 32 -> 128 -> 256 -> D.
    assert (decoder_ins, decoder_outs) == ([32, 128, 256], [128, 256, 100])


def test_anomaly_score_enforces_eval_mode_and_restores_state(
    ae: DenoisingAutoencoder,
) -> None:
    """anomaly_score runs deterministically and leaves training mode unchanged."""
    x = torch.randn(8, ae.input_dim)
    ae.train()
    assert ae.training is True
    bn = next(m for m in ae.encoder.modules() if isinstance(m, torch.nn.BatchNorm1d))
    running_mean_before = bn.running_mean.clone()
    scores = ae.anomaly_score(x)
    assert ae.training is True
    assert torch.equal(bn.running_mean, running_mean_before)
    assert tuple(scores.shape) == (8,)
    assert torch.equal(scores, ae.anomaly_score(x))


def test_anomaly_score_eval_mode_does_not_toggle_training(ae: DenoisingAutoencoder) -> None:
    """Calling anomaly_score in eval mode leaves the model in eval mode."""
    ae.eval()
    ae.anomaly_score(torch.randn(4, ae.input_dim))
    assert ae.training is False


def test_corrupt_keeps_shape_and_adjusts_values(ae: DenoisingAutoencoder) -> None:
    """Corruption preserves shape but modifies values."""
    x = torch.zeros(8, ae.input_dim)
    corrupted = ae.corrupt(x)
    assert tuple(corrupted.shape) == tuple(x.shape)
    assert not torch.equal(corrupted, x)


def test_corrupt_has_noiseless_reconstruction_in_eval(ae: DenoisingAutoencoder) -> None:
    """Evaluation mode never corrupts the input."""
    x = torch.randn(8, ae.input_dim)
    assert ae.training is False
    assert torch.equal(ae(x, corrupt=True), ae(x))


def test_anomaly_score_shape_and_positivity(
    ae: DenoisingAutoencoder, fake_matrix: np.ndarray
) -> None:
    """Scores are per-sample, non-negative and match manual computation."""
    x = torch.from_numpy(fake_matrix)
    scores = ae.anomaly_score(x)
    assert tuple(scores.shape) == (x.size(0),)
    assert (scores >= 0).all()
    with torch.no_grad():
        residual = x - ae(x)
    manual = residual.pow(2).sum(-1) + 0.4 * residual.abs().sum(-1)
    assert torch.allclose(scores, manual)


def test_anomaly_score_reductions(ae: DenoisingAutoencoder, fake_matrix: np.ndarray) -> None:
    """Mean/sum reductions return scalars equal to aggregation of per-sample."""
    x = torch.from_numpy(fake_matrix)
    scores = ae.anomaly_score(x)
    assert torch.isclose(ae.anomaly_score(x, reduction="mean"), scores.mean())
    assert torch.isclose(ae.anomaly_score(x, reduction="sum"), scores.sum())


def test_anomaly_score_negative_gamma_raises(
    ae: DenoisingAutoencoder, fake_matrix: np.ndarray
) -> None:
    """Negative l1_gamma raises ValueError."""
    x = torch.from_numpy(fake_matrix)
    with pytest.raises(ValueError):
        ae.anomaly_score(x, l1_gamma=-1.0)


def test_anomaly_score_bad_reduction_raises(
    ae: DenoisingAutoencoder, fake_matrix: np.ndarray
) -> None:
    """Unknown reduction raises ValueError."""
    x = torch.from_numpy(fake_matrix)
    with pytest.raises(ValueError):
        ae.anomaly_score(x, reduction="invalid")


def test_loss_value_and_shapes(ae: DenoisingAutoencoder) -> None:
    """Loss is a positive scalar matching the weighted composite formula."""
    x = torch.rand(8, ae.input_dim)
    x_hat = torch.rand(8, ae.input_dim)
    loss = ae.loss(x, x_hat)
    assert loss.item() > 0
    mse = torch.nn.functional.mse_loss(x_hat, x)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(x_hat, x.clamp(0, 1))
    expected = 0.5 * mse + 0.5 * bce
    assert torch.isclose(loss, expected)


def test_loss_shape_mismatch_raises(ae: DenoisingAutoencoder) -> None:
    """Mismatched tensors raise ValueError."""
    with pytest.raises(ValueError):
        ae.loss(torch.rand(2, 3), torch.rand(2, 4))


def test_invalid_architecture_raises() -> None:
    """Non-decreasing encoder widths raise ValueError."""
    with pytest.raises(ValueError):
        DenoisingAutoencoder(input_dim=8, encoder_hidden_dims=[4, 8], latent_dim=2)
    with pytest.raises(ValueError):
        DenoisingAutoencoder(input_dim=8, encoder_hidden_dims=[4], latent_dim=8)


def test_invalid_dimensions_raise() -> None:
    """Non-positive input/latent dims raise ValueError."""
    with pytest.raises(ValueError):
        DenoisingAutoencoder(input_dim=-1)
    with pytest.raises(ValueError):
        DenoisingAutoencoder(input_dim=8, latent_dim=0)


def test_state_meta_roundtrip(ae: DenoisingAutoencoder) -> None:
    """Rebuilding from state_meta yields an identical parameter count."""
    rebuilt = DenoisingAutoencoder(**ae.state_meta())
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(p.numel() for p in ae.parameters())


def test_save_and_load_checkpoint(ae: DenoisingAutoencoder, tmp_path: Path) -> None:
    """Checkpointed weights survive a save/load roundtrip."""
    ae.eval()
    ckpt = tmp_path / "ae.pt"
    save_checkpoint(ae, ckpt)
    loaded = load_checkpoint(ckpt)
    x = torch.randn(4, ae.input_dim)
    assert torch.allclose(ae(x), loaded(x))


def test_load_missing_checkpoint_raises(tmp_path: Path) -> None:
    """load_checkpoint raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_load_legit_features_filters_rows_and_columns(tmp_path: Path) -> None:
    """_load_legit_features keeps only legit rows and numeric feature cols."""
    from src.training.train_autoencoder import _load_legit_features

    import pandas as pd

    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 0, 1, 0],
            "feat_a": [0.1, 0.2, 0.3, 0.4],
            "feat_b": [1, 2, 3, 4],
            "cat": ["x", "y", "x", "z"],
        }
    )
    path = tmp_path / "train.parquet"
    df.to_parquet(path, index=False)

    loaded = _load_legit_features(path, non_feature_cols=["TransactionID", "isFraud"])
    assert loaded.shape == (3, 2)
    assert loaded.dtype == np.float32
    assert list(loaded[:, 0]) == [0.1, 0.2, 0.4]


def test_load_legit_features_missing_raises(tmp_path: Path) -> None:
    """_load_legit_features raises FileNotFoundError for missing files."""
    from src.training.train_autoencoder import _load_legit_features

    with pytest.raises(FileNotFoundError):
        _load_legit_features(tmp_path / "nope.parquet", non_feature_cols=["isFraud"])


def test_load_legit_features_no_numeric_raises(tmp_path: Path) -> None:
    """_load_legit_features raises ValueError when no numeric features exist."""
    from src.training.train_autoencoder import _load_legit_features

    import pandas as pd

    df = pd.DataFrame({"isFraud": [0, 0], "cat": ["x", "y"]})
    path = tmp_path / "train.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="No numeric feature"):
        _load_legit_features(path, non_feature_cols=["isFraud"])


def test_train_reduces_loss(config_small, tmp_path: Path) -> None:
    """A few training steps lower the validation loss and store a checkpoint."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((64, 8)).astype(np.float32)
    val_x = rng.standard_normal((16, 8)).astype(np.float32)
    out = tmp_path / "trained.pt"

    @torch.no_grad()
    def val_loss_of(model: DenoisingAutoencoder) -> float:
        model.eval()
        recon = model(torch.from_numpy(val_x))
        return model.loss(torch.from_numpy(val_x), recon).item()

    model = train_autoencoder(train_x, config_small, val_x, out_path=out, device="cpu")
    assert out.is_file()
    assert val_loss_of(model) < val_loss_of(
        DenoisingAutoencoder(input_dim=8, encoder_hidden_dims=[4], latent_dim=2)
    )
    reloaded = load_checkpoint(out)
    assert torch.allclose(model(torch.from_numpy(val_x[:4])), reloaded(torch.from_numpy(val_x[:4])))


def _tiny_train_config(tmp_path: Path, **training_overrides: float | int) -> Config:
    """Build a minimal CPU training config with overridable training knobs."""
    training = {
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "epochs": 2,
        "batch_size": 16,
        "num_workers": 0,
        "pin_memory": False,
        "early_stopping_patience": 5,
        "min_delta": 1.0e-6,
    }
    training.update(training_overrides)
    return Config(
        {
            "autoencoder": {
                "input_dim": 8,
                "encoder_hidden_dims": [4],
                "latent_dim": 2,
                "dropout": 0.0,
                "noise_std": 0.1,
                "feature_dropout_prob": 0.0,
                "activation_slope": 0.2,
                "training": training,
                "loss": {"mse_weight": 0.5, "bce_weight": 0.5},
            },
            "seed": 42,
        },
        base_dir=tmp_path,
    )


def test_load_checkpoint_without_meta_raises(tmp_path: Path) -> None:
    """load_checkpoint raises KeyError when architecture metadata is absent."""
    payload = {"state_dict": {}}
    ckpt = tmp_path / "nometa.pt"
    torch.save(payload, ckpt)
    with pytest.raises(KeyError):
        load_checkpoint(ckpt)


def test_train_accepts_torch_tensors(tmp_path: Path) -> None:
    """Tensor inputs (not only numpy) train and checkpoint normally."""
    torch.manual_seed(0)
    cfg = _tiny_train_config(tmp_path)
    train_x = torch.randn(32, 8)
    val_x = torch.randn(8, 8)
    out = tmp_path / "tensors.pt"
    model = train_autoencoder(train_x, cfg, val_x, out_path=out, device="cpu")
    assert out.is_file()
    assert model.input_dim == 8


def test_train_rejects_bad_dimensions(tmp_path: Path) -> None:
    """1D or empty training matrices raise ValueError."""
    cfg = _tiny_train_config(tmp_path)
    with pytest.raises(ValueError):
        train_autoencoder(np.zeros(8, dtype=np.float32), cfg, device="cpu")
    with pytest.raises(ValueError):
        train_autoencoder(np.zeros((0, 8), dtype=np.float32), cfg, device="cpu")


def test_train_splits_validation_when_val_missing(tmp_path: Path) -> None:
    """Omitting val_x falls back to the trailing 10% of the training rows."""
    torch.manual_seed(1)
    cfg = _tiny_train_config(tmp_path)
    rng = np.random.default_rng(1)
    train_x = rng.standard_normal((40, 8)).astype(np.float32)
    model = train_autoencoder(train_x, cfg, val_x=None, out_path=tmp_path / "split.pt")
    assert model.input_dim == 8


def test_train_empty_val_falls_back_to_train_sample(tmp_path: Path) -> None:
    """An empty validation matrix falls back to a single training sample."""
    torch.manual_seed(2)
    cfg = _tiny_train_config(tmp_path)
    rng = np.random.default_rng(2)
    train_x = rng.standard_normal((32, 8)).astype(np.float32)
    val_x = np.zeros((0, 8), dtype=np.float32)
    model = train_autoencoder(train_x, cfg, val_x, out_path=tmp_path / "empty_val.pt")
    assert model.input_dim == 8


def test_train_early_stopping_breaks(tmp_path: Path) -> None:
    """An impossible min_delta stalls validation and triggers the early-stop break."""
    torch.manual_seed(3)
    cfg = _tiny_train_config(tmp_path, epochs=5, early_stopping_patience=1, min_delta=1.0)
    rng = np.random.default_rng(3)
    train_x = rng.standard_normal((32, 8)).astype(np.float32)
    val_x = rng.standard_normal((8, 8)).astype(np.float32)
    model = train_autoencoder(train_x, cfg, val_x, out_path=tmp_path / "early.pt")
    assert (tmp_path / "early.pt").is_file()
    assert model.input_dim == 8


def test_invalid_feature_dropout_prob_raises() -> None:
    """feature_dropout_prob outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        DenoisingAutoencoder(input_dim=8, feature_dropout_prob=1.5)


def test_main_trains_from_parquet_cli(tmp_path: Path) -> None:
    """main() trains end-to-end from tiny parquet splits and a yaml config."""
    import logging

    import pandas as pd
    import yaml

    import src.utils.logger as logger_module
    from src.training.train_autoencoder import main

    rng = np.random.default_rng(7)
    cols = {f"f{i}": rng.standard_normal(48) for i in range(8)}
    cols["isFraud"] = [0] * 40 + [1] * 8
    train_df = pd.DataFrame(cols)
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path, index=False)
    val_cols = {f"f{i}": rng.standard_normal(24) for i in range(8)}
    val_cols["isFraud"] = [0] * 20 + [1] * 4
    val_path = tmp_path / "val.parquet"
    pd.DataFrame(val_cols).to_parquet(val_path, index=False)

    config_payload = {
        "logging": {"level": "INFO", "log_file": "cli.log"},
        "seed": 42,
        "data": {
            "non_feature_cols": ["isFraud"],
            "train_data_path": str(train_path),
            "val_data_path": str(val_path),
        },
        "autoencoder": {
            "input_dim": 8,
            "encoder_hidden_dims": [4],
            "latent_dim": 2,
            "dropout": 0.0,
            "noise_std": 0.1,
            "feature_dropout_prob": 0.0,
            "activation_slope": 0.2,
            "training": {
                "lr": 1.0e-3,
                "weight_decay": 1.0e-5,
                "epochs": 2,
                "batch_size": 16,
                "num_workers": 0,
                "pin_memory": False,
                "early_stopping_patience": 5,
                "min_delta": 1.0e-6,
            },
            "loss": {"mse_weight": 0.5, "bce_weight": 0.5},
        },
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    out = tmp_path / "cli.pt"

    root_logger = logging.getLogger()
    prev_handlers = list(root_logger.handlers)
    prev_configured = logger_module._configured
    logger_module._configured = False
    try:
        main(["--config", str(config_file), "--out", str(out), "--device", "cpu"])
    finally:
        for handler in list(root_logger.handlers):
            if handler not in prev_handlers:
                root_logger.removeHandler(handler)
                handler.close()
        logger_module._configured = prev_configured

    assert out.is_file()
    reloaded = load_checkpoint(out)
    assert reloaded.input_dim == 8


def test_train_read_only_arrays_emit_no_warning(tmp_path: Path) -> None:
    """Read-only inputs (e.g. pandas views) train without the torch warning."""
    import warnings

    torch.manual_seed(4)
    cfg = _tiny_train_config(tmp_path)
    rng = np.random.default_rng(4)
    train_x = rng.standard_normal((32, 8)).astype(np.float32)
    train_x.setflags(write=False)
    val_x = rng.standard_normal((8, 8)).astype(np.float32)
    val_x.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = train_autoencoder(
            train_x, cfg, val_x, out_path=tmp_path / "readonly.pt", device="cpu"
        )
    assert model.input_dim == 8
