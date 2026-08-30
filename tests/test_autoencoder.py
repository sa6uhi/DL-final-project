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
    from src.utils.config import Config

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


def test_load_feature_matrix_npy(tmp_path: Path) -> None:
    """_load_feature_matrix loads .npy feature matrices."""
    from src.training.train_autoencoder import _load_feature_matrix

    data = np.zeros((10, 8), dtype=np.float32)
    path = tmp_path / "features.npy"
    np.save(path, data)
    loaded = _load_feature_matrix(path)
    assert np.array_equal(loaded, data)


def test_load_feature_matrix_missing_raises(tmp_path: Path) -> None:
    """_load_feature_matrix raises FileNotFoundError for missing files."""
    from src.training.train_autoencoder import _load_feature_matrix

    with pytest.raises(FileNotFoundError):
        _load_feature_matrix(tmp_path / "nope.npy")


def test_load_feature_matrix_unsupported_raises(tmp_path: Path) -> None:
    """_load_feature_matrix raises ValueError for unsupported suffixes."""
    from src.training.train_autoencoder import _load_feature_matrix

    path = tmp_path / "features.pt"
    path.write_bytes(b"fake")
    with pytest.raises(ValueError):
        _load_feature_matrix(path)


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
