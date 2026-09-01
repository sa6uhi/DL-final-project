"""Training entry point for the semi-supervised Deep Denoising Autoencoder.

Trains the DAE strictly on legitimate transactions (``y=0``) sourced from the
processed parquet splits, with early stopping on validation reconstruction
loss and checkpointing to ``models/checkpoints/autoencoder.pt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import DenoisingAutoencoder
from src.utils.config import Config, load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_CHECKPOINT = Path("models/checkpoints/autoencoder.pt")


def _make_loader(
    x: torch.Tensor, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Wrap a feature tensor into a torch DataLoader.

    Args:
        x: Feature tensor of shape ``(n_samples, input_dim)``.
        batch_size: Batch size for sampling.
        shuffle: Whether to shuffle samples between epochs.
        num_workers: Number of dataloader worker processes.
        pin_memory: Whether to pin memory for GPU transfers.

    Returns:
        DataLoader yielding ``(features,)`` batches.
    """
    dataset = TensorDataset(x)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def save_checkpoint(model: DenoisingAutoencoder, path: str | Path) -> None:
    """Persist model weights and architecture metadata to disk.

    Args:
        model: Trained autoencoder instance.
        path: Destination ``.pt`` checkpoint path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": model.state_meta()}, out)
    logger.info("Saved autoencoder checkpoint to %s", out)


def load_checkpoint(path: str | Path, device: str = "cpu") -> DenoisingAutoencoder:
    """Rebuild a DenoisingAutoencoder from a serialized checkpoint.

    Args:
        path: Checkpoint file produced by :func:`save_checkpoint`.
        device: Target device for the restored model.

    Returns:
        The restored model in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint file is missing.
        KeyError: If the checkpoint lacks architecture metadata.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = payload.get("meta")
    if meta is None:
        raise KeyError(f"Checkpoint {ckpt_path} has no architecture metadata")
    model = DenoisingAutoencoder(**meta)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def train_autoencoder(
    train_x: np.ndarray | torch.Tensor,
    config: Config,
    val_x: np.ndarray | torch.Tensor | None = None,
    out_path: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> DenoisingAutoencoder:
    """Train the denoising autoencoder on legitimate transactions.

    Args:
        train_x: Feature matrix (``n_train, input_dim``) of legit records.
        config: Central project configuration (hyperparameters under the
            ``autoencoder`` and ``paths`` keys).
        val_x: Feature matrix (``n_val, input_dim``) of legit records used
            for early stopping. If omitted, the last 10% of training data
            is used.
        out_path: Checkpoint destination.
        device: ``"cpu"`` or ``"cuda"`` for training.

    Returns:
        The trained autoencoder (best validation checkpoint weights).

    Raises:
        ValueError: If ``train_x`` is empty or feature dims are inconsistent.
    """
    acfg = config.autoencoder

    if isinstance(train_x, torch.Tensor):
        train_tensor = train_x.float()
    else:
        train_tensor = torch.from_numpy(np.asarray(train_x, dtype=np.float32))
    if train_tensor.dim() != 2 or train_tensor.size(0) == 0:
        raise ValueError(f"train_x must be a non-empty 2D array, got {tuple(train_tensor.shape)}")

    if isinstance(val_x, torch.Tensor):
        val_tensor = val_x.float()
    elif val_x is not None:
        val_tensor = torch.from_numpy(np.asarray(val_x, dtype=np.float32))
    else:
        n_val = max(1, int(train_tensor.size(0) * 0.1))
        train_tensor, val_tensor = train_tensor[:-n_val], train_tensor[-n_val:]
    if val_tensor.size(0) == 0:
        val_tensor = train_tensor[:1]

    model = DenoisingAutoencoder(
        input_dim=int(acfg.input_dim),
        encoder_hidden_dims=[int(h) for h in acfg.encoder_hidden_dims],
        latent_dim=int(acfg.latent_dim),
        dropout=float(acfg.dropout),
        noise_std=float(acfg.noise_std),
        feature_dropout_prob=float(acfg.feature_dropout_prob),
        activation_slope=float(acfg.activation_slope),
    )
    model.init_weights()
    model.to(device)

    train_cfg = acfg.training
    train_loader = _make_loader(
        train_tensor,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(train_cfg.num_workers),
        pin_memory=bool(train_cfg.pin_memory),
    )
    val_loader = _make_loader(
        val_tensor,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )
    loss_cfg = acfg.loss
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg.epochs))

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    patience = int(train_cfg.early_stopping_patience)
    min_delta = float(train_cfg.min_delta)

    for epoch in range(1, int(train_cfg.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        for features in train_loader:
            x = features[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            x_hat = model(x, corrupt=True)
            loss = model.loss(
                x,
                x_hat,
                mse_weight=float(loss_cfg.mse_weight),
                bce_weight=float(loss_cfg.bce_weight),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        scheduler.step()

        val_loss = _validate(model, val_loader, device, loss_cfg)
        logger.info(
            "Epoch %3d/%d | train_loss %.5f | val_loss %.5f",
            epoch,
            int(train_cfg.epochs),
            epoch_loss / train_tensor.size(0),
            val_loss,
        )

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience %d)", epoch, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    save_checkpoint(model, out_path)
    logger.info("Training complete | best val_loss %.5f", best_loss)
    return model


def _validate(
    model: nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, ...]],
    device: str,
    loss_cfg: Any,
) -> float:
    """Compute the average composite loss over the validation loader.

    Args:
        model: The autoencoder under evaluation.
        val_loader: DataLoader over validation features.
        device: Device to evaluate on.
        loss_cfg: Loss weights from the central configuration.

    Returns:
        Mean validation loss over all samples.
    """
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for features in val_loader:
            x = features[0].to(device)
            x_hat = model(x)
            total += model.loss(
                x,
                x_hat,
                mse_weight=float(loss_cfg.mse_weight),
                bce_weight=float(loss_cfg.bce_weight),
            ).item() * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def _load_legit_features(
    path: str | Path, non_feature_cols: Sequence[str], target_col: str = "isFraud"
) -> np.ndarray:
    """Load a feature matrix of legitimate transactions from a parquet split.

    Selects all numeric (continuous) columns except identifiers and the
    target, then keeps only rows labelled ``isFraud == 0`` since the DAE is
    trained exclusively on legitimate transactions.

    Args:
        path: Path to a processed ``.parquet`` split.
        non_feature_cols: Columns that are never model features (ids, target,
            sequence tensor).
        target_col: Name of the binary fraud label column.

    Returns:
        Float32 feature matrix of shape ``(n_legit, n_features)``.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ValueError: If no numeric feature columns are found.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data split not found: {file_path}")

    df = pd.read_parquet(file_path)
    numeric_cols = [
        col
        for col in df.columns
        if col not in non_feature_cols and df[col].dtype in ("float64", "float32", "int64", "int32")
    ]
    if not numeric_cols:
        raise ValueError(f"No numeric feature columns found in {file_path}")

    legit = df[df[target_col] == 0]
    features = legit[numeric_cols].to_numpy(dtype=np.float32)
    logger.info(
        "Loaded %d legit transactions with %d features from %s",
        features.shape[0],
        features.shape[1],
        file_path,
    )
    return features


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for training the autoencoder.

    Args:
        argv: Command line arguments; uses ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(description="Train the Deep DAE autoencoder on legit data")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-data", type=str, default=None, help="parquet of legit features")
    parser.add_argument(
        "--val-data", type=str, default=None, help="optional parquet of legit features"
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(
        level=str(config.logging.level), log_file=str(config.get_path("logging.log_file"))
    )
    seed_everything(int(config.seed))

    non_feature_cols = list(config.data.non_feature_cols)
    train_path = args.train_data or config.get_path("data.train_data_path")
    val_path = args.val_data or config.get_path("data.val_data_path")

    train_x = _load_legit_features(train_path, non_feature_cols)
    val_x = _load_legit_features(val_path, non_feature_cols)
    train_autoencoder(train_x, config, val_x, out_path=args.out, device=args.device)


if __name__ == "__main__":
    main()
