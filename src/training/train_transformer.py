"""Training entry point for the Feature-Tokenizer Cross-Attention Transformer.

Trains FT-CAT -- or one of its ablation variants -- on the processed IEEE-CIS
sequence splits, early-stopping on validation PR-AUC and checkpointing to
``models/checkpoints/ft_transformer.pt``.

Rows are materialised into contiguous tensors once, up front, rather than
being pulled through a row-wise ``Dataset``.
:class:`~src.data.dataset.TransactionSequenceDataset` indexes pandas per
sample, which costs hundreds of microseconds a row and would dominate GPU
epoch time; the tensors built here are identical in shape, dtype, and row
order, so the data contract is unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import average_precision, roc_auc, tpr_at_fpr
from src.models.ft_transformer import (
    VARIANTS,
    FTCATransformer,
    TabularMLP,
    build_ft_model,
    count_parameters,
)
from src.models.losses import build_loss
from src.training.feature_selection import FeatureSpec, resolve_feature_set
from src.training.trainer_utils import (
    AmpPolicy,
    EarlyStopping,
    EpochMeter,
    build_warmup_cosine_scheduler,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    stratified_subsample,
)
from src.utils.config import Config, load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_CHECKPOINT = Path("models/checkpoints/ft_transformer.pt")

# Column written by ``build_historical_sequences``; each cell holds the K most
# recent prior transactions, index 0 being the most recent (lag 1).
SEQUENCE_COLUMN = "sequence_array"
DEFAULT_MAX_FPR = 0.01


@dataclass(frozen=True)
class TensorBundle:
    """Materialised model inputs for one split.

    Attributes:
        x_cont: Continuous features, ``(n_rows, n_continuous)`` float32.
        x_cat: Categorical codes, ``(n_rows, n_categorical)`` int64.
        seq: History windows, ``(n_rows, seq_len, seq_dim)`` float32.
        y: Binary fraud labels, ``(n_rows,)`` float32.
    """

    x_cont: torch.Tensor
    x_cat: torch.Tensor
    seq: torch.Tensor
    y: torch.Tensor

    def __post_init__(self) -> None:
        """Validate that every tensor describes the same rows.

        Raises:
            ValueError: If the tensors disagree on row count or rank.
        """
        rows = {
            "x_cont": self.x_cont.shape[0],
            "x_cat": self.x_cat.shape[0],
            "seq": self.seq.shape[0],
            "y": self.y.shape[0],
        }
        if len(set(rows.values())) != 1:
            raise ValueError(f"tensors disagree on row count: {rows}")
        if self.x_cont.dim() != 2 or self.x_cat.dim() != 2:
            raise ValueError("x_cont and x_cat must be 2D")
        if self.seq.dim() != 3:
            raise ValueError(f"seq must be 3D, got {tuple(self.seq.shape)}")
        if self.y.dim() != 1:
            raise ValueError(f"y must be 1D, got {tuple(self.y.shape)}")

    def __len__(self) -> int:
        """Return the number of rows in the bundle."""
        return int(self.y.shape[0])

    @property
    def n_positive(self) -> int:
        """Number of fraudulent rows."""
        return int(self.y.sum().item())

    @property
    def positive_rate(self) -> float:
        """Fraction of fraudulent rows, or ``0.0`` when empty."""
        return self.n_positive / len(self) if len(self) else 0.0

    def labels_numpy(self) -> np.ndarray:
        """Return the labels as a 1D int64 numpy array."""
        return self.y.detach().cpu().numpy().astype(np.int64)


@dataclass(frozen=True)
class TrainingHistory:
    """Per-epoch record of one training run.

    Attributes:
        train_loss: Training loss per epoch.
        val_loss: Validation loss per epoch.
        val_pr_auc: Validation PR-AUC per epoch (the early-stopping criterion).
        val_roc_auc: Validation ROC-AUC per epoch.
        best_epoch: 1-indexed epoch with the highest validation PR-AUC.
        best_pr_auc: The highest validation PR-AUC observed.
        epochs_run: Number of epochs actually executed.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_pr_auc: float = 0.0
    epochs_run: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the history."""
        return asdict(self)


def _to_float_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Extract continuous columns as a NaN-free float32 matrix.

    Args:
        df: Source frame.
        cols: Continuous column names, in tokenizer order.

    Returns:
        Array of shape ``(len(df), len(cols))``.
    """
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    matrix = df.loc[:, list(cols)].to_numpy(dtype=np.float32, copy=True)
    # The preprocessor emits explicit ``_is_nan`` mask columns, so residual
    # NaNs carry no extra signal but would poison every downstream gradient.
    if not np.isfinite(matrix).all():
        logger.warning("Continuous block holds non-finite values; substituting zeros")
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def _to_code_matrix(
    df: pd.DataFrame, cols: Sequence[str], cardinalities: Sequence[int]
) -> np.ndarray:
    """Extract categorical columns as embedding-safe int64 codes.

    ``FraudPreprocessor`` maps categories unseen at fit time to the reserved
    ``<UNK>`` index 0, so codes should already be in range. Clipping is a
    defensive guard: an out-of-range code would otherwise surface as an opaque
    device-side assertion inside an embedding lookup.

    Args:
        df: Source frame.
        cols: Categorical column names, in tokenizer order.
        cardinalities: Embedding table size for each column.

    Returns:
        Array of shape ``(len(df), len(cols))``.

    Raises:
        ValueError: If ``cols`` and ``cardinalities`` have different lengths.
    """
    if len(cols) != len(cardinalities):
        raise ValueError(f"cols ({len(cols)}) and cardinalities ({len(cardinalities)}) must align")
    if not cols:
        return np.zeros((len(df), 0), dtype=np.int64)

    matrix = df.loc[:, list(cols)].to_numpy(dtype=np.int64, copy=True)
    for index, (name, card) in enumerate(zip(cols, cardinalities)):
        column = matrix[:, index]
        out_of_range = (column < 0) | (column >= card)
        if bool(out_of_range.any()):
            logger.warning(
                "Column %r holds %d codes outside [0, %d); clamping to <UNK>",
                name,
                int(out_of_range.sum()),
                card,
            )
            column[out_of_range] = 0
            matrix[:, index] = column
    return matrix


def _stack_sequences(series: pd.Series, seq_len: int, seq_dim: int) -> np.ndarray:
    """Stack the per-row history windows into one contiguous array.

    Args:
        series: The ``sequence_array`` column.
        seq_len: Expected window length ``K``.
        seq_dim: Expected per-timestep feature width ``D``.

    Returns:
        Array of shape ``(len(series), seq_len, seq_dim)``.

    Raises:
        ValueError: If any window does not match ``(seq_len, seq_dim)``.
    """
    n_rows = len(series)
    # Parquet stores the column as list<list<double>> and hands it back as
    # nested object arrays, so a bulk np.asarray cannot type it. Filling a
    # preallocated buffer row by row is the one path that works for both that
    # form and the plain nested lists an in-memory frame carries, at roughly
    # two seconds for a 400k-row split and no large temporary.
    stacked = np.zeros((n_rows, seq_len, seq_dim), dtype=np.float32)
    for row_index, window in enumerate(series.to_numpy()):
        block = np.vstack(window).astype(np.float32, copy=False)
        if block.shape != (seq_len, seq_dim):
            raise ValueError(
                f"row {row_index} has history of shape {tuple(block.shape)}, "
                f"expected ({seq_len}, {seq_dim})"
            )
        stacked[row_index] = block
    return np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)


def materialize_tensors(
    df: pd.DataFrame, spec: FeatureSpec, label_col: str = "isFraud"
) -> TensorBundle:
    """Convert a processed split into the four model input tensors.

    Args:
        df: Processed split carrying feature columns and ``sequence_array``.
        spec: Resolved feature contract defining column order and widths.
        label_col: Name of the binary target column.

    Returns:
        A :class:`TensorBundle` matching the FT-CAT forward signature.

    Raises:
        KeyError: If required columns are absent from ``df``.
    """
    required = list(spec.continuous_cols) + list(spec.categorical_cols)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"split is missing {len(missing)} feature columns, e.g. {missing[:5]}")
    for col in (SEQUENCE_COLUMN, label_col):
        if col not in df.columns:
            raise KeyError(f"split has no {col!r} column")

    codes = _to_code_matrix(df, spec.categorical_cols, spec.categorical_cardinalities)
    bundle = TensorBundle(
        x_cont=torch.from_numpy(_to_float_matrix(df, spec.continuous_cols)),
        x_cat=torch.from_numpy(codes),
        seq=torch.from_numpy(_stack_sequences(df[SEQUENCE_COLUMN], spec.seq_len, spec.seq_dim)),
        y=torch.from_numpy(df[label_col].to_numpy(dtype=np.float32, copy=True)),
    )
    logger.info(
        "Materialised %d rows: %d continuous, %d categorical, history (%d, %d), %.3f%% fraud",
        len(bundle),
        bundle.x_cont.shape[1],
        bundle.x_cat.shape[1],
        bundle.seq.shape[1],
        bundle.seq.shape[2],
        100.0 * bundle.positive_rate,
    )
    return bundle


def subsample_bundle(bundle: TensorBundle, n_rows: int, seed: int) -> TensorBundle:
    """Draw a class-proportional row subsample from a bundle.

    Args:
        bundle: The bundle to shrink.
        n_rows: Target row count; ``<= 0`` or ``>= len(bundle)`` is a no-op.
        seed: Seed controlling the draw.

    Returns:
        A bundle holding the sampled rows, in their original order.
    """
    if n_rows <= 0 or n_rows >= len(bundle):
        return bundle
    index = torch.from_numpy(stratified_subsample(bundle.labels_numpy(), n_rows, seed))
    sampled = TensorBundle(
        x_cont=bundle.x_cont[index],
        x_cat=bundle.x_cat[index],
        seq=bundle.seq[index],
        y=bundle.y[index],
    )
    logger.info(
        "Subsampled %d -> %d rows (%.3f%% fraud preserved)",
        len(bundle),
        len(sampled),
        100.0 * sampled.positive_rate,
    )
    return sampled


def build_loader(
    bundle: TensorBundle, batch_size: int, shuffle: bool, pin_memory: bool
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Wrap a bundle in a DataLoader.

    Worker processes are deliberately not used: the bundle already lives in
    memory as tensors, so Windows process spawning would be pure overhead.

    Args:
        bundle: Materialised split.
        batch_size: Rows per batch.
        shuffle: Whether to reshuffle between epochs.
        pin_memory: Whether to pin host memory for faster GPU transfer.

    Returns:
        A DataLoader yielding ``(x_cont, x_cat, seq, y)`` batches.
    """
    dataset = TensorDataset(bundle.x_cont, bundle.x_cat, bundle.seq, bundle.y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    device: str,
    amp: AmpPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one pass over ``loader``, training when an optimizer is supplied.

    Args:
        model: The model to run.
        loader: Batches of ``(x_cont, x_cat, seq, y)``.
        criterion: Loss module taking ``(logits, targets)``.
        device: Device to run on.
        amp: Mixed-precision policy.
        optimizer: Optimizer for training; ``None`` runs evaluation.
        grad_clip: Optional gradient-norm clip applied during training.

    Returns:
        Tuple of ``(mean_loss, scores, labels)`` where ``scores`` are sigmoid
        probabilities.
    """
    training = optimizer is not None
    model.train(training)
    meter = EpochMeter()
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for x_cont, x_cat, seq, y in loader:
        x_cont = x_cont.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        seq = seq.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(training), amp.autocast():
            logits = model(x_cont, x_cat, seq)
            loss = criterion(logits, y)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            amp.backward(loss)
            amp.step(optimizer, parameters=model.parameters(), grad_clip=grad_clip)

        meter.update(float(loss.detach().item()), n=int(y.shape[0]))
        score_chunks.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        label_chunks.append(y.detach().cpu().numpy())

    return meter.average, np.concatenate(score_chunks), np.concatenate(label_chunks)


def _safe_tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float) -> float:
    """Report TPR at a target FPR, tolerating splits too coarse to resolve it.

    The smallest FPR a split can express is ``1 / n_negatives``; on a small
    validation slice that floor can sit above ``max_fpr``, leaving the metric
    genuinely undefined. It is a reporting statistic rather than the stopping
    criterion, so an undefined value is surfaced as NaN instead of aborting
    the run.

    Args:
        scores: Per-sample fraud probabilities.
        labels: Binary ground-truth labels.
        max_fpr: Target false-positive rate.

    Returns:
        The TPR at the operating point, or NaN when none exists.
    """
    try:
        return float(tpr_at_fpr(scores, labels, max_fpr=max_fpr))
    except ValueError as exc:
        logger.warning("TPR@%.3g FPR undefined for this split (%s)", max_fpr, exc)
        return float("nan")


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    device: str,
    amp: AmpPolicy,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> dict[str, float]:
    """Score a split and summarise it with imbalance-aware metrics.

    Args:
        model: Model to evaluate.
        loader: Evaluation batches.
        criterion: Loss module, for the reported loss.
        device: Device to run on.
        amp: Mixed-precision policy.
        max_fpr: Operating point for the TPR report.

    Returns:
        Dict with ``loss``, ``pr_auc``, ``roc_auc`` and ``tpr_at_fpr``; the
        last is NaN when the split is too coarse to resolve ``max_fpr``.

    Raises:
        ValueError: If the split holds only one class, which makes ranking
            metrics undefined.
    """
    loss, scores, labels = run_epoch(model, loader, criterion, device, amp)
    positives = int(labels.sum())
    if positives == 0 or positives == labels.shape[0]:
        raise ValueError(
            f"evaluation split is single-class (pos={positives}, n={labels.shape[0]}); "
            "ranking metrics are undefined"
        )
    return {
        "loss": float(loss),
        "pr_auc": float(average_precision(scores, labels)),
        "roc_auc": float(roc_auc(scores, labels)),
        "tpr_at_fpr": _safe_tpr_at_fpr(scores, labels, max_fpr),
    }


def train_transformer(
    config: Config,
    train_bundle: TensorBundle,
    val_bundle: TensorBundle,
    spec: FeatureSpec,
    variant: str = "ft_cat",
    device: str | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    criterion: nn.Module | None = None,
) -> tuple[nn.Module, TrainingHistory]:
    """Train one FT-CAT variant with early stopping on validation PR-AUC.

    PR-AUC rather than loss or accuracy is the stopping criterion because the
    positive class is roughly 3.5% of the data, where accuracy is degenerate
    and loss is a poor proxy for ranking quality.

    Args:
        config: Loaded project configuration.
        train_bundle: Materialised training split.
        val_bundle: Materialised validation split.
        spec: Feature contract the bundles were built with.
        variant: One of :data:`~src.models.ft_transformer.VARIANTS`.
        device: Target device; resolved automatically when ``None``.
        epochs: Epoch budget; falls back to the configured value.
        seed: Seed for this run; falls back to ``config.seed``.
        criterion: Loss override; falls back to ``build_loss(config)``.

    Returns:
        Tuple of ``(model, history)`` with best-epoch weights restored.

    Raises:
        ValueError: If ``variant`` is unknown or the epoch budget is empty.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")

    run_seed = int(config.seed if seed is None else seed)
    seed_everything(run_seed)

    resolved_device = resolve_device(device)
    train_cfg = config.transformer.training
    total_epochs = int(train_cfg.epochs if epochs is None else epochs)
    if total_epochs <= 0:
        raise ValueError(f"epochs must be positive, got {total_epochs}")

    model = build_ft_model(
        config,
        variant=variant,
        n_continuous=spec.n_continuous,
        categorical_cardinalities=spec.categorical_cardinalities,
        seq_dim=spec.seq_dim,
    ).to(resolved_device)

    objective = (build_loss(config) if criterion is None else criterion).to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.lr),
        weight_decay=float(train_cfg.weight_decay),
    )
    # A short sweep run may be shorter than the configured warmup window.
    warmup_epochs = min(int(train_cfg.warmup_epochs), total_epochs)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs)
    amp = AmpPolicy(resolved_device, bool(train_cfg.amp))
    stopper = EarlyStopping(
        patience=int(train_cfg.early_stopping_patience),
        min_delta=float(train_cfg.min_delta),
        mode="max",
    )

    pin_memory = bool(train_cfg.pin_memory) and resolved_device.startswith("cuda")
    batch_size = int(train_cfg.batch_size)
    train_loader = build_loader(train_bundle, batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = build_loader(val_bundle, batch_size, shuffle=False, pin_memory=pin_memory)

    logger.info(
        "Training %r on %s | %d params | %d train / %d val rows | %d epochs | seed %d",
        variant,
        resolved_device,
        count_parameters(model),
        len(train_bundle),
        len(val_bundle),
        total_epochs,
        run_seed,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_pr_aucs: list[float] = []
    val_roc_aucs: list[float] = []
    epochs_run = 0

    for epoch in range(1, total_epochs + 1):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            objective,
            resolved_device,
            amp,
            optimizer=optimizer,
            grad_clip=float(train_cfg.grad_clip),
        )
        metrics = evaluate(model, val_loader, objective, resolved_device, amp)
        scheduler.step()
        epochs_run = epoch

        train_losses.append(float(train_loss))
        val_losses.append(metrics["loss"])
        val_pr_aucs.append(metrics["pr_auc"])
        val_roc_aucs.append(metrics["roc_auc"])

        improved = stopper.update(metrics["pr_auc"], model)
        logger.info(
            "Epoch %3d/%d | train_loss %.5f | val_loss %.5f | val_pr_auc %.5f | "
            "val_roc_auc %.5f | lr %.2e%s",
            epoch,
            total_epochs,
            train_loss,
            metrics["loss"],
            metrics["pr_auc"],
            metrics["roc_auc"],
            optimizer.param_groups[0]["lr"],
            "  *" if improved else "",
        )
        if stopper.should_stop:
            logger.info("Early stopping after %d epochs without improvement", stopper.patience)
            break

    stopper.restore(model)
    history = TrainingHistory(
        train_loss=train_losses,
        val_loss=val_losses,
        val_pr_auc=val_pr_aucs,
        val_roc_auc=val_roc_aucs,
        best_epoch=stopper.best_epoch,
        best_pr_auc=float(stopper.best_score),
        epochs_run=epochs_run,
    )
    logger.info(
        "Finished %r: best val PR-AUC %.5f at epoch %d",
        variant,
        history.best_pr_auc,
        history.best_epoch,
    )
    return model, history


def build_model_from_meta(meta: dict[str, Any]) -> nn.Module:
    """Rebuild an FT-CAT variant from checkpoint metadata.

    Args:
        meta: The ``meta`` payload written by :func:`save_checkpoint`.

    Returns:
        An untrained model with the recorded architecture.
    """
    kwargs = dict(meta)
    if "use_cross_attention" in kwargs:
        return FTCATransformer(**kwargs)
    return TabularMLP(**kwargs)


def load_ft_transformer(
    path: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained FT-CAT checkpoint.

    The entry point Members C and D use: the returned payload carries the
    ``feature_spec`` describing exactly which columns, in which order, the
    model expects.

    Args:
        path: Checkpoint written by this module.
        device: Device to map the weights onto.

    Returns:
        Tuple of ``(model in eval mode, checkpoint payload)``.
    """
    return load_checkpoint(path, build_model_from_meta, device=device)


def _load_split(path: str | Path) -> pd.DataFrame:
    """Read one processed parquet split.

    Args:
        path: Parquet file path.

    Returns:
        The loaded frame.

    Raises:
        FileNotFoundError: If the split does not exist.
    """
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Processed split not found: {split_path}. "
            "Run `python -m src.data.prepare_data` first."
        )
    frame = pd.read_parquet(split_path)
    logger.info("Loaded %s (%d rows, %d columns)", split_path, len(frame), frame.shape[1])
    return frame


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for training the FT-CAT transformer.

    Args:
        argv: Command line arguments; uses ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(description="Train the FT-CAT fraud transformer")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu; auto when omitted")
    parser.add_argument("--variant", type=str, default="ft_cat", choices=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--subsample",
        action="store_true",
        help="train on config.transformer.subsample rows instead of the full split",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="skip the held-out test evaluation after training",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(
        level=str(config.logging.level), log_file=str(config.get_path("logging.log_file"))
    )
    run_seed = int(config.seed if args.seed is None else args.seed)
    seed_everything(run_seed)

    train_df = _load_split(args.train_data or config.get_path("data.train_data_path"))
    val_df = _load_split(args.val_data or config.get_path("data.val_data_path"))

    spec = resolve_feature_set(config, train_df=train_df)
    train_bundle = materialize_tensors(train_df, spec)
    val_bundle = materialize_tensors(val_df, spec)
    del train_df, val_df

    if args.subsample:
        sub_cfg = config.transformer.subsample
        train_bundle = subsample_bundle(train_bundle, int(sub_cfg.train_rows), run_seed)
        val_bundle = subsample_bundle(val_bundle, int(sub_cfg.val_rows), run_seed)

    model, history = train_transformer(
        config,
        train_bundle,
        val_bundle,
        spec,
        variant=args.variant,
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
    )

    device = resolve_device(args.device)
    amp = AmpPolicy(device, bool(config.transformer.training.amp))
    objective = build_loss(config).to(device)
    batch_size = int(config.transformer.training.batch_size)
    metrics: dict[str, dict[str, float]] = {
        "val": evaluate(
            model,
            build_loader(val_bundle, batch_size, shuffle=False, pin_memory=False),
            objective,
            device,
            amp,
        )
    }
    logger.info("Validation metrics: %s", json.dumps(metrics["val"]))

    if not args.skip_test:
        test_df = _load_split(args.test_data or config.get_path("data.test_data_path"))
        test_bundle = materialize_tensors(test_df, spec)
        metrics["test"] = evaluate(
            model,
            build_loader(test_bundle, batch_size, shuffle=False, pin_memory=False),
            objective,
            device,
            amp,
        )
        logger.info("Out-of-time test metrics: %s", json.dumps(metrics["test"]))

    checkpoint = save_checkpoint(
        model,
        args.out,
        extra={
            "history": history.to_dict(),
            "feature_spec": spec.to_dict(),
            "variant": args.variant,
            "metrics": metrics,
            "seed": run_seed,
        },
    )
    logger.info("Checkpoint %s is %.1f KB on disk", checkpoint, checkpoint.stat().st_size / 1024.0)


if __name__ == "__main__":
    main()
