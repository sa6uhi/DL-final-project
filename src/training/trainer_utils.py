"""Reusable training primitives shared by the supervised training entry points.

Factored out of the individual training scripts so that early stopping,
learning-rate scheduling, mixed-precision policy, and checkpoint I/O behave
identically wherever they are used. Everything here is deliberately
model-agnostic: the helpers take tensors, optimizers, and ``nn.Module``
instances rather than any particular architecture.

The mixed-precision helper is intentionally conservative. ``torch.amp`` only
pays off on CUDA; enabling autocast on CPU degrades throughput on the
project's target hardware, so :class:`AmpPolicy` silently downgrades to a
no-op there rather than pretending AMP is active.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from src.training.feature_selection import stratified_indices
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_MODES: frozenset[str] = frozenset({"min", "max"})


class EpochMeter:
    """Accumulate a sample-weighted running average over one epoch.

    Batches at the end of an epoch are often smaller than the rest, so a plain
    mean over batch losses would over-weight them. This meter weights each
    observation by the number of samples it summarises.
    """

    def __init__(self) -> None:
        """Start an empty meter."""
        self._total: float = 0.0
        self._count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        """Record ``value`` as the mean over ``n`` samples."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self._total += float(value) * n
        self._count += n

    @property
    def count(self) -> int:
        """Total number of samples recorded."""
        return self._count

    @property
    def average(self) -> float:
        """Sample-weighted mean, or ``0.0`` when nothing was recorded."""
        return self._total / self._count if self._count else 0.0


class EarlyStopping:
    """Track the best validation score and decide when training should stop.

    Keeps a detached CPU copy of the best-scoring weights so the caller can
    restore them once the loop ends; the final epoch is rarely the best one
    under an imbalanced objective such as PR-AUC.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max") -> None:
        """Configure the stopper."""
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self._best_score: float = -math.inf if mode == "max" else math.inf
        self._best_epoch: int = 0
        self._epoch: int = 0
        self._epochs_no_improve: int = 0
        self._best_state: dict[str, torch.Tensor] | None = None

    @property
    def best_score(self) -> float:
        """Best score observed so far."""
        return self._best_score

    @property
    def best_epoch(self) -> int:
        """1-indexed epoch at which :attr:`best_score` was observed."""
        return self._best_epoch

    @property
    def epochs_no_improve(self) -> int:
        """Consecutive epochs since the last improvement."""
        return self._epochs_no_improve

    @property
    def should_stop(self) -> bool:
        """Whether patience has been exhausted."""
        return self._epochs_no_improve >= self.patience

    def _is_improvement(self, score: float) -> bool:
        """Return whether ``score`` beats the incumbent by ``min_delta``."""
        if self.mode == "max":
            return score > self._best_score + self.min_delta
        return score < self._best_score - self.min_delta

    def update(self, score: float, model: nn.Module | None = None) -> bool:
        """Record one epoch's validation score."""
        if math.isnan(score):
            raise ValueError("early stopping received a NaN score")

        self._epoch += 1
        if self._is_improvement(score):
            self._best_score = float(score)
            self._best_epoch = self._epoch
            self._epochs_no_improve = 0
            if model is not None:
                self._best_state = {
                    key: value.detach().to("cpu").clone()
                    for key, value in model.state_dict().items()
                }
            return True

        self._epochs_no_improve += 1
        return False

    def restore(self, model: nn.Module) -> bool:
        """Load the best-scoring weights back into ``model``."""
        if self._best_state is None:
            logger.warning("No best-weight snapshot to restore; keeping final weights")
            return False
        model.load_state_dict(self._best_state)
        logger.info(
            "Restored best weights from epoch %d (score %.6f)", self._best_epoch, self._best_score
        )
        return True


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_scale: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear-warmup then cosine-annealing per-epoch LR schedule.

    The learning rate ramps linearly from ``base_lr / warmup_epochs`` up to
    ``base_lr`` across the warmup window, then decays along a half cosine to
    ``base_lr * min_scale``. Attention models are unstable in the first few
    hundred steps at the full learning rate, which is what the warmup buys.
    """
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be positive, got {total_epochs}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
    if warmup_epochs > total_epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) cannot exceed total_epochs ({total_epochs})"
        )
    if not 0.0 <= min_scale <= 1.0:
        raise ValueError(f"min_scale must lie in [0, 1], got {min_scale}")

    def lr_lambda(epoch: int) -> float:
        """Return the LR multiplier for a 0-indexed epoch."""
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        decay_span = max(1, total_epochs - warmup_epochs)
        # Clamped so that any epochs beyond the plan sit at the floor rather
        # than climbing back up the cosine.
        progress = min(1.0, (epoch - warmup_epochs) / decay_span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_device(requested: str | None = None) -> str:
    """Resolve a device string, degrading to CPU when CUDA is unavailable.
    """
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


class AmpPolicy:
    """Mixed-precision policy that is a no-op anywhere except CUDA.

    Wraps the autocast context, the gradient scaler, gradient clipping, and
    the optimizer step so a training loop reads the same whether or not AMP
    is active. ``config.transformer.training.amp`` may be ``true`` while the
    project trains on CPU; this class is what makes that harmless.
    """

    def __init__(self, device: str, requested: bool = True) -> None:
        """Decide whether AMP can actually be used on ``device``.
        """
        self.device_type = torch.device(device).type
        self.enabled = bool(requested) and self.device_type == "cuda"
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler(self.device_type) if self.enabled else None
        )
        if requested and not self.enabled:
            logger.info("AMP requested but disabled on %r; training in fp32", self.device_type)

    def autocast(self) -> AbstractContextManager[Any]:
        """Return the forward-pass autocast context (a null context on CPU)."""
        if not self.enabled:
            return nullcontext()
        return torch.amp.autocast(self.device_type)

    def backward(self, loss: torch.Tensor) -> None:
        """Run the backward pass, scaling the loss when AMP is active.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        grad_clip: float | None = None,
    ) -> None:
        """Clip gradients and advance the optimizer.

        Gradients are unscaled before clipping so the clip threshold means the
        same thing with and without AMP.
        """
        if grad_clip is not None and grad_clip > 0:
            if parameters is None:
                raise ValueError("grad_clip requires the parameters to clip")
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip)

        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


def stratified_subsample(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw class-proportional row indices, never emptying the minority class.

    Thin wrapper over
    :func:`src.training.feature_selection.stratified_indices` so that callers
    subsampling for a sweep and callers subsampling for MI ranking share one
    implementation and therefore one sampling behaviour.
    """
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {array.shape}")
    return stratified_indices(array, n, seed)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist model weights plus the metadata needed to rebuild the model.

    Uses the same ``{"state_dict", "meta"}`` envelope as the autoencoder
    checkpoints so downstream consumers (serialization, gating, serving) can
    load any of the project's checkpoints through one code path.
    """
    if not hasattr(model, "state_meta"):
        raise AttributeError(f"{type(model).__name__} does not expose state_meta()")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict(), "meta": model.state_meta()}
    if extra:
        payload.update(extra)
    torch.save(payload, out)
    logger.info("Saved checkpoint to %s", out)
    return out


def load_checkpoint(
    path: str | Path,
    model_factory: Callable[[dict[str, Any]], nn.Module],
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("meta", "state_dict"):
        if key not in payload:
            raise KeyError(f"Checkpoint {ckpt_path} has no {key!r} entry")

    model = model_factory(payload["meta"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from %s onto %s", ckpt_path, device)
    return model, payload
