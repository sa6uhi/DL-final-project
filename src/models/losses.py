"""Loss functions for supervised fraud classification under extreme class skew."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_REDUCTIONS: frozenset[str] = frozenset({"none", "mean", "sum"})

DEFAULT_GAMMA: float = 2.0
DEFAULT_ALPHA: float = 0.25


def _validate_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str,
) -> None:
    """Validate shared shape/range preconditions for every loss in this module."""
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"Unsupported reduction: {reduction!r}; expected one of {sorted(VALID_REDUCTIONS)}"
        )
    if logits.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}"
        )
    if targets.numel() > 0:
        t_min = float(targets.min())
        t_max = float(targets.max())
        if t_min < 0.0 or t_max > 1.0:
            raise ValueError(f"targets must lie in [0, 1], got range [{t_min}, {t_max}]")


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply the requested reduction to a per-sample loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
    alpha: float | None = DEFAULT_ALPHA,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary Focal Loss computed from raw logits."""
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")

    targets = targets.to(dtype=logits.dtype)
    _validate_inputs(logits, targets, reduction)

    # bce == -log(p_t), computed with the log-sum-exp trick internally.
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t == exp(-bce); recovering it this way avoids a second sigmoid pass and
    # inherits the stability of the fused BCE kernel.
    p_t = torch.exp(-bce)
    eps = torch.finfo(logits.dtype).eps
    one_minus_p_t = (1.0 - p_t).clamp_min(eps)
    modulator = one_minus_p_t.pow(gamma)

    loss = modulator * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    return _reduce(loss, reduction)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Class-weighted binary cross-entropy, the Exp-4 comparison baseline."""
    targets = targets.to(dtype=logits.dtype)
    _validate_inputs(logits, targets, reduction)

    weight_tensor: torch.Tensor | None = None
    if pos_weight is not None:
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        if bool((pos_weight <= 0).any()):
            raise ValueError(f"pos_weight must be positive, got {pos_weight}")
        weight_tensor = pos_weight.to(dtype=logits.dtype, device=logits.device)

    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight_tensor, reduction="none"
    )
    return _reduce(loss, reduction)


def compute_pos_weight(targets: torch.Tensor) -> float:
    """Derive the imbalance ratio ``n_negative / n_positive`` from labels."""
    if targets.numel() == 0:
        logger.warning("compute_pos_weight received an empty tensor; defaulting to 1.0")
        return 1.0
    n_pos = float((targets > 0.5).sum())
    if n_pos == 0.0:
        logger.warning("compute_pos_weight found no positive labels; defaulting to 1.0")
        return 1.0
    n_neg = float(targets.numel()) - n_pos
    return n_neg / n_pos


class FocalLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`focal_loss_with_logits`."""

    def __init__(
        self,
        gamma: float = DEFAULT_GAMMA,
        alpha: float | None = DEFAULT_ALPHA,
        reduction: str = "mean",
    ) -> None:
        """Initialize the criterion."""
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        if reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"Unsupported reduction: {reduction!r}; expected one of {sorted(VALID_REDUCTIONS)}"
            )
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the focal loss for a batch."""
        return focal_loss_with_logits(
            logits, targets, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        """Return the hyperparameters shown in ``repr(module)``."""
        return f"gamma={self.gamma}, alpha={self.alpha}, reduction={self.reduction!r}"


class WeightedBCELoss(nn.Module):
    """``nn.Module`` wrapper around :func:`weighted_bce_with_logits`."""

    def __init__(
        self,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize the criterion."""
        super().__init__()
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError(f"pos_weight must be positive, got {pos_weight}")
        if reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"Unsupported reduction: {reduction!r}; expected one of {sorted(VALID_REDUCTIONS)}"
            )
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted BCE loss for a batch."""
        return weighted_bce_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        """Return the hyperparameters shown in ``repr(module)``."""
        return f"pos_weight={self.pos_weight}, reduction={self.reduction!r}"


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Construct the criterion described by ``config['transformer']['loss']``."""
    loss_cfg = config["transformer"]["loss"]

    name = str(loss_cfg.get("name", "focal")).lower()
    if name == "focal":
        alpha = loss_cfg.get("alpha", DEFAULT_ALPHA)
        criterion: nn.Module = FocalLoss(
            gamma=float(loss_cfg.get("gamma", DEFAULT_GAMMA)),
            alpha=None if alpha is None else float(alpha),
        )
    elif name in {"bce", "weighted_bce"}:
        raw_weight = loss_cfg.get("pos_weight")
        criterion = WeightedBCELoss(pos_weight=None if raw_weight is None else float(raw_weight))
    else:
        raise ValueError(f"Unknown loss name: {name!r}; expected 'focal' or 'bce'")

    logger.info("Built training criterion: %s", criterion)
    return criterion
