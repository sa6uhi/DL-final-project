"""Unit tests for the Focal Loss family in :mod:`src.models.losses`."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from src.models.losses import (
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    FocalLoss,
    WeightedBCELoss,
    build_loss,
    compute_pos_weight,
    focal_loss_with_logits,
    weighted_bce_with_logits,
)
from src.utils.seed import seed_everything


@pytest.fixture()
def logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic imbalanced logits/labels pair for loss assertions."""
    seed_everything(42)
    logits = torch.randn(64, dtype=torch.float32)
    targets = torch.zeros(64, dtype=torch.float32)
    targets[:6] = 1.0
    return logits, targets


def test_gamma_zero_alpha_half_is_half_bce(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """At gamma=0 and alpha=0.5 the focal loss collapses to 0.5 * BCE."""
    logits, targets = logits_and_targets
    focal = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=0.5, reduction="none")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(focal, 0.5 * bce, atol=1e-6)


def test_gamma_zero_no_alpha_is_exactly_bce(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """With alpha disabled and gamma=0, focal loss is plain BCE."""
    logits, targets = logits_and_targets
    focal = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=None, reduction="none")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(focal, bce, atol=1e-6)


def test_alpha_weights_positive_class() -> None:
    """alpha scales positives and (1 - alpha) scales negatives."""
    logits = torch.tensor([0.3, 0.3], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=0.25, reduction="none")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    assert torch.allclose(loss[0], 0.25 * bce[0], atol=1e-6)
    assert torch.allclose(loss[1], 0.75 * bce[1], atol=1e-6)


def test_easy_examples_downweighted_more_as_gamma_grows() -> None:
    """Rising gamma suppresses a well-classified example faster than a hard one."""
    easy = torch.tensor([4.0], dtype=torch.float32)
    hard = torch.tensor([0.05], dtype=torch.float32)
    target = torch.ones(1, dtype=torch.float32)

    ratios = []
    for gamma in (0.0, 1.0, 2.0, 5.0):
        easy_loss = focal_loss_with_logits(easy, target, gamma=gamma, alpha=None)
        hard_loss = focal_loss_with_logits(hard, target, gamma=gamma, alpha=None)
        ratios.append(float(easy_loss / hard_loss))

    # The easy/hard loss ratio must shrink monotonically as gamma increases.
    assert all(left > right for left, right in zip(ratios, ratios[1:])), ratios


def test_focal_never_exceeds_bce_for_default_gamma(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The modulator is in [0, 1], so unweighted focal loss cannot exceed BCE."""
    logits, targets = logits_and_targets
    focal = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=None, reduction="none")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert bool((focal <= bce + 1e-6).all())



# Numerical stability

@pytest.mark.parametrize("magnitude", [20.0, 50.0, 80.0, 200.0])
def test_saturated_logits_stay_finite(magnitude: float) -> None:
    """Extreme logits in both directions produce finite losses, never NaN/inf."""
    logits = torch.tensor([magnitude, -magnitude, magnitude, -magnitude], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    loss = focal_loss_with_logits(logits, targets, reduction="none")
    assert bool(torch.isfinite(loss).all()), loss


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 3.0, 5.0])
def test_gradients_finite_for_all_sweep_gammas(gamma: float) -> None:
    """Every gamma in the Exp-4 sweep yields finite gradients."""
    logits = torch.tensor([60.0, -60.0, 0.1, -0.1], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    focal_loss_with_logits(logits, targets, gamma=gamma, alpha=0.25).backward()

    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all()), logits.grad


def test_gradient_is_nonzero_for_hard_examples(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Training signal actually flows: the gradient norm is strictly positive."""
    logits, targets = logits_and_targets
    logits = logits.clone().requires_grad_(True)
    focal_loss_with_logits(logits, targets).backward()

    assert logits.grad is not None
    assert float(logits.grad.norm()) > 0.0


# Reductions, dtypes and shapes

def test_reduction_modes_are_consistent(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """mean and sum agree with the unreduced tensor."""
    logits, targets = logits_and_targets
    none = focal_loss_with_logits(logits, targets, reduction="none")
    mean = focal_loss_with_logits(logits, targets, reduction="mean")
    total = focal_loss_with_logits(logits, targets, reduction="sum")

    assert none.shape == logits.shape
    assert mean.ndim == 0 and total.ndim == 0
    assert torch.allclose(mean, none.mean(), atol=1e-6)
    assert torch.allclose(total, none.sum(), atol=1e-6)


def test_integer_targets_are_accepted() -> None:
    """int64 labels (as produced by the data pipeline) are cast, not rejected."""
    logits = torch.randn(16, dtype=torch.float32)
    int_targets = torch.randint(0, 2, (16,), dtype=torch.int64)
    loss = focal_loss_with_logits(logits, int_targets)
    assert torch.isfinite(loss)


def test_two_dimensional_logits_supported() -> None:
    """The loss is shape-agnostic, so (batch, 1) logits work unchanged."""
    logits = torch.randn(8, 1, dtype=torch.float32)
    targets = torch.zeros(8, 1, dtype=torch.float32)
    loss = focal_loss_with_logits(logits, targets, reduction="none")
    assert loss.shape == (8, 1)


def test_empty_batch_does_not_raise() -> None:
    """A zero-length batch reduces cleanly instead of hitting the range check."""
    logits = torch.zeros(0, dtype=torch.float32)
    targets = torch.zeros(0, dtype=torch.float32)
    assert focal_loss_with_logits(logits, targets, reduction="sum").item() == 0.0


# Defensive validation

def test_negative_gamma_rejected() -> None:
    """gamma must be non-negative."""
    logits = torch.randn(4)
    targets = torch.zeros(4)
    with pytest.raises(ValueError, match="gamma must be non-negative"):
        focal_loss_with_logits(logits, targets, gamma=-1.0)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_alpha_out_of_range_rejected(alpha: float) -> None:
    """alpha must lie in [0, 1]."""
    logits = torch.randn(4)
    targets = torch.zeros(4)
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        focal_loss_with_logits(logits, targets, alpha=alpha)


def test_shape_mismatch_rejected() -> None:
    """Mismatched logits/targets shapes raise rather than silently broadcasting."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        focal_loss_with_logits(torch.randn(4), torch.zeros(5))


def test_unsupported_reduction_rejected() -> None:
    """Only none/mean/sum are accepted."""
    with pytest.raises(ValueError, match="Unsupported reduction"):
        focal_loss_with_logits(torch.randn(4), torch.zeros(4), reduction="median")


def test_targets_outside_unit_interval_rejected() -> None:
    """Labels must be probabilities; -1 style encodings are caught early."""
    with pytest.raises(ValueError, match=r"targets must lie in \[0, 1\]"):
        focal_loss_with_logits(torch.randn(3), torch.tensor([-1.0, 0.0, 1.0]))

# Weighted BCE control

def test_weighted_bce_matches_torch_reference(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The wrapper delegates faithfully to the torch implementation."""
    logits, targets = logits_and_targets
    ours = weighted_bce_with_logits(logits, targets, pos_weight=9.0, reduction="none")
    theirs = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=torch.tensor(9.0), reduction="none"
    )
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_weighted_bce_without_weight_is_plain_bce(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """pos_weight=None leaves the loss unweighted."""
    logits, targets = logits_and_targets
    ours = weighted_bce_with_logits(logits, targets, reduction="none")
    theirs = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_weighted_bce_accepts_tensor_weight(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A tensor pos_weight is honoured as well as a float."""
    logits, targets = logits_and_targets
    loss = weighted_bce_with_logits(logits, targets, pos_weight=torch.tensor(4.0))
    assert torch.isfinite(loss)


def test_weighted_bce_rejects_non_positive_weight() -> None:
    """pos_weight must be strictly positive."""
    with pytest.raises(ValueError, match="pos_weight must be positive"):
        weighted_bce_with_logits(torch.randn(4), torch.zeros(4), pos_weight=0.0)


def test_compute_pos_weight_reports_imbalance_ratio() -> None:
    """10 negatives against 2 positives yields a ratio of 5."""
    targets = torch.cat([torch.zeros(10), torch.ones(2)])
    assert compute_pos_weight(targets) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "targets",
    [torch.zeros(0), torch.zeros(8)],
    ids=["empty", "no-positives"],
)
def test_compute_pos_weight_degrades_gracefully(targets: torch.Tensor) -> None:
    """Degenerate label tensors fall back to 1.0 instead of dividing by zero."""
    assert compute_pos_weight(targets) == 1.0

# Module wrappers and the config-driven factory

def test_focal_module_matches_functional(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """FocalLoss is a thin wrapper over the functional form."""
    logits, targets = logits_and_targets
    module = FocalLoss(gamma=3.0, alpha=0.1, reduction="sum")
    expected = focal_loss_with_logits(logits, targets, gamma=3.0, alpha=0.1, reduction="sum")
    assert torch.allclose(module(logits, targets), expected, atol=1e-6)


def test_weighted_bce_module_matches_functional(
    logits_and_targets: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """WeightedBCELoss is a thin wrapper over the functional form."""
    logits, targets = logits_and_targets
    module = WeightedBCELoss(pos_weight=7.0)
    expected = weighted_bce_with_logits(logits, targets, pos_weight=7.0)
    assert torch.allclose(module(logits, targets), expected, atol=1e-6)


def test_module_reprs_expose_hyperparameters() -> None:
    """extra_repr surfaces hyperparameters for training logs."""
    assert "gamma=2.0" in repr(FocalLoss(gamma=2.0))
    assert "pos_weight=3.0" in repr(WeightedBCELoss(pos_weight=3.0))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gamma": -0.5}, "gamma must be non-negative"),
        ({"alpha": 2.0}, r"alpha must lie in \[0, 1\]"),
        ({"reduction": "avg"}, "Unsupported reduction"),
    ],
)
def test_focal_module_validates_construction(kwargs: dict[str, object], match: str) -> None:
    """Invalid hyperparameters fail fast at construction, not at first batch."""
    with pytest.raises(ValueError, match=match):
        FocalLoss(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pos_weight": -1.0}, "pos_weight must be positive"),
        ({"reduction": "avg"}, "Unsupported reduction"),
    ],
)
def test_weighted_bce_module_validates_construction(kwargs: dict[str, object], match: str) -> None:
    """Invalid hyperparameters fail fast at construction."""
    with pytest.raises(ValueError, match=match):
        WeightedBCELoss(**kwargs)  # type: ignore[arg-type]


def test_build_loss_reads_project_config(config: object) -> None:
    """The real config/config.yaml yields the specified gamma=2.0, alpha=0.25."""
    criterion = build_loss(config)  # type: ignore[arg-type]
    assert isinstance(criterion, FocalLoss)
    assert criterion.gamma == DEFAULT_GAMMA
    assert criterion.alpha == DEFAULT_ALPHA


def test_build_loss_supports_bce_variant() -> None:
    """Selecting the BCE control returns a WeightedBCELoss."""
    cfg = {"transformer": {"loss": {"name": "bce", "pos_weight": 12.0}}}
    criterion = build_loss(cfg)
    assert isinstance(criterion, WeightedBCELoss)
    assert criterion.pos_weight == 12.0


def test_build_loss_allows_null_alpha_and_pos_weight() -> None:
    """YAML nulls disable class weighting rather than crashing on float(None)."""
    focal = build_loss({"transformer": {"loss": {"name": "focal", "alpha": None}}})
    bce = build_loss({"transformer": {"loss": {"name": "bce", "pos_weight": None}}})
    assert isinstance(focal, FocalLoss) and focal.alpha is None
    assert isinstance(bce, WeightedBCELoss) and bce.pos_weight is None


def test_build_loss_rejects_unknown_name() -> None:
    """An unrecognized objective name is a configuration error."""
    with pytest.raises(ValueError, match="Unknown loss name"):
        build_loss({"transformer": {"loss": {"name": "hinge"}}})


def test_build_loss_requires_transformer_section() -> None:
    """A config missing the transformer block raises KeyError."""
    with pytest.raises(KeyError):
        build_loss({})
