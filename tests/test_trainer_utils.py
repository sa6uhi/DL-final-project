"""Unit tests for the shared training primitives in ``src.training.trainer_utils``.

Covers early stopping semantics (including best-weight capture and restore),
the warmup + cosine learning-rate schedule, the CPU-safe mixed-precision
policy, checkpoint round-tripping, and class-proportional subsampling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

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

WARMUP_EPOCHS: int = 5
TOTAL_EPOCHS: int = 30
BASE_LR: float = 3.0e-4


class TinyModel(nn.Module):
    """Minimal module exposing ``state_meta`` for checkpoint tests."""

    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        """Build the layer and record its constructor arguments."""
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._meta: dict[str, Any] = {"in_features": in_features, "out_features": out_features}

    def state_meta(self) -> dict[str, Any]:
        """Return the constructor arguments needed to rebuild this model."""
        return dict(self._meta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer."""
        return self.linear(x)


class MetaFreeModel(nn.Module):
    """Module deliberately lacking ``state_meta`` to exercise the guard."""

    def __init__(self) -> None:
        """Build a trivial parameterised module."""
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer."""
        return self.linear(x)


@pytest.fixture()
def tiny_model() -> TinyModel:
    """Return a deterministically initialised tiny model."""
    torch.manual_seed(0)
    return TinyModel()


# ---------------------------------------------------------------------------
# EpochMeter
# ---------------------------------------------------------------------------


def test_epoch_meter_weights_by_sample_count() -> None:
    """A short final batch must not distort the epoch average."""
    meter = EpochMeter()
    meter.update(1.0, n=100)
    meter.update(3.0, n=1)
    assert meter.count == 101
    assert meter.average == pytest.approx((1.0 * 100 + 3.0) / 101)


def test_epoch_meter_empty_is_zero() -> None:
    """An untouched meter reports zero rather than dividing by zero."""
    assert EpochMeter().average == 0.0
    assert EpochMeter().count == 0


def test_epoch_meter_rejects_negative_n() -> None:
    """Negative sample counts are a programming error, not a silent no-op."""
    with pytest.raises(ValueError, match="non-negative"):
        EpochMeter().update(1.0, n=-1)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


def test_early_stopping_fires_after_patience_epochs() -> None:
    """Stop exactly ``patience`` non-improving epochs after the best score."""
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.update(0.5) is True
    assert stopper.should_stop is False
    assert stopper.update(0.4) is False
    assert stopper.should_stop is False
    assert stopper.update(0.3) is False
    assert stopper.should_stop is True
    assert stopper.best_epoch == 1
    assert stopper.best_score == pytest.approx(0.5)


def test_early_stopping_min_mode_prefers_lower_scores() -> None:
    """``mode='min'`` treats a decreasing metric as improvement."""
    stopper = EarlyStopping(patience=1, mode="min")
    assert stopper.update(1.0) is True
    assert stopper.update(0.5) is True
    assert stopper.update(0.9) is False
    assert stopper.best_score == pytest.approx(0.5)
    assert stopper.best_epoch == 2


def test_early_stopping_min_delta_rejects_marginal_gains() -> None:
    """Improvements smaller than ``min_delta`` do not reset patience."""
    stopper = EarlyStopping(patience=3, min_delta=0.01, mode="max")
    stopper.update(0.50)
    assert stopper.update(0.505) is False
    assert stopper.epochs_no_improve == 1
    assert stopper.update(0.52) is True
    assert stopper.epochs_no_improve == 0


def test_early_stopping_captures_and_restores_best_weights(tiny_model: TinyModel) -> None:
    """The restored weights are those of the best epoch, not the last."""
    stopper = EarlyStopping(patience=5, mode="max")
    stopper.update(0.9, tiny_model)
    best_weight = tiny_model.linear.weight.detach().clone()

    with torch.no_grad():
        tiny_model.linear.weight.add_(1.0)
    stopper.update(0.1, tiny_model)
    assert not torch.allclose(tiny_model.linear.weight, best_weight)

    assert stopper.restore(tiny_model) is True
    assert torch.allclose(tiny_model.linear.weight, best_weight)


def test_early_stopping_restore_without_snapshot_is_a_noop(tiny_model: TinyModel) -> None:
    """Restoring with no captured snapshot reports failure instead of raising."""
    stopper = EarlyStopping(patience=1, mode="max")
    stopper.update(0.5)
    assert stopper.restore(tiny_model) is False


def test_early_stopping_snapshot_is_detached_from_live_model(tiny_model: TinyModel) -> None:
    """The snapshot must be a clone; later in-place edits must not corrupt it."""
    stopper = EarlyStopping(patience=5, mode="max")
    stopper.update(0.9, tiny_model)
    original = tiny_model.linear.weight.detach().clone()

    with torch.no_grad():
        tiny_model.linear.weight.mul_(-3.0)
    stopper.restore(tiny_model)
    assert torch.allclose(tiny_model.linear.weight, original)


def test_early_stopping_rejects_nan_score() -> None:
    """A NaN metric would poison every later comparison, so reject it."""
    with pytest.raises(ValueError, match="NaN"):
        EarlyStopping(patience=1).update(float("nan"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"patience": -1}, "patience"),
        ({"patience": 1, "min_delta": -0.1}, "min_delta"),
        ({"patience": 1, "mode": "maximise"}, "mode"),
    ],
)
def test_early_stopping_validates_constructor_arguments(
    kwargs: dict[str, Any], message: str
) -> None:
    """Invalid configuration is rejected at construction time."""
    with pytest.raises(ValueError, match=message):
        EarlyStopping(**kwargs)


# ---------------------------------------------------------------------------
# Warmup + cosine scheduler
# ---------------------------------------------------------------------------


def _collect_lrs(warmup: int, total: int, min_scale: float = 0.0) -> list[float]:
    """Run a scheduler for ``total`` epochs and return the LR seen each epoch."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup, total, min_scale)
    # No gradients exist, so this step is a no-op on the weights; it only marks
    # the optimizer as stepped so the scheduler does not warn about ordering.
    optimizer.step()
    seen: list[float] = []
    for _ in range(total):
        seen.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return seen


def test_scheduler_warms_up_then_decays() -> None:
    """LR rises monotonically over warmup and decays monotonically after."""
    lrs = _collect_lrs(WARMUP_EPOCHS, TOTAL_EPOCHS)
    warmup_phase, decay_phase = lrs[:WARMUP_EPOCHS], lrs[WARMUP_EPOCHS - 1 :]

    assert all(b > a for a, b in zip(warmup_phase, warmup_phase[1:]))
    assert warmup_phase[0] == pytest.approx(BASE_LR / WARMUP_EPOCHS)
    assert max(lrs) == pytest.approx(BASE_LR)
    assert all(b <= a + 1e-12 for a, b in zip(decay_phase, decay_phase[1:]))


def test_scheduler_peaks_exactly_at_base_lr_once() -> None:
    """The cosine restarts from the peak, so base LR is reached at warmup end."""
    lrs = _collect_lrs(WARMUP_EPOCHS, TOTAL_EPOCHS)
    assert lrs[WARMUP_EPOCHS - 1] == pytest.approx(BASE_LR)


def test_scheduler_never_emits_negative_lr() -> None:
    """A cosine floor bug would silently invert the update direction."""
    assert all(lr >= 0.0 for lr in _collect_lrs(WARMUP_EPOCHS, TOTAL_EPOCHS))


def test_scheduler_respects_min_scale_floor() -> None:
    """With a floor set, the final LR must not fall below it."""
    floor = 0.1
    lrs = _collect_lrs(WARMUP_EPOCHS, TOTAL_EPOCHS, min_scale=floor)
    assert min(lrs) >= BASE_LR * floor - 1e-12


def test_scheduler_without_warmup_starts_at_base_lr() -> None:
    """``warmup_epochs=0`` degrades to plain cosine annealing."""
    lrs = _collect_lrs(0, 10)
    assert lrs[0] == pytest.approx(BASE_LR)
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))


def test_scheduler_tolerates_stepping_past_the_plan() -> None:
    """Extra epochs clamp at the floor instead of climbing back up."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    scheduler = build_warmup_cosine_scheduler(optimizer, 2, 5)
    optimizer.step()
    for _ in range(12):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("warmup", "total", "min_scale", "message"),
    [
        (5, 0, 0.0, "total_epochs"),
        (-1, 10, 0.0, "warmup_epochs"),
        (11, 10, 0.0, "cannot exceed"),
        (5, 10, 1.5, "min_scale"),
    ],
)
def test_scheduler_validates_arguments(
    warmup: int, total: int, min_scale: float, message: str
) -> None:
    """Impossible schedules are rejected rather than silently misbehaving."""
    optimizer = torch.optim.AdamW(nn.Linear(2, 1).parameters(), lr=BASE_LR)
    with pytest.raises(ValueError, match=message):
        build_warmup_cosine_scheduler(optimizer, warmup, total, min_scale)


# ---------------------------------------------------------------------------
# Device resolution and AMP policy
# ---------------------------------------------------------------------------


def test_resolve_device_honours_explicit_cpu() -> None:
    """An explicit CPU request is passed through untouched."""
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_falls_back_when_cuda_absent() -> None:
    """Requesting CUDA on a CPU-only box degrades instead of crashing."""
    resolved = resolve_device("cuda")
    assert resolved == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_auto_selects_available_hardware() -> None:
    """Omitting the request picks CUDA only when it is actually present."""
    assert resolve_device(None) == ("cuda" if torch.cuda.is_available() else "cpu")


def test_amp_is_disabled_on_cpu_even_when_requested() -> None:
    """config.transformer.training.amp=true must be harmless on CPU."""
    policy = AmpPolicy("cpu", requested=True)
    assert policy.enabled is False
    assert policy.scaler is None


def test_amp_autocast_on_cpu_is_a_null_context() -> None:
    """The disabled policy still yields a usable context manager."""
    policy = AmpPolicy("cpu", requested=True)
    with policy.autocast():
        result = torch.ones(2) * 2
    assert result.dtype is torch.float32


def test_amp_policy_runs_a_full_optimizer_step() -> None:
    """backward + step must update parameters with AMP disabled."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    policy = AmpPolicy("cpu", requested=False)
    before = model.weight.detach().clone()

    with policy.autocast():
        loss = model(torch.randn(8, 4)).pow(2).mean()
    policy.backward(loss)
    policy.step(optimizer, model.parameters(), grad_clip=5.0)

    assert not torch.allclose(model.weight, before)


def test_amp_policy_clips_gradients_to_the_requested_norm() -> None:
    """Clipping must bound the global grad norm before the optimizer steps."""
    model = nn.Linear(4, 1)
    for param in model.parameters():
        param.grad = torch.full_like(param, 100.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    AmpPolicy("cpu", requested=False).step(optimizer, model.parameters(), grad_clip=1.0)
    total_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
    assert float(total_norm) <= 1.0 + 1e-5


def test_amp_policy_rejects_clipping_without_parameters() -> None:
    """Asking to clip without saying what to clip is a programming error."""
    optimizer = torch.optim.AdamW(nn.Linear(2, 1).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="grad_clip requires"):
        AmpPolicy("cpu", requested=False).step(optimizer, None, grad_clip=1.0)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_weights_and_extra(tmp_path: Path, tiny_model: TinyModel) -> None:
    """A saved model reloads to bit-identical weights and keeps ``extra``."""
    spec = {"continuous_cols": ["a", "b"], "seq_len": 5}
    out = save_checkpoint(tiny_model, tmp_path / "nested" / "model.pt", extra={"spec": spec})
    assert out.is_file()

    restored, payload = load_checkpoint(out, lambda meta: TinyModel(**meta))
    assert payload["spec"] == spec
    assert payload["meta"] == tiny_model.state_meta()
    for key, value in tiny_model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_checkpoint_restores_model_in_eval_mode(tmp_path: Path, tiny_model: TinyModel) -> None:
    """Inference consumers must never receive a model still in train mode."""
    save_checkpoint(tiny_model, tmp_path / "model.pt")
    restored, _ = load_checkpoint(tmp_path / "model.pt", lambda meta: TinyModel(**meta))
    assert restored.training is False


def test_save_checkpoint_requires_state_meta(tmp_path: Path) -> None:
    """Without ``state_meta`` the checkpoint could never be rebuilt."""
    with pytest.raises(AttributeError, match="state_meta"):
        save_checkpoint(MetaFreeModel(), tmp_path / "model.pt")


def test_load_checkpoint_reports_missing_file(tmp_path: Path) -> None:
    """A missing checkpoint fails loudly with the offending path."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "absent.pt", lambda meta: TinyModel(**meta))


@pytest.mark.parametrize("dropped", ["meta", "state_dict"])
def test_load_checkpoint_rejects_incomplete_envelope(
    tmp_path: Path, tiny_model: TinyModel, dropped: str
) -> None:
    """Both envelope keys are required; a truncated payload must not load."""
    path = tmp_path / "model.pt"
    payload = {"state_dict": tiny_model.state_dict(), "meta": tiny_model.state_meta()}
    del payload[dropped]
    torch.save(payload, path)

    with pytest.raises(KeyError, match=dropped):
        load_checkpoint(path, lambda meta: TinyModel(**meta))


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------


def test_stratified_subsample_preserves_class_balance() -> None:
    """Fraud prevalence in the sample must track the population prevalence."""
    labels = np.zeros(10_000, dtype=np.int64)
    labels[:350] = 1  # ~3.5% fraud, matching the IEEE-CIS skew
    rng = np.random.default_rng(0)
    rng.shuffle(labels)

    idx = stratified_subsample(labels, 1_000, seed=42)
    assert 900 <= idx.size <= 1_100
    assert labels[idx].mean() == pytest.approx(labels.mean(), abs=0.01)


def test_stratified_subsample_keeps_the_minority_class() -> None:
    """Rounding up the quota means fraud is never sampled away entirely."""
    labels = np.zeros(50_000, dtype=np.int64)
    labels[:5] = 1
    idx = stratified_subsample(labels, 100, seed=42)
    assert labels[idx].sum() >= 1


def test_stratified_subsample_returns_sorted_unique_indices() -> None:
    """Row order must be preserved so tensors stay aligned across splits."""
    labels = np.array([0, 1] * 500, dtype=np.int64)
    idx = stratified_subsample(labels, 200, seed=7)
    assert np.array_equal(idx, np.sort(idx))
    assert np.unique(idx).size == idx.size


def test_stratified_subsample_is_deterministic_for_a_seed() -> None:
    """Two calls with one seed must agree, or sweeps are not reproducible."""
    labels = np.array([0] * 900 + [1] * 100, dtype=np.int64)
    assert np.array_equal(
        stratified_subsample(labels, 250, seed=42), stratified_subsample(labels, 250, seed=42)
    )


@pytest.mark.parametrize("requested", [0, -5, 10_000])
def test_stratified_subsample_returns_everything_when_not_reducing(requested: int) -> None:
    """Non-reducing requests short-circuit to the full index range."""
    labels = np.array([0] * 90 + [1] * 10, dtype=np.int64)
    assert stratified_subsample(labels, requested, seed=1).size == labels.size


def test_stratified_subsample_rejects_2d_labels() -> None:
    """A 2D label array is a caller bug that would silently mis-sample."""
    with pytest.raises(ValueError, match="labels must be 1D"):
        stratified_subsample(np.zeros((4, 2), dtype=np.int64), 2, seed=1)


def test_stratified_subsample_handles_single_class_input() -> None:
    """A split with no fraud must still subsample rather than crash."""
    labels = np.zeros(1_000, dtype=np.int64)
    idx = stratified_subsample(labels, 100, seed=3)
    assert idx.size == 100
    assert math.isclose(labels[idx].sum(), 0.0)
