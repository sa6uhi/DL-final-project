"""Unit tests for Exp 4, the Focal Loss sensitivity sweep (Member B).

Exercises criterion enumeration, the sweep driver and its weighted-BCE control,
grid assembly over repeated seeds, the argmax helper, the figure writer, and
the CLI end to end. Everything runs on tiny synthetic parquet splits pinned to
the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.loss_focal_sensitivity import (
    CONTROL_LOSS,
    FOCAL_LOSS,
    SweepCell,
    _parse_float_list,
    best_cell,
    build_criteria,
    build_grid,
    control_score,
    main,
    plot_focal_sweep,
    run_sweep,
    save_results,
)
from src.models.losses import FocalLoss, WeightedBCELoss
from src.training.feature_selection import FeatureSpec
from src.training.train_transformer import materialize_tensors
from src.utils.config import Config

from tests.test_transformer_cat_ablation import (
    _write_cli_fixtures,
    make_config,
    make_frame,
    make_spec,
)

GAMMAS = [0.5, 2.0]
ALPHAS = [0.25, 0.5]


@pytest.fixture()
def spec() -> FeatureSpec:
    """Feature contract for the synthetic frames."""
    return make_spec()


@pytest.fixture()
def config() -> Config:
    """A fast transformer configuration."""
    return make_config()


@pytest.fixture()
def bundles(spec: FeatureSpec) -> tuple[Any, Any, Any]:
    """Materialised train, validation, and test bundles."""
    return (
        materialize_tensors(make_frame(seed=0), spec),
        materialize_tensors(make_frame(n_rows=96, seed=1), spec),
        materialize_tensors(make_frame(n_rows=96, seed=2), spec),
    )


def make_cell(
    loss: str = FOCAL_LOSS,
    gamma: float | None = 2.0,
    alpha: float | None = 0.25,
    pr_auc: float = 0.5,
    seed: int = 42,
) -> SweepCell:
    """Build a hand-specified sweep cell for the pure-function tests."""
    return SweepCell(
        loss=loss,
        gamma=gamma,
        alpha=alpha,
        pos_weight=None if loss == FOCAL_LOSS else 9.0,
        seed=seed,
        best_val_pr_auc=pr_auc,
        best_epoch=1,
        epochs_run=1,
        train_seconds=1.0,
        test_metrics={"pr_auc": pr_auc, "roc_auc": 0.7, "tpr_at_fpr": 0.3, "loss": 0.1},
    )


def test_build_criteria_enumerates_the_grid_plus_control() -> None:
    """Every gamma-alpha pair is enumerated, with the control appended last."""
    criteria = build_criteria(GAMMAS, ALPHAS, pos_weight=9.0)
    assert len(criteria) == len(GAMMAS) * len(ALPHAS) + 1
    assert [name for name, *_ in criteria[:-1]] == [FOCAL_LOSS] * 4
    assert criteria[-1][0] == CONTROL_LOSS
    assert isinstance(criteria[-1][4], WeightedBCELoss)
    assert all(isinstance(entry[4], FocalLoss) for entry in criteria[:-1])


def test_build_criteria_carries_the_parameters_into_the_module() -> None:
    """Each focal criterion is configured with its own gamma and alpha."""
    for name, gamma, alpha, _, criterion in build_criteria(GAMMAS, ALPHAS, 9.0, False):
        assert name == FOCAL_LOSS
        assert criterion.gamma == gamma
        assert criterion.alpha == alpha


def test_build_criteria_can_skip_the_control() -> None:
    """The control arm is optional."""
    criteria = build_criteria(GAMMAS, ALPHAS, pos_weight=9.0, include_control=False)
    assert len(criteria) == len(GAMMAS) * len(ALPHAS)
    assert all(name == FOCAL_LOSS for name, *_ in criteria)


@pytest.mark.parametrize(
    ("gammas", "alphas", "message"),
    [([], ALPHAS, "at least one gamma"), (GAMMAS, [], "at least one alpha")],
)
def test_build_criteria_rejects_empty_axes(
    gammas: list[float], alphas: list[float], message: str
) -> None:
    """Both sweep axes must be non-empty."""
    with pytest.raises(ValueError, match=message):
        build_criteria(gammas, alphas, pos_weight=9.0)


def test_run_sweep_trains_every_cell_and_the_control(
    config: Config, spec: FeatureSpec, bundles: tuple[Any, Any, Any]
) -> None:
    """The sweep produces one cell per criterion, control included."""
    train_bundle, val_bundle, test_bundle = bundles
    cells = run_sweep(
        config,
        train_bundle,
        val_bundle,
        test_bundle,
        spec,
        gammas=GAMMAS,
        alphas=[0.25],
        seeds=[42],
        epochs=1,
        device="cpu",
    )
    assert len(cells) == len(GAMMAS) + 1
    focal = [cell for cell in cells if cell.loss == FOCAL_LOSS]
    control = [cell for cell in cells if cell.loss == CONTROL_LOSS]
    assert [cell.gamma for cell in focal] == GAMMAS
    assert all(cell.alpha == 0.25 for cell in focal)
    assert all(cell.pos_weight is None for cell in focal)
    assert len(control) == 1
    # pos_weight is derived from the training labels, which are imbalanced.
    assert control[0].pos_weight is not None and control[0].pos_weight > 1.0


def test_run_sweep_can_skip_the_control(
    config: Config, spec: FeatureSpec, bundles: tuple[Any, Any, Any]
) -> None:
    """``include_control=False`` trains only focal cells."""
    train_bundle, val_bundle, test_bundle = bundles
    cells = run_sweep(
        config,
        train_bundle,
        val_bundle,
        test_bundle,
        spec,
        gammas=[2.0],
        alphas=[0.25],
        seeds=[42],
        epochs=1,
        device="cpu",
        include_control=False,
    )
    assert [cell.loss for cell in cells] == [FOCAL_LOSS]


def test_run_sweep_rejects_empty_seeds(
    config: Config, spec: FeatureSpec, bundles: tuple[Any, Any, Any]
) -> None:
    """At least one seed is required."""
    train_bundle, val_bundle, test_bundle = bundles
    with pytest.raises(ValueError, match="at least one seed"):
        run_sweep(
            config,
            train_bundle,
            val_bundle,
            test_bundle,
            spec,
            gammas=[2.0],
            alphas=[0.25],
            seeds=[],
            epochs=1,
            device="cpu",
        )


def test_build_grid_places_cells_at_their_coordinates() -> None:
    """Grid rows are alphas and columns are gammas."""
    cells = [
        make_cell(gamma=0.5, alpha=0.25, pr_auc=0.1),
        make_cell(gamma=2.0, alpha=0.5, pr_auc=0.9),
    ]
    grid = build_grid(cells, GAMMAS, ALPHAS)
    assert grid.shape == (len(ALPHAS), len(GAMMAS))
    assert grid[0, 0] == pytest.approx(0.1)
    assert grid[1, 1] == pytest.approx(0.9)


def test_build_grid_averages_repeated_seeds() -> None:
    """Cells trained under several seeds are averaged."""
    cells = [
        make_cell(gamma=2.0, alpha=0.25, pr_auc=0.4, seed=42),
        make_cell(gamma=2.0, alpha=0.25, pr_auc=0.6, seed=43),
    ]
    assert build_grid(cells, [2.0], [0.25])[0, 0] == pytest.approx(0.5)


def test_build_grid_leaves_untrained_cells_nan() -> None:
    """Combinations never trained are NaN, not zero."""
    grid = build_grid([make_cell(gamma=0.5, alpha=0.25)], GAMMAS, ALPHAS)
    assert np.isnan(grid[1, 1])


def test_build_grid_ignores_the_control() -> None:
    """The BCE control has no grid coordinates and must not leak into it."""
    grid = build_grid([make_cell(loss=CONTROL_LOSS, gamma=None, alpha=None)], GAMMAS, ALPHAS)
    assert np.isnan(grid).all()


def test_control_score_averages_the_control_arm() -> None:
    """The control score is the mean over control cells."""
    cells = [
        make_cell(loss=CONTROL_LOSS, gamma=None, alpha=None, pr_auc=0.2),
        make_cell(loss=CONTROL_LOSS, gamma=None, alpha=None, pr_auc=0.4),
    ]
    assert control_score(cells) == pytest.approx(0.3)


def test_control_score_is_none_without_a_control() -> None:
    """A sweep without the control reports ``None`` rather than zero."""
    assert control_score([make_cell()]) is None


def test_best_cell_maximises_the_metric() -> None:
    """The argmax helper returns the highest-scoring focal cell."""
    cells = [make_cell(gamma=0.5, pr_auc=0.3), make_cell(gamma=2.0, pr_auc=0.8)]
    assert best_cell(cells).gamma == 2.0


def test_best_cell_ignores_the_control() -> None:
    """A stronger control does not become the best focal cell."""
    cells = [make_cell(gamma=2.0, pr_auc=0.3), make_cell(loss=CONTROL_LOSS, pr_auc=0.99)]
    assert best_cell(cells).loss == FOCAL_LOSS


def test_best_cell_requires_focal_cells() -> None:
    """A control-only sweep has no focal argmax."""
    with pytest.raises(ValueError, match="no focal cells"):
        best_cell([make_cell(loss=CONTROL_LOSS, gamma=None, alpha=None)])


def test_plot_focal_sweep_writes_a_png(tmp_path: Path) -> None:
    """The figure is written, including the control reference line."""
    cells = [
        make_cell(gamma=gamma, alpha=alpha, pr_auc=0.1 * index)
        for index, (gamma, alpha) in enumerate([(g, a) for g in GAMMAS for a in ALPHAS], start=1)
    ]
    cells.append(make_cell(loss=CONTROL_LOSS, gamma=None, alpha=None, pr_auc=0.25))
    output = tmp_path / "nested" / "focal_loss_sweep.png"
    assert plot_focal_sweep(cells, GAMMAS, ALPHAS, output) == output
    assert output.stat().st_size > 0


def test_plot_focal_sweep_without_control(tmp_path: Path) -> None:
    """The figure still renders when no control was trained."""
    output = tmp_path / "no_control.png"
    plot_focal_sweep([make_cell(gamma=2.0, alpha=0.25)], [2.0], [0.25], output)
    assert output.is_file()


def test_plot_focal_sweep_rejects_an_empty_grid(tmp_path: Path) -> None:
    """A grid with no trained cells has nothing to draw."""
    control_only = [make_cell(loss=CONTROL_LOSS, gamma=None, alpha=None)]
    with pytest.raises(ValueError, match="no focal cells to plot"):
        plot_focal_sweep(control_only, GAMMAS, ALPHAS, tmp_path / "x.png")


def test_save_results_records_grid_control_and_argmax(tmp_path: Path) -> None:
    """The JSON carries the grid, the control, the argmax, and every cell."""
    cells = [
        make_cell(gamma=0.5, alpha=0.25, pr_auc=0.2),
        make_cell(gamma=2.0, alpha=0.25, pr_auc=0.7),
        make_cell(loss=CONTROL_LOSS, gamma=None, alpha=None, pr_auc=0.3),
    ]
    output = save_results(cells, GAMMAS, [0.25], tmp_path / "focal.json", context={"epochs": 1})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment"] == "loss_focal_sensitivity"
    assert payload["best"]["gamma"] == 2.0
    assert payload["best"]["pr_auc"] == pytest.approx(0.7)
    assert payload["control"] == pytest.approx(0.3)
    assert len(payload["cells"]) == 3
    assert payload["gammas"] == GAMMAS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, [1.0, 2.0]), ("0.5,2", [0.5, 2.0]), ("0.5, 2 ,", [0.5, 2.0])],
)
def test_parse_float_list(raw: str | None, expected: list[float]) -> None:
    """Sweep axes parse from CSV and fall back to the configured default."""
    assert _parse_float_list(raw, [1.0, 2.0]) == expected


def test_parse_float_list_rejects_non_floats() -> None:
    """A malformed list is a clear error, not a silent empty sweep."""
    with pytest.raises(ValueError, match="could not parse float list"):
        _parse_float_list("0.5,abc", [1.0])


def _write_sweep_config(tmp_path: Path) -> Path:
    """Write CLI fixtures extended with a focal sweep block.

    Args:
        tmp_path: Directory to populate.

    Returns:
        Path of the written configuration file.
    """
    import yaml

    config_path = _write_cli_fixtures(tmp_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["transformer"]["focal_sweep"] = {
        "gammas": [2.0],
        "alphas": [0.25],
        "epochs": 1,
        "seeds": [42],
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return config_path


def test_main_writes_figure_and_results(tmp_path: Path) -> None:
    """The CLI runs end to end and writes both artefacts."""
    config_path = _write_sweep_config(tmp_path)
    figure = tmp_path / "sweep.png"
    results = tmp_path / "sweep.json"
    main(
        [
            "--config",
            str(config_path),
            "--figure",
            str(figure),
            "--out",
            str(results),
            "--no-control",
        ]
    )
    assert figure.stat().st_size > 0
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["control"] is None
    assert payload["context"]["epochs"] == 1
    assert len(payload["cells"]) == 1


def test_main_smoke_mode_overrides_the_budget(tmp_path: Path) -> None:
    """``--smoke`` forces a two-gamma, single-epoch sweep."""
    config_path = _write_sweep_config(tmp_path)
    results = tmp_path / "smoke.json"
    main(
        [
            "--config",
            str(config_path),
            "--figure",
            str(tmp_path / "smoke.png"),
            "--out",
            str(results),
            "--smoke",
        ]
    )
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["context"]["smoke"] is True
    assert payload["context"]["epochs"] == 1
    assert payload["control"] is not None


def test_main_honours_explicit_axes(tmp_path: Path) -> None:
    """Explicit CLI axes override the configured sweep."""
    config_path = _write_sweep_config(tmp_path)
    results = tmp_path / "explicit.json"
    main(
        [
            "--config",
            str(config_path),
            "--figure",
            str(tmp_path / "explicit.png"),
            "--out",
            str(results),
            "--gammas",
            "0.5,2.0",
            "--alphas",
            "0.25",
            "--seeds",
            "42",
            "--epochs",
            "1",
            "--no-control",
        ]
    )
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["gammas"] == [0.5, 2.0]
    assert len(payload["cells"]) == 2
