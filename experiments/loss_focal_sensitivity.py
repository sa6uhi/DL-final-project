"""Experiment 4: Focal Loss parameter sensitivity.

The project trains FT-CAT under Focal Loss at the paper defaults
(``gamma=2.0``, ``alpha=0.25``). Those constants are inherited from dense
object detection, not from a 3.5% fraud prevalence, so this experiment asks
whether they actually hold up here: it sweeps ``gamma`` over the focusing range
and ``alpha`` over the class-weighting range, and compares every cell against a
class-weighted BCE control whose ``pos_weight`` is derived from the observed
imbalance ratio.

The two parameters do different jobs, which is why both must move:

* ``gamma`` down-weights already-easy examples, concentrating gradient on the
  borderline transactions that decide the precision-recall trade-off.
* ``alpha`` rebalances the positive and negative terms outright.

Only the loss changes between cells. Architecture, optimizer, schedule, rows,
and seed are held fixed, so a difference in test PR-AUC is attributable to the
objective and nothing else.

Usage:
    python experiments/loss_focal_sensitivity.py                 # full sweep
    python experiments/loss_focal_sensitivity.py --smoke         # fast check
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from src.models.losses import FocalLoss, WeightedBCELoss, compute_pos_weight
from src.training.feature_selection import FeatureSpec
from src.training.train_transformer import (
    DEFAULT_MAX_FPR,
    TensorBundle,
    build_loader,
    evaluate,
    train_transformer,
)
from src.training.trainer_utils import AmpPolicy, resolve_device
from src.utils.config import Config, load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

from experiments.transformer_cat_ablation import prepare_bundles

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_FIGURE = Path("figures/focal_loss_sweep.png")
DEFAULT_RESULTS = Path("results/transformer/focal_sweep.json")

# Name recorded for the weighted-BCE reference arm.
CONTROL_LOSS = "weighted_bce"
FOCAL_LOSS = "focal"

SMOKE_EPOCHS = 1
SMOKE_GAMMAS: tuple[float, ...] = (0.5, 2.0)
SMOKE_ALPHAS: tuple[float, ...] = (0.25,)
SMOKE_SEEDS: tuple[int, ...] = (42,)
SMOKE_TRAIN_ROWS = 4000
SMOKE_VAL_ROWS = 2000


@dataclass(frozen=True)
class SweepCell:
    """Outcome of training one loss configuration.

    Attributes:
        loss: Either ``"focal"`` or ``"weighted_bce"``.
        gamma: Focusing parameter; ``None`` for the BCE control.
        alpha: Class-weighting parameter; ``None`` for the BCE control.
        pos_weight: Positive-class weight; ``None`` for focal cells.
        seed: Seed the run was initialised with.
        best_val_pr_auc: Best validation PR-AUC seen during training.
        best_epoch: 1-indexed epoch that produced ``best_val_pr_auc``.
        epochs_run: Epochs executed before early stopping.
        train_seconds: Wall-clock training time.
        test_metrics: Out-of-time test metrics for the restored best weights.
    """

    loss: str
    gamma: float | None
    alpha: float | None
    pos_weight: float | None
    seed: int
    best_val_pr_auc: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    test_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the cell."""
        return asdict(self)


def build_criteria(
    gammas: Sequence[float],
    alphas: Sequence[float],
    pos_weight: float,
    include_control: bool = True,
) -> list[tuple[str, float | None, float | None, float | None, nn.Module]]:
    """Enumerate the criteria the sweep will train under.

    Args:
        gammas: Focusing parameters to sweep.
        alphas: Class-weighting parameters to sweep.
        pos_weight: Imbalance ratio driving the weighted-BCE control.
        include_control: Whether to append the weighted-BCE control arm.

    Returns:
        ``(loss_name, gamma, alpha, pos_weight, criterion)`` tuples, focal
        cells first in ``gamma``-major order, control last.

    Raises:
        ValueError: If either sweep axis is empty.
    """
    if not gammas:
        raise ValueError("at least one gamma is required")
    if not alphas:
        raise ValueError("at least one alpha is required")

    criteria: list[tuple[str, float | None, float | None, float | None, nn.Module]] = []
    for gamma in gammas:
        for alpha in alphas:
            criteria.append(
                (
                    FOCAL_LOSS,
                    float(gamma),
                    float(alpha),
                    None,
                    FocalLoss(float(gamma), float(alpha)),
                )
            )
    if include_control:
        criteria.append(
            (CONTROL_LOSS, None, None, float(pos_weight), WeightedBCELoss(float(pos_weight)))
        )
    return criteria


def run_sweep(
    config: Config,
    train_bundle: TensorBundle,
    val_bundle: TensorBundle,
    test_bundle: TensorBundle,
    spec: FeatureSpec,
    gammas: Sequence[float],
    alphas: Sequence[float],
    seeds: Sequence[int],
    epochs: int,
    variant: str = "ft_cat",
    device: str | None = None,
    include_control: bool = True,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> list[SweepCell]:
    """Train one model per loss configuration and score it on the test split.

    The control ``pos_weight`` is derived from the *training* labels only --
    reading it off the validation or test split would leak their class balance
    into the objective.

    Args:
        config: Loaded project configuration.
        train_bundle: Materialised training rows.
        val_bundle: Materialised early-stopping rows.
        test_bundle: Materialised out-of-time test rows.
        spec: Feature contract the bundles were built with.
        gammas: Focusing parameters to sweep.
        alphas: Class-weighting parameters to sweep.
        seeds: Seeds to repeat every cell over.
        epochs: Epoch budget per cell.
        variant: Architecture arm held fixed across the sweep.
        device: Target device; resolved automatically when ``None``.
        include_control: Whether to train the weighted-BCE control.
        max_fpr: Operating point for the TPR report.

    Returns:
        One :class:`SweepCell` per ``(criterion, seed)`` pair.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        raise ValueError("at least one seed is required")

    resolved_device = resolve_device(device)
    amp = AmpPolicy(resolved_device, bool(config.transformer.training.amp))
    batch_size = int(config.transformer.training.batch_size)
    test_loader = build_loader(test_bundle, batch_size, shuffle=False, pin_memory=False)

    pos_weight = compute_pos_weight(train_bundle.y)
    logger.info("Derived weighted-BCE pos_weight %.2f from the training labels", pos_weight)

    criteria = build_criteria(gammas, alphas, pos_weight, include_control=include_control)
    total = len(criteria) * len(seeds)

    cells: list[SweepCell] = []
    for index, (name, gamma, alpha, weight, criterion) in enumerate(criteria):
        for offset, seed in enumerate(seeds):
            run_number = index * len(seeds) + offset + 1
            descriptor = (
                f"focal(gamma={gamma}, alpha={alpha})"
                if name == FOCAL_LOSS
                else f"weighted_bce(pos_weight={weight:.1f})"
            )
            logger.info("[%d/%d] Training under %s, seed %d", run_number, total, descriptor, seed)

            started = time.perf_counter()
            model, history = train_transformer(
                config,
                train_bundle,
                val_bundle,
                spec,
                variant=variant,
                device=resolved_device,
                epochs=epochs,
                seed=int(seed),
                criterion=criterion,
            )
            elapsed = time.perf_counter() - started

            metrics = evaluate(model, test_loader, criterion, resolved_device, amp, max_fpr=max_fpr)
            cells.append(
                SweepCell(
                    loss=name,
                    gamma=gamma,
                    alpha=alpha,
                    pos_weight=weight,
                    seed=int(seed),
                    best_val_pr_auc=float(history.best_pr_auc),
                    best_epoch=int(history.best_epoch),
                    epochs_run=int(history.epochs_run),
                    train_seconds=float(elapsed),
                    test_metrics=metrics,
                )
            )
            logger.info(
                "[%d/%d] %s -> test PR-AUC %.5f | val PR-AUC %.5f | %.1fs",
                run_number,
                total,
                descriptor,
                metrics["pr_auc"],
                history.best_pr_auc,
                elapsed,
            )
    return cells


def build_grid(
    cells: Sequence[SweepCell],
    gammas: Sequence[float],
    alphas: Sequence[float],
    metric: str = "pr_auc",
) -> np.ndarray:
    """Assemble the focal cells into a ``(len(alphas), len(gammas))`` matrix.

    Cells repeated over several seeds are averaged. Combinations that were
    never trained come back as NaN so the heatmap renders them as blanks
    instead of implying a zero score.

    Args:
        cells: Sweep cells produced by :func:`run_sweep`.
        gammas: Column axis, in display order.
        alphas: Row axis, in display order.
        metric: Key of ``test_metrics`` to tabulate.

    Returns:
        The metric grid, rows indexed by ``alpha`` and columns by ``gamma``.
    """
    grid = np.full((len(alphas), len(gammas)), np.nan, dtype=float)
    for row, alpha in enumerate(alphas):
        for column, gamma in enumerate(gammas):
            matches = [
                cell.test_metrics[metric]
                for cell in cells
                if cell.loss == FOCAL_LOSS
                and cell.gamma is not None
                and cell.alpha is not None
                and np.isclose(cell.gamma, gamma)
                and np.isclose(cell.alpha, alpha)
            ]
            if matches:
                grid[row, column] = float(np.mean(matches))
    return grid


def control_score(cells: Sequence[SweepCell], metric: str = "pr_auc") -> float | None:
    """Return the mean control score, or ``None`` when no control was trained.

    Args:
        cells: Sweep cells produced by :func:`run_sweep`.
        metric: Key of ``test_metrics`` to average.

    Returns:
        The weighted-BCE arm mean, or ``None``.
    """
    values = [cell.test_metrics[metric] for cell in cells if cell.loss == CONTROL_LOSS]
    return float(np.mean(values)) if values else None


def best_cell(cells: Sequence[SweepCell], metric: str = "pr_auc") -> SweepCell:
    """Return the focal cell with the highest score.

    Args:
        cells: Sweep cells produced by :func:`run_sweep`.
        metric: Key of ``test_metrics`` to maximise.

    Returns:
        The best-scoring focal cell.

    Raises:
        ValueError: If no focal cells are present.
    """
    focal = [cell for cell in cells if cell.loss == FOCAL_LOSS]
    if not focal:
        raise ValueError("no focal cells in the sweep")
    return max(focal, key=lambda cell: cell.test_metrics[metric])


def plot_focal_sweep(
    cells: Sequence[SweepCell],
    gammas: Sequence[float],
    alphas: Sequence[float],
    output_path: str | Path,
    metric: str = "pr_auc",
    show_plot: bool = False,
) -> Path:
    """Render the sweep as an annotated heatmap plus per-alpha curves.

    Args:
        cells: Sweep cells produced by :func:`run_sweep`.
        gammas: Gamma axis, in display order.
        alphas: Alpha axis, in display order.
        output_path: Destination for the PNG.
        metric: Key of ``test_metrics`` to plot.
        show_plot: Whether to display the figure interactively.

    Returns:
        The path the figure was written to.

    Raises:
        ValueError: If the sweep contains no focal cells.
    """
    grid = build_grid(cells, gammas, alphas, metric=metric)
    if np.isnan(grid).all():
        raise ValueError("no focal cells to plot")

    fig, (heat_ax, line_ax) = plt.subplots(1, 2, figsize=(13, 4.8))

    image = heat_ax.imshow(grid, aspect="auto", cmap="viridis")
    heat_ax.set_xticks(range(len(gammas)))
    heat_ax.set_xticklabels([f"{gamma:g}" for gamma in gammas])
    heat_ax.set_yticks(range(len(alphas)))
    heat_ax.set_yticklabels([f"{alpha:g}" for alpha in alphas])
    heat_ax.set_xlabel("Focusing parameter gamma")
    heat_ax.set_ylabel("Class weight alpha")
    heat_ax.set_title(f"Test {metric} across the focal grid")
    finite = grid[np.isfinite(grid)]
    midpoint = (float(finite.min()) + float(finite.max())) / 2.0
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            if np.isnan(value):
                continue
            heat_ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < midpoint else "black",
            )
    fig.colorbar(image, ax=heat_ax, shrink=0.85, label=f"Test {metric}")

    for row, alpha in enumerate(alphas):
        line_ax.plot(
            list(gammas),
            grid[row],
            marker="o",
            linewidth=1.8,
            label=f"alpha = {alpha:g}",
        )

    control = control_score(cells, metric=metric)
    if control is not None:
        line_ax.axhline(
            control,
            linestyle="--",
            linewidth=1.5,
            color="#B4553F",
            label=f"Weighted BCE control ({control:.3f})",
        )

    line_ax.set_xlabel("Focusing parameter gamma")
    line_ax.set_ylabel(f"Test {metric}")
    line_ax.set_title("Focal Loss sensitivity vs weighted BCE")
    line_ax.grid(alpha=0.25)
    line_ax.legend(fontsize=9)

    fig.suptitle("Exp 4: Focal Loss parameter sensitivity (FT-CAT, out-of-time test split)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    logger.info("Saved focal sweep figure to %s", output)
    if show_plot:
        plt.show()
    plt.close(fig)
    return output


def save_results(
    cells: Sequence[SweepCell],
    gammas: Sequence[float],
    alphas: Sequence[float],
    output_path: str | Path,
    context: dict[str, Any] | None = None,
    metric: str = "pr_auc",
) -> Path:
    """Write the sweep cells, the grid, and the argmax to JSON.

    Args:
        cells: Sweep cells produced by :func:`run_sweep`.
        gammas: Gamma axis, in display order.
        alphas: Alpha axis, in display order.
        output_path: Destination JSON path.
        context: Optional run metadata.
        metric: Key of ``test_metrics`` used for the grid and argmax.

    Returns:
        The path the JSON was written to.
    """
    winner = best_cell(cells, metric=metric)
    payload: dict[str, Any] = {
        "experiment": "loss_focal_sensitivity",
        "context": dict(context or {}),
        "metric": metric,
        "gammas": [float(gamma) for gamma in gammas],
        "alphas": [float(alpha) for alpha in alphas],
        "grid": build_grid(cells, gammas, alphas, metric=metric).tolist(),
        "control": control_score(cells, metric=metric),
        "best": {"gamma": winner.gamma, "alpha": winner.alpha, metric: winner.test_metrics[metric]},
        "cells": [cell.to_dict() for cell in cells],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved focal sweep results to %s", output)
    return output


def _parse_float_list(raw: str | None, fallback: Sequence[float]) -> list[float]:
    """Parse a comma-separated float list, falling back to a default.

    Args:
        raw: Comma-separated string such as ``"0.5,2.0"``; ``None`` uses the
            fallback.
        fallback: Values used when ``raw`` is ``None``.

    Returns:
        The parsed floats.

    Raises:
        ValueError: If any element is not a float.
    """
    if raw is None:
        return [float(value) for value in fallback]
    try:
        return [float(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"could not parse float list from {raw!r}: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the Focal Loss sensitivity sweep.

    Args:
        argv: Command line arguments; uses ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(description="Exp 4: Focal Loss parameter sensitivity")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--gammas", type=str, default=None, help="comma-separated gamma values")
    parser.add_argument("--alphas", type=str, default=None, help="comma-separated alpha values")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variant", type=str, default="ft_cat")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu; auto when omitted")
    parser.add_argument("--figure", type=str, default=str(DEFAULT_FIGURE))
    parser.add_argument("--out", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="skip the weighted-BCE reference arm",
    )
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="recompute the MI feature ranking instead of reusing the cache",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny sweep (2 gammas, 1 alpha, 1 epoch) that exercises every path",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(
        level=str(config.logging.level), log_file=str(config.get_path("logging.log_file"))
    )
    seed_everything(int(config.seed))

    sweep_cfg = config.transformer.focal_sweep
    subsample_cfg = config.transformer.subsample

    gammas = _parse_float_list(args.gammas, sweep_cfg.gammas)
    alphas = _parse_float_list(args.alphas, sweep_cfg.alphas)
    seeds = [
        int(value)
        for value in (
            [part for part in args.seeds.split(",") if part.strip()]
            if args.seeds
            else sweep_cfg.seeds
        )
    ]
    epochs = int(args.epochs if args.epochs is not None else sweep_cfg.epochs)
    train_rows = int(subsample_cfg.train_rows)
    val_rows = int(subsample_cfg.val_rows)

    if args.smoke:
        gammas, alphas = list(SMOKE_GAMMAS), list(SMOKE_ALPHAS)
        seeds, epochs = list(SMOKE_SEEDS), SMOKE_EPOCHS
        train_rows, val_rows = SMOKE_TRAIN_ROWS, SMOKE_VAL_ROWS
        logger.warning(
            "Smoke mode: %d gamma(s) x %d alpha(s), %d epoch(s), %d train rows",
            len(gammas),
            len(alphas),
            epochs,
            train_rows,
        )

    train_bundle, val_bundle, test_bundle, spec, _ = prepare_bundles(
        config,
        args.train_data or config.get_path("data.train_data_path"),
        args.test_data or config.get_path("data.test_data_path"),
        train_rows=train_rows,
        val_rows=val_rows,
        seed=int(config.seed),
        refresh_features=args.refresh_features,
    )

    cells = run_sweep(
        config,
        train_bundle,
        val_bundle,
        test_bundle,
        spec,
        gammas=gammas,
        alphas=alphas,
        seeds=seeds,
        epochs=epochs,
        variant=args.variant,
        device=args.device,
        include_control=not args.no_control,
    )

    plot_focal_sweep(cells, gammas, alphas, args.figure)
    save_results(
        cells,
        gammas,
        alphas,
        args.out,
        context={
            "variant": args.variant,
            "gammas": gammas,
            "alphas": alphas,
            "seeds": seeds,
            "epochs": epochs,
            "device": resolve_device(args.device),
            "n_train_rows": len(train_bundle),
            "n_val_rows": len(val_bundle),
            "n_test_rows": len(test_bundle),
            "smoke": bool(args.smoke),
        },
    )

    winner = best_cell(cells)
    control = control_score(cells)
    logger.info(
        "Exp 4 complete. Best focal cell: gamma=%s alpha=%s (test PR-AUC %.5f); control %s",
        winner.gamma,
        winner.alpha,
        winner.test_metrics["pr_auc"],
        f"{control:.5f}" if control is not None else "not trained",
    )


if __name__ == "__main__":
    main()
