"""Experiment 3: FT-CAT architecture ablation.

Answers the question the temporal cross-attention layer exists to justify: does
contextualising a transaction against the cardholder's ``K=5`` most recent
transactions actually buy detection performance over an otherwise identical
model that sees the current row alone?

Three arms, all trained by the same engine on the same rows with the same loss:

* ``ft_cat``       -- feature tokenizer + self-attention + temporal cross-attention.
* ``ft_self_only`` -- the same model with the cross-attention layer removed.
* ``mlp``          -- a flat MLP consuming the identical inputs, history included,
  but with no attention anywhere. Isolates attention structure from features.

Every arm is trained across several seeds because a single-seed gap on a 3.5%
positive class is well within run-to-run noise; the figure reports mean and
standard deviation. Member A's classical baselines are scored on the identical
test rows and drawn as a reference line, so the deep arms are not compared
against a number computed under different conditions.

Selection is on the model-development validation slice carved out of the
training block; the reported numbers come from the untouched out-of-time test
split.

Usage:
    python experiments/transformer_cat_ablation.py                 # full run
    python experiments/transformer_cat_ablation.py --smoke         # fast check
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import average_precision
from src.models.ft_transformer import VARIANTS, count_parameters
from src.models.losses import build_loss
from src.training.feature_selection import FeatureSpec, resolve_feature_set
from src.training.model_development import create_model_development_split
from src.training.train_transformer import (
    DEFAULT_MAX_FPR,
    TensorBundle,
    build_loader,
    evaluate,
    materialize_tensors,
    subsample_bundle,
    train_transformer,
)
from src.training.trainer_utils import AmpPolicy, resolve_device
from src.utils.config import Config, load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_FIGURE = Path("figures/transformer_cat_ablation.png")
DEFAULT_RESULTS = Path("results/transformer/cat_ablation.json")

# Metrics carried into the figure, paired with their display labels.
PLOT_METRICS: tuple[tuple[str, str], ...] = (
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("tpr_at_fpr", "TPR @ 1% FPR"),
)

VARIANT_LABELS: dict[str, str] = {
    "ft_cat": "FT-CAT (cross-attn)",
    "ft_self_only": "FT (self-attn only)",
    "mlp": "MLP (no attention)",
}

VARIANT_COLORS: dict[str, str] = {
    "ft_cat": "#3F7D5A",
    "ft_self_only": "#7FA98F",
    "mlp": "#A8B5AE",
}

# A smoke run must exercise every code path -- training, evaluation, baseline
# scoring, aggregation, plotting -- while staying inside a couple of minutes.
SMOKE_EPOCHS = 1
SMOKE_SEEDS: tuple[int, ...] = (42,)
SMOKE_TRAIN_ROWS = 4000
SMOKE_VAL_ROWS = 2000


@dataclass(frozen=True)
class RunResult:
    """Outcome of training one ``(variant, seed)`` pair.

    Attributes:
        variant: Architecture arm that was trained.
        seed: Seed the run was initialised with.
        n_parameters: Trainable parameter count of the arm.
        best_val_pr_auc: Best validation PR-AUC seen during training.
        best_epoch: 1-indexed epoch that produced ``best_val_pr_auc``.
        epochs_run: Number of epochs actually executed before early stopping.
        train_seconds: Wall-clock training time.
        test_metrics: Out-of-time test metrics for the restored best weights.
    """

    variant: str
    seed: int
    n_parameters: int
    best_val_pr_auc: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    test_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the result."""
        return asdict(self)


def prepare_bundles(
    config: Config,
    train_path: str | Path,
    test_path: str | Path,
    train_rows: int,
    val_rows: int,
    seed: int,
    refresh_features: bool = False,
) -> tuple[TensorBundle, TensorBundle, TensorBundle, FeatureSpec, pd.DataFrame]:
    """Materialise the train, validation, and test tensors for the sweep.

    Mirrors :func:`src.training.train_transformer.main` exactly so the sweep
    trains on the same rows, in the same order, under the same feature
    contract as the production checkpoint. The training block is subdivided
    chronologically: the earlier part trains, the later part early-stops. The
    original validation split is deliberately untouched -- it belongs to
    Member D for gate development and conformal calibration.

    Args:
        config: Loaded project configuration.
        train_path: Processed training parquet.
        test_path: Processed out-of-time test parquet.
        train_rows: Class-proportional training subsample size.
        val_rows: Class-proportional validation subsample size.
        seed: Seed controlling the subsample draw.
        refresh_features: Recompute the MI feature ranking instead of reusing
            the cached contract.

    Returns:
        Tuple of ``(train_bundle, val_bundle, test_bundle, spec, test_df)``.
        The test frame is returned so the classical baselines can be scored on
        exactly the rows the deep arms were evaluated on.

    Raises:
        FileNotFoundError: If either split is missing.
    """
    for label, path in (("train", train_path), ("test", test_path)):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Processed {label} split not found: {path}. "
                "Run `python -m src.data.prepare_data` first."
            )

    train_df = pd.read_parquet(train_path)
    model_train_df, model_val_df = create_model_development_split(train_df, config)
    spec = resolve_feature_set(config, train_df=model_train_df, force_refresh=refresh_features)

    train_bundle = subsample_bundle(materialize_tensors(model_train_df, spec), train_rows, seed)
    val_bundle = subsample_bundle(materialize_tensors(model_val_df, spec), val_rows, seed)
    del train_df, model_train_df, model_val_df

    test_df = pd.read_parquet(test_path)
    test_bundle = materialize_tensors(test_df, spec)

    logger.info(
        "Prepared bundles: %d train / %d val / %d test rows, %d tokens",
        len(train_bundle),
        len(val_bundle),
        len(test_bundle),
        spec.n_tokens,
    )
    return train_bundle, val_bundle, test_bundle, spec, test_df


def run_ablation(
    config: Config,
    train_bundle: TensorBundle,
    val_bundle: TensorBundle,
    test_bundle: TensorBundle,
    spec: FeatureSpec,
    variants: Sequence[str],
    seeds: Sequence[int],
    epochs: int,
    device: str | None = None,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> list[RunResult]:
    """Train every ``(variant, seed)`` pair and score it on the test split.

    Args:
        config: Loaded project configuration.
        train_bundle: Materialised training rows.
        val_bundle: Materialised early-stopping rows.
        test_bundle: Materialised out-of-time test rows.
        spec: Feature contract the bundles were built with.
        variants: Architecture arms to train.
        seeds: Seeds to repeat each arm over.
        epochs: Epoch budget per run.
        device: Target device; resolved automatically when ``None``.
        max_fpr: Operating point for the TPR report.

    Returns:
        One :class:`RunResult` per ``(variant, seed)`` pair.

    Raises:
        ValueError: If ``variants`` is empty, ``seeds`` is empty, or a variant
            is not one of :data:`~src.models.ft_transformer.VARIANTS`.
    """
    if not variants:
        raise ValueError("at least one variant is required")
    if not seeds:
        raise ValueError("at least one seed is required")
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; expected a subset of {list(VARIANTS)}")

    resolved_device = resolve_device(device)
    amp = AmpPolicy(resolved_device, bool(config.transformer.training.amp))
    criterion = build_loss(config).to(resolved_device)
    batch_size = int(config.transformer.training.batch_size)
    test_loader = build_loader(test_bundle, batch_size, shuffle=False, pin_memory=False)

    results: list[RunResult] = []
    total = len(variants) * len(seeds)
    for index, variant in enumerate(variants):
        for offset, seed in enumerate(seeds):
            run_number = index * len(seeds) + offset + 1
            logger.info("[%d/%d] Training %r with seed %d", run_number, total, variant, seed)

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
            )
            elapsed = time.perf_counter() - started

            metrics = evaluate(model, test_loader, criterion, resolved_device, amp, max_fpr=max_fpr)
            results.append(
                RunResult(
                    variant=variant,
                    seed=int(seed),
                    n_parameters=count_parameters(model),
                    best_val_pr_auc=float(history.best_pr_auc),
                    best_epoch=int(history.best_epoch),
                    epochs_run=int(history.epochs_run),
                    train_seconds=float(elapsed),
                    test_metrics=metrics,
                )
            )
            logger.info(
                "[%d/%d] %r seed %d -> test PR-AUC %.5f | ROC-AUC %.5f | %.1fs",
                run_number,
                total,
                variant,
                seed,
                metrics["pr_auc"],
                metrics["roc_auc"],
                elapsed,
            )
    return results


def aggregate(results: Sequence[RunResult]) -> dict[str, dict[str, float]]:
    """Summarise per-variant mean and standard deviation across seeds.

    The standard deviation uses the population convention (``ddof=0``) so a
    single-seed run reports ``0.0`` rather than NaN, which keeps the error bars
    renderable in smoke mode.

    Args:
        results: Runs produced by :func:`run_ablation`.

    Returns:
        Mapping from variant to its aggregated statistics, including a
        ``<metric>_mean`` and ``<metric>_std`` entry for every plotted metric.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    summary: dict[str, dict[str, float]] = {}
    for variant in dict.fromkeys(result.variant for result in results):
        runs = [result for result in results if result.variant == variant]
        stats: dict[str, float] = {
            "n_runs": float(len(runs)),
            "n_parameters": float(runs[0].n_parameters),
            "train_seconds_mean": float(np.mean([run.train_seconds for run in runs])),
            "val_pr_auc_mean": float(np.mean([run.best_val_pr_auc for run in runs])),
        }
        for key, _ in PLOT_METRICS:
            values = np.array([run.test_metrics[key] for run in runs], dtype=float)
            finite = values[np.isfinite(values)]
            # `tpr_at_fpr` is NaN on splits too coarse to resolve the target
            # FPR (see train_transformer._safe_tpr_at_fpr). Reporting NaN
            # directly is honest and keeps numpy from warning on an all-NaN
            # slice; matplotlib simply omits such a bar.
            stats[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
            stats[f"{key}_std"] = float(finite.std()) if finite.size else float("nan")
        summary[variant] = stats
        logger.info(
            "%s: test PR-AUC %.5f +/- %.5f over %d seeds (%d params)",
            variant,
            stats["pr_auc_mean"],
            stats["pr_auc_std"],
            len(runs),
            runs[0].n_parameters,
        )
    return summary


def _numeric_feature_frame(test_df: pd.DataFrame, non_feature_cols: Sequence[str]) -> pd.DataFrame:
    """Reduce a processed split to the numeric matrix the baselines expect.

    String columns are dropped rather than encoded: Member A's estimators were
    fitted on the numeric block only, and under pandas 3 text columns carry the
    ``str`` dtype, which ``select_dtypes(include="number")`` excludes cleanly.

    Args:
        test_df: Processed test split.
        non_feature_cols: Identifier, target, and sequence columns to drop.

    Returns:
        A NaN-free numeric frame.
    """
    dropped = [col for col in non_feature_cols if col in test_df.columns]
    frame = test_df.drop(columns=dropped)
    return frame.select_dtypes(include=["number", "bool"]).fillna(0)


def score_baselines(
    test_df: pd.DataFrame,
    labels: np.ndarray,
    checkpoint_dir: str | Path,
    non_feature_cols: Sequence[str],
) -> dict[str, float]:
    """Score Member A's pickled classical baselines on the same test rows.

    PR-AUC is computed with this project's own
    :func:`~src.evaluation.metrics.average_precision` rather than scikit-learn's,
    so the baseline number and the deep-arm numbers come out of one
    implementation and the comparison in the figure is like for like.

    A missing or unloadable baseline is logged and skipped: Exp 3 must not fail
    because another lane's artefacts have not been produced yet.

    Args:
        test_df: Processed test split.
        labels: Binary fraud labels aligned with ``test_df``.
        checkpoint_dir: Directory holding ``*.pkl`` baseline estimators.
        non_feature_cols: Identifier, target, and sequence columns to drop.

    Returns:
        Mapping from baseline name to test PR-AUC; empty when none could be
        scored.
    """
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        logger.warning("Baseline checkpoint directory %s does not exist; skipping", directory)
        return {}

    pickles = sorted(directory.glob("*.pkl"))
    if not pickles:
        logger.warning("No baseline pickles under %s; skipping the baseline arm", directory)
        return {}

    features = _numeric_feature_frame(test_df, non_feature_cols)
    scores: dict[str, float] = {}

    for path in pickles:
        try:
            with open(path, "rb") as handle:
                model = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError) as exc:
            logger.warning("Could not unpickle %s (%s); skipping", path.name, exc)
            continue

        if not hasattr(model, "predict_proba"):
            logger.info("%s exposes no predict_proba; not a baseline classifier", path.name)
            continue

        matrix = features
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            missing = [name for name in expected if name not in features.columns]
            if missing:
                logger.warning(
                    "%s expects %d columns absent from the test split, e.g. %s; skipping",
                    path.name,
                    len(missing),
                    missing[:3],
                )
                continue
            matrix = features.loc[:, list(expected)]

        try:
            probabilities = np.asarray(model.predict_proba(matrix))[:, 1]
        except (ValueError, IndexError) as exc:
            logger.warning("%s failed to score the test split (%s); skipping", path.name, exc)
            continue

        name = path.stem
        scores[name] = float(average_precision(probabilities, labels))
        logger.info("Baseline %s test PR-AUC %.5f", name, scores[name])

    return scores


def plot_ablation(
    summary: dict[str, dict[str, float]],
    baselines: dict[str, float],
    output_path: str | Path,
    show_plot: bool = False,
) -> Path:
    """Render the grouped ablation bar chart.

    Args:
        summary: Aggregated statistics from :func:`aggregate`.
        baselines: Baseline PR-AUCs from :func:`score_baselines`; the best is
            drawn as a dashed reference line on the PR-AUC group.
        output_path: Destination for the PNG.
        show_plot: Whether to display the figure interactively.

    Returns:
        The path the figure was written to.

    Raises:
        ValueError: If ``summary`` is empty.
    """
    if not summary:
        raise ValueError("nothing to plot: the aggregated summary is empty")

    variants = list(summary)
    positions = np.arange(len(PLOT_METRICS))
    width = min(0.8 / len(variants), 0.28)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for index, variant in enumerate(variants):
        offset = (index - (len(variants) - 1) / 2) * width
        means = [summary[variant][f"{key}_mean"] for key, _ in PLOT_METRICS]
        errors = [summary[variant][f"{key}_std"] for key, _ in PLOT_METRICS]
        ax.bar(
            positions + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, None),
        )

    if baselines:
        best_name = max(baselines, key=lambda key: baselines[key])
        ax.axhline(
            baselines[best_name],
            linestyle="--",
            linewidth=1.5,
            color="#B4553F",
            label=f"Best baseline: {best_name} ({baselines[best_name]:.3f} PR-AUC)",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in PLOT_METRICS])
    ax.set_ylabel("Out-of-time test score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Exp 3: FT-CAT architecture ablation (mean +/- std over seeds)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    logger.info("Saved ablation figure to %s", output)
    if show_plot:
        plt.show()
    plt.close(fig)
    return output


def save_results(
    results: Sequence[RunResult],
    summary: dict[str, dict[str, float]],
    baselines: dict[str, float],
    output_path: str | Path,
    context: dict[str, Any] | None = None,
) -> Path:
    """Write the raw runs, the aggregation, and the baselines to JSON.

    Args:
        results: Individual runs.
        summary: Per-variant aggregation.
        baselines: Classical baseline PR-AUCs.
        output_path: Destination JSON path.
        context: Optional run metadata (row counts, epoch budget, device).

    Returns:
        The path the JSON was written to.
    """
    payload: dict[str, Any] = {
        "experiment": "transformer_cat_ablation",
        "context": dict(context or {}),
        "runs": [result.to_dict() for result in results],
        "summary": summary,
        "baselines": baselines,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved ablation results to %s", output)
    return output


def _parse_int_list(raw: str | None, fallback: Sequence[int]) -> list[int]:
    """Parse a comma-separated integer list, falling back to a default.

    Args:
        raw: Comma-separated string such as ``"42,43"``; ``None`` uses the
            fallback.
        fallback: Values used when ``raw`` is ``None``.

    Returns:
        The parsed integers.

    Raises:
        ValueError: If any element is not an integer.
    """
    if raw is None:
        return [int(value) for value in fallback]
    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"could not parse integer list from {raw!r}: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the FT-CAT architecture ablation.

    Args:
        argv: Command line arguments; uses ``sys.argv`` when omitted.
    """
    parser = argparse.ArgumentParser(description="Exp 3: FT-CAT architecture ablation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--variants", type=str, default=None, help="comma-separated variant names")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu; auto when omitted")
    parser.add_argument("--figure", type=str, default=str(DEFAULT_FIGURE))
    parser.add_argument("--out", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="skip scoring Member A's classical baselines",
    )
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="recompute the MI feature ranking instead of reusing the cache",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (1 seed, 1 epoch, few thousand rows) that exercises every path",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(
        level=str(config.logging.level), log_file=str(config.get_path("logging.log_file"))
    )
    seed_everything(int(config.seed))

    ablation_cfg = config.transformer.ablation
    subsample_cfg = config.transformer.subsample

    variants = (
        [part.strip() for part in args.variants.split(",") if part.strip()]
        if args.variants
        else list(ablation_cfg.variants)
    )
    seeds = _parse_int_list(args.seeds, ablation_cfg.seeds)
    epochs = int(args.epochs if args.epochs is not None else ablation_cfg.epochs)
    train_rows = int(subsample_cfg.train_rows)
    val_rows = int(subsample_cfg.val_rows)

    if args.smoke:
        seeds, epochs = list(SMOKE_SEEDS), SMOKE_EPOCHS
        train_rows, val_rows = SMOKE_TRAIN_ROWS, SMOKE_VAL_ROWS
        logger.warning("Smoke mode: %d seed(s), %d epoch(s), %d train rows", 1, epochs, train_rows)

    train_bundle, val_bundle, test_bundle, spec, test_df = prepare_bundles(
        config,
        args.train_data or config.get_path("data.train_data_path"),
        args.test_data or config.get_path("data.test_data_path"),
        train_rows=train_rows,
        val_rows=val_rows,
        seed=int(config.seed),
        refresh_features=args.refresh_features,
    )

    results = run_ablation(
        config,
        train_bundle,
        val_bundle,
        test_bundle,
        spec,
        variants=variants,
        seeds=seeds,
        epochs=epochs,
        device=args.device,
    )
    summary = aggregate(results)

    baselines: dict[str, float] = {}
    if not args.no_baselines:
        baselines = score_baselines(
            test_df,
            test_bundle.labels_numpy(),
            config.get_path("paths.checkpoints"),
            list(config.data.non_feature_cols),
        )

    plot_ablation(summary, baselines, args.figure)
    save_results(
        results,
        summary,
        baselines,
        args.out,
        context={
            "variants": variants,
            "seeds": seeds,
            "epochs": epochs,
            "device": resolve_device(args.device),
            "n_train_rows": len(train_bundle),
            "n_val_rows": len(val_bundle),
            "n_test_rows": len(test_bundle),
            "n_tokens": spec.n_tokens,
            "smoke": bool(args.smoke),
        },
    )

    best = max(summary, key=lambda variant: summary[variant]["pr_auc_mean"])
    logger.info(
        "Exp 3 complete. Best arm: %s (test PR-AUC %.5f)", best, summary[best]["pr_auc_mean"]
    )


if __name__ == "__main__":
    main()
