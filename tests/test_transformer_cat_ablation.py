"""Unit tests for Exp 3, the FT-CAT architecture ablation (Member B).

Exercises the sweep driver, the per-variant aggregation, the defensive
behaviour of the classical-baseline arm when Member A's pickles are absent or
unusable, the figure writer, and the CLI end to end. Everything runs on tiny
synthetic parquet splits pinned to the CPU.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.transformer_cat_ablation import (
    PLOT_METRICS,
    RunResult,
    _parse_int_list,
    aggregate,
    main,
    plot_ablation,
    prepare_bundles,
    run_ablation,
    save_results,
    score_baselines,
)
from src.training.feature_selection import FeatureSpec
from src.training.train_transformer import SEQUENCE_COLUMN, materialize_tensors
from src.utils.config import Config

N_ROWS = 160
N_CONT = 4
CARDS = [3, 4]
SEQ_LEN = 5
SEQ_DIM = 3
CONT_COLS = [f"c{i}" for i in range(N_CONT)]
CAT_COLS = [f"k{i}" for i in range(len(CARDS))]
SEQ_COLS = [f"s{i}" for i in range(SEQ_DIM)]
NON_FEATURE_COLS = ["TransactionID", "TransactionDT", "isFraud", SEQUENCE_COLUMN]


def make_frame(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic processed split with a learnable fraud signal.

    Args:
        n_rows: Number of rows to generate.
        seed: Seed for the generator.

    Returns:
        A frame shaped like the output of ``src.data.prepare_data``.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({col: rng.normal(size=n_rows) for col in CONT_COLS})
    for col, card in zip(CAT_COLS, CARDS):
        frame[col] = rng.integers(0, card, size=n_rows, dtype=np.int64)
    frame[SEQUENCE_COLUMN] = [rng.normal(size=(SEQ_LEN, SEQ_DIM)).tolist() for _ in range(n_rows)]
    frame["isFraud"] = (frame["c0"] > frame["c0"].quantile(0.7)).astype(np.int64)
    frame["TransactionDT"] = np.arange(n_rows, dtype=np.int64)
    return frame


def make_spec() -> FeatureSpec:
    """Return the feature contract matching :func:`make_frame`."""
    return FeatureSpec(
        continuous_cols=list(CONT_COLS),
        categorical_cols=list(CAT_COLS),
        categorical_cardinalities=list(CARDS),
        sequence_cols=list(SEQ_COLS),
        seq_len=SEQ_LEN,
    )


def make_config() -> Config:
    """Return a transformer configuration that trains in well under a second."""
    return Config(
        {
            "seed": 42,
            "transformer": {
                "d_model": 8,
                "n_heads": 2,
                "dim_feedforward": 16,
                "n_layers": 1,
                "dropout": 0.0,
                "activation": "gelu",
                "norm_first": True,
                "seq_len": SEQ_LEN,
                "n_continuous": N_CONT,
                "categorical_cardinalities": list(CARDS),
                "training": {
                    "lr": 1.0e-2,
                    "weight_decay": 1.0e-4,
                    "epochs": 2,
                    "batch_size": 32,
                    "warmup_epochs": 1,
                    "grad_clip": 5.0,
                    "amp": False,
                    "pin_memory": False,
                    "early_stopping_patience": 5,
                    "min_delta": 1.0e-4,
                },
                "loss": {"name": "focal", "gamma": 2.0, "alpha": 0.25},
            },
            "sequence": {"feature_cols": list(SEQ_COLS), "k_window": SEQ_LEN},
        }
    )


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


def make_results() -> list[RunResult]:
    """Return two hand-built runs of one variant for aggregation tests."""
    return [
        RunResult(
            variant="ft_cat",
            seed=seed,
            n_parameters=100,
            best_val_pr_auc=0.5,
            best_epoch=1,
            epochs_run=1,
            train_seconds=1.0,
            test_metrics={"pr_auc": value, "roc_auc": value, "tpr_at_fpr": value, "loss": 0.1},
        )
        for seed, value in ((42, 0.4), (43, 0.6))
    ]


class StubClassifier:
    """Minimal stand-in for a fitted scikit-learn baseline.

    Attributes:
        feature_names_in_: Columns the estimator claims to have been fitted on.
        column: Column whose values are returned as the positive-class score.
    """

    def __init__(self, feature_names: list[str], column: str) -> None:
        """Record the fitted column contract."""
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.column = column

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a two-column probability matrix driven by ``column``."""
        positive = frame[self.column].to_numpy(dtype=float)
        # Squash into (0, 1) so the output is shaped like a real probability.
        positive = 1.0 / (1.0 + np.exp(-positive))
        return np.column_stack([1.0 - positive, positive])


class ExplodingClassifier(StubClassifier):
    """A baseline that fails at scoring time."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Raise as a badly-fitted estimator would.

        Raises:
            ValueError: Always.
        """
        raise ValueError("shape mismatch")


def write_pickle(path: Path, payload: object) -> None:
    """Pickle ``payload`` to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def test_run_ablation_covers_every_variant_seed_pair(
    config: Config, spec: FeatureSpec, bundles: tuple[Any, Any, Any]
) -> None:
    """One result is produced per variant and seed, with populated metrics."""
    train_bundle, val_bundle, test_bundle = bundles
    results = run_ablation(
        config,
        train_bundle,
        val_bundle,
        test_bundle,
        spec,
        variants=["ft_cat", "mlp"],
        seeds=[42],
        epochs=1,
        device="cpu",
    )
    assert len(results) == 2
    assert {result.variant for result in results} == {"ft_cat", "mlp"}
    for result in results:
        assert result.n_parameters > 0
        assert result.epochs_run == 1
        assert result.train_seconds >= 0.0
        for key, _ in PLOT_METRICS:
            assert key in result.test_metrics


def test_run_ablation_rejects_unknown_variant(
    config: Config, spec: FeatureSpec, bundles: tuple[Any, Any, Any]
) -> None:
    """An unknown architecture arm is refused before any training starts."""
    train_bundle, val_bundle, test_bundle = bundles
    with pytest.raises(ValueError, match="unknown variants"):
        run_ablation(
            config,
            train_bundle,
            val_bundle,
            test_bundle,
            spec,
            variants=["nope"],
            seeds=[42],
            epochs=1,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("variants", "seeds", "message"),
    [([], [42], "at least one variant"), (["ft_cat"], [], "at least one seed")],
)
def test_run_ablation_rejects_empty_axes(
    config: Config,
    spec: FeatureSpec,
    bundles: tuple[Any, Any, Any],
    variants: list[str],
    seeds: list[int],
    message: str,
) -> None:
    """Both sweep axes must be non-empty."""
    train_bundle, val_bundle, test_bundle = bundles
    with pytest.raises(ValueError, match=message):
        run_ablation(
            config,
            train_bundle,
            val_bundle,
            test_bundle,
            spec,
            variants=variants,
            seeds=seeds,
            epochs=1,
            device="cpu",
        )


def test_aggregate_computes_mean_and_std() -> None:
    """Aggregation reports the population mean and standard deviation."""
    summary = aggregate(make_results())
    assert summary["ft_cat"]["n_runs"] == 2.0
    assert summary["ft_cat"]["pr_auc_mean"] == pytest.approx(0.5)
    assert summary["ft_cat"]["pr_auc_std"] == pytest.approx(0.1)


def test_aggregate_single_seed_reports_zero_std() -> None:
    """A single run reports a zero spread rather than NaN."""
    summary = aggregate(make_results()[:1])
    assert summary["ft_cat"]["pr_auc_std"] == pytest.approx(0.0)


def test_aggregate_rejects_empty_results() -> None:
    """There is nothing to summarise without runs."""
    with pytest.raises(ValueError, match="empty result set"):
        aggregate([])


def test_score_baselines_returns_pr_auc_for_a_stub(tmp_path: Path) -> None:
    """A well-formed pickled estimator is scored on the supplied rows."""
    frame = make_frame(seed=3)
    write_pickle(tmp_path / "LogReg_Balanced.pkl", StubClassifier(CONT_COLS, "c0"))
    scores = score_baselines(
        frame,
        frame["isFraud"].to_numpy(dtype=np.int64),
        tmp_path,
        NON_FEATURE_COLS,
    )
    # The label is a threshold on c0 and the stub scores by c0, so the ranking
    # is perfect and average precision must be 1.0.
    assert scores == {"LogReg_Balanced": pytest.approx(1.0)}


def test_score_baselines_skips_models_with_missing_columns(tmp_path: Path) -> None:
    """An estimator fitted on absent columns is skipped, not crashed on."""
    frame = make_frame(seed=3)
    write_pickle(tmp_path / "stale.pkl", StubClassifier(["not_here"], "not_here"))
    assert score_baselines(frame, frame["isFraud"].to_numpy(), tmp_path, NON_FEATURE_COLS) == {}


def test_score_baselines_skips_models_that_fail_to_score(tmp_path: Path) -> None:
    """A scoring failure is contained to that one baseline."""
    frame = make_frame(seed=3)
    write_pickle(tmp_path / "broken.pkl", ExplodingClassifier(CONT_COLS, "c0"))
    assert score_baselines(frame, frame["isFraud"].to_numpy(), tmp_path, NON_FEATURE_COLS) == {}


def test_score_baselines_skips_non_classifiers(tmp_path: Path) -> None:
    """A pickle without ``predict_proba`` is not a baseline."""
    frame = make_frame(seed=3)
    write_pickle(tmp_path / "notes.pkl", {"unrelated": "payload"})
    assert score_baselines(frame, frame["isFraud"].to_numpy(), tmp_path, NON_FEATURE_COLS) == {}


def test_score_baselines_handles_missing_directory(tmp_path: Path) -> None:
    """A missing checkpoint directory is reported, not raised."""
    frame = make_frame(seed=3)
    missing = tmp_path / "absent"
    assert score_baselines(frame, frame["isFraud"].to_numpy(), missing, NON_FEATURE_COLS) == {}


def test_score_baselines_handles_empty_directory(tmp_path: Path) -> None:
    """A directory holding no pickles yields no baselines."""
    frame = make_frame(seed=3)
    assert score_baselines(frame, frame["isFraud"].to_numpy(), tmp_path, NON_FEATURE_COLS) == {}


def test_plot_ablation_writes_a_png(tmp_path: Path) -> None:
    """The figure is written, including the baseline reference line."""
    output = tmp_path / "nested" / "transformer_cat_ablation.png"
    assert plot_ablation(aggregate(make_results()), {"LightGBM": 0.42}, output) == output
    assert output.stat().st_size > 0


def test_plot_ablation_without_baselines(tmp_path: Path) -> None:
    """The figure still renders when no baseline could be scored."""
    output = tmp_path / "no_baseline.png"
    plot_ablation(aggregate(make_results()), {}, output)
    assert output.is_file()


def test_plot_ablation_rejects_empty_summary(tmp_path: Path) -> None:
    """An empty summary has nothing to draw."""
    with pytest.raises(ValueError, match="summary is empty"):
        plot_ablation({}, {}, tmp_path / "x.png")


def test_save_results_writes_the_expected_payload(tmp_path: Path) -> None:
    """The JSON carries the runs, the aggregation, and the baselines."""
    results = make_results()
    output = save_results(
        results,
        aggregate(results),
        {"LightGBM": 0.42},
        tmp_path / "cat_ablation.json",
        context={"epochs": 1},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment"] == "transformer_cat_ablation"
    assert len(payload["runs"]) == 2
    assert payload["baselines"] == {"LightGBM": 0.42}
    assert payload["context"]["epochs"] == 1
    assert "ft_cat" in payload["summary"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, [1, 2]), ("42,43", [42, 43]), ("42, 43 ,", [42, 43])],
)
def test_parse_int_list(raw: str | None, expected: list[int]) -> None:
    """Seeds parse from CSV and fall back to the configured default."""
    assert _parse_int_list(raw, [1, 2]) == expected


def test_parse_int_list_rejects_non_integers() -> None:
    """A malformed list is a clear error, not a silent empty sweep."""
    with pytest.raises(ValueError, match="could not parse integer list"):
        _parse_int_list("42,abc", [1])


def _write_cli_fixtures(tmp_path: Path) -> Path:
    """Write parquet splits and a config file for the CLI tests.

    Args:
        tmp_path: Directory to populate.

    Returns:
        Path of the written configuration file.
    """
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    make_frame(seed=0).to_parquet(processed / "train.parquet", index=False)
    make_frame(n_rows=96, seed=2).to_parquet(processed / "test.parquet", index=False)

    payload = make_config().to_dict()
    payload["transformer"]["feature_cache_path"] = str(processed / "features.json")
    payload["transformer"]["feature_selection"] = {"top_n_continuous": N_CONT, "mi_sample_size": 64}
    payload["transformer"]["subsample"] = {"train_rows": 0, "val_rows": 0}
    payload["transformer"]["ablation"] = {"variants": ["ft_cat"], "seeds": [42], "epochs": 1}
    payload["paths"] = {"checkpoints": str(tmp_path / "checkpoints"), "figures": str(tmp_path)}
    payload["logging"] = {"level": "INFO", "log_file": str(tmp_path / "system.log")}
    payload["data"] = {
        "train_data_path": str(processed / "train.parquet"),
        "test_data_path": str(processed / "test.parquet"),
        "preprocessor_path": str(processed / "preprocessor.pkl"),
        "non_feature_cols": list(NON_FEATURE_COLS),
    }
    payload["split"] = {"time_col": "TransactionDT", "model_train_fraction": 0.75}
    payload["features"] = {"target_col": "isFraud"}

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return config_path


def test_prepare_bundles_reports_a_missing_split(tmp_path: Path) -> None:
    """A missing parquet points at the data preparation step."""
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        prepare_bundles(
            make_config(),
            tmp_path / "absent_train.parquet",
            tmp_path / "absent_test.parquet",
            train_rows=0,
            val_rows=0,
            seed=42,
        )


def test_main_writes_figure_and_results(tmp_path: Path) -> None:
    """The CLI runs end to end and writes both artefacts."""
    config_path = _write_cli_fixtures(tmp_path)
    figure = tmp_path / "ablation.png"
    results = tmp_path / "ablation.json"
    main(
        [
            "--config",
            str(config_path),
            "--figure",
            str(figure),
            "--out",
            str(results),
            "--no-baselines",
        ]
    )
    assert figure.stat().st_size > 0
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["baselines"] == {}
    assert payload["context"]["epochs"] == 1
    assert len(payload["runs"]) == 1


def test_main_smoke_mode_overrides_the_budget(tmp_path: Path) -> None:
    """``--smoke`` forces a single seed and a single epoch."""
    config_path = _write_cli_fixtures(tmp_path)
    results = tmp_path / "smoke.json"
    main(
        [
            "--config",
            str(config_path),
            "--figure",
            str(tmp_path / "smoke.png"),
            "--out",
            str(results),
            "--no-baselines",
            "--smoke",
        ]
    )
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["context"]["smoke"] is True
    assert payload["context"]["seeds"] == [42]
    assert payload["context"]["epochs"] == 1


def test_main_scores_baselines_when_present(tmp_path: Path) -> None:
    """With a pickle on disk the baseline arm reaches the results file."""
    config_path = _write_cli_fixtures(tmp_path)
    write_pickle(tmp_path / "checkpoints" / "LogReg_Balanced.pkl", StubClassifier(CONT_COLS, "c0"))
    results = tmp_path / "with_baseline.json"
    main(
        [
            "--config",
            str(config_path),
            "--figure",
            str(tmp_path / "with_baseline.png"),
            "--out",
            str(results),
            "--variants",
            "ft_cat",
            "--seeds",
            "42",
            "--epochs",
            "1",
        ]
    )
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["baselines"]["LogReg_Balanced"] == pytest.approx(1.0)
