"""Unit tests for the FT-CAT feature contract.

Covers the ``FeatureSpec`` value object, the train-only mutual-information
ranking, cardinality derivation, and the cached ``resolve_feature_set``
orchestration. All fixtures are synthetic so the suite runs without the real
IEEE-CIS dataset.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.training.feature_selection import (
    FeatureSpec,
    _continuous_candidates,
    build_feature_spec,
    compute_cardinalities,
    infer_column_roles,
    load_column_roles,
    rank_features_by_mi,
    resolve_feature_set,
    stratified_indices,
)
from src.utils.config import Config

TARGET = "isFraud"
NON_FEATURE_COLS = ["TransactionID", "TransactionDT", TARGET, "sequence_array"]


@pytest.fixture()
def synthetic_train(rng: np.random.Generator) -> pd.DataFrame:
    """Small labelled frame with one strong, one weak and one constant column."""
    n = 400
    labels = np.zeros(n, dtype=np.int64)
    labels[: n // 4] = 1
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n, dtype=np.int64),
            "TransactionDT": np.arange(n, dtype=np.int64) * 100,
            TARGET: labels,
            # Separated class means => high mutual information.
            "strong": labels.astype(np.float64) * 5.0 + rng.normal(0, 0.25, n),
            "weak": rng.normal(0, 1.0, n),
            "const": np.zeros(n, dtype=np.float64),
            "strong_is_nan": (labels == 1).astype(np.float64),
            "card4": rng.integers(0, 4, n, dtype=np.int64),
            "card6": rng.integers(0, 3, n, dtype=np.int64),
        }
    )


@pytest.fixture()
def spec() -> FeatureSpec:
    """A minimal valid feature spec."""
    return FeatureSpec(
        continuous_cols=["a", "b"],
        categorical_cols=["c"],
        categorical_cardinalities=[5],
        sequence_cols=["TransactionAmt", "D1"],
        seq_len=5,
    )


def test_spec_properties(spec: FeatureSpec) -> None:
    """Token accounting includes the [CLS] token exactly once."""
    assert spec.n_continuous == 2
    assert spec.n_categorical == 1
    assert spec.n_tokens == 4
    assert spec.seq_dim == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"categorical_cardinalities": [5, 7]}, "equal length"),
        ({"seq_len": 0}, "seq_len must be positive"),
        ({"categorical_cardinalities": [0]}, "reserved <UNK>"),
        ({"mi_scores": [0.1]}, "must align"),
    ],
)
def test_spec_rejects_inconsistent_fields(kwargs: dict, message: str) -> None:
    """Malformed specs fail loudly at construction time."""
    base = {
        "continuous_cols": ["a", "b"],
        "categorical_cols": ["c"],
        "categorical_cardinalities": [5],
        "sequence_cols": ["D1"],
        "seq_len": 5,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        FeatureSpec(**base)


def test_spec_dict_roundtrip(spec: FeatureSpec) -> None:
    """to_dict/from_dict preserve every field."""
    assert FeatureSpec.from_dict(spec.to_dict()) == spec


def test_spec_from_dict_requires_core_keys() -> None:
    """A truncated payload names the missing keys."""
    with pytest.raises(KeyError, match="missing keys"):
        FeatureSpec.from_dict({"continuous_cols": ["a"]})


def test_spec_json_roundtrip(spec: FeatureSpec, tmp_path: Path) -> None:
    """Specs survive a save/load cycle through JSON."""
    target = tmp_path / "nested" / "spec.json"
    spec.save_json(target)
    assert target.is_file()
    assert FeatureSpec.load_json(target) == spec


def test_spec_load_json_missing_file(tmp_path: Path) -> None:
    """A missing cache raises rather than silently returning a default."""
    with pytest.raises(FileNotFoundError, match="cache not found"):
        FeatureSpec.load_json(tmp_path / "absent.json")


def test_infer_column_roles_splits_by_dtype(synthetic_train: pd.DataFrame) -> None:
    """Floats become continuous, ints become categorical, ids are dropped."""
    cont, cat = infer_column_roles(synthetic_train, NON_FEATURE_COLS)
    assert cont == ["strong", "weak", "const", "strong_is_nan"]
    assert cat == ["card4", "card6"]


def test_load_column_roles_reads_preprocessor(tmp_path: Path) -> None:
    """The pickled preprocessor is the authoritative role source."""
    path = tmp_path / "preprocessor.pkl"
    path.write_bytes(pickle.dumps(SimpleNamespace(cont_cols=["a"], cat_cols=["b"])))
    assert load_column_roles(path) == (["a"], ["b"])


def test_load_column_roles_missing_file(tmp_path: Path) -> None:
    """An absent artifact raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Preprocessor artifact not found"):
        load_column_roles(tmp_path / "absent.pkl")


def test_load_column_roles_wrong_object(tmp_path: Path) -> None:
    """An object without the expected attributes is rejected."""
    path = tmp_path / "preprocessor.pkl"
    path.write_bytes(pickle.dumps(42))
    with pytest.raises(AttributeError, match="cont_cols"):
        load_column_roles(path)


def test_continuous_candidates_appends_missingness_masks(
    synthetic_train: pd.DataFrame,
) -> None:
    """Each base column contributes its `_is_nan` mask when one exists."""
    candidates = _continuous_candidates(synthetic_train, ["strong", "weak"])
    assert candidates == ["strong", "weak", "strong_is_nan"]


def test_stratified_indices_returns_all_when_not_downsampling() -> None:
    """A quota at or above the population keeps every row."""
    labels = np.array([0, 1, 0, 1])
    assert np.array_equal(stratified_indices(labels, 10, 42), np.arange(4))
    assert np.array_equal(stratified_indices(labels, 0, 42), np.arange(4))


def test_stratified_indices_keeps_both_classes_and_is_deterministic() -> None:
    """The minority class survives downsampling and draws are reproducible."""
    labels = np.zeros(1000, dtype=np.int64)
    labels[:20] = 1
    first = stratified_indices(labels, 100, 42)
    assert np.array_equal(first, stratified_indices(labels, 100, 42))
    assert labels[first].sum() > 0
    assert first.size <= 105


def test_rank_features_by_mi_prefers_the_informative_column(
    synthetic_train: pd.DataFrame,
) -> None:
    """The label-correlated column outranks noise and constants."""
    selected, scores = rank_features_by_mi(
        synthetic_train, TARGET, ["weak", "strong", "const"], top_n=2, seed=42
    )
    assert selected[0] == "strong"
    assert len(selected) == len(scores) == 2
    assert scores == sorted(scores, reverse=True)


def test_rank_features_by_mi_is_deterministic(synthetic_train: pd.DataFrame) -> None:
    """Two runs at the same seed give identical orderings."""
    args = (synthetic_train, TARGET, ["weak", "strong", "const", "strong_is_nan"])
    assert rank_features_by_mi(*args, top_n=3, seed=42) == rank_features_by_mi(
        *args, top_n=3, seed=42
    )


def test_rank_features_by_mi_keeps_all_when_top_n_non_positive(
    synthetic_train: pd.DataFrame,
) -> None:
    """A non-positive top_n disables truncation."""
    selected, _ = rank_features_by_mi(
        synthetic_train, TARGET, ["weak", "strong", "const"], top_n=0, seed=42
    )
    assert sorted(selected) == ["const", "strong", "weak"]


@pytest.mark.parametrize(
    "target, candidates, message",
    [
        ("absent", ["strong"], "not present in frame"),
        (TARGET, ["nope"], "no column present"),
    ],
)
def test_rank_features_by_mi_validates_inputs(
    synthetic_train: pd.DataFrame, target: str, candidates: list[str], message: str
) -> None:
    """Missing targets and empty candidate pools raise ValueError."""
    with pytest.raises(ValueError, match=message):
        rank_features_by_mi(synthetic_train, target, candidates, top_n=1)


def test_rank_features_by_mi_requires_two_classes(synthetic_train: pd.DataFrame) -> None:
    """A single-class split cannot support a mutual-information ranking."""
    single = synthetic_train.assign(**{TARGET: 0})
    with pytest.raises(ValueError, match="both classes"):
        rank_features_by_mi(single, TARGET, ["strong"], top_n=1)


def test_compute_cardinalities_reserves_the_unk_slot(
    synthetic_train: pd.DataFrame,
) -> None:
    """Table size is max_code + 1 so index 0 stays available for unknowns."""
    assert compute_cardinalities(synthetic_train, ["card4", "card6"]) == [4, 3]


def test_compute_cardinalities_rejects_unknown_column(
    synthetic_train: pd.DataFrame,
) -> None:
    """A column absent from the frame raises KeyError."""
    with pytest.raises(KeyError, match="not present in frame"):
        compute_cardinalities(synthetic_train, ["nope"])


def test_compute_cardinalities_rejects_negative_codes() -> None:
    """Negative codes would index outside the embedding table."""
    frame = pd.DataFrame({"c": np.array([-1, 2], dtype=np.int64)})
    with pytest.raises(ValueError, match="negative category codes"):
        compute_cardinalities(frame, ["c"])


def test_build_feature_spec_assembles_a_consistent_contract(
    synthetic_train: pd.DataFrame,
) -> None:
    """The assembled spec is internally consistent and correctly sized."""
    built = build_feature_spec(
        synthetic_train,
        continuous_candidates=["strong", "weak", "const"],
        categorical_cols=["card4", "card6", "absent"],
        sequence_cols=["TransactionAmt", "D1"],
        seq_len=5,
        top_n=2,
        target_col=TARGET,
        seed=42,
    )
    assert built.n_continuous == 2
    assert built.categorical_cols == ["card4", "card6"]
    assert built.n_tokens == 5
    assert len(built.mi_scores) == 2


def _make_config(tmp_path: Path, top_n: int = 2) -> Config:
    """Build a minimal Config rooted at ``tmp_path``."""
    return Config(
        {
            "seed": 42,
            "transformer": {
                "feature_cache_path": "ft_cat_features.json",
                "feature_selection": {"top_n_continuous": top_n, "mi_sample_size": 1000},
            },
            "data": {
                "train_data_path": "train.parquet",
                "preprocessor_path": "preprocessor.pkl",
                "non_feature_cols": NON_FEATURE_COLS,
            },
            "sequence": {"feature_cols": ["TransactionAmt", "D1"], "k_window": 5},
            "features": {"target_col": TARGET},
        },
        base_dir=tmp_path,
    )


def test_resolve_feature_set_builds_and_caches(
    synthetic_train: pd.DataFrame, tmp_path: Path
) -> None:
    """A cache miss builds the spec from the frame and writes it to disk."""
    config = _make_config(tmp_path)
    built = resolve_feature_set(config, train_df=synthetic_train)
    assert (tmp_path / "ft_cat_features.json").is_file()
    # The second call must hit the cache, so a conflicting frame is ignored.
    assert resolve_feature_set(config, train_df=synthetic_train.head(0)) == built


def test_resolve_feature_set_force_refresh_rebuilds(
    synthetic_train: pd.DataFrame, tmp_path: Path
) -> None:
    """force_refresh overwrites a stale cache with a new contract."""
    config = _make_config(tmp_path, top_n=2)
    first = resolve_feature_set(config, train_df=synthetic_train)
    config.transformer.feature_selection["top_n_continuous"] = 3
    second = resolve_feature_set(config, train_df=synthetic_train, force_refresh=True)
    assert first.n_continuous == 2
    assert second.n_continuous == 3
    assert FeatureSpec.load_json(tmp_path / "ft_cat_features.json") == second


def test_resolve_feature_set_falls_back_to_dtypes(
    synthetic_train: pd.DataFrame, tmp_path: Path
) -> None:
    """Without a preprocessor artifact, roles are inferred from dtypes."""
    config = _make_config(tmp_path, top_n=4)
    built = resolve_feature_set(config, train_df=synthetic_train)
    assert built.categorical_cols == ["card4", "card6"]
    assert set(built.continuous_cols) <= {"strong", "weak", "const", "strong_is_nan"}


def test_resolve_feature_set_reads_parquet_when_no_frame_given(
    synthetic_train: pd.DataFrame, tmp_path: Path
) -> None:
    """The training parquet is the default source on a cache miss."""
    config = _make_config(tmp_path)
    synthetic_train.to_parquet(tmp_path / "train.parquet", index=False)
    assert resolve_feature_set(config).n_continuous == 2


def test_resolve_feature_set_without_cache_or_parquet(tmp_path: Path) -> None:
    """With neither input available the error points at prepare_data."""
    config = _make_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        resolve_feature_set(config)
