"""Deterministic, leak-free feature selection for the FT-CAT tokenizer.

The processed IEEE-CIS splits carry roughly 800 continuous columns (numeric
features plus their ``_is_nan`` missingness indicators). A Feature-Tokenizer
Transformer emits one token per feature, so tokenizing all of them would blow
the project's CPU training envelope. This module ranks the continuous columns
by mutual information against the fraud label -- computed **strictly on the
training split** so no validation or test information leaks into the
architecture -- and caches the resulting contract to disk.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reserved embedding index for categories unseen at fit time. Mirrors
# ``FraudPreprocessor.transform``, which maps unknown categories to 0.
UNK_INDEX: int = 0


@dataclass(frozen=True)
class FeatureSpec:
    """Immutable description of the tensor layout consumed by FT-CAT."""

    continuous_cols: list[str]
    categorical_cols: list[str]
    categorical_cardinalities: list[int]
    sequence_cols: list[str]
    seq_len: int
    mi_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate internal consistency of the spec."""
        if len(self.categorical_cols) != len(self.categorical_cardinalities):
            raise ValueError(
                f"categorical_cols ({len(self.categorical_cols)}) and cardinalities "
                f"({len(self.categorical_cardinalities)}) must have equal length"
            )
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if any(card <= UNK_INDEX for card in self.categorical_cardinalities):
            raise ValueError(
                f"every cardinality must exceed the reserved <UNK> index {UNK_INDEX}, "
                f"got {self.categorical_cardinalities}"
            )
        if self.mi_scores and len(self.mi_scores) != len(self.continuous_cols):
            raise ValueError(
                f"mi_scores ({len(self.mi_scores)}) must align with continuous_cols "
                f"({len(self.continuous_cols)})"
            )

    @property
    def n_continuous(self) -> int:
        """Number of continuous feature tokens."""
        return len(self.continuous_cols)

    @property
    def n_categorical(self) -> int:
        """Number of categorical embedding tokens."""
        return len(self.categorical_cols)

    @property
    def n_tokens(self) -> int:
        """Total token count fed to the encoder, including ``[CLS]``."""
        return self.n_continuous + self.n_categorical + 1

    @property
    def seq_dim(self) -> int:
        """Per-timestep feature width ``D`` of the history tensor."""
        return len(self.sequence_cols)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the spec."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Persist the spec to ``path`` as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Saved FT-CAT feature spec to %s", out)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSpec":
        """Rebuild a spec from its dictionary form."""
        required = (
            "continuous_cols",
            "categorical_cols",
            "categorical_cardinalities",
            "sequence_cols",
            "seq_len",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"feature spec payload is missing keys: {missing}")
        return cls(
            continuous_cols=list(payload["continuous_cols"]),
            categorical_cols=list(payload["categorical_cols"]),
            categorical_cardinalities=[int(c) for c in payload["categorical_cardinalities"]],
            sequence_cols=list(payload["sequence_cols"]),
            seq_len=int(payload["seq_len"]),
            mi_scores=[float(s) for s in payload.get("mi_scores", [])],
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "FeatureSpec":
        """Load a spec previously written by :meth:`save_json`."""
        cache_path = Path(path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Feature spec cache not found: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls.from_dict(payload)


def load_column_roles(preprocessor_path: str | Path) -> tuple[list[str], list[str]]:
    """Read the continuous/categorical column split.

    ``data/processed/preprocessor.pkl`` is the authoritative record of which
    raw columns were scaled as continuous and which were vocabulary-encoded,
    so it is preferred over dtype guessing.
    """
    path = Path(preprocessor_path)
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessor artifact not found: {path}")
    with open(path, "rb") as fh:
        preprocessor = pickle.load(fh)
    for attr in ("cont_cols", "cat_cols"):
        if not hasattr(preprocessor, attr):
            raise AttributeError(f"{path} has no attribute {attr!r}")
    return list(preprocessor.cont_cols), list(preprocessor.cat_cols)


def infer_column_roles(
    df: pd.DataFrame, non_feature_cols: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Infer column roles from dtypes when no preprocessor artifact exists.

    Floating-point columns are treated as continuous; integer columns are
    treated as vocabulary-encoded categoricals, which matches the output of
    ``FraudPreprocessor``. Used by unit tests and by any caller working on
    synthetic frames.
    """
    excluded = set(non_feature_cols)
    continuous, categorical = [], []
    for col in df.columns:
        if col in excluded:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            continuous.append(col)
        elif pd.api.types.is_integer_dtype(dtype):
            categorical.append(col)
    return continuous, categorical


def _stratified_indices(labels: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Draw a class-proportional row sample without replacement."""
    n_rows = labels.shape[0]
    if sample_size <= 0 or sample_size >= n_rows:
        return np.arange(n_rows)

    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for value in (0, 1):
        members = np.flatnonzero(labels == value)
        if members.size == 0:
            continue
        # Round up so the minority class is never sampled away entirely.
        quota = min(members.size, int(np.ceil(sample_size * members.size / n_rows)))
        keep.append(rng.choice(members, size=quota, replace=False))
    return np.sort(np.concatenate(keep))


def rank_features_by_mi(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: Sequence[str],
    top_n: int,
    sample_size: int = 50_000,
    seed: int = 42,
) -> tuple[list[str], list[float]]:
    """Rank candidate columns by mutual information with the fraud label.

    Mutual information is estimated on a class-proportional subsample because
    the k-nearest-neighbour estimator is superlinear in row count. Ties are
    broken by column name so the ordering is stable across runs and machines.
    """
    from sklearn.feature_selection import mutual_info_classif

    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not present in frame")
    present = [col for col in candidate_cols if col in df.columns]
    if not present:
        raise ValueError("candidate_cols contains no column present in the frame")

    labels = df[target_col].to_numpy(dtype=np.int64, copy=False)
    if np.unique(labels).size < 2:
        raise ValueError("mutual information requires both classes in the split")

    idx = _stratified_indices(labels, sample_size, seed)
    matrix = df.iloc[idx][present].to_numpy(dtype=np.float64, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    sampled_labels = labels[idx]

    # Constant columns carry no information and make the estimator warn.
    variances = matrix.var(axis=0)
    informative = np.flatnonzero(variances > 0.0)
    scores = np.zeros(len(present), dtype=np.float64)
    if informative.size:
        scores[informative] = mutual_info_classif(
            matrix[:, informative], sampled_labels, random_state=seed
        )
    logger.info(
        "Ranked %d candidate features by MI on %d rows (%d constant columns skipped)",
        len(present),
        idx.size,
        len(present) - informative.size,
    )

    order = sorted(range(len(present)), key=lambda i: (-scores[i], present[i]))
    keep = len(present) if top_n <= 0 else min(top_n, len(present))
    chosen = order[:keep]
    return [present[i] for i in chosen], [float(scores[i]) for i in chosen]


def compute_cardinalities(df: pd.DataFrame, categorical_cols: Sequence[str]) -> list[int]:
    """Derive embedding table sizes for vocabulary-encoded columns.

    ``FraudPreprocessor`` maps training categories to ``1..N`` and reserves 0
    for unseen categories, so the table must hold ``N + 1`` rows.
    """
    cardinalities: list[int] = []
    for col in categorical_cols:
        if col not in df.columns:
            raise KeyError(f"categorical column {col!r} not present in frame")
        codes = df[col].to_numpy(dtype=np.int64, copy=False)
        if codes.size and int(codes.min()) < 0:
            raise ValueError(f"column {col!r} holds negative category codes")
        max_code = int(codes.max()) if codes.size else UNK_INDEX
        cardinalities.append(max(max_code, UNK_INDEX) + 1)
    return cardinalities


def build_feature_spec(
    train_df: pd.DataFrame,
    continuous_candidates: Sequence[str],
    categorical_cols: Sequence[str],
    sequence_cols: Sequence[str],
    seq_len: int,
    top_n: int,
    target_col: str = "isFraud",
    sample_size: int = 50_000,
    seed: int = 42,
) -> FeatureSpec:
    """Assemble a :class:`FeatureSpec` from a training split."""
    selected, scores = rank_features_by_mi(
        train_df,
        target_col=target_col,
        candidate_cols=continuous_candidates,
        top_n=top_n,
        sample_size=sample_size,
        seed=seed,
    )
    present_cats = [col for col in categorical_cols if col in train_df.columns]
    spec = FeatureSpec(
        continuous_cols=selected,
        categorical_cols=present_cats,
        categorical_cardinalities=compute_cardinalities(train_df, present_cats),
        sequence_cols=list(sequence_cols),
        seq_len=int(seq_len),
        mi_scores=scores,
    )
    logger.info(
        "Built FT-CAT feature spec: %d continuous + %d categorical + 1 CLS = %d tokens",
        spec.n_continuous,
        spec.n_categorical,
        spec.n_tokens,
    )
    return spec


def _continuous_candidates(df: pd.DataFrame, base_cont_cols: Sequence[str]) -> list[str]:
    """Expand base continuous columns with their missingness indicators."""
    available = set(df.columns)
    candidates = [col for col in base_cont_cols if col in available]
    candidates += [mask for col in base_cont_cols if (mask := f"{col}_is_nan") in available]
    return candidates


def resolve_feature_set(
    config: Config,
    train_df: pd.DataFrame | None = None,
    force_refresh: bool = False,
) -> FeatureSpec:
    """Load the cached FT-CAT feature contract, building it when absent."""
    cache_path = config.get_path("transformer.feature_cache_path")
    if cache_path.is_file() and not force_refresh:
        spec = FeatureSpec.load_json(cache_path)
        logger.info(
            "Loaded cached FT-CAT feature spec from %s (%d tokens)", cache_path, spec.n_tokens
        )
        return spec

    if train_df is None:
        train_path = config.get_path("data.train_data_path")
        if not train_path.is_file():
            raise FileNotFoundError(
                f"Cannot build a feature spec: neither cache {cache_path} nor "
                f"training split {train_path} exists. Run src.data.prepare_data first."
            )
        train_df = pd.read_parquet(train_path)

    selection_cfg = config.transformer.feature_selection
    non_feature_cols = list(config.data.non_feature_cols)
    target_col = str(config.features.target_col)

    try:
        base_cont, cat_cols = load_column_roles(config.get_path("data.preprocessor_path"))
        base_cont = [col for col in base_cont if col not in non_feature_cols]
        cat_cols = [col for col in cat_cols if col not in non_feature_cols]
        candidates = _continuous_candidates(train_df, base_cont)
    except (FileNotFoundError, AttributeError, ModuleNotFoundError) as exc:
        logger.warning("Falling back to dtype-based column roles (%s)", exc)
        candidates, cat_cols = infer_column_roles(train_df, non_feature_cols)

    spec = build_feature_spec(
        train_df,
        continuous_candidates=candidates,
        categorical_cols=cat_cols,
        sequence_cols=list(config.sequence.feature_cols),
        seq_len=int(config.sequence.k_window),
        top_n=int(selection_cfg.top_n_continuous),
        target_col=target_col,
        sample_size=int(selection_cfg.mi_sample_size),
        seed=int(config.seed),
    )
    spec.save_json(cache_path)
    return spec
