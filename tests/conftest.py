"""Shared pytest fixtures: deterministic synthetic data for fast unit tests.

All fixtures are generated with a seeded numpy generator sourced from the
central ``config/config.yaml`` so that tests are reproducible and run in
under 15 seconds without the real IEEE-CIS dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.utils.config import Config, load_config
from src.utils.seed import seed_everything

if TYPE_CHECKING:
    import torch


@pytest.fixture(scope="session")
def config() -> Config:
    """Session-wide central configuration object."""
    return load_config()


@pytest.fixture(scope="session")
def rng(config: Config) -> np.random.Generator:
    """Session-wide seeded numpy generator (reproducible draws)."""
    seed_everything(int(config.seed))
    return np.random.default_rng(int(config.seed))


@pytest.fixture()
def n_samples() -> int:
    """Number of synthetic rows used by table fixtures."""
    return 128


@pytest.fixture()
def n_features(config: Config) -> int:
    """Feature dimensionality sourced from the central config."""
    return int(config.autoencoder.input_dim)


@pytest.fixture()
def fake_matrix(rng: np.random.Generator, n_samples: int, n_features: int) -> np.ndarray:
    """Rectangular synthetic feature matrix of shape (n_samples, n_features)."""
    return rng.standard_normal((n_samples, n_features)).astype(np.float32)


@pytest.fixture()
def fake_legit_tensor(fake_matrix: np.ndarray) -> "torch.Tensor":
    """Synthetic legit-only batch as a torch tensor (skipped without torch)."""
    torch = pytest.importorskip("torch")
    return torch.from_numpy(fake_matrix)


@pytest.fixture()
def fake_binary_labels(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Imbalanced binary labels: ~10% positives, rest negatives."""
    labels = np.zeros(n_samples, dtype=np.int64)
    n_pos = max(1, int(0.1 * n_samples))
    labels[:n_pos] = 1
    rng.shuffle(labels)
    return labels
