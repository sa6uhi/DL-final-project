"""Sanity tests for the shared fixtures in ``tests/conftest.py``."""

from __future__ import annotations

import numpy as np


def test_fake_matrix_shape(fake_matrix: np.ndarray, n_features: int) -> None:
    """fake_matrix has the requested row/feature count."""
    assert fake_matrix.dtype == np.float32
    assert fake_matrix.shape == (128, n_features)


def test_fake_matrix_is_deterministic(rng: np.random.Generator) -> None:
    """Draws from the seeded rng are reproducible."""
    a = rng.standard_normal(4)
    b = rng.standard_normal(4)
    c = np.random.default_rng(42).standard_normal(4)
    assert a.shape == b.shape == c.shape


def test_fake_binary_labels_imbalance(fake_binary_labels: np.ndarray) -> None:
    """Labels are binary and imbalanced (~10% positives)."""
    assert set(np.unique(fake_binary_labels)).issubset({0, 1})
    assert fake_binary_labels.mean() < 0.25


def test_config_fixture_has_seed(config) -> None:
    """Config fixture exposes the central seed value."""
    assert int(config.seed) == 42
