"""Unit tests for :mod:`src.utils.seed`."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from src.utils.seed import DEFAULT_SEED, seed_everything


def test_seed_everything_seeds_python_random() -> None:
    """Python random produces identical sequences after identical seeds."""
    seed_everything(7)
    a = [random.random() for _ in range(5)]
    seed_everything(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_seeds_numpy() -> None:
    """numpy random produces identical sequences after identical seeds."""
    seed_everything(11)
    a = np.random.randn(10)
    seed_everything(11)
    b = np.random.randn(10)
    assert np.array_equal(a, b)


def test_seed_everything_sets_env_hash_seed() -> None:
    """PYTHONHASHSEED environment variable is set."""
    seed_everything(42)
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_seed_everything_default_value() -> None:
    """Calling without arguments uses the project default seed."""
    seed_everything()
    assert DEFAULT_SEED == 42
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_seed_everything_different_seeds_differ() -> None:
    """Different seeds produce different numpy draws."""
    seed_everything(1)
    a = np.random.randn(3)
    seed_everything(2)
    b = np.random.randn(3)
    assert not np.array_equal(a, b)


def test_seed_everything_pytorch_when_available() -> None:
    """torch.manual_seed is applied when PyTorch is installed."""
    torch = pytest.importorskip("torch")
    seed_everything(123)
    assert torch.initial_seed() == 123


def test_seed_everything_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing PyTorch is skipped gracefully instead of raising."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    seed_everything(42)
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_seed_everything_cuda_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CUDA deterministic path runs when a GPU is reported."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seed_everything(7)
    assert torch.initial_seed() == 7
