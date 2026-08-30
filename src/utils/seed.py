"""Deterministic random seeding utilities.

Use :func:`seed_everything` at every training entry point so that all runs
are reproducible (``seed_everything(42)`` is the project-wide default).
"""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEED: int = 42


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed all random sources for deterministic reproduction.

    Seeds Python's ``random`` and ``numpy``, sets ``PYTHONHASHSEED``, and --
    when PyTorch is installed -- seeds ``torch``, all CUDA devices, and
    enables deterministic cuDNN backend settings.

    Args:
        seed: Integer seed to use. Defaults to 42.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        import torch
    except ImportError:
        logger.debug("PyTorch not installed; skipped torch seeding.")
    else:
        torch.manual_seed(seed)
        random_seed_tensor = torch.initial_seed()
        assert random_seed_tensor == seed, "Torch manual seed was not applied."
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)

    logger.info("Seeded all random sources with seed=%d", seed)
