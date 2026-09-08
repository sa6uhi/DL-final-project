"""Unit tests for :mod:`src.utils.logger`."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import src.utils.logger as logger_module
from src.utils.logger import get_logger, setup_logging


@pytest.fixture()
def unconfigured_logging() -> "object":
    """Reset the once-per-process logging guard around a test.

    ``setup_logging`` deliberately configures the root logger exactly once per
    process, so a test that asserts on its side effects only sees them when it
    is the first caller. Any earlier test that exercises a CLI entry point
    would otherwise silently disarm it. Clearing the module guard -- and the
    handlers it installed -- makes these tests independent of collection order.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_flag = logger_module._configured

    for handler in saved_handlers:
        root.removeHandler(handler)
    logger_module._configured = False

    yield

    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    logger_module._configured = saved_flag


def test_get_logger_returns_named_logger() -> None:
    """get_logger returns a logger bound to the requested name."""
    logger = get_logger("tests.logger.sample")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "tests.logger.sample"


def test_setup_logging_writes_file_handler(tmp_path: Path, unconfigured_logging: object) -> None:
    """setup_logging writes formatted records into the given file."""
    log_file = tmp_path / "test.log"
    setup_logging(level="DEBUG", log_file=log_file)
    logger = get_logger("tests.logger.file")
    logger.info("informative-message")
    for handler in logger.handlers:
        handler.flush()
    content = log_file.read_text(encoding="utf-8")
    assert "informative-message" in content
    assert "tests.logger.file" in content


def test_setup_logging_is_idempotent(unconfigured_logging: object) -> None:
    """Repeated setup calls do not duplicate handlers."""
    setup_logging(level="INFO")
    root = logging.getLogger()
    count_before = len(root.handlers)
    setup_logging(level="DEBUG")
    assert len(root.handlers) == count_before


def test_log_level_uppercase_handling() -> None:
    """Level names are normalized to uppercase (case-insensitive)."""
    logger = get_logger("tests.logger.level", level="warning")
    assert logger.level == logging.WARNING


def test_logger_emits_records() -> None:
    """Loggers created by the factory emit through the root handlers."""
    logger = get_logger("tests.logger.emission")
    assert logger.isEnabledFor(logging.INFO)
    with_handlers = any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers)
    assert with_handlers
