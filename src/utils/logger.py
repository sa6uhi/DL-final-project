"""Centralized structured logging.

Every module must obtain its logger through :func:`get_logger`; raw
``print()`` statements are forbidden inside ``src/``. Logging is configured
exactly once per process via :func:`setup_logging`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured: bool = False


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the root logger exactly once per process.

    Adds a stderr stream handler with a consistent structured format and,
    optionally, a rotating file handler. Subsequent calls are no-ops.

    Args:
        level: Logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``...).
        log_file: Optional path to a log file.

    Returns:
        None
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger name, conventionally ``__name__`` of the module.
        level: Logging level for this logger.

    Returns:
        The :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
