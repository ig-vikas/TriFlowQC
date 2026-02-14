"""
Logging utilities for TriFlowQC.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_LOGGER_NAME = "triflowqc"
_logger: Optional[logging.Logger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get the TriFlowQC logger or a child logger.

    Parameters
    ----------
    name : str or None
        Sub-logger name (e.g. 'simulator'). If None, returns the root
        TriFlowQC logger.

    Returns
    -------
    logging.Logger
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger(_LOGGER_NAME)
        _logger.setLevel(logging.INFO)

        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)

    if name:
        return _logger.getChild(name)
    return _logger


def set_log_level(level: str) -> None:
    """
    Set the global TriFlowQC log level.

    Parameters
    ----------
    level : str
        'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
    """
    logger = get_logger()
    numeric_level = getattr(logging, level.upper(), None)
    if numeric_level is None:
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


class LogContext:
    """
    Context manager for temporary log level changes.

    Usage
    -----
    >>> with LogContext('DEBUG'):
    ...     # verbose logging
    ...     pass
    """

    def __init__(self, level: str):
        self.level = level
        self._prev_level: int = logging.INFO

    def __enter__(self):
        logger = get_logger()
        self._prev_level = logger.level
        set_log_level(self.level)
        return self

    def __exit__(self, *args):
        logger = get_logger()
        logger.setLevel(self._prev_level)
