"""Shared logging configuration for IDA Chat modules."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FILE = Path("/tmp/ida-chat.log")


def configure_logger(name: str = "ida-chat") -> logging.Logger:
    """Return a consistently configured logger for this project."""
    logger = logging.getLogger(name)

    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == LOG_FILE
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


logger = configure_logger()
