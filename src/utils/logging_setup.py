"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.settings import LOG_LEVEL, PATHS

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(logfile: str | Path | None = "app.log") -> None:
    """Configure root logging once, to stdout and optionally to a file."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        PATHS.ensure()
        handlers.append(logging.FileHandler(PATHS.logs / logfile, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
    )
    # These are noisy at DEBUG and never useful to us.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    setup_logging()
    return logging.getLogger(name)
