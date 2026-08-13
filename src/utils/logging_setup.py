"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

from config.settings import LOG_LEVEL, PATHS


def _silence_accelerate_matmul_warnings() -> None:
    """Suppress a known spurious NumPy warning on Apple silicon.

    NumPy built against Apple's Accelerate BLAS reports divide-by-zero,
    overflow and invalid-value warnings for matmuls above roughly 14x14 even
    when the result is correct - Accelerate leaves floating-point exception
    flags set from its SIMD paths. The same code under OpenBLAS is silent and
    produces identical numbers.

    Matched narrowly on ``matmul`` so real numerical problems still surface.
    Tracking: numpy#28687, numpy#29820, scikit-learn#31395.
    """
    for message in (
        "divide by zero encountered in matmul",
        "overflow encountered in matmul",
        "invalid value encountered in matmul",
    ):
        warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)

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
    _silence_accelerate_matmul_warnings()
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    setup_logging()
    return logging.getLogger(name)
