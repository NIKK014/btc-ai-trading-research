"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_ohlcv(
    periods: int = 600,
    freq: str = "1h",
    seed: int = 7,
    start_price: float = 30_000.0,
) -> pd.DataFrame:
    """Generate a deterministic synthetic OHLCV series.

    A geometric random walk with realistic intrabar ranges. Deterministic via
    ``seed`` so test failures are reproducible.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")

    returns = rng.normal(0.0, 0.006, periods)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])

    spread = np.abs(rng.normal(0.0, 0.004, periods)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=periods)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "turnover": close * volume,
        },
        index=index,
    ).rename_axis("timestamp")


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """A 600-bar hourly synthetic OHLCV frame."""
    return make_ohlcv()
