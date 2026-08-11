"""Timeframe helpers shared by the data, backtesting and metrics layers."""

from __future__ import annotations

from config.settings import BYBIT_INTERVALS, TIMEFRAME_MINUTES

MINUTES_PER_YEAR = 365 * 24 * 60


def validate_timeframe(timeframe: str) -> str:
    """Return ``timeframe`` if supported, else raise with the valid options."""
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. "
            f"Supported: {sorted(TIMEFRAME_MINUTES)}"
        )
    return timeframe


def interval_minutes(timeframe: str) -> int:
    """Candle duration in minutes."""
    return TIMEFRAME_MINUTES[validate_timeframe(timeframe)]


def interval_ms(timeframe: str) -> int:
    """Candle duration in milliseconds."""
    return interval_minutes(timeframe) * 60_000


def bybit_interval(timeframe: str) -> str:
    """Bybit V5 ``interval`` query parameter for a project timeframe label."""
    return BYBIT_INTERVALS[validate_timeframe(timeframe)]


def bars_per_year(timeframe: str) -> float:
    """Number of candles in a calendar year.

    Crypto trades continuously, so unlike equities there is no trading-day
    adjustment. Used to annualise Sharpe and Sortino ratios.
    """
    return MINUTES_PER_YEAR / interval_minutes(timeframe)
