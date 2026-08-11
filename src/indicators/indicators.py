"""Technical indicators, implemented directly in pandas.

Why hand-written
----------------
``pandas-ta`` has NumPy 2.x incompatibilities and an uncertain maintenance
future; ``TA-Lib`` requires a C toolchain build that frequently fails on
macOS. More importantly, every indicator here is a source of look-ahead bias
if implemented carelessly, and a from-scratch implementation can be *proved*
causal (see ``tests/test_indicators.py``).

Causality contract
------------------
**Every function in this module returns a value at index ``t`` computed only
from data at or before ``t``.** There is no ``shift(-n)`` anywhere in this
file, and there never should be. This is enforced by an automated test that
truncates the input and checks that historical values are unchanged.

Deliberately excluded
---------------------
* **Ichimoku** - the Chikou Span is the close shifted *backwards*, so reading
  it at time ``t`` reads price at ``t + 26``. Direct look-ahead.
* **Fibonacci retracements / classical support-resistance** - both are
  normally derived from swing highs and lows over a window that includes
  future bars.

Where a "levels" style feature is genuinely useful (the breakout strategy),
:func:`donchian_high` and :func:`donchian_low` provide a strictly causal
equivalent: the extreme of the *previous* N bars, excluding the current one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "IndicatorSpec",
    "sma",
    "ema",
    "wilder_smooth",
    "rsi",
    "macd",
    "bollinger_bands",
    "true_range",
    "atr",
    "adx",
    "session_vwap",
    "stoch_rsi",
    "obv",
    "volume_ratio",
    "log_returns",
    "realised_volatility",
    "donchian_high",
    "donchian_low",
    "add_indicators",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series, returning NaN rather than +/-inf on a zero divisor."""
    return numerator / denominator.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over ``period`` bars."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.

    Uses ``adjust=False`` so the result is the standard recursive EMA that
    charting platforms display, and so the value at ``t`` depends only on the
    value at ``t-1`` and the current observation.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA), the basis of RSI, ATR and ADX.

    Equivalent to an EMA with ``alpha = 1 / period``.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, 0-100, using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)

    rs = _safe_divide(avg_gain, avg_loss)
    result = 100.0 - (100.0 / (1.0 + rs))
    # A zero average loss means unbroken gains -> RSI is 100 by definition.
    return result.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def stoch_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Stochastic RSI: where RSI sits within its own recent range, 0-100."""
    rsi_series = rsi(close, rsi_period)
    lowest = rsi_series.rolling(stoch_period, min_periods=stoch_period).min()
    highest = rsi_series.rolling(stoch_period, min_periods=stoch_period).max()

    raw_k = 100.0 * _safe_divide(rsi_series - lowest, highest - lowest)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return pd.DataFrame({"stochrsi_k": k, "stochrsi_d": d})


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands plus %B and bandwidth.

    ``bb_pct_b`` is 0 at the lower band and 1 at the upper band, which makes
    it a far better ML feature than the raw band levels (it is scale-free).
    ``bb_width`` is a volatility-regime proxy: narrow bands precede expansion.
    """
    middle = sma(close, period)
    deviation = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + num_std * deviation
    lower = middle - num_std * deviation
    return pd.DataFrame(
        {
            "bb_mid": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": _safe_divide(close - lower, upper - lower),
            "bb_width": _safe_divide(upper - lower, middle),
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: the greatest of the current bar's span and the two gaps."""
    previous_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed. Drives stop placement and the
    volatility-scaled ML barriers."""
    return wilder_smooth(true_range(high, low, close), period)


def realised_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns (per bar, not annualised)."""
    return log_returns(close).rolling(period, min_periods=period).std(ddof=0)


# ---------------------------------------------------------------------------
# Trend strength
# ---------------------------------------------------------------------------


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Average Directional Index with the +DI / -DI components.

    ADX measures trend *strength* regardless of direction, which is what makes
    it useful as a filter: trend-following rules should be gated on ADX being
    high, mean-reversion rules on it being low.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr_series = wilder_smooth(true_range(high, low, close), period)
    plus_di = 100.0 * _safe_divide(wilder_smooth(plus_dm, period), atr_series)
    minus_di = 100.0 * _safe_divide(wilder_smooth(minus_dm, period), atr_series)

    dx = 100.0 * _safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    return pd.DataFrame(
        {
            "adx": wilder_smooth(dx, period),
            "di_plus": plus_di,
            "di_minus": minus_di,
        }
    )


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def session_vwap(frame: pd.DataFrame, anchor: str = "D") -> pd.Series:
    """Volume Weighted Average Price, re-anchored each session.

    VWAP is only meaningful relative to an anchor; a cumulative-since-2020
    VWAP is a meaningless number. Anchoring daily (UTC) matches how intraday
    traders actually use it. Purely cumulative within the session, so it is
    causal by construction.
    """
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    session = frame.index.floor(anchor)

    cumulative_pv = (typical_price * frame["volume"]).groupby(session).cumsum()
    cumulative_volume = frame["volume"].groupby(session).cumsum()
    return _safe_divide(cumulative_pv, cumulative_volume)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by the direction of close."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume relative to its own recent average.

    Scale-free, so it transfers across price regimes - unlike raw volume,
    which trends upward over years and would let a model infer the date.
    """
    return _safe_divide(volume, sma(volume, period))


# ---------------------------------------------------------------------------
# Price action
# ---------------------------------------------------------------------------


def log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Log return over ``period`` bars."""
    return np.log(close / close.shift(period))


def donchian_high(high: pd.Series, period: int = 20) -> pd.Series:
    """Highest high of the ``period`` bars **before** the current one.

    Excluding the current bar is what makes this a usable breakout level: a
    channel that includes the current bar can never be broken by it.
    """
    return high.rolling(period, min_periods=period).max().shift(1)


def donchian_low(low: pd.Series, period: int = 20) -> pd.Series:
    """Lowest low of the ``period`` bars **before** the current one."""
    return low.rolling(period, min_periods=period).min().shift(1)


# ---------------------------------------------------------------------------
# Standard feature set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorSpec:
    """Periods for the standard indicator set.

    Grid search varies these, so nothing downstream should assume the default
    values.
    """

    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_rsi_period: int = 14
    volume_period: int = 20
    donchian_period: int = 20
    volatility_period: int = 20


def add_indicators(
    frame: pd.DataFrame,
    spec: IndicatorSpec | None = None,
    *,
    dropna: bool = False,
) -> pd.DataFrame:
    """Attach the standard indicator set to an OHLCV frame.

    Args:
        frame: OHLCV DataFrame indexed by UTC candle open time.
        spec: Indicator periods. Defaults to :class:`IndicatorSpec`.
        dropna: Drop the leading warm-up rows where long-period indicators
            are still undefined. Leave ``False`` for backtesting (so the index
            still aligns with the price series) and set ``True`` when building
            an ML feature matrix.

    Returns:
        A copy of ``frame`` with indicator columns appended. Scale-free
        columns (``*_pct``, ``*_ratio``, ``bb_pct_b``) are the ones intended
        as ML features; raw price-level columns are for strategy rules and
        plotting.
    """
    spec = spec or IndicatorSpec()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")

    out = frame.copy()
    high, low, close, volume = out["high"], out["low"], out["close"], out["volume"]

    # Trend
    out[f"ema_{spec.ema_fast}"] = ema(close, spec.ema_fast)
    out[f"ema_{spec.ema_mid}"] = ema(close, spec.ema_mid)
    out[f"ema_{spec.ema_slow}"] = ema(close, spec.ema_slow)
    out[f"ema_{spec.ema_trend}"] = ema(close, spec.ema_trend)
    out["ema_fast_slow_spread_pct"] = _safe_divide(
        out[f"ema_{spec.ema_fast}"] - out[f"ema_{spec.ema_mid}"], close
    )
    out["dist_from_trend_pct"] = _safe_divide(close - out[f"ema_{spec.ema_trend}"], close)

    # Momentum
    out["rsi"] = rsi(close, spec.rsi_period)
    out = out.join(macd(close, spec.macd_fast, spec.macd_slow, spec.macd_signal))
    out["macd_hist_pct"] = _safe_divide(out["macd_hist"], close)
    out = out.join(stoch_rsi(close, spec.stoch_rsi_period, spec.stoch_rsi_period))

    # Volatility
    out = out.join(bollinger_bands(close, spec.bb_period, spec.bb_std))
    out["atr"] = atr(high, low, close, spec.atr_period)
    out["atr_pct"] = _safe_divide(out["atr"], close)
    out["realised_vol"] = realised_volatility(close, spec.volatility_period)

    # Trend strength
    out = out.join(adx(high, low, close, spec.adx_period))

    # Volume
    out["vwap"] = session_vwap(out)
    out["vwap_dist_pct"] = _safe_divide(close - out["vwap"], close)
    out["obv"] = obv(close, volume)
    out["obv_slope"] = out["obv"].diff(spec.volume_period)
    out["volume_ratio"] = volume_ratio(volume, spec.volume_period)

    # Price action
    out["ret_1"] = log_returns(close, 1)
    out["ret_4"] = log_returns(close, 4)
    out["donchian_high"] = donchian_high(high, spec.donchian_period)
    out["donchian_low"] = donchian_low(low, spec.donchian_period)
    out["donchian_pos"] = _safe_divide(
        close - out["donchian_low"], out["donchian_high"] - out["donchian_low"]
    )

    if dropna:
        out = out.dropna()
    return out


#: Scale-free columns suitable as ML features. Raw price levels are excluded
#: deliberately: a model trained on absolute BTC price learns the calendar,
#: not the market, and cannot generalise out of sample.
ML_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi",
    "macd_hist_pct",
    "stochrsi_k",
    "stochrsi_d",
    "bb_pct_b",
    "bb_width",
    "atr_pct",
    "realised_vol",
    "adx",
    "di_plus",
    "di_minus",
    "vwap_dist_pct",
    "volume_ratio",
    "ema_fast_slow_spread_pct",
    "dist_from_trend_pct",
    "ret_1",
    "ret_4",
    "donchian_pos",
)
