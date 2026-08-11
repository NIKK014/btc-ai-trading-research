"""Trend-following strategies.

Premise: markets that are moving tend to keep moving. These rules aim to hold
a position for as long as the trend persists, accepting a low win rate in
exchange for large winners. Both are gated on a trend-strength filter, because
trend-following in a range is the classic way to bleed to death by a thousand
whipsaws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import pandas as pd

from src.indicators.indicators import IndicatorSpec
from src.strategies.base import Strategy, StrategyParams, combine


@dataclass(frozen=True)
class EmaRsiParams(StrategyParams):
    ema_fast: int = 12
    ema_slow: int = 26
    rsi_period: int = 14
    #: RSI must be on the correct side of this level to confirm the trend.
    rsi_midline: float = 50.0
    #: Refuse to enter when RSI is already at an extreme - chasing a move that
    #: has already run is how trend systems buy the top.
    rsi_upper_block: float = 78.0
    rsi_lower_block: float = 22.0
    adx_min: float = 0.0


class EmaRsiTrend(Strategy):
    """EMA crossover with RSI confirmation."""

    name = "ema_rsi_trend"
    methodology = "trend"
    params_class = EmaRsiParams
    description = """
    Long while the fast EMA is above the slow EMA and RSI confirms momentum is
    on the same side of its midline. Short on the mirror condition. The state
    is a regime, not an event, so the position is held for as long as the two
    conditions agree.

    RSI serves two purposes here: it filters out crossovers that happen with no
    momentum behind them, and the extreme blocks stop the system entering after
    a move has already become exhausted.
    """

    @property
    def indicator_spec(self) -> IndicatorSpec:
        p: EmaRsiParams = self.params
        return IndicatorSpec(
            ema_fast=p.ema_fast,
            ema_mid=p.ema_slow,
            rsi_period=p.rsi_period,
        )

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: EmaRsiParams = self.params
        fast = frame[f"ema_{p.ema_fast}"]
        slow = frame[f"ema_{p.ema_slow}"]
        rsi = frame["rsi"]
        adx = frame["adx"]

        trending = adx >= p.adx_min

        long_condition = (
            (fast > slow) & (rsi > p.rsi_midline) & (rsi < p.rsi_upper_block) & trending
        )
        short_condition = (
            (fast < slow) & (rsi < p.rsi_midline) & (rsi > p.rsi_lower_block) & trending
        )
        return combine(long_condition, short_condition)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {
            "ema_fast": (9, 12, 20, 26),
            "ema_slow": (21, 26, 50, 100),
            "rsi_period": (7, 14, 21),
            "adx_min": (0.0, 20.0),
        }

    @classmethod
    def is_valid(cls, params: EmaRsiParams) -> bool:
        return params.ema_fast < params.ema_slow


@dataclass(frozen=True)
class MacdAdxParams(StrategyParams):
    trend_ema: int = 200
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    #: Only trade when the market is actually trending. 25 is the conventional
    #: threshold; it is in the grid because conventions deserve testing.
    adx_min: float = 25.0


class MacdAdxTrend(Strategy):
    """Long-term EMA regime filter, MACD trigger, ADX strength gate."""

    name = "macd_adx_trend"
    methodology = "trend"
    params_class = MacdAdxParams
    description = """
    A three-layer trend system. The long EMA defines which side of the market
    we are allowed to trade, the MACD histogram provides the directional
    trigger, and ADX gates everything on the trend being strong enough to be
    worth trading.

    This is the strategy most likely to beat buy-and-hold in a sustained trend
    and most likely to be flat for long stretches, which is the intended
    trade-off. Compare its trade count against the mean-reversion strategies.
    """

    @property
    def indicator_spec(self) -> IndicatorSpec:
        p: MacdAdxParams = self.params
        return IndicatorSpec(
            ema_trend=p.trend_ema,
            macd_fast=p.macd_fast,
            macd_slow=p.macd_slow,
            macd_signal=p.macd_signal,
            adx_period=p.adx_period,
        )

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: MacdAdxParams = self.params
        close = frame["close"]
        trend = frame[f"ema_{p.trend_ema}"]
        histogram = frame["macd_hist"]
        adx = frame["adx"]

        strong = adx >= p.adx_min
        long_condition = (close > trend) & (histogram > 0) & strong
        short_condition = (close < trend) & (histogram < 0) & strong
        return combine(long_condition, short_condition)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {
            "trend_ema": (100, 200),
            "macd_fast": (8, 12),
            "macd_slow": (21, 26),
            "adx_min": (20.0, 25.0, 30.0),
        }

    @classmethod
    def is_valid(cls, params: MacdAdxParams) -> bool:
        return params.macd_fast < params.macd_slow
