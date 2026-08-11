"""Momentum strategies.

Premise: the *rate of change* of price carries information beyond its
direction. Where a trend strategy asks "which way is the market pointing",
a momentum strategy asks "is the push accelerating, and is volume behind it".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import pandas as pd

from src.indicators.indicators import IndicatorSpec
from src.strategies.base import Strategy, StrategyParams, state_machine


@dataclass(frozen=True)
class StochRsiMacdParams(StrategyParams):
    stoch_rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_period: int = 20
    #: Stochastic RSI must emerge from this zone, i.e. momentum turning up
    #: from a washed-out state rather than already extended.
    oversold: float = 20.0
    overbought: float = 80.0
    #: Volume must be at least this multiple of its own recent average.
    min_volume_ratio: float = 1.0
    #: Exit when Stochastic RSI reaches the opposite extreme.
    exit_long_level: float = 80.0
    exit_short_level: float = 20.0


class StochRsiMacdMomentum(Strategy):
    """Stochastic RSI turning out of an extreme, confirmed by MACD and volume."""

    name = "stochrsi_macd_momentum"
    methodology = "momentum"
    params_class = StochRsiMacdParams
    description = """
    Enters when Stochastic RSI crosses up out of oversold (or down out of
    overbought) while the MACD histogram agrees with the direction and volume
    is above its recent average. Exits at the opposite Stochastic RSI extreme.

    Unlike the trend strategies this is an event-driven system with an explicit
    exit, so it holds positions for a bounded time. The volume filter is the
    part worth scrutinising in the results: momentum on thin volume is usually
    noise, and if removing the filter improves performance that is itself an
    interesting finding.
    """

    @property
    def indicator_spec(self) -> IndicatorSpec:
        p: StochRsiMacdParams = self.params
        return IndicatorSpec(
            stoch_rsi_period=p.stoch_rsi_period,
            macd_fast=p.macd_fast,
            macd_slow=p.macd_slow,
            macd_signal=p.macd_signal,
            volume_period=p.volume_period,
        )

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: StochRsiMacdParams = self.params
        k = frame["stochrsi_k"]
        d = frame["stochrsi_d"]
        histogram = frame["macd_hist"]
        volume_ratio = frame["volume_ratio"]

        # Crossovers use the previous bar's values only - strictly causal.
        cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        cross_down = (k < d) & (k.shift(1) >= d.shift(1))
        active_volume = volume_ratio >= p.min_volume_ratio

        entry_long = cross_up & (k.shift(1) < p.oversold) & (histogram > 0) & active_volume
        entry_short = cross_down & (k.shift(1) > p.overbought) & (histogram < 0) & active_volume

        exit_long = (k > p.exit_long_level) | (histogram < 0)
        exit_short = (k < p.exit_short_level) | (histogram > 0)

        return state_machine(entry_long, exit_long, entry_short, exit_short)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {
            "stoch_rsi_period": (14, 21),
            "oversold": (15.0, 20.0, 25.0),
            "overbought": (75.0, 80.0, 85.0),
            "min_volume_ratio": (0.0, 1.0, 1.3),
        }

    @classmethod
    def is_valid(cls, params: StochRsiMacdParams) -> bool:
        return params.oversold < params.overbought and params.macd_fast < params.macd_slow
