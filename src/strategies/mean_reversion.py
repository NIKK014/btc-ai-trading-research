"""Mean-reversion strategies.

Premise: price stretched far from a reference tends to snap back. The critical
design point is that mean reversion is only valid in a *range* - fading a
strong trend is how accounts die. Both strategies here are therefore gated on
a trend-strength ceiling, which is exactly the kind of methodology comparison
this project exists to test: the same RSI that confirms a trend signal in
``trend.py`` is used to fade price here, and only the regime filter
distinguishes the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from src.indicators.indicators import IndicatorSpec
from src.strategies.base import Strategy, StrategyParams, state_machine


@dataclass(frozen=True)
class RsiBollingerParams(StrategyParams):
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    adx_period: int = 14
    #: Mean reversion is only permitted below this trend strength.
    adx_max: float = 25.0


class RsiBollingerReversion(Strategy):
    """Fade Bollinger Band extremes when RSI agrees and no trend is present."""

    name = "rsi_bollinger_reversion"
    methodology = "mean_reversion"
    params_class = RsiBollingerParams
    description = """
    Buys when price closes below the lower Bollinger Band with RSI oversold,
    sells short at the mirror condition, and exits when price returns to the
    middle band - the statistical mean the whole premise rests on.

    The ADX ceiling is the load-bearing part of this strategy. Without it the
    rules will happily short every step of a bull market. Its value is in the
    parameter grid so the results can show how much it matters.
    """

    @property
    def indicator_spec(self) -> IndicatorSpec:
        p: RsiBollingerParams = self.params
        return IndicatorSpec(
            bb_period=p.bb_period,
            bb_std=p.bb_std,
            rsi_period=p.rsi_period,
            adx_period=p.adx_period,
        )

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: RsiBollingerParams = self.params
        close = frame["close"]
        rsi = frame["rsi"]
        ranging = frame["adx"] < p.adx_max

        entry_long = (close < frame["bb_lower"]) & (rsi < p.oversold) & ranging
        entry_short = (close > frame["bb_upper"]) & (rsi > p.overbought) & ranging

        # Exit at the mean, which is the event the strategy was predicting.
        exit_long = close >= frame["bb_mid"]
        exit_short = close <= frame["bb_mid"]

        return state_machine(entry_long, exit_long, entry_short, exit_short)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {
            "bb_period": (14, 20, 30),
            "bb_std": (2.0, 2.5),
            "rsi_period": (7, 14),
            "oversold": (20.0, 25.0, 30.0),
            "adx_max": (20.0, 25.0, 100.0),
        }

    @classmethod
    def is_valid(cls, params: RsiBollingerParams) -> bool:
        return params.oversold < params.overbought


@dataclass(frozen=True)
class VwapStretchParams(StrategyParams):
    rsi_period: int = 14
    atr_period: int = 14
    volume_period: int = 20
    #: Entry requires price to be this many ATRs away from session VWAP.
    stretch_atr: float = 1.5
    oversold: float = 35.0
    overbought: float = 65.0
    min_volume_ratio: float = 1.0
    adx_period: int = 14
    adx_max: float = 30.0


class VwapStretchReversion(Strategy):
    """Fade price stretched from session VWAP, confirmed by RSI and volume."""

    name = "vwap_stretch_reversion"
    methodology = "mean_reversion"
    params_class = VwapStretchParams
    description = """
    Session VWAP is the intraday reference price most institutional flow is
    measured against, which is part of why price tends to revert to it. This
    strategy enters when price is stretched more than a set number of ATRs from
    VWAP, RSI confirms the extreme, and volume shows the move is real rather
    than drift. It exits when price touches VWAP again.

    Measuring the stretch in ATRs rather than percent matters: a fixed
    percentage threshold means something completely different at 15m and 4h,
    and would make the timeframe comparison meaningless.
    """

    @property
    def indicator_spec(self) -> IndicatorSpec:
        p: VwapStretchParams = self.params
        return IndicatorSpec(
            rsi_period=p.rsi_period,
            atr_period=p.atr_period,
            volume_period=p.volume_period,
            adx_period=p.adx_period,
        )

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: VwapStretchParams = self.params
        close = frame["close"]
        vwap = frame["vwap"]
        atr = frame["atr"]
        rsi = frame["rsi"]

        stretch = (close - vwap) / atr.replace(0.0, np.nan)
        confirmed_volume = frame["volume_ratio"] >= p.min_volume_ratio
        ranging = frame["adx"] < p.adx_max

        entry_long = (stretch <= -p.stretch_atr) & (rsi < p.oversold) & confirmed_volume & ranging
        entry_short = (stretch >= p.stretch_atr) & (rsi > p.overbought) & confirmed_volume & ranging

        exit_long = close >= vwap
        exit_short = close <= vwap

        return state_machine(entry_long, exit_long, entry_short, exit_short)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {
            "stretch_atr": (1.0, 1.5, 2.0),
            "rsi_period": (7, 14),
            "oversold": (30.0, 35.0),
            "overbought": (65.0, 70.0),
            "min_volume_ratio": (0.0, 1.0),
        }

    @classmethod
    def is_valid(cls, params: VwapStretchParams) -> bool:
        return params.oversold < params.overbought and params.stretch_atr > 0
