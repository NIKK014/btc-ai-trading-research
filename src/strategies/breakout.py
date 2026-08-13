"""Breakout strategies.

Premise: most of the time price goes nowhere, and the occasional decisive
break out of a range is where the money is. Breakout systems have low win
rates by construction - most breaks fail - so they depend entirely on the
winners being much larger than the losers. Watch the profit factor and the
average-win-to-average-loss ratio in the results, not the win rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import pandas as pd

from src.indicators.indicators import IndicatorSpec
from src.strategies.base import Strategy, StrategyParams, state_machine


@dataclass(frozen=True)
class DonchianBreakoutParams(StrategyParams):
    #: Lookback for the breakout level, excluding the current bar.
    channel_period: int = 20
    #: Shorter opposite channel used as the exit, Turtle-style.
    exit_period: int = 10
    volume_period: int = 20
    min_volume_ratio: float = 1.2
    atr_period: int = 14


class DonchianVolumeBreakout(Strategy):
    """Break of the prior N-bar high or low, confirmed by volume."""

    name = "donchian_volume_breakout"
    methodology = "breakout"
    params_class = DonchianBreakoutParams
    description = """
    Goes long when price closes above the highest high of the preceding N bars
    and volume confirms the move; short on the mirror condition. Exits on a
    break of a shorter opposite channel.

    Donchian channels are used here instead of hand-drawn support and
    resistance because they are strictly causal: the level is computed from the
    N bars *before* the current one, so the current bar is able to break it.
    Classical support/resistance and Fibonacci levels are normally derived from
    swing points identified with hindsight, which quietly leaks the future into
    the signal.
    """

    @property
    def indicator_spec(self) -> IndicatorSpec:
        p: DonchianBreakoutParams = self.params
        return IndicatorSpec(
            donchian_period=p.channel_period,
            volume_period=p.volume_period,
            atr_period=p.atr_period,
        )

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: DonchianBreakoutParams = self.params
        close = frame["close"]

        # Exit channel is shorter than the entry channel, so it must be
        # recomputed here rather than reused from the indicator frame.
        exit_high = frame["high"].rolling(p.exit_period, min_periods=p.exit_period).max().shift(1)
        exit_low = frame["low"].rolling(p.exit_period, min_periods=p.exit_period).min().shift(1)

        confirmed = frame["volume_ratio"] >= p.min_volume_ratio

        entry_long = (close > frame["donchian_high"]) & confirmed
        entry_short = (close < frame["donchian_low"]) & confirmed
        exit_long = close < exit_low
        exit_short = close > exit_high

        return state_machine(entry_long, exit_long, entry_short, exit_short)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {
            "channel_period": (20, 30, 55),
            "exit_period": (10, 20),
            "min_volume_ratio": (1.0, 1.2, 1.5),
        }

    @classmethod
    def is_valid(cls, params: DonchianBreakoutParams) -> bool:
        # An exit channel at least as long as the entry channel would exit the
        # position on the bar it was entered.
        return params.exit_period < params.channel_period


@dataclass(frozen=True)
class SqueezeBreakoutParams(StrategyParams):
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_period: int = 20
    #: Window over which "narrow" is judged. Must be a rolling window, not the
    #: whole series - see the note in the class docstring.
    squeeze_lookback: int = 100
    #: Bandwidth must be in the bottom this-fraction of its recent range.
    squeeze_quantile: float = 0.25
    exit_period: int = 10


class BollingerSqueezeBreakout(Strategy):
    """Volatility contraction followed by a band break."""

    name = "bollinger_squeeze_breakout"
    methodology = "breakout"
    params_class = SqueezeBreakoutParams
    description = """
    Waits for Bollinger bandwidth to compress into the bottom quartile of its
    recent range - a volatility squeeze - and then trades the direction of the
    break out of the bands. Exits on a short opposite Donchian channel.

    Leakage note worth reading: "narrow bands" is a relative judgement, and the
    obvious implementation compares bandwidth against a quantile of the *entire*
    series. That silently uses the future, because the 2026 distribution is not
    knowable in 2021. The threshold here is a rolling quantile over the
    preceding window only, which is a good example of how a leak can hide in a
    single innocuous line.
    """

    @property
    def indicator_spec(self) -> IndicatorSpec:
        p: SqueezeBreakoutParams = self.params
        return IndicatorSpec(
            bb_period=p.bb_period,
            bb_std=p.bb_std,
            atr_period=p.atr_period,
            volume_period=p.volume_period,
        )

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: SqueezeBreakoutParams = self.params
        close = frame["close"]
        width = frame["bb_width"]

        # Rolling quantile: uses only the preceding `squeeze_lookback` bars.
        threshold = width.rolling(p.squeeze_lookback, min_periods=p.squeeze_lookback).quantile(
            p.squeeze_quantile
        )
        # Squeeze condition is evaluated on the previous bar, so the breakout
        # bar itself does not need to still be compressed.
        squeezed = (width <= threshold).astype("float64").shift(1).fillna(0.0).astype(bool)

        entry_long = squeezed & (close > frame["bb_upper"])
        entry_short = squeezed & (close < frame["bb_lower"])

        exit_high = frame["high"].rolling(p.exit_period, min_periods=p.exit_period).max().shift(1)
        exit_low = frame["low"].rolling(p.exit_period, min_periods=p.exit_period).min().shift(1)
        exit_long = close < exit_low
        exit_short = close > exit_high

        return state_machine(entry_long, exit_long, entry_short, exit_short)

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {
            "bb_period": (20, 30),
            "squeeze_lookback": (50, 100, 200),
            "squeeze_quantile": (0.15, 0.25, 0.35),
        }
