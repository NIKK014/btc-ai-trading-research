"""Benchmark "strategies".

Every result in this project is meaningless without something to compare it
against. A strategy returning +40% over six years sounds impressive until you
notice that holding BTC returned considerably more with less effort and no
execution risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import pandas as pd

from src.strategies.base import Signal, Strategy, StrategyParams


@dataclass(frozen=True)
class BuyAndHoldParams(StrategyParams):
    #: ``1`` for long BTC, ``-1`` for a permanently short benchmark.
    direction: int = 1


class BuyAndHold(Strategy):
    """Hold a single position for the entire period.

    Run with stops disabled and full-notional sizing (see
    :class:`~src.backtesting.engine.BacktestEngine`), so it is a genuine
    buy-and-hold, not a stopped-out approximation of one. It still pays entry
    and exit fees, because a real investor would.
    """

    name = "buy_and_hold"
    methodology = "benchmark"
    params_class = BuyAndHoldParams
    description = """
    The benchmark every other strategy must beat on a risk-adjusted basis. If
    no strategy in the leaderboard beats it after fees, that is the headline
    result of the project and should be reported as such rather than buried.
    """

    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        p: BuyAndHoldParams = self.params
        direction = Signal.LONG if p.direction >= 0 else Signal.SHORT
        signals = pd.Series(direction, index=frame.index, dtype="int8")
        # Stay flat until indicators have warmed up, so the benchmark is
        # measured over exactly the same bars as every other strategy.
        warmup = frame["close"].isna() | frame.get("atr", frame["close"]).isna()
        signals[warmup] = Signal.FLAT
        return signals

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        return {}
