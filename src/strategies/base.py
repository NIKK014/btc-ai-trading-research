"""Strategy interface.

A strategy turns an indicator frame into a desired position direction per bar:
+1 long, -1 short, 0 flat.

Emitting a target state rather than buy/sell events keeps strategies stateless
and pushes execution concerns - fill timing, stops, re-entry rules - into the
engine. The same signal series is then replayed identically by the backtester,
the ML filter and the live trader, which is what makes Systems A, B and C
comparable.

Signals at bar t may only use indicator values at t or earlier; the engine
fills at the open of t+1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import pandas as pd

from src.indicators.indicators import IndicatorSpec, add_indicators


class Signal:
    """Desired position direction."""

    LONG = 1
    SHORT = -1
    FLAT = 0


SIGNAL_NAMES = {1: "LONG", -1: "SHORT", 0: "FLAT"}


@dataclass(frozen=True)
class StrategyParams:
    """Base class for strategy parameter sets.

    Subclasses declare their own tunables. Every field is a candidate for
    grid search, so nothing downstream may assume a default value.
    """

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> "StrategyParams":
        return replace(self, **changes)

    def label(self) -> str:
        """Compact one-line rendering, used in leaderboards and filenames."""
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


class Strategy(ABC):
    """Base class for all trading strategies."""

    #: Short identifier used in results tables and the database.
    name: str = "unnamed"
    #: One of: trend, momentum, mean_reversion, breakout, benchmark.
    methodology: str = "unspecified"
    #: Human-readable description of the trading logic, surfaced in the
    #: dashboard and the written methodology.
    description: str = ""

    params_class: type = StrategyParams

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params if params is not None else self.params_class()

    # -- indicator requirements ------------------------------------------

    @property
    def indicator_spec(self) -> IndicatorSpec:
        """Indicator periods this strategy needs, derived from its params.

        Overridden by strategies whose parameters change indicator periods,
        so that a grid search over e.g. RSI period actually recomputes RSI.
        """
        return IndicatorSpec()

    def prepare(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Attach the indicators this strategy needs to a raw OHLCV frame."""
        return add_indicators(ohlcv, self.indicator_spec)

    # -- signal generation -----------------------------------------------

    @abstractmethod
    def _signals(self, frame: pd.DataFrame) -> pd.Series:
        """Return the raw desired direction per bar. Implemented by subclasses."""

    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        """Desired position direction at each bar close.

        Args:
            frame: OHLCV plus the indicators from :meth:`prepare`.

        Returns:
            Integer Series aligned to ``frame.index`` with values in
            ``{-1, 0, 1}``. Bars where indicators are still warming up are
            ``0`` (flat), never forward-filled from a later value.
        """
        signals = self._signals(frame)
        signals = signals.reindex(frame.index).fillna(Signal.FLAT).astype("int8")

        invalid = set(signals.unique()) - {-1, 0, 1}
        if invalid:
            raise ValueError(f"{self.name} emitted invalid signal values: {invalid}")
        return signals.rename("signal")

    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Convenience: add indicators and attach the signal column."""
        frame = self.prepare(ohlcv)
        frame["signal"] = self.generate_signals(frame)
        return frame

    # -- optimisation -----------------------------------------------------

    @classmethod
    def param_grid(cls) -> Dict[str, Sequence[Any]]:
        """Candidate values per parameter for grid search.

        Kept deliberately small. A grid of a few dozen configurations that
        finishes is worth more than thousands that do not, and every extra
        configuration increases the chance the leaderboard winner is lucky
        rather than good.
        """
        return {}

    @classmethod
    def is_valid(cls, params: StrategyParams) -> bool:
        """Whether a parameter combination is internally coherent.

        Grid search takes the cartesian product of every parameter, which
        inevitably produces nonsense - a "fast" EMA slower than the slow one,
        an oversold threshold above the overbought one. Rejecting these up
        front stops the search budget being spent on configurations that could
        never trade sensibly, and stops them polluting the distribution used to
        judge selection bias.
        """
        return True

    @classmethod
    def iter_param_sets(cls) -> Iterator[StrategyParams]:
        """Yield one params instance per point in :meth:`param_grid`."""
        import itertools

        grid = cls.param_grid()
        if not grid:
            yield cls.params_class()
            return
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            yield cls.params_class(**dict(zip(keys, combo)))

    # -- presentation ------------------------------------------------------

    def describe(self) -> str:
        """Multi-line description for docs and the dashboard."""
        return (
            f"{self.name} [{self.methodology}]\n"
            f"{self.description.strip()}\n"
            f"Parameters: {self.params.label()}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params.label()})"


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def combine(long_condition: pd.Series, short_condition: pd.Series) -> pd.Series:
    """Turn two boolean condition series into a single direction series.

    Bars where both conditions are true are flat rather than arbitrarily
    resolved: a strategy that simultaneously wants to be long and short has no
    opinion, and pretending otherwise hides a bug in the rules.
    """
    long_condition = long_condition.fillna(False)
    short_condition = short_condition.fillna(False)
    conflict = long_condition & short_condition

    direction = pd.Series(Signal.FLAT, index=long_condition.index, dtype="int8")
    direction[long_condition & ~conflict] = Signal.LONG
    direction[short_condition & ~conflict] = Signal.SHORT
    return direction


def state_machine(
    entry_long: pd.Series,
    exit_long: pd.Series,
    entry_short: pd.Series,
    exit_short: pd.Series,
) -> pd.Series:
    """Build a persistent position state from separate entry and exit rules.

    Trend rules are naturally persistent - "fast EMA above slow EMA" holds for
    as long as the trend does. Mean-reversion and breakout rules fire on a
    single bar and need an explicit exit, or the position would be held until
    the opposite extreme.

    Written as a forward loop rather than a vectorised trick: bar ``i`` only
    ever reads index ``i``, which makes it obviously causal, and 200k bars cost
    well under a second. A same-bar reversal is allowed.
    """
    index = entry_long.index
    el = entry_long.fillna(False).to_numpy(dtype=bool)
    xl = exit_long.fillna(False).to_numpy(dtype=bool)
    es = entry_short.fillna(False).to_numpy(dtype=bool)
    xs = exit_short.fillna(False).to_numpy(dtype=bool)

    out = np.zeros(len(index), dtype=np.int8)
    state = 0
    for i in range(len(index)):
        if state == Signal.LONG and (xl[i] or es[i]):
            state = Signal.FLAT
        elif state == Signal.SHORT and (xs[i] or el[i]):
            state = Signal.FLAT
        if state == Signal.FLAT:
            if el[i]:
                state = Signal.LONG
            elif es[i]:
                state = Signal.SHORT
        out[i] = state
    return pd.Series(out, index=index, dtype="int8")


def registry() -> Dict[str, type]:
    """Map of strategy name -> class for every implemented strategy."""
    from src.strategies import benchmark, breakout, mean_reversion, momentum, trend

    classes: List[type] = [
        trend.EmaRsiTrend,
        trend.MacdAdxTrend,
        momentum.StochRsiMacdMomentum,
        mean_reversion.RsiBollingerReversion,
        mean_reversion.VwapStretchReversion,
        breakout.DonchianVolumeBreakout,
        breakout.BollingerSqueezeBreakout,
        benchmark.BuyAndHold,
    ]
    return {cls.name: cls for cls in classes}


def build(name: str, **params: Any) -> Strategy:
    """Instantiate a strategy by name with optional parameter overrides."""
    available = registry()
    if name not in available:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(available)}")
    cls = available[name]
    return cls(cls.params_class(**params) if params else None)
