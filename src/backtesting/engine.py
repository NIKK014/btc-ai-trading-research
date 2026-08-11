"""Event-driven backtest engine.

This is the most correctness-critical module in the project: every number in
the final comparison flows through it, and a silently wrong engine produces
beautiful, meaningless results. It is therefore deliberately explicit rather
than clever, and is verified in ``tests/test_engine.py`` against P&L computed
by hand.

Execution model
---------------
For each bar, in this order:

1. **Fill pending orders at the open.** A signal is produced on the close of
   bar ``t`` and filled at the open of bar ``t+1``. Nothing is ever filled at
   the price that generated it.
2. **Check stop and target against the bar's high and low.**
3. **Mark to market at the close.**

Pessimistic assumptions, applied consistently
---------------------------------------------
* **Same-bar ambiguity resolves against us.** If a bar's range contains both
  the stop and the target, OHLCV cannot tell us which came first, so the stop
  is assumed. Guessing the favourable outcome is how backtests lie.
* **Gaps through the stop fill at the open**, not at the stop price - if price
  gapped past your stop you did not get your stop.
* **Gaps through the target fill at the target**, not at the better open.
* **Slippage is applied against us** on every market order. Limit exits at the
  target are assumed to fill without slippage.
* **Taker fees on both sides**, because signals are acted on at market.

Position sizing
---------------
``risk`` mode sizes each position so that a stop-out costs exactly
``risk_per_trade`` of current equity, which is what makes trades comparable
across volatility regimes. ``full_notional`` deploys all equity and is used
for the buy-and-hold benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import BACKTEST, RISK, BacktestConfig, RiskConfig
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

EXIT_STOP = "stop_loss"
EXIT_TARGET = "take_profit"
EXIT_SIGNAL = "signal"
EXIT_REVERSAL = "reversal"
EXIT_DAILY_LIMIT = "daily_loss_limit"
EXIT_END_OF_DATA = "end_of_data"

TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "direction",
    "entry_price",
    "exit_price",
    "size",
    "notional",
    "stop_price",
    "target_price",
    "fees",
    "pnl",
    "return_pct",
    "r_multiple",
    "bars_held",
    "exit_reason",
    "equity_after",
]


@dataclass
class BacktestResult:
    """Everything a backtest produces.

    Attributes:
        equity: Mark-to-market account equity at each bar close.
        trades: One row per closed trade, columns as in :data:`TRADE_COLUMNS`.
        signals: The desired-direction series that was replayed.
        position: Actual held direction at each bar close, which differs from
            ``signals`` wherever a stop fired or an entry was blocked.
        meta: Run parameters, for reproducibility.
    """

    equity: pd.Series
    trades: pd.DataFrame
    signals: pd.Series
    position: pd.Series
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        """Per-bar fractional returns of the equity curve."""
        return self.equity.pct_change().fillna(0.0)

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else float("nan")

    @property
    def exposure(self) -> float:
        """Fraction of bars spent holding a position."""
        return float((self.position != 0).mean()) if len(self.position) else 0.0

    def slice(self, start: Any = None, end: Any = None) -> "BacktestResult":
        """Restrict the result to a date range, rebasing the equity curve.

        Used for regime-split reporting: the same run can be evaluated over the
        bull and bear sub-periods of the test set without re-running it.
        """
        equity = self.equity.loc[start:end]
        if equity.empty:
            raise ValueError(f"No bars in range {start} .. {end}")

        trades = self.trades
        if not trades.empty:
            mask = (trades["entry_time"] >= equity.index[0]) & (
                trades["entry_time"] <= equity.index[-1]
            )
            trades = trades.loc[mask]

        return BacktestResult(
            equity=equity,
            trades=trades.reset_index(drop=True),
            signals=self.signals.loc[start:end],
            position=self.position.loc[start:end],
            meta={**self.meta, "sliced": (str(start), str(end))},
        )


def run_backtest(
    frame: pd.DataFrame,
    *,
    config: BacktestConfig = BACKTEST,
    risk: RiskConfig = RISK,
    use_stops: bool = True,
    sizing: str = "risk",
    signal_column: str = "signal",
    enforce_daily_limit: bool = True,
) -> BacktestResult:
    """Replay a signal series against price history.

    Args:
        frame: Must contain ``open``, ``high``, ``low``, ``close``, ``atr`` and
            the signal column. Use :meth:`Strategy.run` to produce it.
        config: Execution assumptions (fees, slippage, fill delay).
        risk: Position sizing and stop placement.
        use_stops: Disable for the buy-and-hold benchmark.
        sizing: ``"risk"`` or ``"full_notional"``.
        enforce_daily_limit: Apply the daily loss limit.

    Returns:
        A :class:`BacktestResult`.
    """
    required = {"open", "high", "low", "close", signal_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Backtest frame is missing columns: {sorted(missing)}")
    if sizing not in {"risk", "full_notional"}:
        raise ValueError(f"Unknown sizing mode {sizing!r}")
    if use_stops and "atr" not in frame.columns:
        raise ValueError("ATR column required when stops are enabled")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Backtest frame must be sorted chronologically")

    n = len(frame)
    index = frame.index
    open_ = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = (
        frame["atr"].to_numpy(dtype=np.float64)
        if "atr" in frame.columns
        else np.full(n, np.nan)
    )
    signal = frame[signal_column].to_numpy(dtype=np.int8)
    day = index.normalize().to_numpy()

    slip = config.slippage_bps / 10_000.0
    delay = config.fill_delay_bars

    equity_curve = np.empty(n, dtype=np.float64)
    position_series = np.zeros(n, dtype=np.int8)

    cash = float(config.initial_capital)
    position = 0
    size = 0.0
    entry_price = 0.0
    stop_price = np.nan
    target_price = np.nan
    risk_per_unit = 0.0
    entry_fee = 0.0
    entry_bar = -1

    #: Direction we were most recently stopped out of. Re-entry in that
    #: direction is blocked until the signal changes, otherwise the system
    #: would immediately re-enter the position that just failed and grind the
    #: account down through fees.
    blocked_direction = 0

    day_start_equity = cash
    current_day = day[0] if n else None
    daily_limit_hit = False

    trades: List[Dict[str, Any]] = []

    def buy_fill(price: float) -> float:
        return price * (1.0 + slip)

    def sell_fill(price: float) -> float:
        return price * (1.0 - slip)

    def close_position(bar: int, fill_price: float, reason: str) -> None:
        nonlocal cash, position, size, entry_price, stop_price, target_price
        nonlocal risk_per_unit, entry_fee, entry_bar

        exit_fee = abs(size * fill_price) * config.taker_fee
        gross = position * size * (fill_price - entry_price)
        # The entry fee was already taken out of cash when the position was
        # opened, so only the exit fee is deducted here. The trade's reported
        # P&L is net of both.
        pnl = gross - entry_fee - exit_fee
        cash += gross - exit_fee
        initial_risk = size * risk_per_unit if risk_per_unit > 0 else np.nan

        trades.append(
            {
                "entry_time": index[entry_bar],
                "exit_time": index[bar],
                "direction": int(position),
                "entry_price": entry_price,
                "exit_price": fill_price,
                "size": size,
                "notional": size * entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "fees": entry_fee + exit_fee,
                "pnl": pnl,
                "return_pct": pnl / (size * entry_price) if size * entry_price else np.nan,
                "r_multiple": pnl / initial_risk if initial_risk and initial_risk > 0 else np.nan,
                "bars_held": bar - entry_bar,
                "exit_reason": reason,
                "equity_after": cash,
            }
        )

        position = 0
        size = 0.0
        entry_price = 0.0
        stop_price = np.nan
        target_price = np.nan
        risk_per_unit = 0.0
        entry_fee = 0.0
        entry_bar = -1

    def open_position(bar: int, direction: int) -> None:
        nonlocal position, size, entry_price, stop_price, target_price
        nonlocal risk_per_unit, entry_fee, entry_bar, cash

        price = buy_fill(open_[bar]) if direction > 0 else sell_fill(open_[bar])
        if not np.isfinite(price) or price <= 0.0:
            return

        if use_stops:
            bar_atr = atr[bar]
            if not np.isfinite(bar_atr) or bar_atr <= 0.0:
                return  # cannot place a stop, so do not take the trade
            distance = bar_atr * risk.atr_stop_multiple
            units = (cash * risk.risk_per_trade) / distance if sizing == "risk" else (
                cash * config.leverage / price
            )
            cap = (cash * config.leverage * risk.max_position_pct) / price
            units = min(units, cap)
            stop = price - distance if direction > 0 else price + distance
            target = (
                price + distance * risk.reward_risk_ratio
                if direction > 0
                else price - distance * risk.reward_risk_ratio
            )
        else:
            distance = 0.0
            units = cash * config.leverage / price
            stop = np.nan
            target = np.nan

        if not np.isfinite(units) or units <= 0.0:
            return

        fee = abs(units * price) * config.taker_fee
        cash -= fee

        position = direction
        size = units
        entry_price = price
        stop_price = stop
        target_price = target
        risk_per_unit = distance
        entry_fee = fee
        entry_bar = bar

    for i in range(n):
        # -- day rollover -------------------------------------------------
        if day[i] != current_day:
            current_day = day[i]
            day_start_equity = cash + (
                position * size * (close[i - 1] - entry_price) if position else 0.0
            )
            daily_limit_hit = False

        # -- 1. fill pending orders at this bar's open --------------------
        desired = int(signal[i - delay]) if i >= delay else 0

        if blocked_direction != 0 and desired != blocked_direction:
            blocked_direction = 0

        if position != 0 and desired != position:
            reason = EXIT_REVERSAL if desired == -position else EXIT_SIGNAL
            fill = sell_fill(open_[i]) if position > 0 else buy_fill(open_[i])
            close_position(i, fill, reason)

        if (
            position == 0
            and desired != 0
            and desired != blocked_direction
            and not (enforce_daily_limit and daily_limit_hit)
        ):
            open_position(i, desired)

        # -- 2. intrabar stop and target ----------------------------------
        if position != 0 and use_stops:
            hit_stop = low[i] <= stop_price if position > 0 else high[i] >= stop_price
            hit_target = high[i] >= target_price if position > 0 else low[i] <= target_price

            held_direction = position
            if hit_stop and (config.ambiguous_bar_favours_stop or not hit_target):
                # A gap through the stop fills at the open, not at the stop.
                if position > 0:
                    raw = min(stop_price, open_[i])
                    fill = sell_fill(raw)
                else:
                    raw = max(stop_price, open_[i])
                    fill = buy_fill(raw)
                close_position(i, fill, EXIT_STOP)
                blocked_direction = held_direction
            elif hit_target:
                # Limit fill at the target; no credit for a favourable gap.
                close_position(i, target_price, EXIT_TARGET)
                blocked_direction = held_direction

        # -- 3. mark to market --------------------------------------------
        unrealised = position * size * (close[i] - entry_price) if position else 0.0
        equity = cash + unrealised

        # -- 4. daily loss limit ------------------------------------------
        if (
            enforce_daily_limit
            and not daily_limit_hit
            and day_start_equity > 0
            and (equity / day_start_equity - 1.0) <= -risk.max_daily_loss
        ):
            daily_limit_hit = True
            if position != 0:
                fill = sell_fill(close[i]) if position > 0 else buy_fill(close[i])
                close_position(i, fill, EXIT_DAILY_LIMIT)
                equity = cash

        equity_curve[i] = equity
        position_series[i] = position

    # -- close any open position at the final close -----------------------
    if position != 0 and n:
        last = n - 1
        fill = sell_fill(close[last]) if position > 0 else buy_fill(close[last])
        close_position(last, fill, EXIT_END_OF_DATA)
        equity_curve[last] = cash
        position_series[last] = 0

    trades_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)

    return BacktestResult(
        equity=pd.Series(equity_curve, index=index, name="equity"),
        trades=trades_frame,
        signals=frame[signal_column].copy(),
        position=pd.Series(position_series, index=index, name="position"),
        meta={
            "initial_capital": config.initial_capital,
            "taker_fee": config.taker_fee,
            "slippage_bps": config.slippage_bps,
            "leverage": config.leverage,
            "use_stops": use_stops,
            "sizing": sizing,
            "risk_per_trade": risk.risk_per_trade,
            "atr_stop_multiple": risk.atr_stop_multiple,
            "reward_risk_ratio": risk.reward_risk_ratio,
            "max_daily_loss": risk.max_daily_loss if enforce_daily_limit else None,
            "bars": n,
        },
    )
