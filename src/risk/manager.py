"""Deterministic risk management.

This module is downstream of every decision layer and upstream of every order.
The strategy, the model and the LLM judge all decide **direction only**. What
follows - how much, where the stop goes, whether we are allowed to trade at
all - is arithmetic, and deliberately not delegated to anything that can
hallucinate.

The same calculations run in the backtest engine and in live paper trading, so
a position sized here matches the one the research produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

import numpy as np

from config.settings import RISK, RiskConfig
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class PositionPlan:
    """A fully specified, risk-checked order."""

    direction: int
    size: float
    entry_price: float
    stop_price: float
    target_price: float
    risk_amount: float
    notional: float
    reward_risk: float

    def describe(self) -> str:
        side = "LONG" if self.direction > 0 else "SHORT"
        return (
            f"{side} {self.size:.6f} @ {self.entry_price:,.2f} | "
            f"stop {self.stop_price:,.2f} | target {self.target_price:,.2f} | "
            f"risking {self.risk_amount:,.2f} ({self.reward_risk:.1f}:1)"
        )


@dataclass
class RiskState:
    """Mutable state the risk manager needs across a trading session."""

    equity: float
    day_start_equity: float
    trading_day: date
    open_positions: int = 0
    halted_reason: Optional[str] = None

    def roll_day(self, today: date) -> None:
        """Reset daily counters when the UTC date changes."""
        if today != self.trading_day:
            logger.info(
                "New trading day %s: resetting daily loss limit (equity %.2f)",
                today,
                self.equity,
            )
            self.trading_day = today
            self.day_start_equity = self.equity
            self.halted_reason = None


class RiskManager:
    """Position sizing and the checks that can refuse a trade outright."""

    def __init__(self, config: RiskConfig = RISK) -> None:
        self.config = config

    # -- sizing ------------------------------------------------------------

    def stop_distance(self, atr: float) -> float:
        """Distance from entry to stop, in price units."""
        return atr * self.config.atr_stop_multiple

    def position_size(self, equity: float, stop_distance: float) -> float:
        """Units sized so a stop-out costs exactly ``risk_per_trade`` of equity.

        This is what makes trades comparable across volatility regimes: a wide
        stop in a violent market buys fewer units, so the loss if wrong is the
        same as in a quiet one. Sizing by a fixed notional instead would mean
        risking several times more in high volatility, which is precisely
        backwards.
        """
        if stop_distance <= 0 or not np.isfinite(stop_distance):
            return 0.0
        return (equity * self.config.risk_per_trade) / stop_distance

    def plan(
        self,
        direction: int,
        price: float,
        atr: float,
        equity: float,
        leverage: float = 1.0,
    ) -> Optional[PositionPlan]:
        """Build a complete order plan, or ``None`` if the trade is not viable.

        Returns ``None`` rather than a degraded plan when ATR is unavailable:
        no ATR means no stop can be placed, and a position without a stop is
        not a trade this system is willing to take.
        """
        if direction == 0:
            return None
        if not np.isfinite(atr) or atr <= 0:
            logger.warning("No usable ATR (%s); refusing to size a position.", atr)
            return None
        if not np.isfinite(price) or price <= 0:
            logger.warning("No usable price (%s); refusing to size a position.", price)
            return None

        distance = self.stop_distance(atr)
        size = self.position_size(equity, distance)

        # Cap notional so 1x leverage really is 1x.
        cap = (equity * leverage * self.config.max_position_pct) / price
        size = min(size, cap)
        if size <= 0:
            return None

        stop = price - distance if direction > 0 else price + distance
        reward = distance * self.config.reward_risk_ratio
        target = price + reward if direction > 0 else price - reward

        return PositionPlan(
            direction=direction,
            size=size,
            entry_price=price,
            stop_price=stop,
            target_price=target,
            risk_amount=size * distance,
            notional=size * price,
            reward_risk=self.config.reward_risk_ratio,
        )

    # -- gatekeeping -------------------------------------------------------

    def can_open(self, state: RiskState) -> Tuple[bool, str]:
        """Whether a new position may be opened right now.

        Checked before every entry, in live trading and in the backtest.
        """
        if state.halted_reason:
            return False, state.halted_reason

        if state.open_positions >= self.config.max_open_positions:
            return False, (
                f"already holding {state.open_positions} position(s), "
                f"limit is {self.config.max_open_positions}"
            )

        if state.day_start_equity > 0:
            drawdown = state.equity / state.day_start_equity - 1.0
            if drawdown <= -self.config.max_daily_loss:
                return False, (
                    f"daily loss limit hit ({drawdown:.2%} vs "
                    f"{-self.config.max_daily_loss:.2%}); no new trades today"
                )

        if state.equity <= 0:
            return False, "no equity remaining"

        return True, ""

    def check_daily_limit(self, state: RiskState) -> bool:
        """Set the halt flag if the daily loss limit has been breached.

        Returns True when the limit has just been hit, so the caller knows to
        close any open position.
        """
        if state.halted_reason or state.day_start_equity <= 0:
            return False

        drawdown = state.equity / state.day_start_equity - 1.0
        if drawdown <= -self.config.max_daily_loss:
            state.halted_reason = (
                f"daily loss limit: {drawdown:.2%} since {state.trading_day}"
            )
            logger.warning("HALT - %s", state.halted_reason)
            return True
        return False
