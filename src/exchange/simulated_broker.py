"""Local simulated broker.

Why this exists
---------------
Bybit's EU entity cannot offer perpetual futures under MiCA, and its API is
restricted to registered API-broker integrations, so an ordinary EEA account
cannot trade programmatically on either mainnet or testnet. Bybit's global
testnet geo-redirects EEA users to the EU site, closing that route too.

Rather than compromise the research - dropping short trades would invalidate
every backtest result in a falling market - execution moves behind a second
implementation of the same interface.

What this is, and is not
------------------------
It **is** a broker that fills against *real* BTC prices from the public
market-data feed, using exactly the fee, slippage and stop-resolution rules
the backtester uses. Its P&L is therefore directly comparable to the research
results.

It is **not** a connection to an exchange. No order reaches a venue and no
queue position, partial fill or liquidation is modelled. Assumed fills are
optimistic in the way all paper trading is.

Notably, this is *more* faithful to the research than Bybit testnet would have
been: testnet runs its own thin order book whose price drifts far from the
real market, so its P&L would have been meaningless.

Interface compatibility
-----------------------
Mirrors :class:`~src.exchange.bybit_client.BybitPaperClient` method for
method, so :class:`~src.exchange.executor.PaperExecutor` cannot tell them
apart. Switching back to a real exchange is a one-line change.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from config.settings import BACKTEST, DATA, BacktestConfig
from src.data.loader import load_ohlcv
from src.data.public_client import BybitPublicClient
from src.database.repository import Repository
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

STATE_KEY = "simulated_broker"


class SimulatedBroker:
    """Paper broker filling against live public prices."""

    def __init__(
        self,
        repository: Optional[Repository] = None,
        market: Optional[BybitPublicClient] = None,
        config: BacktestConfig = BACKTEST,
        starting_equity: Optional[float] = None,
        timeframe: str = "4h",
    ) -> None:
        self.repository = repository or Repository()
        self.market = market or BybitPublicClient()
        self.config = config
        self.timeframe = timeframe
        #: Wall-clock of the last stop/target settlement. The executor calls
        #: wallet_equity() and position() several times per cycle, and each
        #: settlement refetches candles - four to six network round trips per
        #: poll, and a log nobody can read. Settling at most once per
        #: `settle_interval` seconds collapses that to one.
        self._last_settled = 0.0
        self.settle_interval = 10.0

        state = self.repository.get_state(STATE_KEY)
        if state is None:
            state = {
                "equity": float(starting_equity or config.initial_capital),
                "last_close": None,
                "direction": 0,
                "size": 0.0,
                "entry_price": 0.0,
                "stop_price": None,
                "target_price": None,
                "opened_at": None,
                "last_checked": None,
            }
            self.repository.set_state(STATE_KEY, state)
            logger.info(
                "Simulated broker initialised with %.2f USDT", state["equity"]
            )
        self.state = state

    # -- persistence -------------------------------------------------------

    def _save(self) -> None:
        self.repository.set_state(STATE_KEY, self.state)

    def _slip(self, price: float, direction: int) -> float:
        """Slippage always works against us, as in the backtester."""
        slip = self.config.slippage_bps / 10_000.0
        return price * (1.0 + slip) if direction > 0 else price * (1.0 - slip)

    # -- market data -------------------------------------------------------

    def last_price(self, symbol: str = DATA.symbol, category: str = "linear") -> float:
        """Current real BTC price."""
        return self.market.get_last_price(symbol, category)

    # -- account -----------------------------------------------------------

    def wallet_equity(self, coin: str = "USDT") -> float:
        """Equity including any unrealised P&L, after settling stops."""
        self._settle_protective_orders()
        equity = self.state["equity"]
        if self.state["direction"]:
            try:
                price = self.last_price()
                equity += self.state["direction"] * self.state["size"] * (
                    price - self.state["entry_price"]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not mark to market (%s)", exc)
        return float(equity)

    def position(self, symbol: str = DATA.symbol, category: str = "linear") -> Dict[str, Any]:
        """Current position, after checking whether a stop or target fired."""
        self._settle_protective_orders()
        if not self.state["direction"]:
            return {"direction": 0, "size": 0.0, "entry_price": 0.0, "unrealised_pnl": 0.0}

        unrealised = 0.0
        try:
            price = self.last_price()
            unrealised = self.state["direction"] * self.state["size"] * (
                price - self.state["entry_price"]
            )
        except Exception:  # noqa: BLE001
            pass

        return {
            "direction": int(self.state["direction"]),
            "size": float(self.state["size"]),
            "entry_price": float(self.state["entry_price"]),
            "unrealised_pnl": float(unrealised),
        }

    def set_leverage(self, symbol: str, leverage: float, category: str = "linear") -> None:
        """No-op: the simulator enforces leverage through position sizing."""
        logger.debug("Simulated broker ignores set_leverage(%s)", leverage)

    # -- protective orders -------------------------------------------------

    def _settle_protective_orders(self) -> None:
        """Fill a stop or target that was breached between polls.

        The loop wakes every 30 seconds but price moves continuously, so a
        stop can be breached and recovered before we ever look. Checking only
        the spot price at poll time would silently miss those fills and
        overstate performance.

        Instead the candles since entry are replayed and their **highs and
        lows** examined, using the backtester's rules: a gap through the stop
        fills at the open, a favourable gap through the target still fills at
        the target, and a candle containing both fills at the stop because
        OHLCV cannot say which came first.
        """
        if not self.state["direction"] or not self.state["opened_at"]:
            return

        now = time.monotonic()
        if now - self._last_settled < self.settle_interval:
            return
        self._last_settled = now

        try:
            candles = load_ohlcv(self.timeframe, refresh=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load candles to settle stops (%s)", exc)
            return

        since = pd.Timestamp(self.state["opened_at"])
        window = candles.loc[candles.index >= since]
        if window.empty:
            return

        direction = int(self.state["direction"])
        stop = self.state.get("stop_price")
        target = self.state.get("target_price")

        for timestamp, bar in window.iterrows():
            hit_stop = stop is not None and (
                bar["low"] <= stop if direction > 0 else bar["high"] >= stop
            )
            hit_target = target is not None and (
                bar["high"] >= target if direction > 0 else bar["low"] <= target
            )

            if hit_stop:
                # Pessimistic: ambiguous bars resolve against us, and a gap
                # through the stop fills at the open rather than at the stop.
                fill = min(stop, bar["open"]) if direction > 0 else max(stop, bar["open"])
                self._close_at(self._slip(fill, -direction), "stop_loss", timestamp)
                return
            if hit_target:
                self._close_at(float(target), "take_profit", timestamp)
                return

    def _close_at(self, price: float, reason: str, when: Any = None) -> float:
        """Realise the position at ``price`` and return the P&L."""
        direction = int(self.state["direction"])
        size = float(self.state["size"])
        if not direction or size <= 0:
            return 0.0

        gross = direction * size * (price - self.state["entry_price"])
        fee = abs(size * price) * self.config.taker_fee
        pnl = gross - fee

        self.state["equity"] = float(self.state["equity"] + pnl)
        self.state.update(
            direction=0, size=0.0, entry_price=0.0, stop_price=None,
            target_price=None, opened_at=None,
            # Recorded so the executor can log the real exit price and P&L when
            # it notices the position is gone. Without this the trade log shows
            # a zero-P&L close while the equity curve shows the real result -
            # two sources of truth that disagree.
            last_close={"price": float(price), "pnl": float(pnl), "reason": reason},
        )
        self._save()

        logger.info(
            "SIMULATED %s at %.2f%s | pnl %.2f | equity %.2f",
            reason,
            price,
            f" ({when})" if when is not None else "",
            pnl,
            self.state["equity"],
        )
        return pnl

    # -- orders ------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False,
        category: str = "linear",
        quantity_decimals: int = 3,
    ) -> str:
        """Fill a market order at the current real price plus slippage."""
        if direction == 0:
            raise ValueError("direction must be +1 or -1")

        quantity = round(float(quantity), quantity_decimals)
        if quantity <= 0:
            raise ValueError(f"Quantity rounds to zero: {quantity}")

        price = self._slip(self.last_price(symbol, category), direction)

        if reduce_only:
            self._close_at(price, "reduce_only")
            return f"sim-close-{uuid.uuid4().hex[:8]}"

        fee = abs(quantity * price) * self.config.taker_fee
        self.state.update(
            equity=float(self.state["equity"] - fee),
            direction=int(direction),
            size=quantity,
            entry_price=float(price),
            stop_price=float(stop_loss) if stop_loss is not None else None,
            target_price=float(take_profit) if take_profit is not None else None,
            opened_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save()

        order_id = f"sim-{uuid.uuid4().hex[:8]}"
        logger.info(
            "SIMULATED %s %.4f %s at %.2f (fee %.2f, sl %s, tp %s) -> %s",
            "BUY" if direction > 0 else "SELL",
            quantity,
            symbol,
            price,
            fee,
            f"{stop_loss:.2f}" if stop_loss else "-",
            f"{take_profit:.2f}" if take_profit else "-",
            order_id,
        )
        return order_id

    def close_position(self, symbol: str = DATA.symbol, category: str = "linear") -> Optional[str]:
        """Flatten at the current price."""
        if not self.state["direction"]:
            return None
        price = self._slip(self.last_price(symbol, category), -int(self.state["direction"]))
        self._close_at(price, "manual_close")
        return f"sim-close-{uuid.uuid4().hex[:8]}"

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        equity = self.state["equity"]
        if self.state["direction"]:
            side = "LONG" if self.state["direction"] > 0 else "SHORT"
            return (
                f"Simulated broker | equity {equity:,.2f} | "
                f"{side} {self.state['size']:.4f} @ {self.state['entry_price']:,.2f}"
            )
        return f"Simulated broker | equity {equity:,.2f} | flat"
