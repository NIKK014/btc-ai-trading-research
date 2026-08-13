"""Paper-trading execution.

    signals -> ML filter -> judge -> RISK MANAGER -> executor -> broker

Narrow on purpose: by the time a decision reaches here, direction is settled.
The executor asks the risk manager whether the trade is allowed and how large,
sends the order, and records everything - including decisions that produced no
trade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


from config.settings import BACKTEST, DATA, TRADING_MODE, assert_paper_mode
from src.database.repository import Repository
from src.exchange.bybit_client import BybitPaperClient
from src.risk.manager import PositionPlan, RiskManager, RiskState
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

ACTION_NAMES = {1: "LONG", -1: "SHORT", 0: "FLAT"}


class PaperExecutor:
    """Turns approved decisions into paper orders, and logs everything."""

    def __init__(
        self,
        client: Optional[BybitPaperClient] = None,
        repository: Optional[Repository] = None,
        risk_manager: Optional[RiskManager] = None,
        symbol: str = DATA.symbol,
        timeframe: str = "1h",
        strategy_name: str = "unknown",
        system: str = "C_rules_ml_llm",
    ) -> None:
        assert_paper_mode()

        # Duck-typed on purpose: BybitPaperClient and SimulatedBroker expose
        # the same surface, so execution is independent of the venue.
        self.client = client if client is not None else BybitPaperClient()
        self.repository = repository or Repository()
        self.risk = risk_manager or RiskManager()
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_name = strategy_name
        self.system = system

        equity = self.client.wallet_equity()
        today = datetime.now(timezone.utc).date()
        self.state = RiskState(
            equity=equity,
            day_start_equity=self.repository.get_state("day_start_equity", equity),
            trading_day=today,
        )
        self.repository.set_state("trading_mode", TRADING_MODE)
        self.repository.set_state("symbol", symbol)

        logger.info(
            "PaperExecutor ready | MODE: TESTNET | %s %s | equity %.2f USDT",
            symbol,
            timeframe,
            equity,
        )

    # -- state -------------------------------------------------------------

    def sync(self) -> Dict[str, Any]:
        """Refresh equity and position from the exchange.

        The exchange is the source of truth, not local state. A stop can fire
        between polls, and reconciling every cycle is what keeps the database
        honest about it.
        """
        self.state.equity = self.client.wallet_equity()
        self.state.roll_day(datetime.now(timezone.utc).date())
        self.repository.set_state("day_start_equity", self.state.day_start_equity)

        position = self.client.position(self.symbol)
        self.state.open_positions = 1 if position["direction"] != 0 else 0

        self._reconcile(position)
        return position

    def _reconcile(self, position: Dict[str, Any]) -> None:
        """Close the database record if the exchange closed the position.

        This is how stop-loss and take-profit exits get recorded: they happen
        on Bybit's side without us sending anything, so the only way to learn
        about them is to notice the position is gone.
        """
        recorded = self.repository.open_trade()
        if recorded is None or position["direction"] != 0:
            return

        direction = recorded["direction"]
        size = recorded["size"]
        notional = size * recorded["entry_price"]

        # Prefer the broker's record of how the position actually closed. The
        # fallback infers an exit from the position snapshot, which reports
        # zero P&L - correct only when nothing really happened.
        closed = getattr(self.client, "state", {}).get("last_close") if hasattr(self.client, "state") else None
        if closed:
            exit_price = float(closed["price"])
            pnl = float(closed["pnl"])
            reason = closed.get("reason", "closed_on_exchange")
        else:
            exit_price = position.get("entry_price") or recorded["entry_price"]
            pnl = direction * size * (exit_price - recorded["entry_price"])
            reason = "closed_on_exchange"

        self.repository.close_trade(
            trade_id=recorded["id"],
            exit_time=self._now(),
            exit_price=exit_price,
            pnl=pnl,
            return_pct=pnl / notional if notional else 0.0,
            exit_reason=reason,
        )
        logger.info(
            "Reconciled: position closed (trade %s, %s, pnl %.2f)",
            recorded["id"],
            reason,
            pnl,
        )

    # -- acting ------------------------------------------------------------

    def execute(
        self,
        direction: int,
        price: float,
        atr: float,
        decision_log: Optional[Dict[str, Any]] = None,
    ) -> Optional[PositionPlan]:
        """Act on an approved decision.

        Args:
            direction: ``+1`` long, ``-1`` short, ``0`` flat.
            price: Latest price, for sizing.
            atr: Current ATR, for stop placement.
            decision_log: Upstream context to persist alongside the outcome.

        Returns:
            The executed :class:`PositionPlan`, or ``None`` if nothing was done.
        """
        position = self.sync()
        held = position["direction"]
        log = dict(decision_log or {})
        log.setdefault("timestamp", self._now())
        log.setdefault("symbol", self.symbol)
        log.setdefault("timeframe", self.timeframe)
        log.setdefault("strategy", self.strategy_name)
        log.setdefault("strategy_signal", direction)

        # Already positioned as desired: nothing to do.
        if direction == held and direction != 0:
            self._log(log, final_action=held, blocked="already in position")
            return None

        # Exit or reverse: flatten first.
        if held != 0 and direction != held:
            self.client.close_position(self.symbol)
            self._close_recorded_trade(price, "signal" if direction == 0 else "reversal")
            self.state.open_positions = 0
            if direction == 0:
                self._log(log, final_action=0)
                return None

        if direction == 0:
            self._log(log, final_action=0)
            return None

        # Price the order against the venue we are actually trading on.
        # Signals come from mainnet history; testnet has its own book and its
        # own price. Sizing and stops must use the latter or they will be
        # meaningless.
        venue_price = price
        try:
            venue_price = self.client.last_price(self.symbol)
            divergence = abs(venue_price / price - 1.0) if price else 0.0
            if divergence > 0.05:
                logger.warning(
                    "Venue price %.2f differs from signal price %.2f by %.1f%%. "
                    "Testnet runs its own order book; P&L here is not comparable "
                    "to the research results.",
                    venue_price,
                    price,
                    divergence * 100,
                )
            # ATR is expressed in mainnet price units, so rescale it to the
            # venue's price level, keeping the stop the same *fraction* of price.
            if price:
                atr = atr * (venue_price / price)
        except Exception as exc:  # noqa: BLE001 - fall back to the signal price
            logger.warning("Could not read venue price (%s); using signal price.", exc)

        # Risk gate.
        allowed, reason = self.risk.can_open(self.state)
        if not allowed:
            logger.warning("Trade blocked: %s", reason)
            self._log(log, final_action=0, blocked=reason)
            return None

        plan = self.risk.plan(
            direction=direction,
            price=venue_price,
            atr=atr,
            equity=self.state.equity,
            leverage=BACKTEST.leverage,
        )
        if plan is None:
            self._log(log, final_action=0, blocked="risk manager could not size the position")
            return None

        order_id = self.client.place_market_order(
            symbol=self.symbol,
            direction=direction,
            quantity=plan.size,
            stop_loss=plan.stop_price,
            take_profit=plan.target_price,
        )

        self.repository.record_trade(
            {
                "order_id": order_id,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "strategy": self.strategy_name,
                "system": self.system,
                "direction": direction,
                "entry_time": self._now(),
                "entry_price": plan.entry_price,
                "size": plan.size,
                "stop_price": plan.stop_price,
                "target_price": plan.target_price,
                "fees": plan.notional * BACKTEST.taker_fee,
                "mode": "testnet",
            }
        )
        self.state.open_positions = 1
        self._log(log, final_action=direction)
        logger.info("EXECUTED %s", plan.describe())
        return plan

    def flatten(self, reason: str = "manual") -> None:
        """Close any open position immediately."""
        position = self.client.position(self.symbol)
        if position["direction"] == 0:
            return
        self.client.close_position(self.symbol)
        self._close_recorded_trade(position.get("entry_price", 0.0), reason)
        self.state.open_positions = 0
        logger.info("Position flattened (%s)", reason)

    # -- persistence -------------------------------------------------------

    def record_equity(self, price: Optional[float] = None) -> None:
        position = self.client.position(self.symbol)
        self.repository.record_equity(
            timestamp=self._now(),
            equity=self.state.equity,
            unrealised=position.get("unrealised_pnl", 0.0),
            position=position.get("direction", 0),
            price=price,
        )

    def _close_recorded_trade(self, exit_price: float, reason: str) -> None:
        recorded = self.repository.open_trade()
        if recorded is None:
            return
        direction = recorded["direction"]
        size = recorded["size"]
        pnl = direction * size * (exit_price - recorded["entry_price"])
        notional = size * recorded["entry_price"]
        self.repository.close_trade(
            trade_id=recorded["id"],
            exit_time=self._now(),
            exit_price=exit_price,
            pnl=pnl,
            return_pct=pnl / notional if notional else 0.0,
            exit_reason=reason,
            fees=abs(size * exit_price) * BACKTEST.taker_fee,
        )

    def _log(
        self,
        payload: Dict[str, Any],
        final_action: int,
        blocked: Optional[str] = None,
    ) -> None:
        payload["final_action"] = final_action
        payload["blocked_reason"] = blocked
        self.repository.record_decision(payload)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
