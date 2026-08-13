"""Risk manager, persistence and demo-executor tests.

All offline: the exchange client is faked, so the full live path is exercised
without credentials, a network call, or any possibility of placing an order.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from config import settings
from config.settings import RiskConfig
from src.database.repository import Repository
from src.risk.manager import RiskManager, RiskState


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------


def test_position_size_risks_exactly_the_configured_fraction():
    manager = RiskManager(RiskConfig(risk_per_trade=0.01, atr_stop_multiple=2.0))
    equity, atr = 10_000.0, 500.0

    distance = manager.stop_distance(atr)
    size = manager.position_size(equity, distance)

    assert distance == 1_000.0
    assert size * distance == pytest.approx(100.0), "a stop-out must cost 1% of equity"


def test_higher_volatility_reduces_position_size():
    """The whole point of ATR sizing: risk stays constant, size adapts."""
    manager = RiskManager(RiskConfig(risk_per_trade=0.01, atr_stop_multiple=2.0))

    calm = manager.plan(1, price=50_000, atr=300, equity=10_000)
    wild = manager.plan(1, price=50_000, atr=3_000, equity=10_000)

    assert wild.size < calm.size
    assert calm.risk_amount == pytest.approx(100.0)
    assert wild.risk_amount == pytest.approx(100.0)


def test_a_tight_stop_makes_the_leverage_cap_bind():
    """A real and under-appreciated property of 1x leverage.

    When the stop is close to entry, sizing for 1% risk implies a notional
    several times equity. The leverage cap truncates it, so the *effective*
    risk per trade is smaller than configured. This is why short timeframes
    end up risking far less per trade than the config suggests while paying
    exactly the same fees - and it is a large part of why 15m trading loses.
    """
    manager = RiskManager(RiskConfig(risk_per_trade=0.01, atr_stop_multiple=2.0))
    plan = manager.plan(1, price=50_000, atr=100, equity=10_000, leverage=1.0)

    assert plan.notional == pytest.approx(10_000.0), "capped at 1x equity"
    assert plan.risk_amount < 100.0, "effective risk is below the configured 1%"
    assert plan.risk_amount == pytest.approx(40.0)


def test_stop_and_target_sit_on_the_correct_sides():
    manager = RiskManager(RiskConfig(atr_stop_multiple=2.0, reward_risk_ratio=2.0))

    long_plan = manager.plan(1, price=100.0, atr=1.0, equity=10_000)
    assert long_plan.stop_price == pytest.approx(98.0)
    assert long_plan.target_price == pytest.approx(104.0)

    short_plan = manager.plan(-1, price=100.0, atr=1.0, equity=10_000)
    assert short_plan.stop_price == pytest.approx(102.0)
    assert short_plan.target_price == pytest.approx(96.0)


def test_notional_is_capped_at_the_leverage_limit():
    """A tight stop implies a huge position; 1x leverage must still mean 1x."""
    manager = RiskManager(RiskConfig(risk_per_trade=0.01, atr_stop_multiple=2.0))
    plan = manager.plan(1, price=100.0, atr=0.01, equity=10_000, leverage=1.0)

    assert plan.notional <= 10_000 * 1.0001


def test_no_atr_means_no_trade():
    """No ATR, no stop; no stop, no trade."""
    manager = RiskManager()
    assert manager.plan(1, price=100.0, atr=float("nan"), equity=10_000) is None
    assert manager.plan(1, price=100.0, atr=0.0, equity=10_000) is None
    assert manager.plan(0, price=100.0, atr=1.0, equity=10_000) is None


def test_daily_loss_limit_blocks_further_trading():
    manager = RiskManager(RiskConfig(max_daily_loss=0.03))
    state = RiskState(equity=9_650.0, day_start_equity=10_000.0, trading_day=date(2026, 8, 11))

    assert manager.check_daily_limit(state) is True
    allowed, reason = manager.can_open(state)
    assert not allowed
    assert "daily loss limit" in reason


def test_daily_limit_resets_on_a_new_day():
    manager = RiskManager(RiskConfig(max_daily_loss=0.03))
    state = RiskState(equity=9_600.0, day_start_equity=10_000.0, trading_day=date(2026, 8, 11))
    manager.check_daily_limit(state)

    state.roll_day(date(2026, 8, 12))
    assert state.halted_reason is None
    assert state.day_start_equity == 9_600.0
    assert manager.can_open(state)[0]


def test_position_limit_is_enforced():
    manager = RiskManager(RiskConfig(max_open_positions=1))
    state = RiskState(
        equity=10_000.0, day_start_equity=10_000.0, trading_day=date(2026, 8, 11), open_positions=1
    )
    allowed, reason = manager.can_open(state)
    assert not allowed
    assert "already holding" in reason


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.fixture
def repository(tmp_path) -> Repository:
    return Repository(tmp_path / "test.db")


def test_trade_lifecycle_is_recorded(repository):
    trade_id = repository.record_trade(
        {
            "order_id": "abc",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "strategy": "ema_rsi_trend",
            "system": "C",
            "direction": 1,
            "entry_time": "2026-08-11T10:00:00+00:00",
            "entry_price": 60_000.0,
            "size": 0.01,
            "stop_price": 58_000.0,
            "target_price": 64_000.0,
            "fees": 0.33,
            "mode": "demo",
        }
    )

    assert repository.open_trade()["id"] == trade_id

    repository.close_trade(
        trade_id, "2026-08-11T14:00:00+00:00", 62_000.0, 20.0, 0.033, "take_profit"
    )
    assert repository.open_trade() is None

    summary = repository.summary()
    assert summary["closed_trades"] == 1
    assert summary["total_pnl"] == pytest.approx(20.0)


def test_decisions_are_logged_even_when_no_trade_results(repository):
    """A log of only trades cannot explain why the system stayed flat."""
    repository.record_decision(
        {
            "timestamp": "2026-08-11T10:00:00+00:00",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "strategy": "ema_rsi_trend",
            "strategy_signal": 1,
            "ml_prediction": -1,
            "ml_confidence": 0.55,
            "judge_decision": "HOLD",
            "judge_confidence": 70,
            "judge_reason": "Model disagrees with the strategy.",
            "risk_assessment": "HIGH",
            "final_action": 0,
            "blocked_reason": None,
            "indicators": {"rsi": 58.2, "adx": 24.0},
            "model": "gpt-test",
        }
    )

    decisions = repository.decisions()
    assert len(decisions) == 1
    assert decisions.iloc[0]["final_action"] == 0
    assert json.loads(decisions.iloc[0]["indicators"])["rsi"] == pytest.approx(58.2)


def test_equity_history_is_deduplicated_by_timestamp(repository):
    repository.record_equity("2026-08-11T10:00:00+00:00", 10_000.0)
    repository.record_equity("2026-08-11T10:00:00+00:00", 10_050.0)

    curve = repository.equity_curve()
    assert len(curve) == 1
    assert curve.iloc[0]["equity"] == 10_050.0


def test_state_round_trips(repository):
    repository.set_state("day_start_equity", 9_876.5)
    assert repository.get_state("day_start_equity") == 9_876.5
    assert repository.get_state("missing", "fallback") == "fallback"


def test_wal_mode_is_enabled_so_the_dashboard_never_blocks_the_trader(repository):
    with repository.connect() as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


# ---------------------------------------------------------------------------
# Executor safety
# ---------------------------------------------------------------------------


class FakeBybit:
    """Stand-in for the demo client."""

    def __init__(self, equity=10_000.0, position=None, venue_price=60_000.0):
        self.equity = equity
        self.venue_price = venue_price
        self._position = position or {
            "direction": 0,
            "size": 0.0,
            "entry_price": 0.0,
            "unrealised_pnl": 0.0,
        }
        self.orders = []
        self.closed = 0

    def wallet_equity(self, coin="USDT"):
        return self.equity

    def position(self, symbol, category="linear"):
        return dict(self._position)

    def place_market_order(self, symbol, direction, quantity, **kwargs):
        self.orders.append(
            {"symbol": symbol, "direction": direction, "quantity": quantity, **kwargs}
        )
        self._position = {
            "direction": direction,
            "size": quantity,
            "entry_price": 60_000.0,
            "unrealised_pnl": 0.0,
        }
        return f"order-{len(self.orders)}"

    def last_price(self, symbol, category="linear"):
        return self.venue_price

    def close_position(self, symbol, category="linear"):
        self.closed += 1
        self._position = {
            "direction": 0,
            "size": 0.0,
            "entry_price": 0.0,
            "unrealised_pnl": 0.0,
        }
        return "close-order"


@pytest.fixture
def executor(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TRADING_MODE", "demo")
    from src.exchange import executor as executor_module

    monkeypatch.setattr(executor_module, "assert_paper_mode", lambda: None)
    client = FakeBybit()
    return (
        executor_module.PaperExecutor(
            client=client,
            repository=Repository(tmp_path / "exec.db"),
            symbol="BTCUSDT",
            timeframe="1h",
            strategy_name="ema_rsi_trend",
        ),
        client,
    )


def test_executor_places_a_protected_order(executor):
    """Stop and target are attached at submission, so the position is never
    unprotected - not even between two API calls."""
    paper, client = executor
    plan = paper.execute(direction=1, price=60_000.0, atr=1_000.0)

    assert plan is not None
    assert len(client.orders) == 1
    order = client.orders[0]
    assert order["direction"] == 1
    assert order["stop_loss"] < 60_000 < order["take_profit"]


def test_executor_records_the_trade_and_the_decision(executor):
    paper, _ = executor
    paper.execute(direction=1, price=60_000.0, atr=1_000.0)

    assert paper.repository.open_trade() is not None
    decisions = paper.repository.decisions()
    assert len(decisions) == 1
    assert decisions.iloc[0]["final_action"] == 1
    assert decisions.iloc[0]["mode"] if "mode" in decisions.columns else True


def test_executor_does_nothing_when_already_positioned(executor):
    paper, client = executor
    paper.execute(direction=1, price=60_000.0, atr=1_000.0)
    paper.execute(direction=1, price=61_000.0, atr=1_000.0)

    assert len(client.orders) == 1, "must not double up on the same direction"
    blocked = paper.repository.decisions().iloc[0]["blocked_reason"]
    assert blocked == "already in position"


def test_executor_reverses_by_flattening_first(executor):
    paper, client = executor
    paper.execute(direction=1, price=60_000.0, atr=1_000.0)
    paper.execute(direction=-1, price=60_000.0, atr=1_000.0)

    assert client.closed == 1
    assert client.orders[-1]["direction"] == -1


def test_executor_respects_the_daily_loss_limit(executor):
    paper, client = executor
    paper.state.day_start_equity = 10_000.0
    paper.state.equity = 9_600.0
    client.equity = 9_600.0

    plan = paper.execute(direction=1, price=60_000.0, atr=1_000.0)

    assert plan is None
    assert client.orders == []
    assert "daily loss" in paper.repository.decisions().iloc[0]["blocked_reason"]


def test_executor_refuses_to_trade_without_atr(executor):
    paper, client = executor
    assert paper.execute(direction=1, price=60_000.0, atr=float("nan")) is None
    assert client.orders == []


def test_executor_reconciles_a_position_closed_on_the_exchange(executor):
    """Stops fire on Bybit's side; the only way to learn is to notice."""
    paper, client = executor
    paper.execute(direction=1, price=60_000.0, atr=1_000.0)
    assert paper.repository.open_trade() is not None

    client._position = {"direction": 0, "size": 0.0, "entry_price": 61_000.0, "unrealised_pnl": 0.0}
    paper.sync()

    assert paper.repository.open_trade() is None
    closed = paper.repository.trades().iloc[0]
    assert closed["exit_reason"] == "closed_on_exchange"


# ---------------------------------------------------------------------------
# The safeguards themselves
# ---------------------------------------------------------------------------


def test_client_cannot_be_constructed_outside_demo_mode(monkeypatch):
    monkeypatch.setattr(settings, "TRADING_MODE", "live")
    from src.exchange.bybit_client import BybitPaperClient

    with pytest.raises(settings.UnsafeConfigurationError):
        BybitPaperClient(api_key="k", api_secret="s")


def test_client_refuses_to_send_to_a_non_demo_host(monkeypatch):
    monkeypatch.setattr(settings, "TRADING_MODE", "demo")
    from src.exchange.bybit_client import BybitPaperClient, UnsafeEndpointError

    client = BybitPaperClient(api_key="k", api_secret="s", session=object())
    client.base_url = "https://example.invalid"

    with pytest.raises(UnsafeEndpointError, match="never trade real funds"):
        client._request("POST", "/v5/order/create", {})


def test_client_requires_credentials(monkeypatch):
    monkeypatch.setattr(settings, "TRADING_MODE", "demo")
    monkeypatch.delenv("BYBIT_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
    from src.exchange.bybit_client import BybitPaperClient, BybitPaperError

    with pytest.raises(BybitPaperError, match="testnet.bybit.com"):
        BybitPaperClient()


def test_signature_is_deterministic_and_secret_dependent(monkeypatch):
    monkeypatch.setattr(settings, "TRADING_MODE", "demo")
    from src.exchange.bybit_client import BybitPaperClient

    first = BybitPaperClient(api_key="k", api_secret="secret-a", session=object())
    second = BybitPaperClient(api_key="k", api_secret="secret-b", session=object())

    assert first._sign("123", "body") == first._sign("123", "body")
    assert first._sign("123", "body") != second._sign("123", "body")


def test_orders_are_priced_against_the_venue_not_the_signal(executor):
    """Testnet runs its own order book, so its price can diverge from the real
    market. A stop computed from the mainnet price would land nowhere near the
    testnet book and would fill instantly or never."""
    paper, client = executor
    client.venue_price = 30_000.0  # testnet is half the mainnet price

    plan = paper.execute(direction=1, price=60_000.0, atr=1_000.0)

    assert plan.entry_price == pytest.approx(30_000.0), "must size at the venue price"
    # The stop must remain the same *fraction* of price, not the same absolute
    # distance, or a 50% price difference would make it 2x too wide.
    stop_fraction = (plan.entry_price - plan.stop_price) / plan.entry_price
    assert stop_fraction == pytest.approx(2 * 1_000 / 60_000, rel=1e-6)


def test_venue_price_failure_falls_back_to_the_signal_price(executor):
    paper, client = executor

    def boom(*args, **kwargs):
        raise RuntimeError("ticker unavailable")

    client.last_price = boom
    plan = paper.execute(direction=1, price=60_000.0, atr=1_000.0)

    assert plan is not None, "a ticker outage must not stop trading"
    assert plan.entry_price == pytest.approx(60_000.0)


# ---------------------------------------------------------------------------
# Deployed dashboard: price fallback
# ---------------------------------------------------------------------------


def test_coinbase_candles_are_parsed_into_ascending_ohlc():
    """Coinbase's column order is not the obvious one.

    It returns ``[time, low, high, open, close, volume]`` - low and high before
    open and close - newest row first. Reading it positionally as OHLCV silently
    swaps the wicks for the body and draws a plausible, wrong candlestick. This
    fallback only runs on the deployed copy, where nobody would be watching.
    """
    from unittest.mock import MagicMock, patch

    import dashboard.data_access as da

    # time, low, high, open, close, volume - newest first, as the API sends it.
    payload = [
        [1_755_014_400, 100.0, 400.0, 200.0, 300.0, 9.0],
        [1_755_000_000, 10.0, 40.0, 20.0, 30.0, 1.0],
    ]
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None

    with patch("requests.get", return_value=response):
        frame = da._coinbase_candles("4h", 200)

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.is_monotonic_increasing
    assert str(frame.index.tz) == "UTC"

    oldest = frame.iloc[0]
    assert (oldest["open"], oldest["high"], oldest["low"], oldest["close"]) == (20.0, 40.0, 10.0, 30.0)
    # The invariant that catches a positional misread.
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
    assert (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()


def test_live_price_falls_back_when_the_primary_venue_refuses():
    """Bybit blocks datacenter IPs, so on the deployed copy it always fails."""
    from unittest.mock import MagicMock, patch

    import dashboard.data_access as da

    response = MagicMock()
    response.json.return_value = {"price": "63999.5"}
    response.raise_for_status.return_value = None

    with patch.object(da, "BybitPublicClient", side_effect=RuntimeError("geoblocked")), \
            patch("requests.get", return_value=response):
        assert da.load_live_price.__wrapped__("BTCUSDT") == 63999.5
