"""Simulated broker tests.

The simulator must behave exactly like the backtester, or the live track
record and the research results are measuring different things.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import BacktestConfig
from src.database.repository import Repository
from src.exchange.simulated_broker import SimulatedBroker

FRICTIONLESS = BacktestConfig(initial_capital=10_000.0, taker_fee=0.0, slippage_bps=0.0)
REALISTIC = BacktestConfig(initial_capital=10_000.0, taker_fee=0.00055, slippage_bps=2.0)


class FakeMarket:
    def __init__(self, price=60_000.0):
        self.price = price

    def get_last_price(self, symbol, category="linear"):
        return self.price


@pytest.fixture
def broker(tmp_path):
    return SimulatedBroker(
        repository=Repository(tmp_path / "sim.db"),
        market=FakeMarket(),
        config=FRICTIONLESS,
    )


def candles(rows, freq="4h") -> pd.DataFrame:
    """Candles timestamped just after "now".

    ``_settle_protective_orders`` only considers bars that *started* at or
    after the entry, because a bar straddling the entry contains price action
    from before we were in the trade.
    """
    start = pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=1)
    index = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close"]).assign(
        volume=1.0
    )


def set_candles(monkeypatch, rows) -> None:
    monkeypatch.setattr(
        "src.exchange.simulated_broker.load_ohlcv", lambda *a, **k: candles(rows)
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Never reach the network: default to "no new candles"."""
    monkeypatch.setattr(
        "src.exchange.simulated_broker.load_ohlcv",
        lambda *a, **k: candles([]).iloc[:0],
    )


# ---------------------------------------------------------------------------
# Interface compatibility
# ---------------------------------------------------------------------------


def test_simulator_matches_the_exchange_client_interface():
    """The executor must not be able to tell the two apart."""
    from src.exchange.bybit_client import BybitPaperClient

    required = [
        "wallet_equity",
        "position",
        "last_price",
        "place_market_order",
        "close_position",
        "set_leverage",
    ]
    for name in required:
        assert hasattr(SimulatedBroker, name), f"simulator is missing {name}"
        assert hasattr(BybitPaperClient, name), f"exchange client is missing {name}"


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------


def test_opening_a_position_records_direction_size_and_price(broker):
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)
    position = broker.position()

    assert position["direction"] == 1
    assert position["size"] == pytest.approx(0.1)
    assert position["entry_price"] == pytest.approx(60_000.0)


def test_slippage_and_fees_match_the_backtester(tmp_path):
    """Both sides pay, and slippage always works against us."""
    broker = SimulatedBroker(
        repository=Repository(tmp_path / "s.db"), market=FakeMarket(60_000.0), config=REALISTIC
    )
    broker.place_market_order("BTCUSDT", 1, 0.1)

    slip = REALISTIC.slippage_bps / 10_000.0
    expected_fill = 60_000.0 * (1 + slip)
    expected_fee = 0.1 * expected_fill * REALISTIC.taker_fee

    assert broker.state["entry_price"] == pytest.approx(expected_fill)
    assert broker.state["equity"] == pytest.approx(10_000.0 - expected_fee)


def test_short_entry_slips_downward(tmp_path):
    broker = SimulatedBroker(
        repository=Repository(tmp_path / "s.db"), market=FakeMarket(60_000.0), config=REALISTIC
    )
    broker.place_market_order("BTCUSDT", -1, 0.1)
    assert broker.state["entry_price"] < 60_000.0, "selling must fill lower, not higher"


def test_closing_realises_pnl_into_equity(broker):
    broker.place_market_order("BTCUSDT", 1, 0.1)
    broker.market.price = 62_000.0
    broker.close_position()

    assert broker.state["direction"] == 0
    assert broker.state["equity"] == pytest.approx(10_000.0 + 200.0)


def test_short_profits_when_price_falls(broker):
    broker.place_market_order("BTCUSDT", -1, 0.1)
    broker.market.price = 58_000.0
    broker.close_position()
    assert broker.state["equity"] == pytest.approx(10_000.0 + 200.0)


def test_unrealised_pnl_is_marked_to_market(broker):
    broker.place_market_order("BTCUSDT", 1, 0.1)
    broker.market.price = 61_000.0
    assert broker.position()["unrealised_pnl"] == pytest.approx(100.0)
    assert broker.wallet_equity() == pytest.approx(10_100.0)


def test_quantity_rounding_to_zero_is_rejected(broker):
    with pytest.raises(ValueError, match="rounds to zero"):
        broker.place_market_order("BTCUSDT", 1, 0.0001)


# ---------------------------------------------------------------------------
# Protective orders - the part that must match the backtester
# ---------------------------------------------------------------------------


def test_a_stop_breached_between_polls_is_filled(broker, monkeypatch):
    """Price moves continuously while the loop sleeps for 30 seconds.

    Checking only the spot price at poll time would miss a stop that was
    breached and recovered in between, silently overstating performance.
    """
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)
    broker.market.price = 61_000.0  # recovered by the time we look

    set_candles(monkeypatch, [[60_000, 61_000, 57_500, 61_000]])
    position = broker.position()

    assert position["direction"] == 0, "the stop must have fired"
    assert broker.state["equity"] == pytest.approx(10_000.0 - 200.0)


def test_a_target_breached_between_polls_is_filled(broker, monkeypatch):
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)
    broker.market.price = 61_000.0

    set_candles(monkeypatch, [[60_000, 64_500, 59_500, 61_000]])
    broker.position()
    assert broker.state["equity"] == pytest.approx(10_000.0 + 400.0)


def test_a_candle_hitting_both_barriers_resolves_as_a_stop(broker, monkeypatch):
    """The backtester's pessimism rule, applied live."""
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)

    set_candles(monkeypatch, [[60_000, 65_000, 57_000, 61_000]])
    broker.position()

    assert broker.state["equity"] < 10_000.0, "ambiguous bar must not be given to us"


def test_a_gap_through_the_stop_fills_at_the_open(broker, monkeypatch):
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)

    set_candles(monkeypatch, [[55_000, 55_500, 54_000, 55_000]])
    broker.position()

    # Filled at 55,000 (the gapped open), not at the 58,000 stop.
    assert broker.state["equity"] == pytest.approx(10_000.0 - 500.0)


def test_a_short_stop_fires_on_a_rally(broker, monkeypatch):
    broker.place_market_order("BTCUSDT", -1, 0.1, stop_loss=62_000, take_profit=56_000)

    set_candles(monkeypatch, [[60_000, 62_500, 59_800, 62_000]])
    broker.position()

    assert broker.state["direction"] == 0
    assert broker.state["equity"] == pytest.approx(10_000.0 - 200.0)


def test_an_untouched_position_stays_open(broker, monkeypatch):
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)

    set_candles(monkeypatch, [[60_000, 61_000, 59_000, 60_500]])
    assert broker.position()["direction"] == 1


def test_a_data_outage_does_not_close_the_position(broker, monkeypatch):
    """A failed price fetch must not be mistaken for a stop-out."""
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000)

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.exchange.simulated_broker.load_ohlcv", boom)
    assert broker.position()["direction"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_state_survives_a_restart(tmp_path):
    """The trader must be restartable without losing its position."""
    path = tmp_path / "sim.db"
    first = SimulatedBroker(
        repository=Repository(path), market=FakeMarket(), config=FRICTIONLESS
    )
    first.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000)

    second = SimulatedBroker(
        repository=Repository(path), market=FakeMarket(), config=FRICTIONLESS
    )
    position = second.position()

    assert position["direction"] == 1
    assert position["size"] == pytest.approx(0.1)


def test_starting_equity_is_configurable(tmp_path):
    broker = SimulatedBroker(
        repository=Repository(tmp_path / "s.db"),
        market=FakeMarket(),
        config=FRICTIONLESS,
        starting_equity=50_000.0,
    )
    assert broker.wallet_equity() == pytest.approx(50_000.0)


# ---------------------------------------------------------------------------
# Integration with the executor
# ---------------------------------------------------------------------------


def test_the_executor_drives_the_simulator_unchanged(tmp_path, monkeypatch):
    """The whole point of the interface: the executor is venue-agnostic."""
    from config import settings
    from src.exchange import executor as executor_module

    monkeypatch.setattr(settings, "TRADING_MODE", "testnet")
    monkeypatch.setattr(executor_module, "assert_paper_mode", lambda: None)

    repository = Repository(tmp_path / "exec.db")
    broker = SimulatedBroker(
        repository=repository, market=FakeMarket(), config=FRICTIONLESS
    )
    set_candles(monkeypatch, [[60_000, 60_100, 59_900, 60_000]])

    paper = executor_module.PaperExecutor(
        client=broker, repository=repository, symbol="BTCUSDT", strategy_name="test"
    )
    plan = paper.execute(direction=1, price=60_000.0, atr=1_000.0)

    assert plan is not None
    assert broker.position()["direction"] == 1
    assert repository.open_trade() is not None


def test_settlement_is_throttled_between_calls(broker, monkeypatch):
    """The executor calls wallet_equity() and position() several times per
    cycle. Refetching candles on each one means four to six network round
    trips per poll and an unreadable log."""
    calls = {"n": 0}

    def counting_loader(*args, **kwargs):
        calls["n"] += 1
        return candles([[60_000, 60_100, 59_900, 60_000]])

    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)
    monkeypatch.setattr("src.exchange.simulated_broker.load_ohlcv", counting_loader)

    for _ in range(5):
        broker.position()
        broker.wallet_equity()

    assert calls["n"] == 1, f"expected one settlement, got {calls['n']}"


def test_throttle_expires_so_stops_still_fire(broker, monkeypatch):
    """Throttling must delay settlement, never prevent it."""
    broker.place_market_order("BTCUSDT", 1, 0.1, stop_loss=58_000, take_profit=64_000)
    broker.position()

    broker._last_settled = 0.0  # simulate the interval elapsing
    set_candles(monkeypatch, [[60_000, 61_000, 57_500, 61_000]])

    assert broker.position()["direction"] == 0, "the stop must still fire"
