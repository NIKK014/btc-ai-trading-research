"""Backtest engine correctness tests.

Every number in the final A/B/C comparison flows through this engine, so its
arithmetic is checked against P&L computed by hand rather than against its own
output. Each test constructs a tiny price series where the correct answer can
be worked out on paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import BacktestConfig, RiskConfig
from src.backtesting.engine import (
    EXIT_DAILY_LIMIT,
    EXIT_END_OF_DATA,
    EXIT_REVERSAL,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TARGET,
    run_backtest,
)

FRICTIONLESS = BacktestConfig(initial_capital=10_000.0, taker_fee=0.0, slippage_bps=0.0)
REALISTIC = BacktestConfig(initial_capital=10_000.0, taker_fee=0.00055, slippage_bps=2.0)
STOP_RISK = RiskConfig(atr_stop_multiple=2.0, reward_risk_ratio=2.0, max_daily_loss=0.03)


def make_frame(
    opens,
    highs,
    lows,
    closes,
    signals,
    atr=None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Build a minimal backtest frame from explicit price arrays."""
    index = pd.date_range("2024-01-01", periods=len(opens), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": np.asarray(opens, dtype=float),
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "close": np.asarray(closes, dtype=float),
            "atr": np.full(len(opens), atr, dtype=float)
            if atr is not None
            else np.full(len(opens), np.nan),
            "signal": np.asarray(signals, dtype=np.int8),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Fill timing - the foundation everything else rests on
# ---------------------------------------------------------------------------


def test_entry_fills_at_the_next_bar_open_not_the_signal_bar_close():
    """A signal on the close of bar t must fill at the open of bar t+1."""
    frame = make_frame(
        opens=[100, 200, 300, 400],
        highs=[100, 200, 300, 400],
        lows=[100, 200, 300, 400],
        closes=[100, 200, 300, 400],
        signals=[1, 1, 0, 0],
    )
    result = run_backtest(frame, config=FRICTIONLESS, use_stops=False, sizing="full_notional")

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == frame.index[1], "must not fill on the signal bar"
    assert trade["entry_price"] == pytest.approx(200.0), "must fill at the open, not the close"


def test_exit_fills_at_the_next_bar_open_after_the_signal_clears():
    frame = make_frame(
        opens=[100, 100, 105, 110, 110],
        highs=[100, 100, 105, 110, 110],
        lows=[100, 100, 105, 110, 110],
        closes=[100, 100, 105, 110, 110],
        signals=[1, 1, 0, 0, 0],
    )
    result = run_backtest(frame, config=FRICTIONLESS, use_stops=False, sizing="full_notional")

    trade = result.trades.iloc[0]
    assert trade["exit_time"] == frame.index[3]
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["exit_reason"] == EXIT_SIGNAL


# ---------------------------------------------------------------------------
# P&L arithmetic
# ---------------------------------------------------------------------------


def test_frictionless_pnl_matches_hand_calculation():
    """10,000 at 100 buys 100 units; selling at 110 makes exactly 1,000."""
    frame = make_frame(
        opens=[100, 100, 105, 110, 110],
        highs=[100, 100, 105, 110, 110],
        lows=[100, 100, 105, 110, 110],
        closes=[100, 100, 105, 110, 110],
        signals=[1, 1, 0, 0, 0],
    )
    result = run_backtest(frame, config=FRICTIONLESS, use_stops=False, sizing="full_notional")

    trade = result.trades.iloc[0]
    assert trade["size"] == pytest.approx(100.0)
    assert trade["pnl"] == pytest.approx(1_000.0)
    assert trade["fees"] == pytest.approx(0.0)
    assert result.final_equity == pytest.approx(11_000.0)


def test_fees_and_slippage_match_an_independent_calculation():
    """Recompute the whole trade from first principles, not from the engine."""
    frame = make_frame(
        opens=[100, 100, 105, 110, 110],
        highs=[100, 100, 105, 110, 110],
        lows=[100, 100, 105, 110, 110],
        closes=[100, 100, 105, 110, 110],
        signals=[1, 1, 0, 0, 0],
    )
    result = run_backtest(frame, config=REALISTIC, use_stops=False, sizing="full_notional")

    slip = REALISTIC.slippage_bps / 10_000.0
    fee_rate = REALISTIC.taker_fee

    entry_fill = 100 * (1 + slip)          # buying: we pay up
    size = 10_000 / entry_fill
    entry_fee = size * entry_fill * fee_rate
    exit_fill = 110 * (1 - slip)           # selling: we receive less
    exit_fee = size * exit_fill * fee_rate
    expected_pnl = size * (exit_fill - entry_fill) - entry_fee - exit_fee

    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(entry_fill)
    assert trade["exit_price"] == pytest.approx(exit_fill)
    assert trade["pnl"] == pytest.approx(expected_pnl)
    assert trade["fees"] == pytest.approx(entry_fee + exit_fee)
    assert result.final_equity == pytest.approx(10_000 + expected_pnl)


def test_short_trades_profit_when_price_falls():
    frame = make_frame(
        opens=[100, 100, 95, 90, 90],
        highs=[100, 100, 95, 90, 90],
        lows=[100, 100, 95, 90, 90],
        closes=[100, 100, 95, 90, 90],
        signals=[-1, -1, 0, 0, 0],
    )
    result = run_backtest(frame, config=FRICTIONLESS, use_stops=False, sizing="full_notional")

    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    assert trade["pnl"] == pytest.approx(1_000.0)


def test_fees_strictly_reduce_returns():
    frame = make_frame(
        opens=[100, 100, 105, 110, 110],
        highs=[100, 100, 105, 110, 110],
        lows=[100, 100, 105, 110, 110],
        closes=[100, 100, 105, 110, 110],
        signals=[1, 1, 0, 0, 0],
    )
    cheap = run_backtest(frame, config=FRICTIONLESS, use_stops=False, sizing="full_notional")
    dear = run_backtest(frame, config=REALISTIC, use_stops=False, sizing="full_notional")
    assert dear.final_equity < cheap.final_equity


# ---------------------------------------------------------------------------
# Stops and targets
# ---------------------------------------------------------------------------


def test_stop_loss_fires_at_the_stop_price():
    """ATR 1, multiple 2 -> stop 2 below entry."""
    frame = make_frame(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 101, 100],
        lows=[100, 100, 97, 100],   # breaches a stop at 98
        closes=[100, 100, 99, 100],
        signals=[1, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == EXIT_STOP
    assert trade["stop_price"] == pytest.approx(98.0)
    assert trade["exit_price"] == pytest.approx(98.0)


def test_take_profit_fires_at_the_target_price():
    """Risk 2, reward:risk 2:1 -> target 4 above entry."""
    frame = make_frame(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 105, 100],   # breaches a target at 104
        lows=[100, 100, 99, 100],
        closes=[100, 100, 104, 100],
        signals=[1, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == EXIT_TARGET
    assert trade["target_price"] == pytest.approx(104.0)
    assert trade["exit_price"] == pytest.approx(104.0)


def test_same_bar_stop_and_target_resolves_against_us():
    """The single most important pessimism rule in the engine.

    OHLCV cannot say whether the high or the low came first, so a bar
    containing both the stop and the target must be treated as a stop-out.
    """
    frame = make_frame(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 106, 100],   # target 104 breached
        lows=[100, 100, 96, 100],     # stop 98 also breached
        closes=[100, 100, 100, 100],
        signals=[1, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == EXIT_STOP, "ambiguous bar must not be given to us"
    assert trade["pnl"] < 0


def test_gap_through_the_stop_fills_at_the_open():
    """If price gapped past the stop, you did not get your stop."""
    frame = make_frame(
        opens=[100, 100, 95, 100],    # bar 2 gaps below the stop at 98
        highs=[100, 100, 95, 100],
        lows=[100, 100, 94, 100],
        closes=[100, 100, 95, 100],
        signals=[1, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == EXIT_STOP
    assert trade["exit_price"] == pytest.approx(95.0), "must fill at the gapped open, not 98"


def test_gap_through_the_target_gives_no_bonus():
    """A favourable gap fills at the limit price, not the better open."""
    frame = make_frame(
        opens=[100, 100, 108, 100],   # bar 2 gaps above the target at 104
        highs=[100, 100, 110, 100],
        lows=[100, 100, 107, 100],
        closes=[100, 100, 108, 100],
        signals=[1, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == EXIT_TARGET
    assert trade["exit_price"] == pytest.approx(104.0)


def test_short_stop_sits_above_the_entry():
    frame = make_frame(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 103, 100],   # breaches a short stop at 102
        lows=[100, 100, 99, 100],
        closes=[100, 100, 102, 100],
        signals=[-1, -1, -1, -1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )

    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    assert trade["stop_price"] == pytest.approx(102.0)
    assert trade["exit_reason"] == EXIT_STOP


def test_risk_based_sizing_loses_exactly_the_configured_risk():
    """The point of risk sizing: a stop-out costs 1% of equity, whatever the
    volatility. Verified with fees off so the arithmetic is exact."""
    risk = RiskConfig(risk_per_trade=0.01, atr_stop_multiple=2.0, reward_risk_ratio=2.0)
    frame = make_frame(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 101, 100],
        lows=[100, 100, 97, 100],
        closes=[100, 100, 99, 100],
        signals=[1, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(frame, config=FRICTIONLESS, risk=risk, sizing="risk")

    trade = result.trades.iloc[0]
    assert trade["pnl"] == pytest.approx(-100.0), "1% of 10,000"
    assert trade["r_multiple"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Position state machine
# ---------------------------------------------------------------------------


def test_reversal_closes_and_reopens_in_the_same_bar():
    frame = make_frame(
        opens=[100, 100, 110, 110, 110],
        highs=[100, 100, 110, 110, 110],
        lows=[100, 100, 110, 110, 110],
        closes=[100, 100, 110, 110, 110],
        signals=[1, -1, -1, -1, -1],
    )
    result = run_backtest(frame, config=FRICTIONLESS, use_stops=False, sizing="full_notional")

    assert result.trades.iloc[0]["exit_reason"] == EXIT_REVERSAL
    assert result.trades.iloc[0]["exit_time"] == frame.index[2]
    assert result.trades.iloc[1]["direction"] == -1
    assert result.trades.iloc[1]["entry_time"] == frame.index[2], "reversal is same-bar"


def test_re_entry_is_blocked_until_the_signal_resets():
    """Without this rule a persistent signal re-buys immediately after every
    stop-out and grinds the account away in fees."""
    frame = make_frame(
        opens=[100] * 8,
        highs=[100, 100, 101, 100, 100, 100, 100, 100],
        lows=[100, 100, 97, 100, 100, 100, 100, 100],
        closes=[100] * 8,
        signals=[1] * 8,
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )

    assert len(result.trades) == 1, "signal never reset, so no re-entry is allowed"
    assert result.trades.iloc[0]["exit_reason"] == EXIT_STOP


def test_re_entry_resumes_once_the_signal_changes():
    frame = make_frame(
        opens=[100] * 8,
        highs=[100, 100, 101, 100, 100, 100, 100, 100],
        lows=[100, 100, 97, 100, 100, 100, 100, 100],
        closes=[100] * 8,
        signals=[1, 1, 1, 0, 0, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=STOP_RISK, sizing="full_notional"
    )
    assert len(result.trades) == 2


def test_open_position_is_closed_at_the_end_of_the_data():
    frame = make_frame(
        opens=[100, 100, 110, 120],
        highs=[100, 100, 110, 120],
        lows=[100, 100, 110, 120],
        closes=[100, 100, 110, 120],
        signals=[1, 1, 1, 1],
    )
    result = run_backtest(frame, config=FRICTIONLESS, use_stops=False, sizing="full_notional")

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == EXIT_END_OF_DATA
    assert result.position.iloc[-1] == 0


# ---------------------------------------------------------------------------
# Risk controls
# ---------------------------------------------------------------------------


def test_daily_loss_limit_closes_the_position_and_blocks_re_entry():
    risk = RiskConfig(risk_per_trade=1.0, atr_stop_multiple=50.0, max_daily_loss=0.03)
    # A 5% adverse move inside one UTC day, with the stop far enough away that
    # the daily limit is what triggers, not the stop.
    frame = make_frame(
        opens=[100, 100, 100, 95, 95, 95],
        highs=[100, 100, 100, 95, 95, 95],
        lows=[100, 100, 95, 95, 95, 95],
        closes=[100, 100, 95, 95, 95, 95],
        signals=[1, 1, 1, 1, 1, 1],
        atr=1.0,
    )
    result = run_backtest(
        frame, config=FRICTIONLESS, risk=risk, sizing="full_notional"
    )

    assert result.trades.iloc[0]["exit_reason"] == EXIT_DAILY_LIMIT
    assert len(result.trades) == 1, "no new positions for the rest of the day"


def test_no_trade_is_taken_while_atr_is_undefined():
    """No ATR means no stop can be placed, so the trade must be skipped."""
    frame = make_frame(
        opens=[100] * 4,
        highs=[100] * 4,
        lows=[100] * 4,
        closes=[100] * 4,
        signals=[1] * 4,
        atr=None,
    )
    result = run_backtest(frame, risk=STOP_RISK, config=FRICTIONLESS, sizing="risk")
    assert result.trades.empty


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------


def test_backtest_is_invariant_to_future_data(ohlcv):
    """Appending future bars must not change past equity.

    The same truncation argument used for the indicators, applied to the whole
    execution path. The final bar is excluded because a truncated run
    force-closes its open position there.
    """
    from src.strategies.base import build

    strategy = build("ema_rsi_trend")
    prepared = strategy.run(ohlcv)
    cut = 400

    full = run_backtest(prepared, config=REALISTIC)
    partial = run_backtest(prepared.iloc[:cut], config=REALISTIC)

    pd.testing.assert_series_equal(
        full.equity.iloc[: cut - 1],
        partial.equity.iloc[:-1],
        rtol=1e-9,
        obj="equity curve changed when future bars were appended",
    )


def test_equity_curve_is_finite_and_aligned(ohlcv):
    from src.strategies.base import build

    strategy = build("macd_adx_trend")
    result = run_backtest(strategy.run(ohlcv), config=REALISTIC)

    assert result.equity.index.equals(ohlcv.index)
    assert np.isfinite(result.equity.to_numpy()).all()
    assert (result.equity > 0).all(), "1x leverage with a 1% risk cap cannot bankrupt"


def test_missing_columns_are_rejected(ohlcv):
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(ohlcv)


def test_unsorted_input_is_rejected(ohlcv):
    from src.strategies.base import build

    prepared = build("ema_rsi_trend").run(ohlcv)
    with pytest.raises(ValueError, match="chronologically"):
        run_backtest(prepared.iloc[::-1])
