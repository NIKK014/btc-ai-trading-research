"""Metrics, gates and confidence-interval tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import MetricsConfig
from src.backtesting.metrics import (
    build_leaderboard,
    cagr,
    check_gates,
    difference_ci,
    drawdown_series,
    max_drawdown,
    metrics_by_period,
    profit_factor,
    select_winner,
    sharpe_ratio,
    sortino_ratio,
    trade_returns,
)

FAST_CONFIG = MetricsConfig(bootstrap_samples=200)


def equity_curve(values, freq="1h") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=index, name="equity")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_max_drawdown_matches_hand_calculation():
    """Peak 120, trough 60 -> 50% drawdown, and a later recovery must not hide it."""
    equity = equity_curve([100, 120, 90, 60, 80, 130])
    assert max_drawdown(equity) == pytest.approx(0.5)


def test_drawdown_is_zero_at_new_highs():
    equity = equity_curve([100, 110, 120, 130])
    assert max_drawdown(equity) == pytest.approx(0.0)
    assert (drawdown_series(equity) == 0).all()


def test_cagr_of_a_doubling_over_one_year():
    index = pd.date_range("2024-01-01", "2025-01-01", freq="D", tz="UTC")
    equity = pd.Series(np.linspace(10_000, 20_000, len(index)), index=index)
    assert cagr(equity) == pytest.approx(1.0, rel=0.02)


def test_sharpe_is_zero_for_a_flat_curve():
    returns = pd.Series(np.zeros(500))
    assert sharpe_ratio(returns, 8760) == 0.0


def test_sharpe_scales_with_the_annualisation_factor():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 5_000))
    hourly = sharpe_ratio(returns, 8760)
    daily = sharpe_ratio(returns, 365)
    assert hourly == pytest.approx(daily * np.sqrt(8760 / 365))


def test_sortino_ignores_upside_volatility():
    """Two series with identical downside but different upside: Sortino must
    reward the one with bigger winners, where Sharpe would punish it."""
    mild = pd.Series([0.01, -0.01] * 100)
    explosive = pd.Series([0.05, -0.01] * 100)

    assert sortino_ratio(explosive, 365) > sortino_ratio(mild, 365)
    assert sharpe_ratio(explosive, 365) < sortino_ratio(explosive, 365)


def test_profit_factor_arithmetic():
    assert profit_factor([100, 50, -75]) == pytest.approx(2.0)
    assert profit_factor([100, 200]) == float("inf")
    assert np.isnan(profit_factor([]))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_gates_reject_a_tiny_sample_however_spectacular():
    """The headline purpose of the gates."""
    lucky = {"n_trades": 3, "max_drawdown": 0.05, "profit_factor": 12.0, "total_return": 9.0}
    eligible, reasons = check_gates(lucky)
    assert not eligible
    assert "only 3 trades" in reasons[0]


def test_gates_reject_excessive_drawdown():
    reckless = {"n_trades": 200, "max_drawdown": 0.75, "profit_factor": 2.0}
    eligible, reasons = check_gates(reckless)
    assert not eligible
    assert any("drawdown" in r for r in reasons)


def test_gates_reject_unprofitable_strategies():
    losing = {"n_trades": 200, "max_drawdown": 0.2, "profit_factor": 0.8}
    eligible, reasons = check_gates(losing)
    assert not eligible
    assert any("profit factor" in r for r in reasons)


def test_gates_accept_a_reasonable_strategy():
    sound = {"n_trades": 120, "max_drawdown": 0.22, "profit_factor": 1.4}
    eligible, reasons = check_gates(sound)
    assert eligible
    assert reasons == []


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _row(name, trades, ret, sharpe, dd, pf, wr, sortino):
    return {
        "strategy": name,
        "n_trades": trades,
        "total_return": ret,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "profit_factor": pf,
        "win_rate": wr,
        "sortino": sortino,
    }


def test_ineligible_strategies_are_ranked_below_eligible_ones():
    board = build_leaderboard(
        [
            _row("lucky", 4, 12.0, 6.0, 0.02, 30.0, 1.0, 9.0),
            _row("solid", 150, 0.60, 1.4, 0.20, 1.6, 0.55, 1.9),
        ]
    )
    assert board.iloc[0]["strategy"] == "solid"
    assert bool(board.iloc[0]["eligible"]) is True
    assert bool(board.iloc[1]["eligible"]) is False


def test_select_winner_ignores_ineligible_rows():
    board = build_leaderboard(
        [
            _row("lucky", 4, 12.0, 6.0, 0.02, 30.0, 1.0, 9.0),
            _row("solid", 150, 0.60, 1.4, 0.20, 1.6, 0.55, 1.9),
        ]
    )
    assert select_winner(board)["strategy"] == "solid"


def test_select_winner_returns_none_when_nothing_qualifies():
    board = build_leaderboard([_row("bad", 5, -0.5, -1.0, 0.7, 0.4, 0.2, -1.0)])
    assert select_winner(board) is None


def test_display_score_is_robust_to_an_outlier():
    """Rank-based normalisation must not let one absurd row flatten the rest.

    With min-max scaling the 5000% return would compress every other strategy
    into the bottom of the range; with percentile ranks the ordering of the
    sensible rows is preserved.
    """
    rows = [
        _row("outlier", 100, 50.0, 1.0, 0.30, 1.2, 0.50, 1.0),
        _row("good", 100, 0.80, 2.5, 0.10, 2.2, 0.60, 3.0),
        _row("ok", 100, 0.40, 1.5, 0.20, 1.5, 0.55, 1.8),
    ]
    board = build_leaderboard(rows).set_index("strategy")
    assert board.loc["good", "score"] > board.loc["ok", "score"]
    assert board.loc["good", "score"] > board.loc["outlier", "score"]


def test_leaderboard_handles_infinite_profit_factor():
    board = build_leaderboard([_row("nolosses", 100, 1.0, 2.0, 0.1, np.inf, 1.0, 3.0)])
    assert np.isfinite(board.iloc[0]["score"])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _trades_from_returns(returns, start_equity=10_000.0) -> pd.DataFrame:
    equity = start_equity
    rows = []
    for r in returns:
        pnl = equity * r
        equity += pnl
        rows.append({"pnl": pnl, "equity_after": equity})
    return pd.DataFrame(rows)


def test_trade_returns_recovers_the_input_returns():
    returns = [0.02, -0.01, 0.03, -0.015]
    recovered = trade_returns(_trades_from_returns(returns))
    np.testing.assert_allclose(recovered, returns, rtol=1e-9)


def test_difference_ci_detects_a_large_real_difference():
    rng = np.random.default_rng(3)
    weak = _trades_from_returns(rng.normal(0.000, 0.01, 300))
    strong = _trades_from_returns(rng.normal(0.010, 0.01, 300))

    result = difference_ci(weak, strong, config=FAST_CONFIG)
    assert result["significant"] is True
    assert result["ci_low"] > 0
    assert result["probability_b_better"] > 0.95


def test_difference_ci_reports_no_significance_for_identical_systems():
    """The result the project most needs to be able to state honestly."""
    rng = np.random.default_rng(4)
    a = _trades_from_returns(rng.normal(0.001, 0.02, 150))
    b = _trades_from_returns(rng.normal(0.001, 0.02, 150))

    result = difference_ci(a, b, config=FAST_CONFIG)
    assert result["significant"] is False
    assert result["ci_low"] < 0 < result["ci_high"]


def test_difference_ci_is_empty_without_enough_trades():
    assert difference_ci(pd.DataFrame(), pd.DataFrame(), config=FAST_CONFIG) == {}


# ---------------------------------------------------------------------------
# Integration with the engine
# ---------------------------------------------------------------------------


def test_metrics_and_regime_split_run_on_a_real_backtest(ohlcv):
    from src.backtesting.engine import run_backtest
    from src.backtesting.metrics import compute_metrics, summarise
    from src.strategies.base import build

    strategy = build("ema_rsi_trend")
    result = run_backtest(strategy.run(ohlcv))
    metrics = compute_metrics(result, "1h")

    assert metrics["bars"] == len(ohlcv)
    assert 0.0 <= metrics["exposure"] <= 1.0
    assert np.isfinite(metrics["max_drawdown"])
    assert isinstance(summarise(result, "1h", "test"), str)

    midpoint = ohlcv.index[len(ohlcv) // 2]
    split = metrics_by_period(
        result,
        {"first_half": (ohlcv.index[0], midpoint), "second_half": (midpoint, ohlcv.index[-1])},
        "1h",
    )
    assert list(split["period"]) == ["first_half", "second_half"]
