"""Parameter search tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.optimizer import (
    best_params_per_strategy,
    candidate_params,
    search_strategy,
    selection_bias_report,
)
from src.strategies.breakout import DonchianVolumeBreakout
from src.strategies.momentum import StochRsiMacdMomentum
from src.strategies.trend import EmaRsiTrend


def test_incoherent_parameter_combinations_are_filtered_out():
    """A "fast" EMA slower than the slow EMA is not a strategy to test."""
    params = candidate_params(EmaRsiTrend, max_configs=1_000)
    assert params, "grid produced nothing"
    assert all(p.ema_fast < p.ema_slow for p in params)

    full_grid = 4 * 4 * 3 * 2  # ema_fast x ema_slow x rsi_period x adx_min
    assert len(params) < full_grid, "invalid combinations were not removed"


def test_breakout_exit_channel_must_be_shorter_than_the_entry_channel():
    params = candidate_params(DonchianVolumeBreakout, max_configs=1_000)
    assert all(p.exit_period < p.channel_period for p in params)


def test_momentum_thresholds_must_be_ordered():
    params = candidate_params(StochRsiMacdMomentum, max_configs=1_000)
    assert all(p.oversold < p.overbought for p in params)


def test_sampling_respects_the_budget_and_is_deterministic():
    first = candidate_params(EmaRsiTrend, max_configs=5, seed=1)
    second = candidate_params(EmaRsiTrend, max_configs=5, seed=1)
    different = candidate_params(EmaRsiTrend, max_configs=5, seed=2)

    assert len(first) == 5
    assert [p.label() for p in first] == [p.label() for p in second], "seed must reproduce"
    assert [p.label() for p in first] != [p.label() for p in different]


def test_search_scores_every_configuration(ohlcv):
    board = search_strategy(EmaRsiTrend, ohlcv, "1h", max_configs=6)

    assert len(board) == 6
    assert {"strategy", "params", "sortino", "eligible", "score"} <= set(board.columns)
    assert board["params"].nunique() == 6, "configurations must be distinct"


def test_indicator_caching_does_not_change_results(ohlcv):
    """Configurations sharing an indicator spec reuse one prepared frame.

    That optimisation must be invisible in the output: verify against a
    single-configuration search that cannot have used the cache.
    """
    from src.backtesting.engine import run_backtest
    from src.backtesting.metrics import compute_metrics

    params = candidate_params(EmaRsiTrend, max_configs=4, seed=1)
    board = search_strategy(EmaRsiTrend, ohlcv, "1h", max_configs=4, seed=1)

    target = params[0]
    strategy = EmaRsiTrend(target)
    direct = compute_metrics(run_backtest(strategy.run(ohlcv)), "1h")

    row = board[board["params"] == target.label()].iloc[0]
    assert row["total_return"] == pytest.approx(direct["total_return"])
    assert row["n_trades"] == direct["n_trades"]


def test_selection_bias_report_measures_the_spread():
    board = pd.DataFrame(
        {
            "sortino": [2.0, 1.0, 1.0, 0.0, -1.0],
            "eligible": [True, True, False, False, False],
        }
    )
    report = selection_bias_report(board)

    assert report["configurations"] == 5
    assert report["best"] == 2.0
    assert report["median"] == 1.0
    assert report["spread_over_median"] == 1.0
    assert report["share_positive"] == pytest.approx(0.6)


def test_selection_bias_report_survives_infinities():
    board = pd.DataFrame({"sortino": [np.inf, 1.0, np.nan], "eligible": [True, True, False]})
    report = selection_bias_report(board)
    assert np.isfinite(report["best"])


def test_best_params_prefers_eligible_configurations(ohlcv):
    board = pd.DataFrame(
        {
            "strategy": ["ema_rsi_trend", "ema_rsi_trend"],
            "timeframe": ["1h", "1h"],
            "params": ["lucky", "solid"],
            "sortino": [9.0, 1.5],
            "sharpe": [8.0, 1.4],
            "total_return": [5.0, 0.4],
            "n_trades": [3, 120],
            "eligible": [False, True],
        }
    )
    winners = best_params_per_strategy(board, {"ema_rsi_trend": EmaRsiTrend})
    assert winners["ema_rsi_trend"]["params"] == "solid"
