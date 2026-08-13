"""Parameter search.

Deliberately small
------------------
Searching thousands of configurations does not find a better strategy, it
finds a luckier one. With 50 configurations tested on one validation period,
the best result is roughly what you would expect the maximum of 50 noisy draws
to look like even if every configuration were worthless. The search is
therefore capped, and every report includes the **distribution** across the
grid rather than only its maximum - because the gap between the best result
and the median result is a direct measure of how much of the winner is luck.

Efficiency
----------
Recomputing indicators dominates the cost, and many parameters (RSI
thresholds, volume filters, ADX ceilings) do not change any indicator period.
Configurations are therefore grouped by their :class:`IndicatorSpec` so the
prepared frame is computed once per distinct spec and reused.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config.settings import BACKTEST, METRICS, RISK, BacktestConfig, MetricsConfig, RiskConfig
from src.backtesting.engine import run_backtest
from src.backtesting.metrics import build_leaderboard, compute_metrics
from src.indicators.indicators import add_indicators
from src.strategies.base import StrategyParams
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def candidate_params(
    strategy_cls: type,
    max_configs: int = 40,
    seed: int = 42,
) -> List[StrategyParams]:
    """Valid parameter sets for a strategy, sampled down to ``max_configs``.

    Combinations the strategy declares nonsensical (a fast EMA slower than the
    slow EMA, an oversold level above the overbought level) are dropped before
    sampling, so the budget is not wasted on configurations that could never
    trade sensibly.
    """
    all_params = [p for p in strategy_cls.iter_param_sets() if strategy_cls.is_valid(p)]
    if len(all_params) <= max_configs:
        return all_params

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_params), size=max_configs, replace=False)
    logger.info(
        "%s: sampling %d of %d valid configurations",
        strategy_cls.name,
        max_configs,
        len(all_params),
    )
    return [all_params[i] for i in sorted(chosen)]


def search_strategy(
    strategy_cls: type,
    ohlcv: pd.DataFrame,
    timeframe: str,
    *,
    max_configs: int = 40,
    seed: int = 42,
    config: BacktestConfig = BACKTEST,
    risk: RiskConfig = RISK,
) -> pd.DataFrame:
    """Evaluate every candidate configuration of one strategy on one timeframe.

    Returns:
        A scored leaderboard of configurations, ranked by the primary metric
        among those passing the eligibility gates.
    """
    params_list = candidate_params(strategy_cls, max_configs, seed)

    # Group by indicator spec so identical indicator frames are computed once.
    by_spec: Dict[Tuple, List[StrategyParams]] = {}
    for params in params_list:
        spec_key = tuple(sorted(asdict(strategy_cls(params).indicator_spec).items()))
        by_spec.setdefault(spec_key, []).append(params)

    is_benchmark = strategy_cls.methodology == "benchmark"
    rows: List[Dict[str, Any]] = []

    for group in by_spec.values():
        prepared_base = add_indicators(ohlcv, strategy_cls(group[0]).indicator_spec)
        for params in group:
            strategy = strategy_cls(params)
            frame = prepared_base.copy()
            frame["signal"] = strategy.generate_signals(frame)
            result = run_backtest(
                frame,
                config=config,
                risk=risk,
                use_stops=not is_benchmark,
                sizing="full_notional" if is_benchmark else "risk",
                enforce_daily_limit=not is_benchmark,
            )
            metrics = compute_metrics(result, timeframe)
            metrics.update(
                strategy=strategy_cls.name,
                methodology=strategy_cls.methodology,
                params=params.label(),
                **{f"p_{k}": v for k, v in params.to_dict().items()},
            )
            rows.append(metrics)

    logger.info(
        "%s @ %s: evaluated %d configurations across %d indicator specs",
        strategy_cls.name,
        timeframe,
        len(rows),
        len(by_spec),
    )
    return build_leaderboard(rows)


def selection_bias_report(
    leaderboard: pd.DataFrame,
    metric: str = "sortino",
    config: MetricsConfig = METRICS,
) -> Dict[str, float]:
    """Quantify how much of the winner's edge could be search luck.

    ``spread`` is the winner's metric minus the grid median. A large spread on
    a small grid is a warning, not a triumph: it means the result is highly
    sensitive to parameters, which rarely survives out of sample. A winner that
    is barely better than its neighbours is usually the more robust choice.
    """
    if leaderboard.empty or metric not in leaderboard.columns:
        return {}

    values = leaderboard[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {}

    eligible = leaderboard[leaderboard.get("eligible", True)]
    return {
        "configurations": int(len(leaderboard)),
        "eligible": int(len(eligible)),
        "best": float(values.max()),
        "median": float(values.median()),
        "worst": float(values.min()),
        "spread_over_median": float(values.max() - values.median()),
        "share_positive": float((values > 0).mean()),
    }


def optimise(
    strategies: Dict[str, type],
    ohlcv_by_timeframe: Dict[str, pd.DataFrame],
    *,
    max_configs: int = 40,
    seed: int = 42,
    config: BacktestConfig = BACKTEST,
    risk: RiskConfig = RISK,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Run parameter search for every strategy on every timeframe.

    Returns:
        ``(combined_leaderboard, diagnostics)`` where ``diagnostics`` maps
        ``"strategy@timeframe"`` to its selection-bias report.
    """
    frames: List[pd.DataFrame] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}

    for timeframe, ohlcv in ohlcv_by_timeframe.items():
        for name, cls in strategies.items():
            board = search_strategy(
                cls,
                ohlcv,
                timeframe,
                max_configs=max_configs,
                seed=seed,
                config=config,
                risk=risk,
            )
            if board.empty:
                continue
            diagnostics[f"{name}@{timeframe}"] = selection_bias_report(board)
            frames.append(board)

    if not frames:
        return pd.DataFrame(), diagnostics

    combined = pd.concat(frames, ignore_index=True)
    return build_leaderboard(combined.to_dict("records")), diagnostics


def best_params_per_strategy(
    leaderboard: pd.DataFrame,
    strategies: Dict[str, type],
) -> Dict[str, Dict[str, Any]]:
    """Extract the winning configuration for each strategy from a leaderboard."""
    winners: Dict[str, Dict[str, Any]] = {}
    if leaderboard.empty:
        return winners

    eligible = leaderboard[leaderboard.get("eligible", True)]
    source = eligible if not eligible.empty else leaderboard

    for name in strategies:
        subset = source[source["strategy"] == name]
        if subset.empty:
            continue
        row = subset.sort_values(METRICS.primary_metric, ascending=False).iloc[0]
        # ``p_*`` columns carry the raw parameter values, so the winning
        # configuration can be rebuilt programmatically rather than by parsing
        # its label string.
        values = {
            key[2:]: row[key]
            for key in row.index
            if key.startswith("p_") and pd.notna(row[key])
        }
        winners[name] = {
            "timeframe": row["timeframe"],
            "params": row["params"],
            "param_values": values,
            "sortino": row.get("sortino"),
            "sharpe": row.get("sharpe"),
            "total_return": row.get("total_return"),
            "n_trades": row.get("n_trades"),
            "eligible": bool(row.get("eligible", False)),
        }
    return winners
