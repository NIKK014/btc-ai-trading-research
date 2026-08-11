"""Orchestration: run many strategy/timeframe combinations and rank them.

Keeps the split discipline in one place. Nothing here can accidentally read the
test period, because :func:`get_split` is the only way to obtain date ranges
and it refuses to hand out the test window unless explicitly unlocked.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config.settings import BACKTEST, RISK, SPLIT, BacktestConfig, RiskConfig, SplitConfig
from src.backtesting.engine import BacktestResult, run_backtest
from src.backtesting.metrics import build_leaderboard, compute_metrics
from src.data.loader import load_ohlcv
from src.strategies.base import Strategy, StrategyParams, registry
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


def get_split(
    name: str,
    config: SplitConfig = SPLIT,
    unlock_test: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Date bounds for a named split.

    The test period is deliberately awkward to obtain. Requesting it requires
    ``unlock_test=True``, which exists so that touching the out-of-sample data
    is always a visible, deliberate act in the calling code rather than
    something that can happen by accident during exploration.
    """
    if name == TRAIN:
        return None, config.train_end
    if name == VALIDATION:
        return config.train_end, config.validation_end
    if name == TEST:
        if not unlock_test:
            raise PermissionError(
                "The test period may only be read once, for the final comparison. "
                "Pass unlock_test=True if that is genuinely what you are doing."
            )
        return config.validation_end, None
    raise ValueError(f"Unknown split {name!r}")


def apply_embargo(frame: pd.DataFrame, config: SplitConfig = SPLIT) -> pd.DataFrame:
    """Drop the first ``embargo_bars`` rows of a split.

    Labels look ``horizon`` bars into the future, so samples either side of a
    split boundary overlap. Removing a small buffer at the start of each split
    stops that overlap leaking information across the seam.
    """
    return frame.iloc[config.embargo_bars :] if config.embargo_bars else frame


def backtest_strategy(
    strategy: Strategy,
    ohlcv: pd.DataFrame,
    *,
    config: BacktestConfig = BACKTEST,
    risk: RiskConfig = RISK,
) -> BacktestResult:
    """Prepare indicators, generate signals and replay them.

    Benchmarks run without stops and fully invested, so buy-and-hold is a real
    buy-and-hold rather than a stopped-out approximation of one.
    """
    prepared = strategy.run(ohlcv)
    is_benchmark = strategy.methodology == "benchmark"
    return run_backtest(
        prepared,
        config=config,
        risk=risk,
        use_stops=not is_benchmark,
        sizing="full_notional" if is_benchmark else "risk",
        enforce_daily_limit=not is_benchmark,
    )


def run_grid(
    timeframes: Iterable[str],
    split: str = VALIDATION,
    strategies: Optional[Dict[str, type]] = None,
    *,
    param_sets: Optional[Dict[str, List[StrategyParams]]] = None,
    config: BacktestConfig = BACKTEST,
    risk: RiskConfig = RISK,
    unlock_test: bool = False,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], BacktestResult]]:
    """Backtest every strategy across every timeframe on one split.

    Args:
        timeframes: Timeframe labels to evaluate.
        split: ``"train"``, ``"validation"`` or ``"test"``.
        strategies: Name -> class. Defaults to the full registry.
        param_sets: Optional per-strategy parameter sets, for grid search.

    Returns:
        ``(leaderboard, results)`` where ``results`` is keyed by
        ``(strategy_name, timeframe, params_label)``.
    """
    strategies = strategies or registry()
    start, end = get_split(split, unlock_test=unlock_test)

    rows: List[Dict[str, Any]] = []
    results: Dict[Tuple[str, str, str], BacktestResult] = {}

    for timeframe in timeframes:
        ohlcv = apply_embargo(load_ohlcv(timeframe, start=start, end=end))
        if ohlcv.empty:
            logger.warning("No %s data in split %s", timeframe, split)
            continue

        for name, cls in strategies.items():
            candidates = (
                param_sets.get(name, [cls.params_class()])
                if param_sets
                else [cls.params_class()]
            )
            for params in candidates:
                strategy = cls(params)
                result = backtest_strategy(strategy, ohlcv, config=config, risk=risk)
                metrics = compute_metrics(result, timeframe)
                metrics.update(
                    strategy=name,
                    methodology=strategy.methodology,
                    params=params.label(),
                    split=split,
                )
                rows.append(metrics)
                results[(name, timeframe, params.label())] = result

        logger.info("Completed %s on %s (%d bars)", split, timeframe, len(ohlcv))

    return build_leaderboard(rows), results


LEADERBOARD_COLUMNS = [
    "rank",
    "strategy",
    "methodology",
    "timeframe",
    "n_trades",
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "avg_bars_held",
    "exposure",
    "fees_pct_of_capital",
    "final_equity",
    "score",
    "eligible",
    "gate_failures",
]


def format_leaderboard(leaderboard: pd.DataFrame, top: int = 30) -> str:
    """Render a leaderboard as a readable fixed-width table."""
    if leaderboard.empty:
        return "(no results)"
    columns = [c for c in LEADERBOARD_COLUMNS if c in leaderboard.columns]
    view = leaderboard[columns].head(top).copy()

    for column in ("total_return", "cagr", "max_drawdown", "win_rate", "exposure", "fees_pct_of_capital"):
        if column in view:
            view[column] = view[column].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    for column in ("sharpe", "sortino", "profit_factor", "score", "avg_bars_held"):
        if column in view:
            view[column] = view[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    if "final_equity" in view:
        view["final_equity"] = view["final_equity"].map(lambda v: f"{v:,.0f}")
    if "gate_failures" in view:
        view["gate_failures"] = view["gate_failures"].str.slice(0, 40)

    return view.to_string(index=False)
