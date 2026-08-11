"""Cached, read-only data access for the dashboard.

The dashboard never computes anything expensive and never writes. It reads
SQLite (written by the live loop) and the CSVs in ``data/results/`` (written
by the research scripts). Backtests are cheap enough to recompute on demand -
under a second on 4h data - so equity curves are regenerated rather than
stored, which keeps them in sync with the current configuration.

Every loader degrades gracefully. A missing file returns an empty frame rather
than raising, because a dashboard that crashes because one script has not been
run yet is useless during a presentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from config.settings import DATA, PATHS
from src.backtesting.engine import BacktestResult, run_backtest
from src.backtesting.metrics import compute_metrics
from src.backtesting.runner import VALIDATION, get_split
from src.data.loader import load_ohlcv
from src.database.repository import Repository
from src.strategies.base import build, registry

LIVE_TTL = 20  # seconds; the loop only acts every few hours
RESEARCH_TTL = 300


# ---------------------------------------------------------------------------
# Research artefacts
# ---------------------------------------------------------------------------


@st.cache_data(ttl=RESEARCH_TTL)
def load_results_csv(name: str) -> pd.DataFrame:
    """Read one results CSV, or an empty frame if it does not exist yet."""
    path = PATHS.results / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(ttl=RESEARCH_TTL)
def load_tuned_params() -> Dict[str, Any]:
    path = PATHS.results / "tuned_params.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def available_results() -> Dict[str, bool]:
    """Which research steps have been run, for the setup checklist."""
    return {
        "Strategy leaderboard": not load_results_csv("leaderboard_validation.csv").empty,
        "Parameter search": not load_results_csv("optimizer_validation.csv").empty,
        "ML models": not load_results_csv("ml_models_4h.csv").empty,
        "System A/B/C": not load_results_csv("systems_abc_4h.csv").empty,
    }


# ---------------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------------


@st.cache_resource
def get_repository() -> Repository:
    return Repository()


@st.cache_data(ttl=LIVE_TTL)
def load_trades() -> pd.DataFrame:
    frame = get_repository().trades()
    for column in ("entry_time", "exit_time"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


@st.cache_data(ttl=LIVE_TTL)
def load_decisions(limit: int = 200) -> pd.DataFrame:
    frame = get_repository().decisions(limit=limit)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


@st.cache_data(ttl=LIVE_TTL)
def load_equity_history() -> pd.DataFrame:
    return get_repository().equity_curve()


@st.cache_data(ttl=LIVE_TTL)
def load_live_summary() -> Dict[str, Any]:
    repository = get_repository()
    summary = repository.summary()
    summary["broker_state"] = repository.get_state("simulated_broker", {})
    summary["trading_mode"] = repository.get_state("trading_mode", "unknown")
    return summary


@st.cache_data(ttl=LIVE_TTL)
def load_recent_prices(timeframe: str = "4h", bars: int = 200) -> pd.DataFrame:
    """Recent candles from the local cache. Never hits the network."""
    try:
        return load_ohlcv(timeframe).tail(bars)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# On-demand backtests
# ---------------------------------------------------------------------------


def _tuned_strategy(name: str = "ema_rsi_trend"):
    """Rebuild the winning configuration from the optimiser's saved params."""
    tuned = load_tuned_params().get(name, {})
    values = tuned.get("param_values", {})
    cls = registry().get(name)
    if cls is None:
        return build("ema_rsi_trend"), "4h"

    fields = {f.name: f.type for f in cls.params_class.__dataclass_fields__.values()}
    coerced = {}
    for key, value in values.items():
        if key in fields:
            coerced[key] = int(value) if "int" in str(fields[key]) else float(value)
    strategy = cls(cls.params_class(**coerced)) if coerced else cls()
    return strategy, tuned.get("timeframe", "4h")


@st.cache_data(ttl=RESEARCH_TTL)
def validation_equity_curves(
    strategy_name: str = "ema_rsi_trend",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Equity curves for the tuned strategy and buy-and-hold, on validation.

    Recomputed rather than stored so it always reflects the current
    configuration. Costs well under a second on 4h data.
    """
    strategy, timeframe = _tuned_strategy(strategy_name)
    start, end = get_split(VALIDATION)
    ohlcv = load_ohlcv(timeframe, start=start, end=end)
    if ohlcv.empty:
        return pd.DataFrame(), {}

    strategy_result = run_backtest(strategy.run(ohlcv))

    benchmark = build("buy_and_hold")
    benchmark_result = run_backtest(
        benchmark.run(ohlcv), use_stops=False, sizing="full_notional", enforce_daily_limit=False
    )

    curves = pd.DataFrame(
        {
            strategy.name: strategy_result.equity,
            "buy_and_hold": benchmark_result.equity,
        }
    )
    context = {
        "timeframe": timeframe,
        "strategy": strategy.name,
        "params": strategy.params.label(),
        "strategy_metrics": compute_metrics(strategy_result, timeframe),
        "benchmark_metrics": compute_metrics(benchmark_result, timeframe),
        "trades": strategy_result.trades,
        "result": strategy_result,
    }
    return curves, context


@st.cache_data(ttl=RESEARCH_TTL)
def label_balance_table() -> pd.DataFrame:
    """Class balance under both barrier definitions, across timeframes."""
    from config.settings import LabelConfig
    from src.models.labels import class_balance, triple_barrier_labels

    rows = []
    for mode, config in (
        ("ATR-scaled", LabelConfig()),
        ("Fixed 0.5%", LabelConfig(mode="fixed", fixed_pct=0.005)),
    ):
        for timeframe in DATA.timeframes:
            try:
                frame = load_ohlcv(timeframe)
            except Exception:  # noqa: BLE001
                continue
            if frame.empty:
                continue
            balance = class_balance(triple_barrier_labels(frame, config), config)
            if balance:
                rows.append(
                    {
                        "barrier": mode,
                        "timeframe": timeframe,
                        "LONG": balance["long_share"],
                        "SHORT": balance["short_share"],
                        "HOLD": balance["hold_share"],
                        "discarded_ties": balance["ambiguous_share"],
                        "samples": balance["samples"],
                    }
                )
    return pd.DataFrame(rows)
