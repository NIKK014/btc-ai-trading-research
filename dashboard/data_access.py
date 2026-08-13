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
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from config.settings import DATA, PATHS
from src.data.loader import cache_path, fetch_ohlcv, load_ohlcv
from src.data.public_client import BybitPublicClient
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


@st.cache_data(ttl=10)
def load_live_price(symbol: str = DATA.symbol) -> Optional[float]:
    """Current spot price, for marking an open position to market.

    The candle cache only advances when a 4h candle closes, so marking against
    it can be four hours stale - useless for watching a live position. This is
    the one place the dashboard touches the network, with a short timeout and
    a fallback to the last close, so a slow or failed request degrades to the
    old behaviour rather than stalling the page.
    """
    try:
        return BybitPublicClient(timeout=5, max_retries=1).get_last_price(symbol)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=LIVE_TTL)
def load_recent_prices(timeframe: str = "4h", bars: int = 200) -> pd.DataFrame:
    """Recent candles: from the parquet cache locally, from the API when deployed.

    The cache is checked by path rather than by calling ``load_ohlcv`` blind,
    because ``load_ohlcv`` responds to a missing cache by downloading the full
    history back to 2020 - several hundred requests. That is right for a
    research machine and wrong for a web page, where it would hang the first
    visitor for minutes. A deployed copy has no parquet (too large for Git), so
    it takes the second branch and fetches just the window it needs.
    """
    try:
        if cache_path(timeframe, DATA.symbol, DATA).exists():
            cached = load_ohlcv(timeframe)
            if not cached.empty:
                return cached.tail(bars)
    except Exception:  # noqa: BLE001
        pass

    try:
        hours = {"15m": 0.25, "1h": 1.0, "4h": 4.0}.get(timeframe, 4.0) * bars
        start = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours * 1.2)
        return fetch_ohlcv(timeframe, start).tail(bars)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def has_local_trading_data() -> bool:
    """Whether this is a research machine or the published web copy.

    Keyed on the parquet cache, not the database. The database is created the
    moment anything reads it, so testing for that file would answer "yes"
    everywhere - including on the deployed site, three lines after Streamlit
    itself created it. The 14MB price cache is never in Git, so its presence
    is the honest signal.
    """
    return any(cache_path(tf, DATA.symbol, DATA).exists() for tf in DATA.timeframes)


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
            # Same trap as load_recent_prices: no parquet means load_ohlcv
            # would download six years of history to fill in one table.
            if not cache_path(timeframe, DATA.symbol, DATA).exists():
                continue
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


@st.cache_data(ttl=LIVE_TTL)
def open_position_summary() -> Dict[str, Any]:
    """The live position marked against the current spot price."""
    trades = load_trades()
    if trades.empty:
        return {}
    open_trades = trades[trades["exit_time"].isna()]
    if open_trades.empty:
        return {}

    trade = open_trades.iloc[0]
    price = load_live_price()
    if price is None:
        prices = load_recent_prices()
        price = float(prices["close"].iloc[-1]) if not prices.empty else None
        stale = True
    else:
        stale = False
    if price is None:
        return {}

    direction = int(trade["direction"])
    size = float(trade["size"])
    entry = float(trade["entry_price"])
    pnl = direction * size * (price - entry)
    notional = size * entry

    return {
        "direction": direction,
        "size": size,
        "entry_price": entry,
        "mark_price": float(price),
        "unrealised_pnl": float(pnl),
        "unrealised_pct": float(pnl / notional) if notional else 0.0,
        "stop_price": float(trade["stop_price"]) if pd.notna(trade["stop_price"]) else None,
        "target_price": float(trade["target_price"]) if pd.notna(trade["target_price"]) else None,
        "entry_time": trade["entry_time"],
        "stale": stale,
    }
