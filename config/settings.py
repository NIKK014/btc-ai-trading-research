"""Central configuration for the BTC AI trading research system.

Every tunable value in the project lives here. Downstream modules import from
this file rather than hardcoding constants, so an experiment can be re-run with
different assumptions by changing one place.

Safety note
-----------
``TRADING_MODE`` and the demo REST host are treated as safety-critical. The
demo host is a module-level constant rather than an environment variable
precisely so that a malformed ``.env`` cannot redirect order flow. See
``src/exchange/`` for the enforcement layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:  # python-dotenv is convenient but the project must not hard-fail without it
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

#: The ONLY host this project ever sends authenticated (order-placing) requests
#: to. Deliberately a constant, not an env var. The production trading host
#: does not appear anywhere in this codebase.
#:
#: Bybit Testnet rather than Demo Trading, because Bybit EU cannot offer
#: perpetual futures under MiCA and this research requires the ability to go
#: short. Testnet is a wholly separate environment with its own balances -
#: there is no mechanism by which a testnet order can touch real funds, which
#: makes the safety guarantee stronger than demo trading, not weaker.
BYBIT_PAPER_TRADE_URL = "https://api-testnet.bybit.com"

#: Public market-data host. Used exclusively by the unauthenticated read-only
#: client in ``src/data/``, which has no request-signing code and therefore
#: cannot place an order even if it were called with credentials.
#:
#: Deliberately mainnet: testnet's order book is thin and its price history is
#: unusable for research. Signals are therefore derived from real market data
#: while orders execute against testnet's own book. See
#: ``docs/limitations.md`` - the live demo proves the pipeline works, it does
#: not produce meaningful P&L.
BYBIT_PUBLIC_DATA_URL = "https://api.bybit.com"

TRADING_MODE = os.getenv("TRADING_MODE", "").strip().lower()

#: Modes in which no real funds can possibly be at risk.
PAPER_MODES = frozenset({"testnet", "demo", "paper"})


class UnsafeConfigurationError(RuntimeError):
    """Raised when the application is not provably in a paper-only mode."""


def assert_paper_mode() -> None:
    """Fail loudly unless the process is configured for paper trading.

    Called at the top of any entrypoint that can place orders. Raises rather
    than warns: refusing to start is the correct behaviour here.
    """
    if TRADING_MODE not in PAPER_MODES:
        raise UnsafeConfigurationError(
            f"TRADING_MODE must be one of {sorted(PAPER_MODES)} "
            f"(got {TRADING_MODE!r}). This project may never execute trades "
            "with real funds. Set TRADING_MODE=testnet in your .env file."
        )


#: Retained so older call sites keep working; identical behaviour.
assert_paper_mode = assert_paper_mode


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: Project timeframe label -> Bybit V5 ``interval`` parameter.
BYBIT_INTERVALS: dict[str, str] = {
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}

#: Timeframe label -> duration in minutes. Used for pagination and for
#: annualising risk metrics.
TIMEFRAME_MINUTES: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


@dataclass(frozen=True)
class DataConfig:
    """Historical market data acquisition and caching."""

    symbol: str = "BTCUSDT"
    category: str = "linear"  # USDT perpetual futures
    timeframes: tuple[str, ...] = ("15m", "1h", "4h")
    history_start: str = "2020-01-01"
    #: Small slice used during development so iteration stays fast.
    dev_start: str = "2024-01-01"
    cache_dir: Path = PROJECT_ROOT / "data" / "raw"
    #: Bybit's per-request maximum.
    request_limit: int = 1000
    #: Politeness delay between paginated requests, seconds.
    request_sleep: float = 0.08
    request_timeout: int = 20
    max_retries: int = 4


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    """Execution assumptions. Deliberately pessimistic where ambiguous."""

    initial_capital: float = 10_000.0
    #: Bybit linear perpetual taker fee, one side. Round trip is 2x this.
    taker_fee: float = 0.00055
    maker_fee: float = 0.0002
    #: Slippage applied against us on both entry and exit, in basis points.
    slippage_bps: float = 2.0
    leverage: float = 1.0
    #: Signals are computed on the close of bar t and filled at the open of
    #: bar t+1. Never fill on the signal bar's own close.
    fill_delay_bars: int = 1
    #: When a stop and a target are both breached inside one candle, OHLCV
    #: cannot tell us which came first. Assume the stop. Systematically
    #: pessimistic beats systematically optimistic.
    ambiguous_bar_favours_stop: bool = True


@dataclass(frozen=True)
class RiskConfig:
    """Deterministic risk management. The LLM never touches these values."""

    risk_per_trade: float = 0.01  # 1% of equity at risk per position
    max_daily_loss: float = 0.03  # 3% -> stop trading for the day
    max_open_positions: int = 1
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    reward_risk_ratio: float = 2.0  # take profit at 2R
    max_position_pct: float = 1.0  # cap notional at 100% of equity at 1x


# ---------------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train / validation / test split.

    The test period is touched exactly once, for the final A/B/C comparison.
    All strategy selection, parameter search and ML thresholding happens on
    validation only.
    """

    train_end: str = "2023-12-31"
    validation_end: str = "2025-06-30"
    #: Bars dropped either side of every split boundary. Labels look 4 bars
    #: into the future, so adjacent samples overlap; without an embargo that
    #: overlap leaks across the seam.
    embargo_bars: int = 4


@dataclass(frozen=True)
class LabelConfig:
    """Triple-barrier target definition for the ML models.

    A fixed percentage barrier is not comparable across timeframes (0.5% in
    one hour is a real move; 0.5% in sixteen hours is noise), so the default
    scales the barrier by ATR. The fixed variant is retained for a sensitivity
    check.
    """

    horizon_bars: int = 4
    mode: str = "atr"  # "atr" | "fixed"
    atr_multiple: float = 1.0
    atr_period: int = 14
    fixed_pct: float = 0.005
    #: If both barriers are breached within the same candle we cannot know the
    #: order, so the sample is labelled HOLD and excluded from training.
    drop_ambiguous: bool = True


@dataclass(frozen=True)
class MetricsConfig:
    """Scoring and eligibility.

    Selection uses a single primary metric behind hard gates. The composite
    score exists for the leaderboard display only - it double-counts return by
    construction and is not a sound basis for choosing a winner.
    """

    primary_metric: str = "sortino"
    #: Below this many trades nothing is statistically measurable.
    min_trades: int = 30
    max_drawdown_limit: float = 0.40
    min_profit_factor: float = 1.0
    bootstrap_samples: int = 1_000
    confidence_level: float = 0.95
    display_score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "total_return": 0.30,
            "sharpe": 0.25,
            "max_drawdown": 0.20,
            "profit_factor": 0.15,
            "win_rate": 0.10,
        }
    )


# ---------------------------------------------------------------------------
# Models and agents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLConfig:
    """Supervised learning settings."""

    random_state: int = 42
    class_weight: str = "balanced"
    rf_n_estimators: int = 300
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 50
    logreg_max_iter: int = 2_000
    #: Minimum predicted probability for the ML filter to confirm a signal.
    #: Tuned on validation only.
    confidence_threshold: float = 0.40


@dataclass(frozen=True)
class LLMConfig:
    """LLM trading judge.

    The prompt deliberately contains no timestamps and no absolute prices. An
    LLM has memorised a great deal of Bitcoin price history, so a date or a
    price level is a look-ahead channel through the model weights.
    """

    model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    temperature: float = 0.0
    max_concurrency: int = 10
    max_retries: int = 2
    #: Responses are cached on a hash of the prompt payload, so re-running a
    #: backtest costs nothing and the presentation cannot be broken by an
    #: API outage.
    cache_enabled: bool = True


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathsConfig:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    results: Path = PROJECT_ROOT / "data" / "results"
    models: Path = PROJECT_ROOT / "data" / "models"
    database: Path = PROJECT_ROOT / "data" / "trading.db"
    logs: Path = PROJECT_ROOT / "logs"

    def ensure(self) -> None:
        """Create every project directory if it does not already exist."""
        for path in (self.data, self.raw, self.results, self.models, self.logs):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LiveConfig:
    """Live demo paper-trading loop."""

    timeframe: str = "1h"
    poll_seconds: int = 30
    #: Refuse to act on a candle that has not closed yet.
    require_closed_candle: bool = True


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

DATA = DataConfig()
BACKTEST = BacktestConfig()
RISK = RiskConfig()
SPLIT = SplitConfig()
LABELS = LabelConfig()
METRICS = MetricsConfig()
ML = MLConfig()
LLM = LLMConfig()
PATHS = PathsConfig()
LIVE = LiveConfig()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
