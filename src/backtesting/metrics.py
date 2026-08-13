"""Performance metrics, eligibility gates and confidence intervals.

Three principles.

Profit is not the ranking criterion. 300% with a 70% drawdown is not better
than 80% with 15%, and 500% over four trades says nothing at all.

Selection uses one metric behind hard gates. The composite score is for making
the leaderboard readable - it double-counts return by construction and is not a
sound basis for choosing a winner.

Every headline number gets a confidence interval. With a few hundred trades the
difference between two systems is usually inside the noise, and saying so is a
result rather than a failure.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.settings import METRICS, MetricsConfig
from src.backtesting.engine import BacktestResult
from src.utils.timeframes import bars_per_year

EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Annualised Sharpe ratio, risk-free rate assumed zero.

    Computed over *all* bars, including those spent flat. That is deliberate:
    a strategy that is in the market 5% of the time is taking less risk and
    should not be flattered by ignoring the bars where it earned nothing. It
    also keeps the comparison against buy-and-hold honest.
    """
    values = returns.to_numpy(dtype=np.float64)
    if values.size < 2:
        return float("nan")
    deviation = values.std(ddof=1)
    if deviation < EPSILON:
        return 0.0
    return float(values.mean() / deviation * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    """Annualised Sortino ratio: like Sharpe but only penalising downside.

    Upside volatility is not risk. This is the project's primary selection
    metric.
    """
    values = returns.to_numpy(dtype=np.float64)
    if values.size < 2:
        return float("nan")
    downside = values[values < 0.0]
    if downside.size == 0:
        return float("inf") if values.mean() > 0 else 0.0
    downside_deviation = np.sqrt(np.mean(downside**2))
    if downside_deviation < EPSILON:
        return 0.0
    return float(values.mean() / downside_deviation * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at each bar."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline, as a positive fraction."""
    if equity.empty:
        return float("nan")
    return float(-drawdown_series(equity).min())


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate implied by the equity curve."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0:
        return float("nan")
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def profit_factor(pnl: Sequence[float]) -> float:
    """Gross profit divided by gross loss."""
    values = np.asarray(pnl, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses < EPSILON:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Full metric set
# ---------------------------------------------------------------------------


def compute_metrics(result: BacktestResult, timeframe: str) -> Dict[str, Any]:
    """Compute every reported metric for one backtest run."""
    equity = result.equity
    trades = result.trades
    periods = bars_per_year(timeframe)
    initial = float(result.meta.get("initial_capital", equity.iloc[0] if len(equity) else np.nan))

    pnl = trades["pnl"].to_numpy(dtype=np.float64) if not trades.empty else np.array([])
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    metrics: Dict[str, Any] = {
        "timeframe": timeframe,
        "bars": len(equity),
        "start": equity.index[0] if len(equity) else pd.NaT,
        "end": equity.index[-1] if len(equity) else pd.NaT,
        "initial_capital": initial,
        "final_equity": result.final_equity,
        "total_return": result.final_equity / initial - 1.0 if initial else np.nan,
        "cagr": cagr(equity),
        "sharpe": sharpe_ratio(result.returns, periods),
        "sortino": sortino_ratio(result.returns, periods),
        "max_drawdown": max_drawdown(equity),
        "exposure": result.exposure,
        "n_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()) if pnl.size else np.nan,
        "profit_factor": profit_factor(pnl),
        "avg_trade": float(pnl.mean()) if pnl.size else np.nan,
        "avg_trade_pct": float(trades["return_pct"].mean()) if not trades.empty else np.nan,
        "avg_win": float(wins.mean()) if wins.size else np.nan,
        "avg_loss": float(losses.mean()) if losses.size else np.nan,
        "payoff_ratio": float(wins.mean() / abs(losses.mean()))
        if wins.size and losses.size
        else np.nan,
        "expectancy_r": float(trades["r_multiple"].mean()) if not trades.empty else np.nan,
        "best_trade": float(pnl.max()) if pnl.size else np.nan,
        "worst_trade": float(pnl.min()) if pnl.size else np.nan,
        "avg_bars_held": float(trades["bars_held"].mean()) if not trades.empty else np.nan,
        "fees_paid": float(trades["fees"].sum()) if not trades.empty else 0.0,
    }

    # Fee drag is worth surfacing explicitly: on short timeframes it is often
    # the single largest determinant of whether a strategy is viable.
    metrics["fees_pct_of_capital"] = metrics["fees_paid"] / initial if initial else np.nan
    metrics["calmar"] = (
        metrics["cagr"] / metrics["max_drawdown"]
        if metrics["max_drawdown"] and metrics["max_drawdown"] > EPSILON
        else np.nan
    )
    return metrics


def exit_reason_breakdown(result: BacktestResult) -> pd.DataFrame:
    """How trades ended, and what each exit type earned on average.

    Diagnostic rather than headline: a strategy whose stops account for most of
    its exits is being stopped out by noise and probably needs a wider stop, and
    one that almost never reaches its target has a reward:risk ratio that the
    market is not offering.
    """
    if result.trades.empty:
        return pd.DataFrame(columns=["count", "share", "total_pnl", "avg_pnl"])

    grouped = result.trades.groupby("exit_reason")["pnl"]
    summary = pd.DataFrame(
        {
            "count": grouped.size(),
            "total_pnl": grouped.sum(),
            "avg_pnl": grouped.mean(),
        }
    )
    summary["share"] = summary["count"] / summary["count"].sum()
    return summary[["count", "share", "total_pnl", "avg_pnl"]].sort_values(
        "count", ascending=False
    )


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def check_gates(
    metrics: Dict[str, Any], config: MetricsConfig = METRICS
) -> Tuple[bool, List[str]]:
    """Apply hard eligibility gates.

    A configuration failing any gate is disqualified regardless of its score.
    This is what stops "three lucky trades, 900% return" topping the
    leaderboard.

    Benchmarks are exempt - buy-and-hold takes one trade and would fail the
    minimum-trades gate, but it is the bar everything is measured against.
    :func:`select_winner` excludes it from being chosen.

    Returns:
        ``(eligible, reasons_for_failure)``.
    """
    if metrics.get("methodology") == "benchmark":
        return True, []

    failures: List[str] = []

    n_trades = metrics.get("n_trades", 0)
    if n_trades < config.min_trades:
        failures.append(f"only {n_trades} trades (need {config.min_trades})")

    drawdown = metrics.get("max_drawdown")
    if drawdown is not None and np.isfinite(drawdown) and drawdown > config.max_drawdown_limit:
        failures.append(f"drawdown {drawdown:.1%} exceeds {config.max_drawdown_limit:.0%}")

    factor = metrics.get("profit_factor")
    if factor is None or (np.isfinite(factor) and factor <= config.min_profit_factor):
        failures.append(f"profit factor {factor:.2f} at or below {config.min_profit_factor}")

    return (not failures), failures


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

#: Metrics where a smaller value is better.
LOWER_IS_BETTER = {"max_drawdown"}


def build_leaderboard(
    rows: Iterable[Dict[str, Any]],
    config: MetricsConfig = METRICS,
) -> pd.DataFrame:
    """Assemble scored, ranked results from many backtest runs.

    The display score normalises each component by **percentile rank within the
    cohort** rather than min-max. Min-max scaling lets a single outlier compress
    every other strategy into a narrow band and silently dominate the ranking;
    rank-based scaling is robust to that.

    Rows are sorted by the primary metric among eligible strategies, with
    ineligible ones listed afterwards so they remain visible and auditable
    rather than quietly deleted.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame

    # Rebuilding a board from already-scored rows is common (the optimiser
    # concatenates per-strategy boards), so derived columns are recomputed
    # from scratch rather than inherited.
    frame = frame.drop(columns=["rank", "eligible", "gate_failures", "score"], errors="ignore")

    gate_results = frame.apply(lambda row: check_gates(row.to_dict(), config), axis=1)
    frame["eligible"] = [ok for ok, _ in gate_results]
    frame["gate_failures"] = ["; ".join(reasons) for _, reasons in gate_results]

    score = pd.Series(0.0, index=frame.index)
    for metric, weight in config.display_score_weights.items():
        if metric not in frame.columns:
            continue
        values = frame[metric].replace([np.inf, -np.inf], np.nan)
        ranked = values.rank(pct=True, na_option="bottom")
        if metric in LOWER_IS_BETTER:
            ranked = 1.0 - ranked
        score += weight * ranked
    frame["score"] = score

    primary = config.primary_metric
    sort_columns = ["eligible", primary if primary in frame.columns else "score"]
    frame = frame.sort_values(sort_columns, ascending=[False, False]).reset_index(drop=True)
    frame.insert(0, "rank", frame.index + 1)
    return frame


def select_winner(
    leaderboard: pd.DataFrame, config: MetricsConfig = METRICS
) -> Optional[pd.Series]:
    """Return the top eligible configuration, or ``None`` if there is none.

    "None" is a legitimate outcome and must be reported rather than worked
    around by relaxing the gates until something passes.
    """
    eligible = leaderboard[leaderboard["eligible"]] if "eligible" in leaderboard else leaderboard
    if "methodology" in eligible.columns:
        eligible = eligible[eligible["methodology"] != "benchmark"]
    if eligible.empty:
        return None
    metric = config.primary_metric if config.primary_metric in eligible.columns else "score"
    return eligible.sort_values(metric, ascending=False).iloc[0]


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def trade_returns(trades: pd.DataFrame) -> np.ndarray:
    """Per-trade return on the equity that was at risk when it was opened.

    Expressing each trade as a fraction of contemporaneous equity makes trades
    from different points in the curve comparable, which is what allows them to
    be resampled.
    """
    if trades.empty:
        return np.array([])
    equity_before = trades["equity_after"] - trades["pnl"]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (trades["pnl"] / equity_before).to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _path_metrics(sample: np.ndarray) -> Dict[str, float]:
    """Total return and max drawdown of a compounded sequence of trade returns."""
    curve = np.cumprod(1.0 + sample)
    peak = np.maximum.accumulate(curve)
    return {
        "total_return": float(curve[-1] - 1.0),
        "max_drawdown": float(-(curve / peak - 1.0).min()),
        "win_rate": float((sample > 0).mean()),
        "profit_factor": profit_factor(sample),
    }


def bootstrap_trade_metrics(
    trades: pd.DataFrame,
    config: MetricsConfig = METRICS,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Confidence intervals by resampling trades with replacement.

    Each resample draws ``n`` trades from the observed ``n`` and compounds them
    into an alternative equity path. The spread of the resulting distribution
    answers the question that actually matters: *how much of this result is
    the strategy, and how much is the particular order the trades happened to
    arrive in?*

    Note the assumption: resampling treats trades as independent, which
    discards any serial correlation between them. It is a lower bound on the
    true uncertainty, not an upper one.
    """
    returns = trade_returns(trades)
    if returns.size < 2:
        return {}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, returns.size, size=(config.bootstrap_samples, returns.size))
    samples = returns[draws]

    collected: Dict[str, List[float]] = {}
    for row in samples:
        for key, value in _path_metrics(row).items():
            collected.setdefault(key, []).append(value)

    alpha = (1.0 - config.confidence_level) / 2.0
    intervals: Dict[str, Tuple[float, float]] = {}
    for key, values in collected.items():
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size:
            intervals[key] = (
                float(np.quantile(array, alpha)),
                float(np.quantile(array, 1.0 - alpha)),
            )
    return intervals


def difference_ci(
    trades_a: pd.DataFrame,
    trades_b: pd.DataFrame,
    metric: str = "total_return",
    config: MetricsConfig = METRICS,
    seed: int = 42,
) -> Dict[str, float]:
    """Confidence interval on the difference between two systems.

    This is the function that answers the project's actual research questions.
    Bootstrap both systems independently, take the difference of the paired
    resamples, and report the interval. **If that interval contains zero, the
    two systems are not distinguishable at this sample size** - which for
    "did the LLM help?" is a perfectly good answer, and a far more defensible
    one than quoting two point estimates and declaring a winner.
    """
    a = trade_returns(trades_a)
    b = trade_returns(trades_b)
    if a.size < 2 or b.size < 2:
        return {}

    rng = np.random.default_rng(seed)
    samples = config.bootstrap_samples
    a_draws = a[rng.integers(0, a.size, size=(samples, a.size))]
    b_draws = b[rng.integers(0, b.size, size=(samples, b.size))]

    a_values = np.array([_path_metrics(row)[metric] for row in a_draws])
    b_values = np.array([_path_metrics(row)[metric] for row in b_draws])
    differences = b_values - a_values
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return {}

    alpha = (1.0 - config.confidence_level) / 2.0
    low = float(np.quantile(differences, alpha))
    high = float(np.quantile(differences, 1.0 - alpha))
    return {
        "metric": metric,
        "observed_a": float(_path_metrics(a)[metric]),
        "observed_b": float(_path_metrics(b)[metric]),
        "mean_difference": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
        "significant": bool(low > 0.0 or high < 0.0),
        "probability_b_better": float((differences > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Regime-split reporting
# ---------------------------------------------------------------------------


def metrics_by_period(
    result: BacktestResult,
    periods: Dict[str, Tuple[Any, Any]],
    timeframe: str,
) -> pd.DataFrame:
    """Evaluate one run separately over named sub-periods.

    The test period in this project spans a large drawdown, so aggregate test
    numbers conflate "the strategy works" with "the market fell". Splitting the
    report by regime separates the two, and directly answers the question every
    examiner asks: *did it only work because the market went up?*
    """
    rows = []
    for label, (start, end) in periods.items():
        try:
            window = result.slice(start, end)
        except ValueError:
            continue
        row = compute_metrics(window, timeframe)
        row["period"] = label
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    columns = ["period", "n_trades", "total_return", "sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor"]
    return frame[[c for c in columns if c in frame.columns]]


def summarise(result: BacktestResult, timeframe: str, label: str = "") -> str:
    """Human-readable one-block summary, for logs and notebooks."""
    m = compute_metrics(result, timeframe)
    eligible, failures = check_gates(m)
    lines = [
        f"{label or 'Backtest'} [{timeframe}]",
        f"  Return {m['total_return']:>8.1%}   CAGR {m['cagr']:>7.1%}   Final {m['final_equity']:>12,.0f}",
        f"  Sharpe {m['sharpe']:>7.2f}   Sortino {m['sortino']:>6.2f}   MaxDD {m['max_drawdown']:>7.1%}",
        f"  Trades {m['n_trades']:>7,}   WinRate {m['win_rate']:>6.1%}   PF {m['profit_factor']:>9.2f}",
        f"  Exposure {m['exposure']:>5.1%}   Fees {m['fees_pct_of_capital']:>9.1%} of capital",
        f"  Eligible: {'yes' if eligible else 'NO - ' + '; '.join(failures)}",
    ]
    return "\n".join(lines)
