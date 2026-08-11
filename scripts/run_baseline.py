"""System A: rule-based strategy leaderboard across timeframes.

Runs every strategy on every timeframe over the train and validation splits,
ranks them, and writes the results to ``data/results/``. The test period is
not touched.

    python scripts/run_baseline.py
    python scripts/run_baseline.py --timeframes 1h 4h --split validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.settings import DATA, METRICS, PATHS  # noqa: E402
from src.backtesting.metrics import (  # noqa: E402
    bootstrap_trade_metrics,
    exit_reason_breakdown,
    select_winner,
)
from src.backtesting.runner import (  # noqa: E402
    TRAIN,
    VALIDATION,
    format_leaderboard,
    run_grid,
)
from src.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger("run_baseline")


def report(split: str, timeframes, results_dir: Path) -> pd.DataFrame:
    leaderboard, results = run_grid(timeframes, split=split)
    if leaderboard.empty:
        logger.error("No results for split %s", split)
        return leaderboard

    path = results_dir / f"leaderboard_{split}.csv"
    leaderboard.to_csv(path, index=False)

    print(f"\n{'=' * 100}\nSYSTEM A - {split.upper()}\n{'=' * 100}")
    print(format_leaderboard(leaderboard))

    # The benchmark is the number that matters. A strategy that underperforms
    # buy-and-hold on a risk-adjusted basis has not earned its complexity.
    benchmark = leaderboard[leaderboard["methodology"] == "benchmark"]
    winner = select_winner(leaderboard)

    print(f"\n{'-' * 100}")
    if winner is None:
        print("No strategy passed the eligibility gates. That is a legitimate result:")
        print("it means nothing we tested is distinguishable from noise once minimum")
        print("sample size, drawdown and profitability requirements are enforced.")
    else:
        print(
            f"Best eligible: {winner['strategy']} @ {winner['timeframe']} "
            f"-> return {winner['total_return']:.1%}, Sharpe {winner['sharpe']:.2f}, "
            f"Sortino {winner['sortino']:.2f}, MaxDD {winner['max_drawdown']:.1%}, "
            f"{int(winner['n_trades'])} trades"
        )
        if not benchmark.empty:
            best_benchmark = benchmark.loc[benchmark["sharpe"].idxmax()]
            print(
                f"Buy and hold @ {best_benchmark['timeframe']}: "
                f"return {best_benchmark['total_return']:.1%}, "
                f"Sharpe {best_benchmark['sharpe']:.2f}, "
                f"MaxDD {best_benchmark['max_drawdown']:.1%}"
            )
            verdict = (
                "BEATS" if winner["sharpe"] > best_benchmark["sharpe"] else "DOES NOT BEAT"
            )
            print(f"-> Best strategy {verdict} buy-and-hold on Sharpe.")

        key = (winner["strategy"], winner["timeframe"], winner["params"])
        best_result = results.get(key)
        if best_result is not None and not best_result.trades.empty:
            print("\nExit reasons for the best strategy:")
            print(exit_reason_breakdown(best_result).to_string())

            intervals = bootstrap_trade_metrics(best_result.trades)
            if intervals:
                print(
                    f"\nBootstrap {METRICS.confidence_level:.0%} confidence intervals "
                    f"({METRICS.bootstrap_samples:,} resamples of the trade sequence):"
                )
                for metric, (low, high) in intervals.items():
                    print(f"  {metric:<16} [{low:>8.3f}, {high:>8.3f}]")
                total = intervals.get("total_return")
                if total and total[0] < 0 < total[1]:
                    print(
                        "  -> The interval for total return contains zero: this result is\n"
                        "     not distinguishable from luck at this sample size."
                    )

    # Fee drag by timeframe is one of the clearest findings the project produces.
    print(f"\n{'-' * 100}\nMedian fee drag and trade count by timeframe (non-benchmark):")
    strategies_only = leaderboard[leaderboard["methodology"] != "benchmark"]
    summary = strategies_only.groupby("timeframe").agg(
        median_trades=("n_trades", "median"),
        median_fees_pct=("fees_pct_of_capital", "median"),
        median_return=("total_return", "median"),
        eligible=("eligible", "sum"),
    )
    print(summary.to_string(float_format=lambda v: f"{v:,.2f}"))
    print(f"\nSaved -> {path}")
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframes", nargs="+", default=list(DATA.timeframes))
    parser.add_argument("--split", nargs="+", default=[TRAIN, VALIDATION])
    args = parser.parse_args()

    PATHS.ensure()
    for split in args.split:
        report(split, args.timeframes, PATHS.results)

    print(
        "\nReminder: the test period has not been touched. Strategy selection, "
        "parameter\nsearch and ML thresholds all happen on validation only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
