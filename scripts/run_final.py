"""THE FINAL RUN - the untouched out-of-sample test period.

This script reads the test data. Everything else in the project deliberately
cannot: :func:`get_split` raises unless ``unlock_test=True``, which appears
exactly once, here.

Run it once. Re-running it after seeing the results and changing something
turns the test set into a second validation set, and the out-of-sample claim
becomes false. The results are written to ``data/results/final_test.csv``.

    python scripts/run_final.py
    python scripts/run_final.py --skip-llm     # A and B only, no API cost

What it reports
---------------
1. Every system on the test period, against buy-and-hold.
2. The same numbers split by market regime, because the test period contains a
   large drawdown and aggregate figures conflate "the strategy works" with
   "the market fell".
3. Validation vs test degradation - how much of the validation result was real.
4. Bootstrap confidence intervals on every difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd  # noqa: E402

from config.settings import METRICS, PATHS, SPLIT  # noqa: E402
from src.agents.deterministic_judge import AlwaysAgreeJudge  # noqa: E402
from src.agents.harness import agreement_stats, build_snapshots, describe_agreement, run_arm  # noqa: E402
from src.agents.trading_judge import TradingJudge  # noqa: E402
from src.backtesting.engine import run_backtest  # noqa: E402
from src.backtesting.metrics import (  # noqa: E402
    bootstrap_trade_metrics,
    compute_metrics,
    difference_ci,
    metrics_by_period,
)
from src.backtesting.runner import TEST, VALIDATION, apply_embargo, get_split  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.database.repository import Repository  # noqa: E402
from src.models.features import build_dataset  # noqa: E402
from src.models.predict import apply_ml_filter, predictions_frame  # noqa: E402
from src.models.train import (  # noqa: E402
    Splits,
    chronological_splits,
    comparison_table,
    train_models,
)
from src.strategies.base import build  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402
from run_ml import load_tuned  # noqa: E402

logger = get_logger("run_final")

RULE = "=" * 100


def fit_for_test(features, target, splits: Splits):
    """Refit on train + validation, the standard final-model protocol.

    Hyperparameters and the confidence threshold were chosen on validation and
    are held fixed. The model itself is refitted on everything before the test
    period, because throwing away eighteen months of data at the final step
    would handicap the model for no methodological benefit. The test period
    itself is never seen during fitting.
    """
    combined = splits.train.union(splits.validation)
    return train_models(features, target, Splits(train=combined, validation=splits.validation, test=splits.test))


def regime_periods(ohlcv: pd.DataFrame) -> dict:
    """Split the test window at its price peak into an up- and a down-phase.

    The test period runs through a large drawdown. Aggregate numbers therefore
    conflate strategy skill with market direction, and splitting at the peak
    answers the question every examiner asks: did it only work because the
    market went up?
    """
    if ohlcv.empty:
        return {}
    peak = ohlcv["close"].idxmax()
    start, end = ohlcv.index[0], ohlcv.index[-1]
    if peak <= start or peak >= end:
        return {"whole test period": (start, end)}
    return {
        f"rising (to peak {peak.date()})": (start, peak),
        f"falling (from {peak.date()})": (peak, end),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="ema_rsi_trend")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt."
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Cap the test window, e.g. 2026-08-11T08:00:01. Without this the test "
            "runs to the last candle on disk, so a later re-run silently scores a "
            "longer period and the numbers move. Pass the original end to "
            "reproduce a published run exactly."
        ),
    )
    args = parser.parse_args()

    PATHS.ensure()

    print(RULE)
    print("  FINAL OUT-OF-SAMPLE TEST")
    print(RULE)
    print(
        "  This reads the test period, which nothing else in the project can.\n"
        "  Run it ONCE. Changing something and re-running turns the test set\n"
        "  into a second validation set and the out-of-sample claim is void.\n"
    )
    if not args.yes:
        if input("  Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("  Aborted. Nothing was read.")
            return 1

    # ------------------------------------------------------------- setup
    strategy, timeframe = load_tuned(args.strategy)
    timeframe = timeframe or "4h"
    print(f"\n  Strategy : {strategy.name} @ {timeframe}")
    print(f"  Params   : {strategy.params.label()}")
    print(f"  Splits   : train -> {SPLIT.train_end}, validation -> {SPLIT.validation_end}, "
          f"then test\n")

    full = load_ohlcv(timeframe)
    features, target, _ = build_dataset(full, strategy.indicator_spec)
    splits = chronological_splits(features.index)

    trained = fit_for_test(features, target, splits)
    model_name = comparison_table(trained).iloc[0]["model"]
    model = trained[model_name]
    print(f"  Model    : {model_name}, refitted on train + validation")
    print(
        "             NOTE: the accuracy figures logged above are measured on data\n"
        "             the model was just trained on, because validation is now part\n"
        "             of the training set. They are NOT generalisation estimates -\n"
        "             quote the clean numbers from scripts/run_ml.py instead.\n"
    )

    test_start, test_end = get_split(TEST, unlock_test=True)
    if args.end:
        test_end = args.end
    test_ohlcv = apply_embargo(load_ohlcv(timeframe, start=test_start, end=test_end))
    if test_ohlcv.empty:
        print("  No test data available.")
        return 1
    print(f"  Test set : {test_ohlcv.index[0]} -> {test_ohlcv.index[-1]} "
          f"({len(test_ohlcv):,} bars)\n")

    prepared = strategy.run(test_ohlcv)
    signals = prepared["signal"]
    test_features = features.loc[features.index.intersection(prepared.index)]
    predictions = predictions_frame(model, test_features)

    # -------------------------------------------------------------- arms
    results = {}
    records = {}

    results["A_rules_only"] = run_backtest(prepared)

    frame_b = prepared.copy()
    frame_b["signal"] = apply_ml_filter(signals, predictions, args.threshold)
    results["B_rules_plus_ml"] = run_backtest(frame_b)

    snapshots = build_snapshots(prepared, signals, predictions, strategy.name, timeframe)
    gated, unjudged_records = run_arm(
        AlwaysAgreeJudge(), prepared, signals, predictions, strategy.name, timeframe, snapshots
    )
    frame_control = prepared.copy()
    frame_control["signal"] = gated
    results["control_unjudged"] = run_backtest(frame_control)
    records["control_unjudged"] = unjudged_records

    if not args.skip_llm:
        try:
            judge = TradingJudge()
            gated_c, llm_records = run_arm(
                judge, prepared, signals, predictions, strategy.name, timeframe, snapshots
            )
            frame_c = prepared.copy()
            frame_c["signal"] = gated_c
            results["C_rules_ml_llm"] = run_backtest(frame_c)
            records["C_rules_ml_llm"] = llm_records
            print(
                f"  LLM      : {judge.calls_made} API calls, {judge.cache_hits} cached, "
                f"{judge.failures} failures\n"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM arm unavailable (%s); continuing without System C.", exc)

    benchmark = build("buy_and_hold")
    results["benchmark_buy_and_hold"] = run_backtest(
        benchmark.run(test_ohlcv), use_stops=False, sizing="full_notional",
        enforce_daily_limit=False,
    )

    # ----------------------------------------------------------- results
    rows = []
    for label, result in results.items():
        metrics = compute_metrics(result, timeframe)
        metrics["system"] = label
        rows.append(metrics)
    table = pd.DataFrame(rows)

    print(RULE)
    print("  1. TEST-SET PERFORMANCE")
    print(RULE)
    display = table[
        ["system", "n_trades", "total_return", "sharpe", "sortino", "max_drawdown",
         "win_rate", "profit_factor", "exposure"]
    ].copy()
    for column in ("total_return", "max_drawdown", "win_rate", "exposure"):
        display[column] = display[column].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    for column in ("sharpe", "sortino", "profit_factor"):
        display[column] = display[column].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    print(display.to_string(index=False))

    # -------------------------------------------------------- by regime
    print(f"\n{RULE}")
    print("  2. BY MARKET REGIME")
    print(RULE)
    periods = regime_periods(test_ohlcv)
    regime_rows = []
    print(
        "  The test period spans a large drawdown, so aggregate figures conflate\n"
        "  strategy skill with market direction. Split at the price peak:\n"
    )
    for label, result in results.items():
        split_table = metrics_by_period(result, periods, timeframe)
        if split_table.empty:
            continue
        print(f"  {label}")
        formatted = split_table.copy()
        for column in ("total_return", "max_drawdown", "win_rate"):
            if column in formatted:
                formatted[column] = formatted[column].map(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "-"
                )
        for column in ("sharpe", "sortino", "profit_factor"):
            if column in formatted:
                formatted[column] = formatted[column].map(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "-"
                )
        print(formatted.to_string(index=False))
        print()
        tagged = split_table.copy()
        tagged.insert(0, "system", label)
        regime_rows.append(tagged)

    # ------------------------------------------------ validation vs test
    print(RULE)
    print("  3. VALIDATION vs TEST - how much of the validation result was real?")
    print(RULE)
    validation_start, validation_end = get_split(VALIDATION)
    validation_ohlcv = apply_embargo(
        load_ohlcv(timeframe, start=validation_start, end=validation_end)
    )
    validation_result = run_backtest(strategy.run(validation_ohlcv))
    validation_metrics = compute_metrics(validation_result, timeframe)
    test_metrics = table[table["system"] == "A_rules_only"].iloc[0]

    degradation = pd.DataFrame(
        {
            "validation": [
                f"{validation_metrics['total_return']:.1%}",
                f"{validation_metrics['sharpe']:.2f}",
                f"{validation_metrics['sortino']:.2f}",
                f"{validation_metrics['max_drawdown']:.1%}",
                int(validation_metrics["n_trades"]),
            ],
            "test": [
                f"{test_metrics['total_return']:.1%}",
                f"{test_metrics['sharpe']:.2f}",
                f"{test_metrics['sortino']:.2f}",
                f"{test_metrics['max_drawdown']:.1%}",
                int(test_metrics["n_trades"]),
            ],
        },
        index=["Total return", "Sharpe", "Sortino", "Max drawdown", "Trades"],
    )
    print(degradation.to_string())
    print(
        "\n  The gap between these columns is what parameter selection bought us on\n"
        "  validation and could not deliver out of sample. It is the most honest\n"
        "  number in the project."
    )

    # ----------------------------------------------------------- the CIs
    print(f"\n{RULE}")
    print(f"  4. DIFFERENCES, WITH {METRICS.confidence_level:.0%} BOOTSTRAP INTERVALS")
    print(RULE)
    baseline = results["A_rules_only"]
    for label, result in results.items():
        if label == "A_rules_only":
            continue
        comparison = difference_ci(baseline.trades, result.trades)
        if not comparison:
            print(f"  {label:<26} too few trades to compare")
            continue
        verdict = "SIGNIFICANT" if comparison["significant"] else "not distinguishable"
        print(
            f"  {label:<26} {comparison['mean_difference']:+7.1%}  "
            f"CI [{comparison['ci_low']:+7.1%}, {comparison['ci_high']:+7.1%}]  "
            f"P(better) {comparison['probability_b_better']:5.1%}  {verdict}"
        )

    if "C_rules_ml_llm" in results and "B_rules_plus_ml" in results:
        head_to_head = difference_ci(
            results["B_rules_plus_ml"].trades, results["C_rules_ml_llm"].trades
        )
        if head_to_head:
            print(
                "\n  THE QUESTION THAT MATTERS - the LLM judge against the four-line rule\n"
                "  (System B is the deterministic agreement rule, reached by another path):"
            )
            print(
                f"    C minus B: {head_to_head['mean_difference']:+.1%}  "
                f"CI [{head_to_head['ci_low']:+.1%}, {head_to_head['ci_high']:+.1%}]  "
                f"P(LLM better) {head_to_head['probability_b_better']:.1%}"
            )
            if not head_to_head["significant"]:
                print(
                    "    -> Not distinguishable. Whatever the LLM contributed, a four-line\n"
                    "       condition contributed the same."
                )

    intervals = bootstrap_trade_metrics(baseline.trades)
    if intervals:
        print("\n  System A, absolute:")
        for metric, (low, high) in intervals.items():
            print(f"    {metric:<16} [{low:>8.3f}, {high:>8.3f}]")

    if records:
        print(f"\n{RULE}")
        print("  5. JUDGE BEHAVIOUR")
        print(RULE)
        for label, entries in records.items():
            print("  " + describe_agreement(agreement_stats(entries), label))

    # -------------------------------------------------------------- save
    path = PATHS.results / "final_test.csv"
    table.to_csv(path, index=False)
    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(
            PATHS.results / "final_test_by_regime.csv", index=False
        )
    pd.DataFrame(
        {
            "metric": ["total_return", "sharpe", "sortino", "max_drawdown", "n_trades"],
            "validation": [validation_metrics[k] for k in
                           ("total_return", "sharpe", "sortino", "max_drawdown", "n_trades")],
            "test": [test_metrics[k] for k in
                     ("total_return", "sharpe", "sortino", "max_drawdown", "n_trades")],
        }
    ).to_csv(PATHS.results / "final_degradation.csv", index=False)

    # The CSV is the source of truth; the database copy is a convenience for
    # the dashboard, so a storage problem must not lose the run.
    try:
        repository = Repository()
        for label, result in results.items():
            repository.record_backtest(
                system=label,
                strategy=strategy.name,
                timeframe=timeframe,
                split="test",
                metrics=compute_metrics(result, timeframe),
                params=strategy.params.label(),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write results to SQLite (%s); CSV is unaffected.", exc)

    summary = {
        "strategy": strategy.name,
        "timeframe": timeframe,
        "params": strategy.params.label(),
        "model": model_name,
        "threshold": args.threshold,
        "test_start": str(test_ohlcv.index[0]),
        "test_end": str(test_ohlcv.index[-1]),
        "validation": {k: validation_metrics[k] for k in ("total_return", "sharpe", "sortino", "max_drawdown", "n_trades")},
    }
    (PATHS.results / "final_test_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Saved -> {path}")
    print("  Saved -> data/results/final_test_summary.json")
    print("  Recorded in SQLite (backtests table).\n")
    print(RULE)
    print("  The test period has now been used. Do not re-run this after changing")
    print("  anything, or the out-of-sample claim no longer holds.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
