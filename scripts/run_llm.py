"""System C: does an LLM judge improve on rules and ML?

Runs five arms over the same strategy signals on the validation split:

    A              rules only
    always_agree   control - must reproduce A exactly
    B              rules + ML filter
    deterministic  control - four lines of arithmetic, same inputs as C
    C              rules + LLM judge

    python scripts/run_llm.py                # uses cached decisions if present
    python scripts/run_llm.py --dry-run      # no API calls, control arms only

Requires OPENAI_API_KEY in .env. Every decision is cached to
``data/cache/llm_decisions.json``, so re-runs are free and the presentation
cannot be broken by an API outage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.settings import LLM, ML, PATHS  # noqa: E402
from src.agents.deterministic_judge import AlwaysAgreeJudge, DeterministicJudge  # noqa: E402
from src.agents.harness import (  # noqa: E402
    agreement_stats,
    build_snapshots,
    describe_agreement,
    run_arm,
)
from src.agents.schema import records_to_frame  # noqa: E402
from src.agents.trading_judge import TradingJudge  # noqa: E402
from src.backtesting.engine import run_backtest  # noqa: E402
from src.backtesting.metrics import compute_metrics, difference_ci, summarise  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.models.features import build_dataset  # noqa: E402
from src.models.predict import apply_ml_filter, predictions_frame  # noqa: E402
from src.models.train import chronological_splits, comparison_table, train_models  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger("run_llm")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ml import load_tuned  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="ema_rsi_trend")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--threshold", type=float, default=ML.confidence_threshold)
    parser.add_argument("--dry-run", action="store_true", help="Skip the LLM arm entirely.")
    parser.add_argument("--limit", type=int, default=None, help="Cap LLM decision points.")
    args = parser.parse_args()

    PATHS.ensure()
    strategy, tuned_timeframe = load_tuned(args.strategy)
    timeframe = args.timeframe or tuned_timeframe or "4h"

    # ------------------------------------------------------------ setup
    ohlcv = load_ohlcv(timeframe)
    features, target, _ = build_dataset(ohlcv, strategy.indicator_spec)
    splits = chronological_splits(features.index)
    trained = train_models(features, target, splits)
    table = comparison_table(trained)
    best_name = table.iloc[0]["model"]
    model = trained[best_name]

    validation = ohlcv.loc[splits.validation[0] : splits.validation[-1]]
    prepared = strategy.run(validation)
    signals = prepared["signal"]
    predictions = predictions_frame(
        model, features.loc[features.index.intersection(prepared.index)]
    )

    snapshots = build_snapshots(prepared, signals, predictions, strategy.name, timeframe)
    if args.limit:
        snapshots = snapshots[: args.limit]

    print(f"\n{'=' * 100}")
    print(f"SYSTEM COMPARISON - {strategy.name} @ {timeframe} (validation)")
    print(f"ML model: {best_name}   entry decision points: {len(snapshots)}")
    print("=" * 100)

    # ------------------------------------------------------------- arms
    results = {}
    all_records = {}

    results["A_rules_only"] = run_backtest(prepared)

    for label, judge in (
        ("control_always_agree", AlwaysAgreeJudge()),
        ("control_deterministic", DeterministicJudge(args.threshold)),
    ):
        gated, records = run_arm(
            judge, prepared, signals, predictions, strategy.name, timeframe, snapshots
        )
        frame = prepared.copy()
        frame["signal"] = gated
        results[label] = run_backtest(frame)
        all_records[label] = records

    frame_b = prepared.copy()
    frame_b["signal"] = apply_ml_filter(signals, predictions, args.threshold)
    results["B_rules_plus_ml"] = run_backtest(frame_b)

    if not args.dry_run:
        judge = TradingJudge()
        gated, records = run_arm(
            judge, prepared, signals, predictions, strategy.name, timeframe, snapshots
        )
        frame_c = prepared.copy()
        frame_c["signal"] = gated
        results["C_rules_ml_llm"] = run_backtest(frame_c)
        all_records["C_rules_ml_llm"] = records
        print(
            f"\nLLM: {judge.calls_made} API calls, {judge.cache_hits} cache hits, "
            f"{judge.failures} failures (model {LLM.model})"
        )

    # ---------------------------------------------------------- results
    rows = []
    for label, result in results.items():
        metrics = compute_metrics(result, timeframe)
        metrics["system"] = label
        rows.append(metrics)
        print()
        print(summarise(result, timeframe, label))

    print(f"\n{'-' * 100}\nJUDGE BEHAVIOUR\n{'-' * 100}")
    for label, records in all_records.items():
        print(describe_agreement(agreement_stats(records), label))

    # Verification: System B and the deterministic judge implement the same
    # rule by two different code paths, so their results must be identical. If
    # they ever diverge, one of the two is wrong.
    if "B_rules_plus_ml" in results and "control_deterministic" in results:
        b_return = compute_metrics(results["B_rules_plus_ml"], timeframe)["total_return"]
        d_return = compute_metrics(results["control_deterministic"], timeframe)["total_return"]
        match = abs(b_return - d_return) < 1e-9
        print(
            f"\nCross-check: System B and the deterministic judge agree "
            f"{'exactly' if match else 'NO - THEY DIVERGE, ONE IS WRONG'}. "
            "They are the same rule reached by two code paths."
        )

    # The judged arms are compared against always_agree rather than raw System
    # A. Bars where the ML model has no prediction cannot be judged at all, so
    # every judged arm sees a slightly smaller decision set; always_agree is
    # that same decision set with no judgement applied, which isolates the
    # judge's contribution from the difference in coverage.
    print(f"\n{'-' * 100}\nDIFFERENCES vs UNJUDGED BASELINE (bootstrap 95% CI on total return)\n{'-' * 100}")
    baseline = results.get("control_always_agree", results["A_rules_only"])
    for label, result in results.items():
        if label in {"control_always_agree", "A_rules_only"}:
            continue
        comparison = difference_ci(baseline.trades, result.trades)
        if not comparison:
            print(f"{label:<24} too few trades to compare")
            continue
        verdict = "SIGNIFICANT" if comparison["significant"] else "not distinguishable"
        print(
            f"{label:<24} {comparison['mean_difference']:+7.1%}  "
            f"CI [{comparison['ci_low']:+7.1%}, {comparison['ci_high']:+7.1%}]  "
            f"P(better)={comparison['probability_b_better']:5.1%}  {verdict}"
        )

    if "C_rules_ml_llm" in results and "control_deterministic" in results:
        head_to_head = difference_ci(
            results["control_deterministic"].trades, results["C_rules_ml_llm"].trades
        )
        if head_to_head:
            print(f"\n{'-' * 100}\nTHE QUESTION THAT MATTERS: LLM judge vs a four-line rule")
            print(
                f"  Difference {head_to_head['mean_difference']:+.1%}  "
                f"CI [{head_to_head['ci_low']:+.1%}, {head_to_head['ci_high']:+.1%}]  "
                f"P(LLM better) = {head_to_head['probability_b_better']:.1%}"
            )
            if not head_to_head["significant"]:
                print(
                    "  -> The LLM is not distinguishable from the deterministic rule.\n"
                    "     Whatever it contributed, a four-line condition contributed the same."
                )

    # ------------------------------------------------------------- save
    path = PATHS.results / f"systems_abc_{timeframe}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    for label, records in all_records.items():
        frame = records_to_frame(records)
        if not frame.empty:
            frame.to_csv(PATHS.results / f"decisions_{label}_{timeframe}.csv")
    print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
