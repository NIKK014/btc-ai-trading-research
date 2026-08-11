"""System B: does a machine-learning filter improve the tuned strategy?

Trains Dummy / Logistic Regression / Random Forest on the triple-barrier
target, applies the best model as a filter on System A's signals, and reports
the difference with a bootstrap confidence interval.

    python scripts/run_ml.py
    python scripts/run_ml.py --timeframe 4h --threshold 0.45

Reads train and validation only. The test period is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.settings import DATA, LABELS, ML, PATHS, LabelConfig  # noqa: E402
from src.backtesting.engine import run_backtest  # noqa: E402
from src.backtesting.metrics import compute_metrics, difference_ci, summarise  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.models.features import build_dataset, correlated_pairs  # noqa: E402
from src.models.labels import class_balance, describe_balance, triple_barrier_labels  # noqa: E402
from src.models.predict import (  # noqa: E402
    apply_ml_filter,
    filter_diagnostics,
    predictions_frame,
    sweep_threshold,
)
from src.models.train import chronological_splits, comparison_table, train_models  # noqa: E402
from src.strategies.base import build, registry  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger("run_ml")


def load_tuned(strategy_name: str):
    """Rebuild the strategy from the optimiser's saved winning configuration."""
    path = PATHS.results / "tuned_params.json"
    if not path.exists():
        logger.warning("No tuned parameters found; using defaults. Run run_optimizer.py first.")
        return build(strategy_name), None

    tuned = json.loads(path.read_text(encoding="utf-8"))
    if strategy_name not in tuned:
        return build(strategy_name), None

    entry = tuned[strategy_name]
    values = entry.get("param_values", {})
    cls = registry()[strategy_name]
    fields = {f.name: f.type for f in cls.params_class.__dataclass_fields__.values()}
    coerced = {}
    for key, value in values.items():
        if key not in fields:
            continue
        coerced[key] = int(value) if "int" in str(fields[key]) else float(value)
    return cls(cls.params_class(**coerced)), entry.get("timeframe")


def label_balance_report(timeframes) -> None:
    print(f"\n{'=' * 100}\nTARGET DEFINITION\n{'=' * 100}")
    print(f"ATR-scaled barrier ({LABELS.atr_multiple} x ATR{LABELS.atr_period}, horizon {LABELS.horizon_bars} bars):")
    for timeframe in timeframes:
        frame = load_ohlcv(timeframe)
        labels = triple_barrier_labels(frame)
        print("  " + describe_balance(class_balance(labels), timeframe))

    fixed = LabelConfig(mode="fixed", fixed_pct=0.005)
    print(f"\nSensitivity check - fixed {fixed.fixed_pct:.1%} barrier, same horizon:")
    for timeframe in timeframes:
        frame = load_ohlcv(timeframe)
        labels = triple_barrier_labels(frame, fixed)
        print("  " + describe_balance(class_balance(labels, fixed), timeframe))
    print(
        "\nThe fixed barrier means something different on every timeframe, which\n"
        "is why the ATR-scaled version is used for the experiment."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="ema_rsi_trend")
    parser.add_argument("--timeframe", default=None, help="Defaults to the tuned timeframe.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--skip-balance", action="store_true")
    args = parser.parse_args()

    PATHS.ensure()
    strategy, tuned_timeframe = load_tuned(args.strategy)
    timeframe = args.timeframe or tuned_timeframe or "4h"

    if not args.skip_balance:
        label_balance_report(DATA.timeframes)

    # ---------------------------------------------------------------- data
    ohlcv = load_ohlcv(timeframe)
    features, target, labels = build_dataset(ohlcv, strategy.indicator_spec)
    splits = chronological_splits(features.index)

    print(f"\n{'=' * 100}\nDATASET - {DATA.symbol} {timeframe}\n{'=' * 100}")
    print(splits.describe())
    print(f"features: {features.shape[1]}   usable rows: {len(features):,}")

    duplicates = correlated_pairs(features.loc[splits.train], threshold=0.95)
    if duplicates:
        print("\nNear-duplicate features (importance will be split between them):")
        for pair in duplicates[:5]:
            print(f"  {pair['a']:<26} ~ {pair['b']:<26} r={pair['correlation']:.3f}")

    # --------------------------------------------------------------- models
    trained = train_models(features, target, splits)
    print(f"\n{'=' * 100}\nMODEL COMPARISON (validation)\n{'=' * 100}")
    table = comparison_table(trained)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "\nuplift_vs_dummy is the only column that means anything alone: it is how\n"
        "much better than 'always predict the majority class' the model actually is."
    )

    best_name = table.iloc[0]["model"]
    if best_name == "dummy":
        print("\nNo model beat the dummy baseline. Reported as-is: on this target,")
        print("these features carry no usable signal.")
    best_model = trained[best_name]

    print(f"\nConfusion matrix - {best_name} (validation):")
    print(best_model.confusion.to_string())
    if best_model.importances is not None:
        print(f"\nTop features - {best_name}:")
        print(best_model.importances.head(10).to_string())

    # ------------------------------------------------- System A vs System B
    validation_ohlcv = ohlcv.loc[splits.validation[0] : splits.validation[-1]]
    prepared = strategy.run(validation_ohlcv)
    signals = prepared["signal"]

    validation_features = features.loc[features.index.intersection(prepared.index)]
    predictions = predictions_frame(best_model, validation_features)

    print(f"\n{'=' * 100}\nML FILTER THRESHOLD SWEEP (validation)\n{'=' * 100}")
    sweep = sweep_threshold(signals, predictions)
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    threshold = args.threshold if args.threshold is not None else ML.confidence_threshold
    filtered = apply_ml_filter(signals, predictions, threshold)

    system_a = run_backtest(prepared)
    prepared_b = prepared.copy()
    prepared_b["signal"] = filtered
    system_b = run_backtest(prepared_b)

    print(f"\n{'=' * 100}\nSYSTEM A vs SYSTEM B  (threshold {threshold})\n{'=' * 100}")
    diagnostics = filter_diagnostics(signals, filtered)
    print(
        f"Filter vetoed {diagnostics.get('veto_rate', 0):.1%} of signal bars "
        f"(long {diagnostics.get('long_veto_rate', 0):.1%}, "
        f"short {diagnostics.get('short_veto_rate', 0):.1%})\n"
    )
    print(summarise(system_a, timeframe, "SYSTEM A  (rules only)"))
    print()
    print(summarise(system_b, timeframe, f"SYSTEM B  (rules + {best_name})"))

    comparison = difference_ci(system_a.trades, system_b.trades, metric="total_return")
    if comparison:
        print(f"\n{'-' * 100}")
        print(
            f"Difference in total return (B minus A): "
            f"{comparison['mean_difference']:+.1%}  "
            f"95% CI [{comparison['ci_low']:+.1%}, {comparison['ci_high']:+.1%}]"
        )
        print(f"P(System B better) = {comparison['probability_b_better']:.1%}")
        if comparison["significant"]:
            verdict = "improved" if comparison["mean_difference"] > 0 else "degraded"
            print(f"-> The ML filter {verdict} performance at this sample size.")
        else:
            print(
                "-> The interval contains zero: at this sample size the ML filter\n"
                "   is NOT distinguishable from no filter at all. This is a valid\n"
                "   and reportable result, not a failure of the experiment."
            )

    rows = []
    for label, result in (("A_rules_only", system_a), (f"B_rules_plus_{best_name}", system_b)):
        metrics = compute_metrics(result, timeframe)
        metrics["system"] = label
        rows.append(metrics)
    path = PATHS.results / f"systems_ab_{timeframe}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    table.to_csv(PATHS.results / f"ml_models_{timeframe}.csv", index=False)
    print(f"\nSaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
