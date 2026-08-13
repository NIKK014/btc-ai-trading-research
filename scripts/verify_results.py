"""Prove that a code change did not move any published number.

The results in ``docs/results.md`` and on the dashboard come from a test split
that has been used once and cannot honestly be used again. That makes ordinary
refactoring risky: a tidy-up that silently changes a backtest is not something
the test suite would necessarily catch, and there is no second chance to notice.

So this script recomputes everything the presentation depends on - every
strategy backtest across all three timeframes, the label distributions, the
model comparison table, the ML filter at each threshold and an optimiser run -
and reduces it to one hash.

    python scripts/verify_results.py --save      # before you change anything
    ...edit code...
    python scripts/verify_results.py             # after: pass or fail

A matching hash means the change was cosmetic. A differing hash prints the
first keys that moved, so it is clear what broke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.settings import DATA, PATHS  # noqa: E402
from src.backtesting.engine import run_backtest  # noqa: E402
from src.backtesting.metrics import compute_metrics  # noqa: E402
from src.backtesting.optimizer import search_strategy  # noqa: E402
from src.backtesting.runner import TRAIN, VALIDATION, run_grid  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.models.features import build_dataset  # noqa: E402
from src.models.labels import class_balance, triple_barrier_labels  # noqa: E402
from src.models.predict import apply_ml_filter, predictions_frame  # noqa: E402
from src.models.train import chronological_splits, comparison_table, train_models  # noqa: E402
from src.strategies import build, registry  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "results" / "verify_reference.json"

# Rounded before hashing: BLAS libraries disagree in the last bit or two of a
# float, so an exact comparison would fail on a different machine for reasons
# that have nothing to do with the code.
PRECISION = 8


def _round(value: object) -> object:
    if isinstance(value, float):
        return round(value, PRECISION)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def snapshot() -> dict:
    """Recompute every result the presentation relies on.

    Deliberately goes through ``run_grid``, the same entry point
    ``scripts/run_baseline.py`` uses, rather than calling the engine directly.
    A verifier that reimplements the pipeline only proves that the copy still
    agrees with itself.
    """
    results: dict = {}

    # The leaderboards behind docs/results.md, split by split.
    for split in (TRAIN, VALIDATION):
        leaderboard, _ = run_grid(DATA.timeframes, split=split)
        results[f"leaderboard_{split}"] = _round(
            leaderboard.drop(columns=["params"], errors="ignore")
            .round(PRECISION)
            .to_dict(orient="records")
        )

    for timeframe in DATA.timeframes:
        ohlcv = load_ohlcv(timeframe)
        results[f"labels_{timeframe}"] = _round(class_balance(triple_barrier_labels(ohlcv)))

    # ML and filtering, on the tuned timeframe only - this is the expensive part.
    ohlcv = load_ohlcv("4h")
    strategy = build("ema_rsi_trend")
    features, target, _ = build_dataset(ohlcv, strategy.indicator_spec)
    splits = chronological_splits(features.index)
    trained = train_models(features, target, splits)
    results["ml_table"] = _round(
        comparison_table(trained).round(PRECISION).to_dict(orient="records")
    )

    validation = ohlcv.loc[splits.validation[0] : splits.validation[-1]]
    prepared = strategy.run(validation)
    predictions = predictions_frame(
        trained["random_forest"],
        features.loc[features.index.intersection(prepared.index)],
    )
    for threshold in (0.30, 0.35, 0.40):
        frame = prepared.copy()
        frame["signal"] = apply_ml_filter(prepared["signal"], predictions, threshold)
        results[f"filter_{threshold}"] = _round(
            compute_metrics(run_backtest(frame), "4h")
        )

    grid = search_strategy(registry()["ema_rsi_trend"], ohlcv, "4h", max_configs=8)
    results["opt"] = _round(grid.round(PRECISION).to_dict(orient="records"))

    return results


def digest(results: dict) -> str:
    payload = json.dumps(results, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Write the reference, don't compare.")
    args = parser.parse_args()

    PATHS.ensure()
    print("Recomputing every published result (this takes a couple of minutes)...")
    results = snapshot()
    current = digest(results)

    if args.save:
        REFERENCE.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
        print(f"\nReference saved: {len(results)} keys, sha {current}")
        print(f"  -> {REFERENCE}")
        return 0

    if not REFERENCE.exists():
        print(f"\nNo reference at {REFERENCE}. Run with --save first.")
        return 1

    reference = json.loads(REFERENCE.read_text())
    expected = digest(reference)

    print(f"\nkeys compared: {len(results)}")
    print(f"reference sha: {expected}")
    print(f"current sha  : {current}")

    if current == expected:
        print("\nIDENTICAL - the change did not move any published number.")
        return 0

    print("\nCHANGED - these results moved:")
    for key in sorted(set(reference) | set(results)):
        if reference.get(key) != results.get(key):
            print(f"  {key}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
