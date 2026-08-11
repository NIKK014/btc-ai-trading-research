"""Parameter search for every strategy, on the validation split only.

    python scripts/run_optimizer.py
    python scripts/run_optimizer.py --timeframes 4h --max-configs 60

Writes the full configuration leaderboard and the tuned parameter set to
``data/results/``. The test period is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.settings import DATA, PATHS  # noqa: E402
from src.backtesting.optimizer import (  # noqa: E402
    best_params_per_strategy,
    optimise,
)
from src.backtesting.runner import (  # noqa: E402
    VALIDATION,
    apply_embargo,
    format_leaderboard,
    get_split,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.strategies.base import registry  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger("run_optimizer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframes", nargs="+", default=list(DATA.timeframes))
    parser.add_argument("--max-configs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    PATHS.ensure()
    start, end = get_split(VALIDATION)

    strategies = {k: v for k, v in registry().items() if v.methodology != "benchmark"}
    data = {
        timeframe: apply_embargo(load_ohlcv(timeframe, start=start, end=end))
        for timeframe in args.timeframes
    }

    began = time.time()
    leaderboard, diagnostics = optimise(
        strategies, data, max_configs=args.max_configs, seed=args.seed
    )
    elapsed = time.time() - began

    if leaderboard.empty:
        print("No configurations produced results.")
        return 1

    board_path = PATHS.results / "optimizer_validation.csv"
    leaderboard.to_csv(board_path, index=False)

    print(f"\n{'=' * 100}")
    print(f"PARAMETER SEARCH - VALIDATION ONLY ({len(leaderboard):,} configurations, {elapsed:.0f}s)")
    print("=" * 100)
    print(format_leaderboard(leaderboard, top=20))

    print(f"\n{'-' * 100}")
    print("Selection-bias diagnostics (Sortino across each strategy's own grid)")
    print("A large gap between best and median means the result is highly")
    print("parameter-sensitive, which rarely survives out of sample.\n")

    rows = []
    for key, report in sorted(diagnostics.items()):
        if report:
            rows.append({"strategy@timeframe": key, **report})
    if rows:
        frame = pd.DataFrame(rows)
        print(
            frame.to_string(
                index=False,
                float_format=lambda v: f"{v:,.2f}",
            )
        )

    winners = best_params_per_strategy(leaderboard, strategies)
    tuned_path = PATHS.results / "tuned_params.json"
    tuned_path.write_text(json.dumps(winners, indent=2, default=str), encoding="utf-8")

    print(f"\n{'-' * 100}\nBest configuration per strategy:")
    for name, info in winners.items():
        flag = "" if info["eligible"] else "   (FAILS GATES)"
        print(
            f"  {name:<28} @ {info['timeframe']:<4} "
            f"Sortino {info['sortino']:>6.2f}  return {info['total_return']:>7.1%}  "
            f"{int(info['n_trades']):>5} trades{flag}"
        )
        print(f"      {info['params']}")

    print(f"\nSaved -> {board_path}\nSaved -> {tuned_path}")
    print(
        "\nThese parameters were chosen on validation data. Their validation\n"
        "performance is therefore optimistic by construction - the honest\n"
        "estimate is what they do on the untouched test period."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
