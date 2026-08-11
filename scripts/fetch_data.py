"""Download and cache historical BTCUSDT candles.

Run once at the start of the project, then again with --refresh whenever you
want to pull in newly closed candles.

    python scripts/fetch_data.py                  # full history, all timeframes
    python scripts/fetch_data.py --refresh        # extend existing caches
    python scripts/fetch_data.py --timeframes 1h  # single timeframe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.settings import DATA, PATHS  # noqa: E402
from src.data.loader import cache_path, load_ohlcv, validate_ohlcv  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger("fetch_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(DATA.timeframes),
        help=f"Timeframes to download (default: {' '.join(DATA.timeframes)})",
    )
    parser.add_argument("--symbol", default=DATA.symbol)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Extend existing caches with newly closed candles.",
    )
    args = parser.parse_args()

    PATHS.ensure()
    reports = []

    for timeframe in args.timeframes:
        frame = load_ohlcv(timeframe, symbol=args.symbol, refresh=args.refresh)
        if frame.empty:
            logger.error("No data for %s - skipping validation.", timeframe)
            continue
        report = validate_ohlcv(frame, timeframe)
        report["cache"] = cache_path(timeframe, args.symbol).name
        report["size_mb"] = round(cache_path(timeframe, args.symbol).stat().st_size / 1e6, 2)
        reports.append(report)

    if not reports:
        logger.error("Nothing downloaded.")
        return 1

    summary = pd.DataFrame(reports)[
        ["timeframe", "rows", "start", "end", "gaps", "missing_candles", "nan_values", "size_mb"]
    ]
    print("\n" + summary.to_string(index=False))
    print(
        "\nNote: candle timestamps are UTC OPEN times. Gaps are exchange "
        "downtime and are deliberately left unfilled - interpolating candles "
        "would invent price action that never happened."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
