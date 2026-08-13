"""Export a small slice of recent candles for the published dashboard.

The deployed copy has no price cache - six years of candles are far too large
for Git - so it falls back to fetching a live window. That works from a laptop
and fails from a datacenter: Bybit refuses cloud IP ranges outright, and the
public alternatives are unreliable from the same hosts.

A live chart that is sometimes blank is worse than a recent one that always
renders, particularly with a presentation to give. So this writes the last few
hundred candles to a small CSV that ships with the repo and is used whenever
the live feeds cannot be reached. The dashboard labels it as a snapshot and
shows its date - it is never passed off as live.

    python scripts/export_snapshot.py
    python scripts/export_snapshot.py --bars 300 --timeframe 4h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PATHS  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshot"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--bars", type=int, default=250)
    args = parser.parse_args()

    PATHS.ensure()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_ohlcv(args.timeframe).tail(args.bars)
    if frame.empty:
        print("No cached candles to export. Run scripts/fetch_data.py first.")
        return 1

    path = SNAPSHOT_DIR / f"recent_{args.timeframe}.csv"
    frame[["open", "high", "low", "close", "volume"]].to_csv(path)

    size_kb = path.stat().st_size / 1024
    print(f"Wrote {len(frame)} {args.timeframe} candles ({size_kb:.0f} KB)")
    print(f"  {frame.index[0]}  ->  {frame.index[-1]}")
    print(f"  {path}")
    print("\nCommit and push this to refresh the chart on the published dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
