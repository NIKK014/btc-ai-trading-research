"""Reset the live paper-trading record.

Clears trades, decisions, equity history and the simulated broker's balance,
so the trader starts from a clean slate. Research results in
``data/results/`` are **not** touched.

    python scripts/reset_live.py            # asks first
    python scripts/reset_live.py --yes

Useful when the live log has been polluted - by two loops running at once, by
a dry run mixed with a real one, or simply to start a clean track record
before a presentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PATHS  # noqa: E402
from src.database.repository import Repository  # noqa: E402

LOCK_FILE = PATHS.root / "data" / "live.lock"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    repository = Repository()
    summary = repository.summary()

    print("=" * 70)
    print("  RESET LIVE TRADING RECORD")
    print("=" * 70)
    print(f"\n  Trades logged     {summary['total_trades']}")
    print(f"  Decisions logged  {summary['decisions_logged']}")
    print("\n  This clears trades, decisions, equity history and the simulated")
    print("  broker balance. Research results in data/results/ are untouched.\n")

    if LOCK_FILE.exists():
        print(f"  WARNING: a trading loop may still be running (lock: {LOCK_FILE}).")
        print("  Stop it first:  pkill -f 'main.py'\n")

    if not args.yes and input("  Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        print("  Aborted. Nothing changed.")
        return 1

    with repository.connect() as connection:
        for table in ("trades", "decisions", "portfolio_history"):
            connection.execute(f"DELETE FROM {table}")
        connection.execute(
            "DELETE FROM system_state WHERE key IN ('simulated_broker', 'day_start_equity')"
        )

    if LOCK_FILE.exists():
        LOCK_FILE.unlink()

    print("\n  Cleared. The broker will reinitialise at its configured starting")
    print("  equity on the next run.\n")
    print("  Restart with:")
    print("    nohup python main.py --system C > logs/live.log 2>&1 &")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
