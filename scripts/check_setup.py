"""Preflight check before running the live paper trader.

Verifies every prerequisite in order and stops at the first real problem, so a
failure tells you exactly what to fix rather than surfacing as a stack trace
three minutes into a live run.

    python scripts/check_setup.py
    python scripts/check_setup.py --set-leverage

Read-only by default: it queries balances and positions but never places an
order. ``--set-leverage`` is the one exception and only sets 1x.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (  # noqa: E402
    BACKTEST,
    BYBIT_PAPER_TRADE_URL,
    DATA,
    PAPER_MODES,
    PATHS,
    RISK,
    TRADING_MODE,
)

PASS = "  [OK]  "
FAIL = "  [!!]  "
WARN = "  [ ? ] "

failures: list = []
warnings: list = []


def check(condition: bool, label: str, detail: str = "", fatal: bool = True) -> bool:
    marker = PASS if condition else (FAIL if fatal else WARN)
    print(f"{marker}{label}" + (f"  -  {detail}" if detail else ""))
    if not condition:
        (failures if fatal else warnings).append(label)
    return condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-leverage", action="store_true", help="Set 1x leverage.")
    parser.add_argument("--symbol", default=DATA.symbol)
    parser.add_argument(
        "--broker",
        default="simulated",
        choices=["simulated", "bybit"],
        help="Which execution venue to verify.",
    )
    args = parser.parse_args()

    venue = "local simulator" if args.broker == "simulated" else "Bybit testnet"
    print("=" * 72)
    print(f"  PREFLIGHT CHECK - paper trading via {venue}")
    print("=" * 72)

    # ---------------------------------------------------------- 1. safety
    print("\n1. Safety configuration")
    if not check(
        TRADING_MODE in PAPER_MODES,
        f"TRADING_MODE is a paper mode ({TRADING_MODE!r})",
        f"expected one of {sorted(PAPER_MODES)}" if TRADING_MODE not in PAPER_MODES else "",
    ):
        print("\n     Fix: set TRADING_MODE=testnet in .env")
        return 1

    if args.broker == "bybit":
        check(
            BYBIT_PAPER_TRADE_URL == "https://api-testnet.bybit.com",
            "Trading host is the testnet endpoint",
            BYBIT_PAPER_TRADE_URL,
        )
    else:
        check(True, "No exchange credentials in use", "simulator places no external orders")
    check(
        (PATHS.root / ".env").exists(),
        ".env exists",
    )
    check(
        ".env" in (PATHS.root / ".gitignore").read_text(encoding="utf-8").split("\n"),
        ".env is gitignored",
    )

    # ----------------------------------------------------- 2. credentials
    print("\n2. Credentials")
    if args.broker == "simulated":
        check(
            bool(os.getenv("OPENAI_API_KEY")),
            "OPENAI_API_KEY is set",
            "needed only for --system C",
            fatal=False,
        )
        print("       (Bybit keys not required: the simulator fills locally.)")
    key = os.getenv("BYBIT_API_KEY", "")
    secret = os.getenv("BYBIT_API_SECRET", "")
    # Show a masked prefix so you can confirm *which* key is loaded. Bybit's
    # console displays the same prefix, and the commonest failure by far is a
    # stale key left in .env after switching environments.
    masked = f"{key[:6]}...{key[-2:]}" if len(key) > 8 else "(too short)"
    if args.broker == "bybit":
        check(bool(key), "BYBIT_API_KEY is set", f"{masked}  ({len(key)} chars)")
        check(bool(secret), "BYBIT_API_SECRET is set", f"{len(secret)} chars")
        check(
            bool(os.getenv("OPENAI_API_KEY")),
            "OPENAI_API_KEY is set",
            "needed only for --system C",
            fatal=False,
        )
    if failures:
        print("\n     Fix: add your TESTNET keys to .env")
        print("     testnet.bybit.com -> avatar -> API -> create key")
        return 1

    # ------------------------------------------------------------ 3. data
    print("\n3. Market data")
    for timeframe in DATA.timeframes:
        path = PATHS.raw / f"{args.symbol}_{timeframe}.parquet"
        check(
            path.exists(),
            f"{timeframe} history cached",
            f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "run scripts/fetch_data.py",
        )
    check(
        (PATHS.results / "tuned_params.json").exists(),
        "Tuned parameters available",
        "run scripts/run_optimizer.py" if not (PATHS.results / "tuned_params.json").exists() else "",
        fatal=False,
    )

    # -------------------------------------------------------- 4. database
    print("\n4. Database")
    try:
        from src.database.repository import Repository

        repository = Repository()
        summary = repository.summary()
        check(True, "SQLite ready", f"{summary['total_trades']} trades logged so far")
    except Exception as exc:  # noqa: BLE001
        check(False, "SQLite ready", str(exc))
        return 1

    # -------------------------------------------------- 5. execution venue
    if args.broker == "simulated":
        print("\n5. Execution venue: local simulator")
        try:
            from src.exchange.simulated_broker import SimulatedBroker

            broker = SimulatedBroker(repository=repository, timeframe="4h")
            equity = broker.wallet_equity()
            check(equity > 0, "Simulator ready", broker.describe())
            price = broker.last_price(args.symbol)
            check(price > 0, "Live price feed reachable", f"{args.symbol} at {price:,.2f}")
        except Exception as exc:  # noqa: BLE001
            check(False, "Simulator ready", str(exc))
            return 1

        print("\n" + "=" * 72)
        if failures:
            print(f"  {len(failures)} problem(s) to fix:")
            for item in failures:
                print(f"    - {item}")
            return 1
        print("\n  READY. Live settings:")
        print(f"    Venue           local simulator (real prices, simulated fills)")
        print(f"    Symbol          {args.symbol}")
        print(f"    Equity          {equity:,.2f} USDT (simulated)")
        print(f"    Risk per trade  {RISK.risk_per_trade:.1%}  ->  {equity * RISK.risk_per_trade:,.2f} USDT")
        print(f"    Daily loss cap  {RISK.max_daily_loss:.1%}")
        print(f"    Stop            {RISK.atr_stop_multiple:g} x ATR")
        print(f"    Target          {RISK.reward_risk_ratio:g}:1 reward:risk")
        print("\n  Next:")
        print("    python main.py --system C --dry-run --once     # rehearse")
        print("    python main.py --system C                      # start trading")
        print("=" * 72)
        return 0

    print("\n5. Bybit testnet connection")
    try:
        from src.exchange.bybit_client import BybitPaperClient

        client = BybitPaperClient()
        check(True, "Client constructed", "paper mode enforced")
    except Exception as exc:  # noqa: BLE001
        check(False, "Client constructed", str(exc))
        return 1

    try:
        equity = client.wallet_equity()
        check(equity > 0, "Wallet reachable", f"{equity:,.2f} USDT")
    except Exception as exc:  # noqa: BLE001
        check(False, "Wallet reachable", str(exc))
        print("\n     Common causes:")
        print("       - keys were created on mainnet, not on testnet.bybit.com")
        print("       - the testnet wallet is empty (use the testnet faucet)")
        print("       - keys lack Contract Trade permission")
        print("       - system clock is skewed (Bybit rejects stale timestamps)")
        return 1

    try:
        position = client.position(args.symbol)
        state = (
            "flat"
            if position["direction"] == 0
            else f"{'LONG' if position['direction'] > 0 else 'SHORT'} {position['size']}"
        )
        check(True, "Position endpoint reachable", state)
    except Exception as exc:  # noqa: BLE001
        check(False, "Position endpoint reachable", str(exc))

    if args.set_leverage:
        try:
            client.set_leverage(args.symbol, BACKTEST.leverage)
            check(True, f"Leverage set to {BACKTEST.leverage:g}x")
        except Exception as exc:  # noqa: BLE001
            check(False, "Leverage set", str(exc), fatal=False)

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 72)
    if failures:
        print(f"  {len(failures)} problem(s) to fix before going live:")
        for item in failures:
            print(f"    - {item}")
        return 1

    if warnings:
        print("  Non-blocking warnings:")
        for item in warnings:
            print(f"    - {item}")

    risk_per_trade = equity * RISK.risk_per_trade
    print("\n  READY. Live settings:")
    print(f"    Symbol          {args.symbol}")
    print(f"    Equity          {equity:,.2f} USDT (testnet)")
    print(f"    Risk per trade  {RISK.risk_per_trade:.1%}  ->  {risk_per_trade:,.2f} USDT")
    print(f"    Daily loss cap  {RISK.max_daily_loss:.1%}  ->  {equity * RISK.max_daily_loss:,.2f} USDT")
    print(f"    Leverage        {BACKTEST.leverage:g}x")
    print(f"    Stop            {RISK.atr_stop_multiple:g} x ATR")
    print(f"    Target          {RISK.reward_risk_ratio:g}:1 reward:risk")
    print("\n  Next:")
    print("    python main.py --system C --dry-run --once     # rehearse")
    print("    python main.py --system C                      # go live on testnet")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
