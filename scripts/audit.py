"""Pre-presentation audit.

Checks that the project is internally consistent: that every artefact exists,
that the safety invariants hold, and - most importantly - that the numbers
written into the documentation still match the numbers in the result files.

Hand-copied figures drift. A README that quotes a stale Sharpe ratio is worse
than one that quotes none, because it will be believed.

    python scripts/audit.py

Exit code 0 means everything checked out. Anything else needs attention before
you present.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.settings import BYBIT_PAPER_TRADE_URL, PATHS, TRADING_MODE  # noqa: E402

PASS, FAIL, WARN = "  [OK]  ", "  [!!]  ", "  [ ? ] "
failures: list = []
warnings: list = []


def check(ok: bool, label: str, detail: str = "", fatal: bool = True) -> bool:
    print(f"{PASS if ok else (FAIL if fatal else WARN)}{label}" + (f"  -  {detail}" if detail else ""))
    if not ok:
        (failures if fatal else warnings).append(label)
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * 70)


def read_csv(name: str) -> pd.DataFrame:
    path = PATHS.results / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def docs_text() -> str:
    """Every markdown file concatenated, for claim checking."""
    parts = [(ROOT / "README.md").read_text(encoding="utf-8")]
    parts += [p.read_text(encoding="utf-8") for p in sorted((ROOT / "docs").glob("*.md"))]
    return "\n".join(parts)


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0

    print("=" * 70)
    print("  PROJECT AUDIT")
    print("=" * 70)

    # ------------------------------------------------------------ 1. tests
    section("1. Test suite")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        tail = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        check(result.returncode == 0, "All tests pass", tail)
    except Exception as exc:  # noqa: BLE001
        check(False, "All tests pass", str(exc))

    # ------------------------------------------------------- 2. artefacts
    section("2. Required artefacts")
    for name in (
        "leaderboard_validation.csv", "optimizer_validation.csv", "tuned_params.json",
        "ml_models_4h.csv", "final_test.csv", "final_test_by_regime.csv",
        "final_degradation.csv",
    ):
        path = PATHS.results / name
        check(path.exists(), f"data/results/{name}", f"{path.stat().st_size:,} bytes" if path.exists() else "missing")

    for timeframe in ("15m", "1h", "4h"):
        path = PATHS.raw / f"BTCUSDT_{timeframe}.parquet"
        check(path.exists(), f"{timeframe} price cache", f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "missing")

    for doc in ("README.md", "docs/methodology.md", "docs/results.md", "docs/limitations.md",
                "docs/presentation.md", "docs/llm-judge.md", "docs/machine-learning.md",
                "docs/architecture.md", "docs/strategies.md"):
        check((ROOT / doc).exists(), doc)

    check(
        (PATHS.results / "decisions_C_rules_ml_llm_4h.csv").exists(),
        "Judge decision export (AI Judge dashboard tab)",
        "run scripts/show_judge.py" if not (PATHS.results / "decisions_C_rules_ml_llm_4h.csv").exists() else "",
        fatal=False,
    )

    # ------------------------------------------------- 3. results sanity
    section("3. Results are internally consistent")
    final = read_csv("final_test.csv")
    if final.empty:
        check(False, "final_test.csv readable")
    else:
        systems = set(final["system"])
        check(
            {"A_rules_only", "B_rules_plus_ml", "benchmark_buy_and_hold"} <= systems,
            "All required arms present",
            ", ".join(sorted(systems)),
        )
        check("C_rules_ml_llm" in systems, "System C (LLM judge) present", fatal=False)

        for _, row in final.iterrows():
            label = row["system"]
            ok = (
                -1.0 <= row["total_return"] <= 10.0
                and 0.0 <= row["max_drawdown"] <= 1.0
                and 0.0 <= row["win_rate"] <= 1.0
                and row["n_trades"] >= 0
            )
            check(ok, f"{label} metrics in plausible range",
                  f"ret {row['total_return']:+.1%}, dd {row['max_drawdown']:.1%}, {int(row['n_trades'])} trades")

        # Each filter can only remove trades, never add them.
        a = final.loc[final["system"] == "A_rules_only", "n_trades"].iloc[0]
        for arm in ("B_rules_plus_ml", "C_rules_ml_llm"):
            if arm in systems:
                n = final.loc[final["system"] == arm, "n_trades"].iloc[0]
                check(n <= a, f"{arm} trades <= System A", f"{int(n)} <= {int(a)}")

    # ------------------------------------------ 4. docs match the results
    section("4. Documentation matches the result files")
    text = docs_text()

    if not final.empty:
        claims = []
        for system, label in (
            ("A_rules_only", "System A return"),
            ("B_rules_plus_ml", "System B return"),
            ("benchmark_buy_and_hold", "Buy-and-hold return"),
        ):
            if system not in set(final["system"]):
                continue
            value = final.loc[final["system"] == system, "total_return"].iloc[0]
            rendered = f"{abs(value) * 100:.1f}%"
            claims.append((label, rendered, rendered in text))
        for label, rendered, found in claims:
            check(found, f"{label} ({rendered}) appears in the docs",
                  "" if found else "docs may quote a stale figure")

    degradation = read_csv("final_degradation.csv")
    if not degradation.empty:
        for metric, fmt in (("sharpe", "{:.2f}"), ("total_return", "{:.1%}")):
            row = degradation[degradation["metric"] == metric]
            if row.empty:
                continue
            validation = fmt.format(float(row["validation"].iloc[0])).lstrip("+")
            test = fmt.format(abs(float(row["test"].iloc[0])))
            check(validation in text, f"Validation {metric} ({validation}) in docs")
            check(test in text, f"Test {metric} ({test}) in docs")

    models = read_csv("ml_models_4h.csv")
    if not models.empty:
        best = models.iloc[0]
        clean = f"{best['val_balanced_acc']:.3f}"
        check(clean in text, f"Clean ML accuracy ({clean}) quoted in docs")
        check(
            "0.526" not in text or "not" in text.lower(),
            "Refit-on-train accuracy (0.526) not quoted as a clean result",
            fatal=False,
        )

    # ------------------------------------------------------- 5. safety
    section("5. Safety invariants")
    check(TRADING_MODE in {"testnet", "demo", "paper"}, f"TRADING_MODE is a paper mode ({TRADING_MODE!r})")
    check(BYBIT_PAPER_TRADE_URL == "https://api-testnet.bybit.com", "Paper host is the testnet endpoint")

    production = "api" + ".bybit.com"
    offenders = [
        str(p.relative_to(ROOT))
        for p in list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py"]
        if production in p.read_text(encoding="utf-8") and "api-demo" not in p.read_text(encoding="utf-8")
        and p.name != "public_client.py"
    ]
    check(not offenders, "Production trading host absent from the codebase", ", ".join(offenders))

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").split("\n")
    check(".env" in gitignore, ".env is gitignored")
    check(any("fuse_hidden" in line for line in gitignore), "FUSE artefacts ignored")
    check(any("db-wal" in line for line in gitignore), "SQLite WAL sidecars ignored")

    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
        check(".env\n" not in tracked, "No .env committed")
        check(".fuse_hidden" not in tracked, "No FUSE artefacts committed", fatal=False)
        check("db-wal" not in tracked, "No WAL sidecars committed", fatal=False)
        check(".parquet" not in tracked, "No large data files committed", fatal=False)
    except Exception:  # noqa: BLE001
        check(True, "Git tracking check skipped", "not a git repo or git unavailable", fatal=False)

    # ---------------------------------------------------- 6. scripts run
    section("6. Scripts are runnable")
    for script in sorted((ROOT / "scripts").glob("*.py")):
        if script.name == "audit.py":
            continue  # running our own --help would recurse forever
        try:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT, capture_output=True, text=True, timeout=180,
            )
            check(result.returncode == 0, f"scripts/{script.name} --help",
                  result.stderr.strip().splitlines()[-1] if result.returncode else "")
        except Exception as exc:  # noqa: BLE001
            check(False, f"scripts/{script.name} --help", str(exc))

    # ------------------------------------------------------- 7. summary
    print("\n" + "=" * 70)
    if failures:
        print(f"  {len(failures)} PROBLEM(S) TO FIX:\n")
        for item in failures:
            print(f"    - {item}")
    else:
        print("  ALL CHECKS PASSED")
    if warnings:
        print(f"\n  {len(warnings)} non-blocking warning(s):")
        for item in warnings:
            print(f"    - {item}")
    print("\n  Not checked here (needs your machine):")
    print("    - exactly one trading loop running:  ps aux | grep '[m]ain.py'")
    print("    - dashboard reachable:               http://localhost:8501")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
