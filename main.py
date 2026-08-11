"""Live paper-trading loop - Bybit Testnet only.

Runs as its own process, separate from the Streamlit dashboard. The loop
writes to SQLite; the dashboard only reads. Putting the loop inside Streamlit
would restart it on every widget interaction.

    python main.py                      # System C: rules + ML + LLM judge
    python main.py --system A           # rules only
    python main.py --dry-run            # decide and log, place no orders
    python main.py --once               # single cycle, for testing

Safety: this process refuses to start unless TRADING_MODE is a paper mode,
and the only host it can reach with credentials is Bybit's testnet endpoint.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.settings import (
    DATA,
    LIVE,
    ML,
    PATHS,
    TRADING_MODE,
    assert_paper_mode,
)
from src.agents.deterministic_judge import DeterministicJudge
from src.agents.harness import approvals_from_records
from src.agents.trading_judge import TradingJudge, build_snapshot
from src.data.loader import load_ohlcv
from src.database.repository import Repository
from src.exchange.executor import PaperExecutor
from src.exchange.simulated_broker import SimulatedBroker
from src.models.features import build_dataset, build_features
from src.models.predict import predictions_frame
from src.models.train import chronological_splits, comparison_table, train_models
from src.utils.logging_setup import get_logger

logger = get_logger("live")

RUNNING = True
LOCK_FILE = PATHS.root / "data" / "live.lock"


def acquire_lock() -> bool:
    """Refuse to start if another trading loop is already running.

    Two concurrent loops race: both see a flat position, both open one, and
    each then reconciles away the other's trade. The symptom is duplicated
    decisions and paired trades seconds apart - subtle enough to survive
    unnoticed into a presentation.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            existing = int(LOCK_FILE.read_text().strip())
            os.kill(existing, 0)  # raises unless the process is alive
        except (ValueError, ProcessLookupError, PermissionError):
            logger.warning("Removing stale lock file from PID %s", LOCK_FILE.read_text().strip())
        else:
            logger.error(
                "Another trading loop is already running (PID %s). Refusing to "
                "start a second one.\n  Stop it with:  pkill -f 'python main.py'",
                existing,
            )
            return False
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except OSError:
        pass


def handle_shutdown(signum, frame) -> None:
    """Stop after the current cycle. Open positions keep their exchange-side
    stop and target, so shutting down never leaves a naked position."""
    global RUNNING
    logger.info("Shutdown requested; finishing the current cycle.")
    RUNNING = False


def load_strategy(name: str):
    sys.path.insert(0, str(PATHS.root / "scripts"))
    from run_ml import load_tuned

    return load_tuned(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="ema_rsi_trend")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--system", default="C", choices=["A", "B", "C"])
    parser.add_argument("--threshold", type=float, default=ML.confidence_threshold)
    parser.add_argument("--dry-run", action="store_true", help="Log decisions, place no orders.")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    parser.add_argument("--poll", type=int, default=LIVE.poll_seconds)
    parser.add_argument(
        "--broker",
        default="simulated",
        choices=["simulated", "bybit"],
        help="simulated: local fills at real prices. bybit: live testnet API.",
    )
    args = parser.parse_args()

    # Refuse to start outside a paper mode, before anything else happens.
    assert_paper_mode()
    PATHS.ensure()
    if not args.dry_run and not acquire_lock():
        return 1

    print("=" * 78)
    print(f"  TRADING MODE: {TRADING_MODE.upper()}   -   PAPER TRADING ONLY")
    print("  This process cannot place an order with real funds.")
    print("=" * 78)

    strategy, tuned_timeframe = load_strategy(args.strategy)
    timeframe = args.timeframe or tuned_timeframe or LIVE.timeframe

    # ----------------------------------------------------------- prepare
    logger.info("Training the model on all history available up to now...")
    history = load_ohlcv(timeframe, refresh=True)
    features, target, _ = build_dataset(history, strategy.indicator_spec)
    splits = chronological_splits(features.index)
    trained = train_models(features, target, splits)
    model_name = comparison_table(trained).iloc[0]["model"]
    model = trained[model_name]
    logger.info("Using %s for live predictions", model_name)

    judge = None
    if args.system == "C":
        judge = TradingJudge()
    elif args.system == "B":
        judge = DeterministicJudge(args.threshold)

    repository = Repository()
    executor: Optional[PaperExecutor] = None
    if not args.dry_run:
        broker = None
        if args.broker == "simulated":
            broker = SimulatedBroker(repository=repository, timeframe=timeframe)
            logger.info(broker.describe())
        executor = PaperExecutor(
            client=broker,
            repository=repository,
            symbol=DATA.symbol,
            timeframe=timeframe,
            strategy_name=strategy.name,
            system=f"{args.system}_live",
        )

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    last_candle: Optional[pd.Timestamp] = None
    logger.info("Live loop starting | system %s | %s %s", args.system, DATA.symbol, timeframe)

    # -------------------------------------------------------------- loop
    while RUNNING:
        try:
            candles = load_ohlcv(timeframe, refresh=True)
            if candles.empty:
                time.sleep(args.poll)
                continue

            latest = candles.index[-1]
            if last_candle is not None and latest == last_candle and not args.once:
                time.sleep(args.poll)
                continue
            last_candle = latest

            prepared = strategy.run(candles)
            row = prepared.iloc[-1]
            strategy_signal = int(row["signal"])
            price = float(row["close"])
            atr = float(row["atr"])

            # ML prediction for the latest closed candle.
            #
            # Uses build_features, NOT build_dataset. build_dataset drops the
            # trailing rows whose *label* window is incomplete - correct for
            # training, fatal here, because the most recent candle can never
            # have a label and would therefore never receive a prediction.
            # Predicting needs features only; labels are a training concern.
            live_features = build_features(candles, strategy.indicator_spec).dropna()
            ml_prediction, ml_confidence = 0, 0.0
            if latest in live_features.index:
                prediction_row = predictions_frame(model, live_features.loc[[latest]]).iloc[0]
                ml_prediction = int(prediction_row["prediction"])
                ml_confidence = float(prediction_row["confidence"])

            # Decide.
            action = strategy_signal
            judge_decision = None
            if judge is not None and strategy_signal != 0:
                snapshot = build_snapshot(
                    row=row,
                    strategy_name=strategy.name,
                    timeframe=timeframe,
                    strategy_signal=strategy_signal,
                    ml_prediction=ml_prediction,
                    ml_confidence=ml_confidence,
                )
                judge_decision, _ = judge.decide(snapshot)
                action = judge_decision.signal if judge_decision.decision != "HOLD" else 0
                if judge_decision.signal != strategy_signal:
                    action = 0  # the judge gates proposals; it cannot invent them

            logger.info(
                "%s | close %.2f | strategy %+d | ml %+d (%.0f%%) | action %+d%s",
                latest,
                price,
                strategy_signal,
                ml_prediction,
                ml_confidence * 100,
                action,
                f" | {judge_decision.decision} - {judge_decision.reason}" if judge_decision else "",
            )

            decision_log = {
                "timestamp": latest.isoformat(),
                "symbol": DATA.symbol,
                "timeframe": timeframe,
                "strategy": strategy.name,
                "strategy_signal": strategy_signal,
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "judge_decision": judge_decision.decision if judge_decision else None,
                "judge_confidence": judge_decision.confidence if judge_decision else None,
                "judge_reason": judge_decision.reason if judge_decision else None,
                "risk_assessment": judge_decision.risk_assessment if judge_decision else None,
                "indicators": {
                    key: row.get(key)
                    for key in ("rsi", "adx", "atr_pct", "bb_pct_b", "volume_ratio")
                },
                "model": model_name,
            }

            if executor is not None:
                executor.execute(direction=action, price=price, atr=atr, decision_log=decision_log)
                executor.record_equity(price=price)
            else:
                decision_log["final_action"] = action
                decision_log["blocked_reason"] = "dry run - no orders placed"
                repository.record_decision(decision_log)

        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001 - the loop must survive
            logger.exception("Cycle failed, continuing: %s", exc)

        if args.once:
            break
        time.sleep(args.poll)

    release_lock()
    logger.info("Live loop stopped. Open positions retain their exchange-side stop and target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
