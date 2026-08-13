"""Watch the LLM trading judge work.

Three modes, all useful in a presentation.

    python scripts/show_judge.py                 # replay the test period from cache
    python scripts/show_judge.py --live          # judge the market right now
    python scripts/show_judge.py --prompt-only   # show the prompt, call nothing

``--live`` is the demo: it takes the latest closed candle, prints the exact
prompt the model receives, calls it, and prints the validated response. Note
what is *not* in the prompt - no date, no price. That is the leakage control
you can point at on stage.

The replay mode reads from ``data/cache/llm_decisions.json`` and issues zero
API calls, so it works with the network unplugged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd  # noqa: E402

from config.settings import LLM, PATHS  # noqa: E402
from src.agents.harness import agreement_stats, build_snapshots, describe_agreement  # noqa: E402
from src.agents.schema import records_to_frame  # noqa: E402
from src.agents.trading_judge import (  # noqa: E402
    SYSTEM_PROMPT,
    TradingJudge,
    build_snapshot,
    prompt_for,
)
from src.backtesting.runner import TEST, apply_embargo, get_split  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.models.features import build_dataset, build_features  # noqa: E402
from src.models.predict import predictions_frame  # noqa: E402
from src.models.train import chronological_splits, comparison_table, train_models  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402
from run_final import fit_for_test  # noqa: E402
from run_ml import load_tuned  # noqa: E402

logger = get_logger("show_judge")

RULE = "=" * 78
THIN = "-" * 78


def banner(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def show_prompt(snapshot) -> None:
    banner("WHAT THE MODEL SEES")
    print("\n--- system prompt " + "-" * 60)
    print(SYSTEM_PROMPT)
    print("\n--- user prompt " + "-" * 62)
    print(prompt_for(snapshot))
    print(THIN)
    print(
        "Note what is absent: no date, no timestamp, no absolute price.\n"
        "An LLM has read a great deal of Bitcoin history. Told the date or the\n"
        "price level, it could partially recall what happened next - look-ahead\n"
        "bias through model weights, invisible to every pandas-level control.\n"
        "Two tests assert these never appear."
    )


def show_decision(decision, cached: bool) -> None:
    banner("WHAT THE MODEL RETURNS")
    print(f"\n  Decision        {decision.decision}")
    print(f"  Confidence      {decision.confidence}/100")
    print(f"  Risk            {decision.risk_assessment}")
    print(f"  Reasoning       {decision.reason}")
    print(f"\n  Source          {'cache (no API call)' if cached else 'live API call'}")
    print(
        "\n  Validated against a Pydantic schema, not parsed out of prose. The day\n"
        "  it writes \"I would lean long, though...\" a regex parser either crashes\n"
        "  or silently returns the wrong direction."
    )


def prepare(strategy_name: str, refit_for_test: bool = False):
    """Load data, fit the model, and return everything the judge needs.

    ``refit_for_test`` must be True when replaying the test period. The final
    run refits on train + validation, so a model fitted on train alone produces
    different ML predictions, different snapshots, different prompt hashes -
    and therefore a completely fresh set of API calls whose decisions do not
    correspond to the published System C results.
    """
    strategy, timeframe = load_tuned(strategy_name)
    timeframe = timeframe or "4h"
    ohlcv = load_ohlcv(timeframe)
    features, target, _ = build_dataset(ohlcv, strategy.indicator_spec)
    splits = chronological_splits(features.index)
    trained = fit_for_test(features, target, splits) if refit_for_test else train_models(
        features, target, splits
    )
    model = trained[comparison_table(trained).iloc[0]["model"]]
    return strategy, timeframe, ohlcv, model, features


def run_live(args) -> int:
    """Judge the current market state."""
    strategy, timeframe, ohlcv, model, _ = prepare(args.strategy)
    prepared = strategy.run(ohlcv)
    row = prepared.iloc[-1]
    latest = prepared.index[-1]

    live_features = build_features(ohlcv, strategy.indicator_spec).dropna()
    ml_prediction, ml_confidence = 0, 0.0
    if latest in live_features.index:
        prediction = predictions_frame(model, live_features.loc[[latest]]).iloc[0]
        ml_prediction = int(prediction["prediction"])
        ml_confidence = float(prediction["confidence"])

    strategy_signal = int(row["signal"])
    banner("CURRENT MARKET STATE")
    print(f"\n  Candle          {latest}  (this IS shown to you, not to the model)")
    print(f"  Close           {row['close']:,.2f}")
    print(f"  Strategy signal {['FLAT', 'LONG', 'SHORT'][0 if strategy_signal == 0 else (1 if strategy_signal > 0 else 2)]}")
    print(f"  ML prediction   {['HOLD', 'LONG', 'SHORT'][0 if ml_prediction == 0 else (1 if ml_prediction > 0 else 2)]} "
          f"at {ml_confidence:.0%} confidence")

    if strategy_signal == 0:
        print(
            "\n  The strategy has no signal right now, so in normal operation the judge\n"
            "  would NOT be consulted - that is what keeps a 13,580-candle backtest to\n"
            "  127 API calls instead of 13,580.\n"
            "\n  Forcing a hypothetical LONG so you can see the mechanism."
        )
        strategy_signal = 1

    snapshot = build_snapshot(
        row=row,
        strategy_name=strategy.name,
        timeframe=timeframe,
        strategy_signal=strategy_signal,
        ml_prediction=ml_prediction,
        ml_confidence=ml_confidence,
    )
    show_prompt(snapshot)

    if args.prompt_only:
        print("\n  --prompt-only: no API call made.")
        return 0

    judge = TradingJudge()
    decision, cached = judge.decide(snapshot)
    judge.cache.save()
    show_decision(decision, cached)

    print(f"\n{THIN}")
    print(
        f"  The judge gates the strategy's proposal. It cannot invent a trade the\n"
        f"  strategy never suggested, and it never sizes the position - that is the\n"
        f"  deterministic risk manager, downstream."
    )
    return 0


def run_replay(args) -> int:
    """Replay the judged test period from cache. Zero API calls."""
    # Mirror scripts/run_final.py exactly: same model fit, same feature path,
    # same snapshot construction. Any divergence changes the prompt hash and
    # produces decisions that are NOT the ones behind the published results.
    strategy, timeframe, ohlcv, model, features = prepare(args.strategy, refit_for_test=True)

    start, end = get_split(TEST, unlock_test=True)
    test_ohlcv = apply_embargo(load_ohlcv(timeframe, start=start, end=end))
    prepared = strategy.run(test_ohlcv)
    signals = prepared["signal"]

    predictions = predictions_frame(
        model, features.loc[features.index.intersection(prepared.index)]
    )
    snapshots = build_snapshots(prepared, signals, predictions, strategy.name, timeframe)

    judge = TradingJudge()
    records = judge.decide_many(snapshots, save_cache=True)

    banner(f"REPLAY - {len(records)} JUDGED DECISIONS ON THE TEST PERIOD")
    print(
        f"\n  API calls this run : {judge.calls_made}"
        f"{'  (all served from cache)' if judge.calls_made == 0 else ''}"
    )
    print(f"  Cache hits         : {judge.cache_hits}")
    print(f"  Model              : {LLM.model}")
    if judge.calls_made:
        print(
            "\n  Note: new candles have arrived since the final run, so a few decision\n"
            "  points are new and had to be judged. Everything already seen was served\n"
            "  from cache. Counts drift upward over time; the frozen record of the\n"
            "  experiment is data/results/final_test.csv."
        )
    print()
    print("  " + describe_agreement(agreement_stats(records), "LLM judge"))
    print(
        "\n  Neither degenerate: a judge agreeing ~100% is a rubber stamp, one agreeing\n"
        "  ~50% on a near-binary choice is a coin flip. This one deliberated."
    )

    frame = records_to_frame(records)
    if frame.empty:
        print("\n  No decisions to show.")
        return 1

    path = PATHS.results / f"decisions_C_rules_ml_llm_{timeframe}.csv"
    frame.to_csv(path)

    banner("WORKED EXAMPLES")
    approved = frame[frame["decision"] == frame["strategy_signal"]]
    vetoed = frame[frame["decision"] == "HOLD"]

    for title, subset in (("APPROVED", approved), ("VETOED", vetoed)):
        print(f"\n  --- {title} ({len(subset)} of {len(frame)}) " + "-" * (44 - len(title)))
        for timestamp, row in subset.head(args.examples).iterrows():
            print(
                f"\n  {timestamp:%Y-%m-%d %H:%M}   strategy {row['strategy_signal']:<5} "
                f"ml {row['ml_prediction']:<5} ({row['ml_confidence']:.0%})"
                f"  ->  {row['decision']} @ {row['confidence']}"
            )
            print(f"      {row['reason']}")

    banner("DISAGREEMENTS - where the judge overruled the strategy")
    overruled = frame[
        (frame["decision"] == "HOLD") & (frame["ml_prediction"] == frame["strategy_signal"])
    ]
    print(
        f"\n  {len(overruled)} decisions where the strategy AND the model agreed, and the\n"
        f"  judge still stood aside. These are the ones worth reading - they are the\n"
        f"  only place the LLM added anything the deterministic rule would not have.\n"
    )
    for timestamp, row in overruled.head(args.examples).iterrows():
        print(f"  {timestamp:%Y-%m-%d %H:%M}  both said {row['strategy_signal']}, judge said HOLD @ {row['confidence']}")
        print(f"      {row['reason']}\n")

    print(THIN)
    print(f"  Saved -> {path}")
    print("  Also visible in the dashboard under the 'AI Judge' tab.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="ema_rsi_trend")
    parser.add_argument("--live", action="store_true", help="Judge the market right now.")
    parser.add_argument("--prompt-only", action="store_true", help="Show the prompt, call nothing.")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    PATHS.ensure()
    if args.live or args.prompt_only:
        return run_live(args)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
