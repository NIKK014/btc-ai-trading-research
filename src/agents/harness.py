"""Comparison harness for Systems A, B and C.

Every arm is routed through the same code path: identical strategy signals,
identical snapshots, identical entry-gating. The only thing that varies is the
object making the decision. That is what makes the comparison an experiment
rather than three separately-tuned demos.

Arms
----
``A``            Rules only, no judge.
``B``            Rules gated by the ML model's agreement.
``C``            Rules gated by the LLM judge.
``deterministic``  The control: four lines of arithmetic, same inputs as C.
``always_agree``   A sanity arm that must reproduce A exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.agents.schema import JudgeRecord, MarketSnapshot
from src.agents.trading_judge import build_snapshot, entry_decision_points
from src.models.predict import gate_entries
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_snapshots(
    prepared: pd.DataFrame,
    signals: pd.Series,
    predictions: pd.DataFrame,
    strategy_name: str,
    timeframe: str,
) -> List[Tuple[str, MarketSnapshot]]:
    """One snapshot per entry decision point.

    Bars where the model has no prediction (feature warm-up) are skipped
    rather than defaulted, so no arm is fed a fabricated ML opinion.
    """
    points = entry_decision_points(signals)
    snapshots: List[Tuple[str, MarketSnapshot]] = []

    for timestamp in points:
        if timestamp not in prepared.index or timestamp not in predictions.index:
            continue
        row = prepared.loc[timestamp]
        prediction_row = predictions.loc[timestamp]
        try:
            snapshot = build_snapshot(
                row=row,
                strategy_name=strategy_name,
                timeframe=timeframe,
                strategy_signal=int(signals.loc[timestamp]),
                ml_prediction=int(prediction_row["prediction"]),
                ml_confidence=float(prediction_row["confidence"]),
            )
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping snapshot at %s: %s", timestamp, exc)
            continue
        snapshots.append((timestamp.isoformat(), snapshot))

    logger.info(
        "Built %d snapshots from %d entry points (%d bars)",
        len(snapshots),
        len(points),
        len(signals),
    )
    return snapshots


def approvals_from_records(
    records: Sequence[JudgeRecord],
    index: pd.Index,
) -> pd.Series:
    """Boolean approval mask from a decision log.

    An entry is approved only when the judge's decision matches the direction
    the strategy asked about. HOLD, or a decision in the opposite direction,
    both mean "do not open this trade" - the judge is a gate on the strategy's
    proposal, not an independent signal generator. Letting it invent trades
    the strategy never proposed would break the like-for-like comparison.
    """
    approved: Dict[pd.Timestamp, bool] = {}
    for record in records:
        timestamp = pd.Timestamp(record.timestamp)
        approved[timestamp] = (
            record.decision == record.strategy_signal and record.decision != "HOLD"
        )
    mask = pd.Series(approved, dtype="boolean").reindex(index)
    return mask.fillna(False).astype(bool)


def run_arm(
    judge: Any,
    prepared: pd.DataFrame,
    signals: pd.Series,
    predictions: pd.DataFrame,
    strategy_name: str,
    timeframe: str,
    snapshots: Optional[List[Tuple[str, MarketSnapshot]]] = None,
) -> Tuple[pd.Series, List[JudgeRecord]]:
    """Run one judge over the strategy's entry points.

    Returns:
        ``(gated_signals, decision_records)``.
    """
    snapshots = snapshots if snapshots is not None else build_snapshots(
        prepared, signals, predictions, strategy_name, timeframe
    )
    records = judge.decide_many(snapshots)
    approved = approvals_from_records(records, signals.index)
    return gate_entries(signals, approved), records


def agreement_stats(records: Sequence[JudgeRecord]) -> Dict[str, float]:
    """How often the judge simply agreed with what it was shown.

    The single most diagnostic number for System C. A judge agreeing ~100% of
    the time is a rubber stamp adding cost and latency for nothing; one
    agreeing ~50% of the time on a binary-ish choice is closer to a coin flip
    than to judgement. Either reading is more informative than the P&L.
    """
    if not records:
        return {}

    total = len(records)
    agreed = sum(1 for r in records if r.decision == r.strategy_signal)
    held = sum(1 for r in records if r.decision == "HOLD")
    matched_ml = sum(1 for r in records if r.decision == r.ml_prediction)
    overruled_ml = sum(
        1
        for r in records
        if r.decision == r.strategy_signal and r.ml_prediction != r.strategy_signal
    )

    return {
        "decisions": total,
        "agreed_with_strategy": agreed / total,
        "chose_hold": held / total,
        "matched_ml": matched_ml / total,
        "backed_strategy_over_ml": overruled_ml / total,
        "mean_confidence": sum(r.confidence for r in records) / total,
        "cached": sum(1 for r in records if r.cached) / total,
    }


def describe_agreement(stats: Dict[str, float], label: str) -> str:
    """Readable agreement summary with an interpretation warning."""
    if not stats:
        return f"{label}: no decisions"

    line = (
        f"{label:<22} {int(stats['decisions']):>4} decisions | "
        f"agreed {stats['agreed_with_strategy']:>5.1%} | "
        f"HOLD {stats['chose_hold']:>5.1%} | "
        f"matched ML {stats['matched_ml']:>5.1%} | "
        f"mean confidence {stats['mean_confidence']:>4.0f}"
    )
    if stats["agreed_with_strategy"] > 0.95:
        line += "\n  -> effectively a rubber stamp: it approves almost everything"
    elif stats["chose_hold"] > 0.95:
        line += "\n  -> effectively always flat: it approves almost nothing"
    return line
