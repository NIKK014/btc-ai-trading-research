"""The control arm.

Why this exists
---------------
A judge that vetoes trades will change performance whether or not it
understands anything. Filtering alone reshapes the return distribution, cuts
the sample size and moves every metric. So "System C beat System B" is not
evidence that the LLM reasoned well - it may only be evidence that fewer
trades were taken.

The honest comparison is against a judge that is transparently *not*
intelligent: four lines of arithmetic, given exactly the same inputs. If the
LLM cannot beat this, the finding is that a language model added nothing a
junior developer could not have written in a minute - which is a real,
publishable result, and far more interesting than a vague claim of
improvement.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from config.settings import ML
from src.agents.schema import DECISION_TO_SIGNAL, JudgeDecision, JudgeRecord, MarketSnapshot


class DeterministicJudge:
    """Trade only when the strategy and the model agree, above a threshold.

    Exposes the same interface as :class:`~src.agents.trading_judge.TradingJudge`
    so the two can be swapped in the comparison harness without any other code
    changing - which is what makes the comparison fair.
    """

    name = "deterministic_judge"

    def __init__(self, confidence_threshold: float = ML.confidence_threshold) -> None:
        self.confidence_threshold = confidence_threshold
        self.calls_made = 0
        self.cache_hits = 0
        self.failures = 0

    def decide(self, snapshot: MarketSnapshot) -> Tuple[JudgeDecision, bool]:
        """The entire decision rule."""
        self.calls_made += 1
        agrees = snapshot.ml_prediction == snapshot.strategy_signal
        confident = snapshot.ml_confidence >= self.confidence_threshold

        if agrees and confident and snapshot.strategy_signal != "HOLD":
            return (
                JudgeDecision(
                    decision=snapshot.strategy_signal,
                    confidence=int(round(snapshot.ml_confidence * 100)),
                    reason="Strategy and model agree above the confidence threshold.",
                    risk_assessment="MODERATE",
                ),
                False,
            )

        reason = (
            "Strategy and model disagree; standing aside."
            if not agrees
            else "Model confidence below threshold; standing aside."
        )
        return (
            JudgeDecision(
                decision="HOLD",
                confidence=int(round((1.0 - snapshot.ml_confidence) * 100)),
                reason=reason,
                risk_assessment="MODERATE",
            ),
            False,
        )

    def decide_many(
        self,
        snapshots: Sequence[Tuple[str, MarketSnapshot]],
        save_cache: bool = False,
    ) -> List[JudgeRecord]:
        records = []
        for timestamp, snapshot in snapshots:
            decision, cached = self.decide(snapshot)
            records.append(
                JudgeRecord.from_decision(
                    timestamp=timestamp,
                    snapshot=snapshot,
                    decision=decision,
                    model=self.name,
                    prompt_hash="",
                    cached=cached,
                )
            )
        return records


class AlwaysAgreeJudge:
    """A second control: approve every signal unconditionally.

    Reproduces System A exactly when routed through the judge harness, which
    proves the harness itself adds no bias. If this arm does not match System
    A's numbers, the comparison machinery is broken and every other result is
    suspect.
    """

    name = "always_agree"

    def __init__(self) -> None:
        self.calls_made = 0
        self.cache_hits = 0
        self.failures = 0

    def decide(self, snapshot: MarketSnapshot) -> Tuple[JudgeDecision, bool]:
        self.calls_made += 1
        return (
            JudgeDecision(
                decision=snapshot.strategy_signal,
                confidence=100,
                reason="Control arm: approves every signal without judgement.",
                risk_assessment="MODERATE",
            ),
            False,
        )

    def decide_many(
        self,
        snapshots: Sequence[Tuple[str, MarketSnapshot]],
        save_cache: bool = False,
    ) -> List[JudgeRecord]:
        return [
            JudgeRecord.from_decision(
                timestamp=timestamp,
                snapshot=snapshot,
                decision=self.decide(snapshot)[0],
                model=self.name,
                prompt_hash="",
            )
            for timestamp, snapshot in snapshots
        ]


def decisions_to_signals(records: Sequence[JudgeRecord], index) -> "pd.Series":
    """Convert a decision log into an engine-compatible signal series.

    Decisions are made at entry points only, so an approved decision persists
    until the strategy itself stands down - the same "gate the entry, then let
    it run" semantics used by the ML filter.
    """
    import pandas as pd

    approved = {
        pd.Timestamp(record.timestamp): DECISION_TO_SIGNAL[record.decision]
        for record in records
    }
    return pd.Series(approved).reindex(index).fillna(0).astype("int8")
