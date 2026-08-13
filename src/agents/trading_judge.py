"""The LLM trading judge.

Decides direction only. Position size, stops and the daily loss limit are
computed downstream by the risk manager - nothing that can hallucinate is
allowed to size a position.

The prompt contains no dates and no absolute prices. An LLM has read a great
deal of Bitcoin history, so a date or price level invites it to recall what
happened next: look-ahead bias through model weights, invisible to any
pandas-level control. See MarketSnapshot.

Responses are cached on a hash of the prompt, and the judge is consulted only
where a trade would actually open - 127 calls on the test period rather than
13,580.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config.settings import LLM, PATHS, RISK, LLMConfig
from src.agents.schema import JudgeDecision, JudgeRecord, MarketSnapshot
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a disciplined trading judge for a Bitcoin research system.

You are given a rule-based strategy's signal, a machine-learning model's
prediction, and current market conditions expressed as relative values. Your
only job is to decide whether to act on the signal.

Return exactly one decision:
  LONG  - open or hold a long position
  SHORT - open or hold a short position
  HOLD  - stay out; the evidence does not justify risking capital

Principles:
- HOLD is a legitimate and often correct answer. You are not required to trade.
- Disagreement between the strategy and the model is a reason for caution.
- High volatility relative to the target distance makes the trade worse, not
  more exciting.
- You do not control position size, stop placement or leverage. Those are
  fixed by a separate risk manager and are shown to you as constraints.
- Judge only what is in front of you. You have no knowledge of what happens
  next, and no information beyond this snapshot.

Respond with JSON only, matching this schema exactly:
{"decision": "LONG"|"SHORT"|"HOLD",
 "confidence": <integer 0-100>,
 "reason": "<one or two sentences, under 300 characters>",
 "risk_assessment": "LOW"|"MODERATE"|"HIGH"}"""


def prompt_for(snapshot: MarketSnapshot) -> str:
    """The user-side prompt for one decision."""
    return (
        "Evaluate this trading opportunity.\n\n"
        f"{snapshot.to_prompt_block()}\n\n"
        "Respond with JSON only."
    )


def prompt_hash(snapshot: MarketSnapshot, model: str) -> str:
    """Stable cache key.

    Includes the model name so switching models does not silently reuse
    another model's answers.
    """
    payload = json.dumps(snapshot.model_dump(), sort_keys=True) + "|" + model
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class DecisionCache:
    """Disk-backed cache of judge responses, keyed by prompt hash."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (PATHS.data / "cache" / "llm_decisions.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._store = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info("Loaded %d cached judge decisions", len(self._store))
            except json.JSONDecodeError:
                logger.warning("Cache at %s is corrupt; starting empty", self.path)

    def get(self, key: str) -> Optional[JudgeDecision]:
        payload = self._store.get(key)
        return JudgeDecision(**payload) if payload else None

    def put(self, key: str, decision: JudgeDecision) -> None:
        self._store[key] = decision.model_dump()

    def save(self) -> None:
        self.path.write_text(json.dumps(self._store, indent=1), encoding="utf-8")
        logger.info("Saved %d judge decisions to %s", len(self._store), self.path)

    def __len__(self) -> int:
        return len(self._store)


class TradingJudge:
    """LLM judge with caching, validation and bounded concurrency."""

    def __init__(
        self,
        config: LLMConfig = LLM,
        cache: Optional[DecisionCache] = None,
        client: Any = None,
    ) -> None:
        self.config = config
        self.cache = cache if cache is not None else DecisionCache()
        self._client = client
        self.calls_made = 0
        self.cache_hits = 0
        self.failures = 0

    # -- client -----------------------------------------------------------

    @property
    def client(self):
        """Lazily constructed OpenAI client, so tests never need a key."""
        if self._client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to .env before running the judge."
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    # -- single decision ---------------------------------------------------

    def _call_model(self, snapshot: MarketSnapshot) -> JudgeDecision:
        """One API call, with a retry on malformed output."""
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_for(snapshot)},
                    ],
                )
                self.calls_made += 1
                content = response.choices[0].message.content
                return JudgeDecision(**json.loads(content))
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last_error = exc
                logger.warning("Judge call failed (attempt %d): %s", attempt + 1, exc)

        raise RuntimeError(f"Judge failed after {self.config.max_retries + 1} attempts") from last_error

    def decide(self, snapshot: MarketSnapshot) -> Tuple[JudgeDecision, bool]:
        """Return ``(decision, was_cached)`` for one snapshot.

        On failure the judge falls back to HOLD rather than to the strategy's
        signal. Standing aside when the decision layer is broken is the safe
        default, and it keeps a failed API call from being silently recorded
        as agreement.
        """
        key = prompt_hash(snapshot, self.config.model)

        if self.config.cache_enabled:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached, True

        try:
            decision = self._call_model(snapshot)
        except RuntimeError as exc:
            self.failures += 1
            logger.error("Judge unavailable, defaulting to HOLD: %s", exc)
            return (
                JudgeDecision(
                    decision="HOLD",
                    confidence=0,
                    reason="Judge unavailable; standing aside rather than guessing.",
                    risk_assessment="HIGH",
                ),
                False,
            )

        if self.config.cache_enabled:
            self.cache.put(key, decision)
        return decision, False

    # -- batch -------------------------------------------------------------

    def decide_many(
        self,
        snapshots: Sequence[Tuple[str, MarketSnapshot]],
        save_cache: bool = True,
    ) -> List[JudgeRecord]:
        """Judge many decision points concurrently.

        Cached snapshots are resolved first without touching the network, so a
        re-run of an unchanged backtest issues zero API calls.
        """
        if not snapshots:
            return []

        logger.info("Judging %d decision points (%d cached)", len(snapshots), len(self.cache))

        def work(item: Tuple[str, MarketSnapshot]) -> JudgeRecord:
            timestamp, snapshot = item
            decision, cached = self.decide(snapshot)
            return JudgeRecord.from_decision(
                timestamp=timestamp,
                snapshot=snapshot,
                decision=decision,
                model=self.config.model,
                prompt_hash=prompt_hash(snapshot, self.config.model),
                cached=cached,
            )

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as pool:
            records = list(pool.map(work, snapshots))

        if save_cache and self.config.cache_enabled:
            self.cache.save()

        logger.info(
            "Judge complete: %d API calls, %d cache hits, %d failures",
            self.calls_made,
            self.cache_hits,
            self.failures,
        )
        return records


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------

SIGNAL_TO_DECISION = {1: "LONG", -1: "SHORT", 0: "HOLD"}


def build_snapshot(
    row: pd.Series,
    strategy_name: str,
    timeframe: str,
    strategy_signal: int,
    ml_prediction: int,
    ml_confidence: float,
    current_position: int = 0,
) -> MarketSnapshot:
    """Assemble one snapshot from an indicator row.

    Note what is *not* copied across: the timestamp, the close price, the EMA
    levels, the Bollinger band prices. Only relative quantities travel into
    the prompt.
    """
    stop_distance = float(row.get("atr_pct", 0.0)) * RISK.atr_stop_multiple
    return MarketSnapshot(
        timeframe=timeframe,
        strategy_name=strategy_name,
        strategy_signal=SIGNAL_TO_DECISION[int(strategy_signal)],
        rsi=float(row["rsi"]),
        adx=float(row["adx"]),
        macd_histogram_pct=float(row["macd_hist_pct"]),
        bollinger_pct_b=float(min(max(row["bb_pct_b"], -2.0), 3.0)),
        atr_pct=float(row["atr_pct"]),
        distance_from_trend_pct=float(row["dist_from_trend_pct"]),
        distance_from_vwap_pct=float(row["vwap_dist_pct"]),
        volume_ratio=float(row["volume_ratio"]),
        return_last_bar_pct=float(row["ret_1"]),
        return_last_four_bars_pct=float(row["ret_4"]),
        ml_prediction=SIGNAL_TO_DECISION[int(ml_prediction)],
        ml_confidence=float(ml_confidence),
        current_position=SIGNAL_TO_DECISION[int(current_position)],
        account_risk_pct=RISK.risk_per_trade,
        stop_distance_pct=stop_distance,
        target_distance_pct=stop_distance * RISK.reward_risk_ratio,
        reward_risk_ratio=RISK.reward_risk_ratio,
    )


def entry_decision_points(signals: pd.Series) -> pd.DatetimeIndex:
    """Bars where the strategy wants to open a position it does not hold.

    This is what keeps the LLM backtest affordable: on 13,000 bars of 4h data
    the strategy attempts a few hundred entries, so the judge is consulted a
    few hundred times rather than 13,000.
    """
    previous = signals.shift(1).fillna(0).astype("int8")
    is_entry = (signals != 0) & (signals != previous)
    return signals.index[is_entry]
