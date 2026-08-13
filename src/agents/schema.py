"""Structured input and output for the trading judge.

Output is validated against a Pydantic schema rather than parsed from prose.
The day the model writes "I would lean long, though..." a regex parser either
crashes or silently returns the wrong direction.

MarketSnapshot deliberately carries no timestamp and no absolute price - see
the note on its class.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Decision = Literal["LONG", "SHORT", "HOLD"]
DECISION_TO_SIGNAL: Dict[str, int] = {"LONG": 1, "SHORT": -1, "HOLD": 0}


class MarketSnapshot(BaseModel):
    """What the judge is allowed to know at a decision point.

    Deliberately contains **no timestamp and no absolute price**. An LLM has
    memorised a great deal of Bitcoin price history, so telling it the date, or
    that BTC is at $118,320, invites it to recall what happened next. That is
    look-ahead bias travelling through the model's weights rather than through
    the dataframe, and no amount of careful pandas prevents it.

    Every field below is scale-free or relative, so a snapshot from 2021 and
    one from 2026 are indistinguishable as to *when* they are.
    """

    timeframe: str
    strategy_name: str
    strategy_signal: Decision

    rsi: float = Field(ge=0, le=100)
    adx: float = Field(ge=0, le=100)
    macd_histogram_pct: float = Field(description="MACD histogram as a fraction of price")
    bollinger_pct_b: float = Field(description="0 at the lower band, 1 at the upper")
    atr_pct: float = Field(description="ATR as a fraction of price")
    distance_from_trend_pct: float
    distance_from_vwap_pct: float
    volume_ratio: float = Field(description="Volume relative to its recent average")
    return_last_bar_pct: float
    return_last_four_bars_pct: float

    ml_prediction: Decision
    ml_confidence: float = Field(ge=0, le=1)

    current_position: Decision = "HOLD"
    account_risk_pct: float
    stop_distance_pct: float
    target_distance_pct: float
    reward_risk_ratio: float

    def to_prompt_block(self) -> str:
        """Render as the compact, labelled block the judge receives."""
        return "\n".join(
            [
                f"Timeframe: {self.timeframe}",
                f"Strategy: {self.strategy_name}",
                f"Strategy signal: {self.strategy_signal}",
                "",
                "Indicators (all relative, no absolute prices):",
                f"  RSI: {self.rsi:.1f}",
                f"  ADX (trend strength): {self.adx:.1f}",
                f"  MACD histogram: {self.macd_histogram_pct * 100:+.3f}% of price",
                f"  Bollinger %B: {self.bollinger_pct_b:.2f}",
                f"  ATR (volatility): {self.atr_pct * 100:.2f}% of price",
                f"  Distance from trend EMA: {self.distance_from_trend_pct * 100:+.2f}%",
                f"  Distance from session VWAP: {self.distance_from_vwap_pct * 100:+.2f}%",
                f"  Volume vs average: {self.volume_ratio:.2f}x",
                f"  Last bar return: {self.return_last_bar_pct * 100:+.2f}%",
                f"  Last 4 bars return: {self.return_last_four_bars_pct * 100:+.2f}%",
                "",
                "Machine-learning model:",
                f"  Prediction: {self.ml_prediction}",
                f"  Confidence: {self.ml_confidence:.0%}",
                "",
                "Risk framework (fixed, not yours to change):",
                f"  Current position: {self.current_position}",
                f"  Account risk per trade: {self.account_risk_pct:.1%}",
                f"  Stop distance: {self.stop_distance_pct * 100:.2f}% from entry",
                f"  Target distance: {self.target_distance_pct * 100:.2f}% from entry",
                f"  Reward:risk: {self.reward_risk_ratio:.1f}:1",
            ]
        )


class JudgeDecision(BaseModel):
    """The judge's validated response."""

    decision: Decision
    confidence: int = Field(ge=0, le=100)
    reason: str = Field(max_length=400)
    risk_assessment: Literal["LOW", "MODERATE", "HIGH"]

    @field_validator("reason")
    @classmethod
    def reason_must_say_something(cls, value: str) -> str:
        if len(value.strip()) < 10:
            raise ValueError("reason must be a real explanation")
        return value.strip()

    @property
    def signal(self) -> int:
        """The decision as an engine-compatible direction."""
        return DECISION_TO_SIGNAL[self.decision]


class JudgeRecord(BaseModel):
    """One fully logged decision, for the audit trail and the dashboard.

    Everything needed to answer "why did it take that trade?" months later,
    and to replay the entire backtest without calling the API again.
    """

    timestamp: str
    timeframe: str
    strategy_name: str
    strategy_signal: Decision
    ml_prediction: Decision
    ml_confidence: float
    decision: Decision
    confidence: int
    reason: str
    risk_assessment: str
    model: str
    cached: bool = False
    prompt_hash: str = ""
    error: Optional[str] = None

    @classmethod
    def from_decision(
        cls,
        timestamp: str,
        snapshot: MarketSnapshot,
        decision: JudgeDecision,
        model: str,
        prompt_hash: str,
        cached: bool = False,
    ) -> "JudgeRecord":
        return cls(
            timestamp=timestamp,
            timeframe=snapshot.timeframe,
            strategy_name=snapshot.strategy_name,
            strategy_signal=snapshot.strategy_signal,
            ml_prediction=snapshot.ml_prediction,
            ml_confidence=snapshot.ml_confidence,
            decision=decision.decision,
            confidence=decision.confidence,
            reason=decision.reason,
            risk_assessment=decision.risk_assessment,
            model=model,
            cached=cached,
            prompt_hash=prompt_hash,
        )


def records_to_frame(records: List[JudgeRecord]):
    """Decision log as a DataFrame, indexed by timestamp."""
    import pandas as pd

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([record.model_dump() for record in records])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()
