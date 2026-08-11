"""Judge tests.

All offline. The LLM client is injected, so the full System C path -
snapshot construction, schema validation, caching, entry gating - is exercised
without an API key or a network call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from config.settings import LLMConfig
from src.agents.deterministic_judge import AlwaysAgreeJudge, DeterministicJudge
from src.agents.harness import (
    agreement_stats,
    approvals_from_records,
    build_snapshots,
    run_arm,
)
from src.agents.schema import JudgeDecision, MarketSnapshot
from src.agents.trading_judge import (
    DecisionCache,
    TradingJudge,
    build_snapshot,
    entry_decision_points,
    prompt_for,
    prompt_hash,
)


def snapshot(**overrides) -> MarketSnapshot:
    defaults = dict(
        timeframe="4h",
        strategy_name="ema_rsi_trend",
        strategy_signal="LONG",
        rsi=58.0,
        adx=27.0,
        macd_histogram_pct=0.002,
        bollinger_pct_b=0.72,
        atr_pct=0.021,
        distance_from_trend_pct=0.035,
        distance_from_vwap_pct=0.004,
        volume_ratio=1.3,
        return_last_bar_pct=0.006,
        return_last_four_bars_pct=0.018,
        ml_prediction="LONG",
        ml_confidence=0.62,
        current_position="HOLD",
        account_risk_pct=0.01,
        stop_distance_pct=0.042,
        target_distance_pct=0.084,
        reward_risk_ratio=2.0,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


class FakeClient:
    """Minimal stand-in for the OpenAI client."""

    def __init__(self, payload=None, fail_times=0):
        self.payload = payload or {
            "decision": "LONG",
            "confidence": 78,
            "reason": "Trend and model agree with room to the target.",
            "risk_assessment": "MODERATE",
        }
        self.fail_times = fail_times
        self.calls = 0
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["messages"][1]["content"])
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated API error")
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


# ---------------------------------------------------------------------------
# The leakage control
# ---------------------------------------------------------------------------


def test_prompt_contains_no_dates_or_absolute_prices():
    """The subtlest leak in the project: an LLM recognising the date.

    A model that has memorised BTC history can partially recall what came next
    if told the date or the price level. Every value it sees must be relative.
    """
    text = prompt_for(snapshot())

    for year in ("2020", "2021", "2022", "2023", "2024", "2025", "2026"):
        assert year not in text
    for token in ("$", "USD", "USDT", "price:", "close:"):
        assert token not in text
    for month in ("January", "March", "December", "Jan ", "Dec "):
        assert month not in text

    assert "RSI" in text and "%" in text, "relative values must still be present"


def test_snapshot_schema_has_no_price_or_time_fields():
    fields = set(MarketSnapshot.model_fields)
    forbidden = {"timestamp", "date", "close", "price", "open", "high", "low", "entry_price"}
    assert not (fields & forbidden)


def test_snapshot_built_from_a_row_carries_no_price(ohlcv):
    from src.indicators.indicators import add_indicators

    prepared = add_indicators(ohlcv, dropna=True)
    built = build_snapshot(
        prepared.iloc[-1], "ema_rsi_trend", "4h", 1, 1, 0.6
    )
    rendered = built.to_prompt_block()
    close = f"{prepared['close'].iloc[-1]:.0f}"
    assert close not in rendered


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def test_valid_response_is_parsed_and_converted_to_a_signal():
    judge = TradingJudge(config=LLMConfig(cache_enabled=False), client=FakeClient())
    decision, cached = judge.decide(snapshot())

    assert decision.decision == "LONG"
    assert decision.signal == 1
    assert not cached


def test_malformed_output_is_retried_then_falls_back_to_hold():
    """A broken judge must stand aside, never guess."""
    client = FakeClient(payload="not json at all")
    judge = TradingJudge(config=LLMConfig(cache_enabled=False, max_retries=1), client=client)
    decision, _ = judge.decide(snapshot())

    assert client.calls == 2, "should retry once"
    assert decision.decision == "HOLD"
    assert judge.failures == 1


def test_transient_failure_is_retried_and_then_succeeds():
    client = FakeClient(fail_times=1)
    judge = TradingJudge(config=LLMConfig(cache_enabled=False, max_retries=2), client=client)
    decision, _ = judge.decide(snapshot())

    assert decision.decision == "LONG"
    assert judge.failures == 0


def test_out_of_range_confidence_is_rejected():
    with pytest.raises(ValidationError):
        JudgeDecision(
            decision="LONG", confidence=150, reason="over the limit", risk_assessment="LOW"
        )


def test_invalid_decision_value_is_rejected():
    with pytest.raises(ValidationError):
        JudgeDecision(
            decision="MAYBE", confidence=50, reason="not a valid action", risk_assessment="LOW"
        )


def test_empty_reasoning_is_rejected():
    with pytest.raises(ValidationError):
        JudgeDecision(decision="HOLD", confidence=50, reason="no", risk_assessment="LOW")


# ---------------------------------------------------------------------------
# Caching - what makes the backtest affordable and reproducible
# ---------------------------------------------------------------------------


def test_identical_snapshots_hit_the_cache_and_skip_the_api(tmp_path):
    cache = DecisionCache(tmp_path / "cache.json")
    client = FakeClient()
    judge = TradingJudge(config=LLMConfig(), cache=cache, client=client)

    judge.decide(snapshot())
    judge.decide(snapshot())

    assert client.calls == 1, "second identical call must be served from cache"
    assert judge.cache_hits == 1


def test_cache_survives_a_restart(tmp_path):
    """A re-run of an unchanged backtest must cost nothing."""
    path = tmp_path / "cache.json"
    client = FakeClient()
    first = TradingJudge(config=LLMConfig(), cache=DecisionCache(path), client=client)
    first.decide(snapshot())
    first.cache.save()

    second = TradingJudge(config=LLMConfig(), cache=DecisionCache(path), client=FakeClient())
    decision, cached = second.decide(snapshot())

    assert cached
    assert decision.decision == "LONG"


def test_cache_key_distinguishes_different_market_states():
    a = prompt_hash(snapshot(rsi=30.0), "m")
    b = prompt_hash(snapshot(rsi=70.0), "m")
    assert a != b


def test_cache_key_distinguishes_models():
    assert prompt_hash(snapshot(), "model-a") != prompt_hash(snapshot(), "model-b")


def test_corrupt_cache_file_does_not_crash(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert len(DecisionCache(path)) == 0


# ---------------------------------------------------------------------------
# Cost control
# ---------------------------------------------------------------------------


def test_judge_is_only_consulted_at_entry_points():
    """Calling on every bar would be thousands of pointless requests."""
    signals = pd.Series(
        [0, 1, 1, 1, 0, 0, -1, -1, 0, 1],
        index=pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC"),
        dtype="int8",
    )
    points = entry_decision_points(signals)

    assert len(points) == 3, "three position openings in ten bars"
    assert list(points) == [signals.index[1], signals.index[6], signals.index[9]]


# ---------------------------------------------------------------------------
# Control arms
# ---------------------------------------------------------------------------


def test_deterministic_judge_approves_only_on_agreement():
    judge = DeterministicJudge(confidence_threshold=0.5)

    agree, _ = judge.decide(snapshot(ml_prediction="LONG", ml_confidence=0.7))
    assert agree.decision == "LONG"

    disagree, _ = judge.decide(snapshot(ml_prediction="SHORT", ml_confidence=0.7))
    assert disagree.decision == "HOLD"

    unsure, _ = judge.decide(snapshot(ml_prediction="LONG", ml_confidence=0.2))
    assert unsure.decision == "HOLD"


def test_always_agree_arm_reproduces_the_strategy_exactly(ohlcv):
    """A sanity check on the harness itself.

    If routing System A's signals through the judge machinery changes them,
    the comparison apparatus is biased and every System C result is suspect.
    """
    from src.indicators.indicators import add_indicators
    from src.models.predict import gate_entries
    from src.strategies.base import build

    strategy = build("ema_rsi_trend")
    prepared = strategy.run(ohlcv)
    signals = prepared["signal"]

    predictions = pd.DataFrame(
        {
            "prediction": signals,
            "prob_long": 0.6,
            "prob_short": 0.6,
            "prob_hold": 0.2,
            "confidence": 0.6,
        },
        index=prepared.index,
    )

    gated, records = run_arm(
        AlwaysAgreeJudge(), prepared, signals, predictions, "ema_rsi_trend", "1h"
    )
    expected = gate_entries(signals, pd.Series(True, index=signals.index))
    pd.testing.assert_series_equal(gated, expected, check_names=False)
    assert len(records) == len(entry_decision_points(signals))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def test_a_judge_cannot_invent_trades_the_strategy_never_proposed():
    """The judge is a gate on proposals, not an independent signal source."""
    index = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    signals = pd.Series([0, 1, 0, 0], index=index, dtype="int8")

    from src.agents.schema import JudgeRecord

    records = [
        JudgeRecord(
            timestamp=index[1].isoformat(),
            timeframe="4h",
            strategy_name="s",
            strategy_signal="LONG",
            ml_prediction="LONG",
            ml_confidence=0.6,
            decision="SHORT",  # judge wants the opposite side
            confidence=80,
            reason="Disagrees with the proposal entirely.",
            risk_assessment="HIGH",
            model="test",
        )
    ]
    approved = approvals_from_records(records, index)
    assert not approved.iloc[1], "opposite-direction decision must not open a trade"


def test_agreement_stats_flag_a_rubber_stamp():
    from src.agents.schema import JudgeRecord

    records = [
        JudgeRecord(
            timestamp=f"2024-01-0{i + 1}T00:00:00+00:00",
            timeframe="4h",
            strategy_name="s",
            strategy_signal="LONG",
            ml_prediction="LONG",
            ml_confidence=0.6,
            decision="LONG",
            confidence=90,
            reason="Approved without much thought.",
            risk_assessment="LOW",
            model="test",
        )
        for i in range(9)
    ]
    stats = agreement_stats(records)
    assert stats["agreed_with_strategy"] == 1.0
    from src.agents.harness import describe_agreement

    assert "rubber stamp" in describe_agreement(stats, "llm")


def test_build_snapshots_skips_bars_without_predictions(ohlcv):
    from src.indicators.indicators import add_indicators
    from src.strategies.base import build

    strategy = build("ema_rsi_trend")
    prepared = strategy.run(ohlcv)
    signals = prepared["signal"]

    half = prepared.index[: len(prepared) // 2]
    predictions = pd.DataFrame(
        {"prediction": 1, "prob_long": 0.6, "prob_short": 0.2, "prob_hold": 0.2, "confidence": 0.6},
        index=half,
    )

    snapshots = build_snapshots(prepared, signals, predictions, "ema_rsi_trend", "1h")
    assert all(pd.Timestamp(ts) in half for ts, _ in snapshots)
