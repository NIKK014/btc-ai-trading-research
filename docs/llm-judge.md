# The LLM trading judge

## What it does

The judge receives a structured snapshot of one trading opportunity and returns
LONG, SHORT or HOLD, with a confidence score and written reasoning.

**It decides direction. It never decides exposure.** Position size, stop
placement, reward:risk and the daily loss limit are computed by a deterministic
risk manager downstream. An LLM that could size its own positions would be one
hallucination away from a catastrophic trade, and no amount of prompt
engineering makes that acceptable.

---

## The leakage problem nobody expects

An LLM has read a great deal of Bitcoin price history and commentary. Tell it
that BTC is at **$118,320 on 3 March** and it may partially recall what
happened next.

That is look-ahead bias travelling **through the model's weights** rather than
through a dataframe — invisible to every pandas-level control in the project.

### The control

Every value the judge sees is relative or scale-free. The schema has no
timestamp field and no price field.

```
Timeframe: 4h
Strategy: ema_rsi_trend
Strategy signal: LONG

Indicators (all relative, no absolute prices):
  RSI: 58.0
  ADX (trend strength): 27.0
  MACD histogram: +0.200% of price
  Bollinger %B: 0.72
  ATR (volatility): 2.10% of price
  Distance from trend EMA: +3.50%
  Distance from session VWAP: +0.40%
  Volume vs average: 1.30x
  Last bar return: +0.60%
  Last 4 bars return: +1.80%

Machine-learning model:
  Prediction: LONG
  Confidence: 62%

Risk framework (fixed, not yours to change):
  Current position: HOLD
  Account risk per trade: 1.0%
  Stop distance: 4.20% from entry
  Target distance: 8.40% from entry
  Reward:risk: 2.0:1
```

Two tests enforce this. `test_prompt_contains_no_dates_or_absolute_prices`
asserts no year, month, currency symbol or price token ever appears in the
rendered prompt. `test_snapshot_schema_has_no_price_or_time_fields` asserts the
Pydantic model cannot carry them in the first place.

---

## Structured output

The response is validated against a Pydantic schema, not parsed out of prose:

```json
{"decision": "LONG", "confidence": 78,
 "reason": "Trend and model agree with room to the target.",
 "risk_assessment": "MODERATE"}
```

`decision` is constrained to three literals, `confidence` to 0–100,
`risk_assessment` to three levels, and `reason` must be a real explanation.

Regex-parsing an LLM's prose is how a demo becomes a liability: the day it
writes *"I would lean long, though..."* the parser either crashes or silently
returns the wrong direction.

**On failure the judge falls back to HOLD**, never to the strategy's signal. A
broken decision layer should stand aside, and a failed call must not be quietly
recorded as agreement.

---

## Making the backtest affordable

Naively, judging 13,580 candles means 13,580 API calls, most of them asking
"there is no signal, what should we do?"

**The judge is consulted only at entry decision points** — bars where the
strategy wants to open a position it does not hold. On the test period that is
**127 calls, not 13,580**.

| | |
|---|---|
| Calls on the test period | 127 |
| Wall-clock time | ~60 seconds (10 concurrent) |
| Failures | 0 |
| Cost | cents |

**Every response is cached on a SHA-256 hash of the prompt payload**, including
the model name so switching models never reuses another model's answers. The
cache is plain JSON in `data/cache/llm_decisions.json`, so it can be inspected
and diffed.

Consequences: re-running an unchanged backtest issues **zero** API calls and
returns identical results, and a live demo cannot be broken by an API outage.
`temperature=0` for reproducibility.

---

## The control arms

A judge that vetoes trades will change performance whether or not it
understands anything. Filtering alone reshapes the return distribution and
shrinks the sample. So "System C beat System A" is not evidence of reasoning.

**The deterministic judge** — four lines of arithmetic, given identical inputs:

```python
if strategy_signal == ml_prediction and ml_confidence >= threshold:
    return strategy_signal
return "HOLD"
```

If the LLM cannot beat this, the finding is that a language model added nothing
a junior developer could not have written in a minute. That is a real,
publishable result.

**The always-agree judge** approves everything unconditionally and must
reproduce System A exactly. It is a check on the *harness*: if routing signals
through the comparison machinery changes them, every System C result is
suspect.

A cross-check verifies that System B and the deterministic judge produce
identical results — the same rule reached by two independent code paths.

---

## Results

| System | Trades | Return | Sharpe | Max DD | Exposure |
|---|---|---|---|---|---|
| A — rules only | 129 | −10.5% | −0.91 | 14.5% | 45.9% |
| B — + ML (= deterministic judge) | 64 | **+1.6%** | 0.22 | 11.2% | 22.9% |
| **C — + LLM judge** | 52 | −0.3% | −0.01 | **7.7%** | 19.0% |

Difference from System A: **+9.7%, 95% CI [−18.1%, +34.8%]**, P(better) 78.4%.
Not distinguishable — and statistically indistinguishable from the ML filter's
own effect.

### Judge behaviour — the most diagnostic number

| Judge | Decisions | Agreed | Chose HOLD | Matched ML | Mean confidence |
|---|---|---|---|---|---|
| Always agree | 127 | 100.0% | 0.0% | 60.6% | 100 |
| **LLM** | 127 | **40.9%** | **59.1%** | 64.6% | 66 |

This is the interesting middle. A judge agreeing ~100% of the time is a rubber
stamp adding cost and latency for nothing; one agreeing ~50% on a binary-ish
choice is closer to a coin flip than to judgement.

**The LLM vetoed nearly 60% of proposals at moderate confidence. It genuinely
deliberated — and still did not beat four lines of arithmetic.**

---

## Verdict

**The LLM judge did not improve trading performance.**

System C returned −0.3% against System B's +1.6%, and System B *is* the
deterministic rule. The LLM matched what a simple condition already achieved,
at the cost of an API dependency, latency and non-determinism.

It did produce the lowest drawdown of any arm (7.7%, one seventh of
buy-and-hold's) — but so did the mechanical filter, in the same direction, and
the intervals overlap heavily.

**What it added that the rule did not:** a written justification for every
decision, logged and inspectable months later. For an auditable system that has
value. It is not the value the experiment set out to measure.

---

## What would test this better

- **More decision points.** 127 calls over 13 months cannot resolve effects of
  this size.
- **Richer context.** The judge sees only technical indicators. Funding rates,
  open interest or sentiment might give it something the rule does not have —
  though each adds a new leakage surface.
- **Prompt ablation.** Removing the ML prediction, or the risk framing, would
  isolate which input the judge is actually using.
- **Multiple models.** Answers "which vendor", not "does reasoning help" — a
  procurement question rather than a research one, and deliberately out of
  scope here.
