# Strategies

Seven implementations across four methodologies, plus a benchmark. The
experiment compares **methodologies**, not individual indicators — the same RSI
confirms a trend in one strategy and signals a reversal in another, and only
the surrounding logic distinguishes them.

| Strategy | Methodology | Indicators |
|---|---|---|
| `ema_rsi_trend` | Trend | EMA fast/slow, RSI, ADX gate |
| `macd_adx_trend` | Trend | EMA200 regime, MACD histogram, ADX gate |
| `stochrsi_macd_momentum` | Momentum | Stochastic RSI, MACD, volume |
| `rsi_bollinger_reversion` | Mean reversion | Bollinger Bands, RSI, ADX ceiling |
| `vwap_stretch_reversion` | Mean reversion | Session VWAP, ATR, RSI, volume |
| `donchian_volume_breakout` | Breakout | Donchian channels, volume |
| `bollinger_squeeze_breakout` | Breakout | Bollinger bandwidth, Donchian exit |
| `buy_and_hold` | Benchmark | — |

---

## The signal contract

A strategy converts an indicator-enriched frame into a **desired position
direction** per bar: `+1` long, `−1` short, `0` flat.

Emitting a target *state* rather than buy/sell *events* keeps the strategy
layer stateless and pushes execution concerns — when the order fills, whether a
stop already closed the position, whether re-entry is allowed — into the
engine, where they belong. The same signal series is then replayed identically
by the backtester, the ML filter and the live trader, which is what makes
Systems A, B and C comparable.

Signals at bar `t` may only reference indicator values at `t` or earlier. The
engine fills at the **open of bar `t+1`**.

---

## Trend

Premise: markets that are moving tend to keep moving. Low win rate by
construction, dependent on winners being much larger than losers.

### `ema_rsi_trend` — the study winner

Long while the fast EMA is above the slow EMA and RSI confirms momentum on the
same side of its midline; short on the mirror. RSI extremes block entry, so the
system does not buy a move that has already run.

Regime-based, not event-based: the position is held as long as both conditions
agree.

Tuned parameters: **EMA 9/100, RSI 21, `adx_min=0`.** The search pushed the
slow EMA to the widest value in the grid and **disabled the ADX filter
entirely** — the trend-strength gate, which conventional wisdom says should
help, contributed nothing here.

### `macd_adx_trend`

Three layers: EMA200 decides which side of the market is tradeable, the MACD
histogram triggers, ADX gates on trend strength (conventionally 25 — in the
grid because conventions deserve testing). Best Sortino 0.28.

---

## Momentum

Premise: the *rate of change* of price carries information beyond its
direction.

### `stochrsi_macd_momentum`

Enters when Stochastic RSI crosses out of an extreme with the MACD histogram
agreeing and volume above average; exits at the opposite extreme.

Best Sortino **0.11** across every timeframe and parameter set. The weakest
methodology tested. Its selectivity is also its problem: on validation it took
9 trades at 4h — below the measurability threshold.

---

## Mean reversion

Premise: price stretched far from a reference snaps back. Only valid in a
range, so both are gated on a trend-strength **ceiling**. Fading a strong trend
is how accounts die.

### `rsi_bollinger_reversion`

Buys below the lower Bollinger Band with RSI oversold, exits at the middle band
— the statistical mean the premise rests on. The ADX ceiling is load-bearing:
without it the rules short every step of a bull market.

### `vwap_stretch_reversion`

Fades price stretched more than *k* ATRs from session VWAP, confirmed by RSI
and volume; exits on a VWAP touch. Measuring stretch in ATRs rather than
percent matters — a fixed percentage means something different on each
timeframe and would make the timeframe comparison meaningless.

VWAP is anchored daily. A VWAP cumulative since 2020 is a meaningless number.

---

## Breakout

Premise: ranges resolve violently. Low win rate by construction — most breaks
fail — so judge these on profit factor and payoff ratio, never win rate.

### `donchian_volume_breakout`

Breaks the highest high of the preceding N bars with volume confirmation; exits
on a shorter opposite channel, Turtle-style.

**Donchian channels replace classical support/resistance deliberately.** The
level is computed from the N bars *before* the current one, so the current bar
can break it. Classical S/R and Fibonacci levels are derived from swing points
identified with hindsight, which quietly leaks the future into the signal.

### `bollinger_squeeze_breakout`

Waits for Bollinger bandwidth to compress into the bottom quartile of its
recent range, then trades the break.

**A leakage lesson lives in one line here.** "Narrow bands" is a relative
judgement, and the obvious implementation compares bandwidth against a quantile
of the *entire* series — which silently uses the future, because the 2026
distribution is not knowable in 2021. The threshold is a **rolling** quantile
over the preceding window only.

---

## Benchmark

### `buy_and_hold`

Runs with stops disabled and full-notional sizing, so it is a genuine
buy-and-hold rather than a stopped-out approximation. It still pays entry and
exit fees, because a real investor would.

Exempt from the eligibility gates (one trade would fail the minimum-trades
rule) but excluded from being *selected* — it is the bar, not a candidate.

**On validation no strategy beat it.** On test it lost 41.0% and every system
beat it comfortably.

---

## Parameter search

Grids are deliberately small: searching thousands of configurations does not
find a better strategy, it finds a luckier one. Capped at 40 per strategy per
timeframe, sampled deterministically.

Incoherent combinations are rejected before sampling — a "fast" EMA slower than
the slow one, an oversold level above overbought, a breakout exit channel
longer than its entry channel — so the budget is not wasted and the
distribution used to judge selection bias is not polluted.

Configurations are grouped by their `IndicatorSpec` so the indicator frame is
computed once per distinct spec. Many parameters (RSI thresholds, volume
filters, ADX ceilings) change no indicator period at all.

**Every search reports the distribution, not just the maximum.** The gap
between the best result and the grid median is a direct measure of how much of
the winner is luck — see [`results.md`](results.md).
