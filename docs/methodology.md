# Methodology

How the experiment was designed, and why each choice was made. If you read one
document, read this one — the results in [`results.md`](results.md) are only
worth as much as the controls described here.

---

## 1. The research questions

1. Which combination of **trading methodology, technical indicators and
   timeframe** produces the best risk-adjusted BTC day-trading performance?
2. Does adding a **machine-learning filter** improve it?
3. Does adding an **LLM trading judge** improve it further — or does it merely
   reproduce what a simple rule already achieves?

Question 3 is phrased deliberately. "Did the LLM help?" is unanswerable without
saying *compared to what*, and the obvious comparison — against no judge at all
— is the wrong one. See §6.

---

## 2. Data

| | |
|---|---|
| Instrument | BTCUSDT perpetual futures |
| Source | Bybit public market-data API |
| Timeframes | 15m, 1h, 4h |
| Period | 2020-03-25 → present |
| Rows | 223,670 (15m) · 55,918 (1h) · 13,980 (4h) |
| Integrity | **Zero gaps, zero duplicates, zero NaNs** across all three series |

History begins 2020-03-25 because that is when Bybit listed the contract — 12
days after the COVID crash low, which is therefore not in the dataset.

**Timestamps are candle open times, in UTC.** A candle indexed `12:00` on the
4h timeframe covers 12:00–15:59 and is only complete at 16:00. The loader
discards the still-forming final candle: acting on it is look-ahead bias in
backtests and a live trading bug.

Gaps from exchange downtime are left unfilled. Interpolating candles would
invent price action that never happened.

### Regime coverage

| Year | Return | Max drawdown |
|---|---|---|
| 2020 | +339% | −20% |
| 2021 | +59% | −55% |
| 2022 | −64% | −67% |
| 2023 | +156% | −22% |
| 2024 | +120% | −32% |
| 2025 | −7% | −35% |
| 2026 YTD | −27% | −40% |

Bull markets, a full bear market, recoveries and two large drawdowns.

---

## 3. Chronological splits

| Split | Period | Purpose |
|---|---|---|
| Train | 2020-03 → 2023-12 | Model fitting |
| Validation | 2024-01 → 2025-06 | Strategy selection, parameter search, thresholds |
| **Test** | 2025-07 → 2026-08 | **Final comparison. Read exactly once.** |

The test period is deliberately awkward to obtain in code. `get_split("test")`
raises `PermissionError` unless called with `unlock_test=True`, which appears
in exactly one file: `scripts/run_final.py`. Touching out-of-sample data is
always a visible, deliberate act.

**Known asymmetry, disclosed:** validation is a bull market (+120%) and test is
a bear market (−41%). Selecting on a bull-only validation set structurally
favours long-biased strategies. This is why every test result is *also*
reported split by regime (§7), and why the asymmetry is stated in the results
rather than buried.

---

## 4. Data-leakage controls

The single largest threat to a project like this. Eleven controls, each with a
specific failure mode it prevents.

| Risk | Control |
|---|---|
| Indicators peeking forward | Hand-written; an automated test recomputes every indicator on truncated data and asserts historical values are unchanged |
| **Ichimoku Chikou Span** | **Excluded.** It is the close shifted *backwards* 26 periods, so reading it at `t` reads price at `t+26` |
| Fibonacci / swing support-resistance | **Excluded.** Derived from swing points identified with hindsight. Replaced by Donchian channels of the *previous* N bars |
| Rolling-quantile thresholds | The Bollinger squeeze uses a rolling quantile over preceding bars, never a whole-series quantile — the 2026 distribution is not knowable in 2021 |
| Shuffled time series | Never shuffled. All splits strictly chronological |
| Overlapping labels at split seams | 4-bar embargo removed at the start of validation and test |
| Scaler fitted on all data | Fitted inside the sklearn pipeline, on training folds only |
| Test set used for tuning | `get_split` raises unless explicitly unlocked |
| Acting on an unclosed candle | Loader drops the still-forming final candle |
| **The LLM having memorised BTC history** | The judge prompt contains no dates and no absolute prices. Enforced by test |
| Same-candle barrier ties | Labelled HOLD and excluded from training, never guessed |
| Live inference using training data | Inference uses the feature builder, not the label-filtered dataset. Enforced by test |

Two of these deserve elaboration.

### The Chikou Span

Ichimoku is standard in retail crypto analysis and would have been an easy
inclusion. Its Chikou Span is defined as the closing price plotted 26 periods
*in the past*. A model reading the Chikou value at bar `t` is reading the close
of bar `t+26`. The indicator was excluded entirely rather than partially used.

### The LLM's memorised history

An LLM has read a great deal of Bitcoin price commentary. Told that BTC is at
$118,320 on 3 March, it may partially recall what happened next — look-ahead
bias travelling through model weights rather than through a dataframe, where no
amount of careful pandas would catch it.

Every value the judge receives is therefore relative: RSI, % distance from EMA,
ATR as % of price, MACD histogram as a fraction of price, ML probability.
`test_prompt_contains_no_dates_or_absolute_prices` asserts no year, month,
currency symbol or price token ever reaches the model, and a second test
asserts the schema has no timestamp or price fields at all.

---

## 5. The machine-learning target

**Triple-barrier labelling**: for each bar, does price reach the upper or lower
barrier first within the next 4 candles? Three classes — LONG, SHORT, HOLD.

### Why the barrier is scaled by ATR

The original design used a fixed ±0.5%. Measured on real data:

| Barrier | 15m HOLD | 1h HOLD | 4h HOLD | Ties discarded (4h) |
|---|---|---|---|---|
| Fixed 0.5% | 53.7% | 18.6% | **2.8%** | **26.1%** |
| ATR-scaled (1.0 × ATR14) | 27.8% | 31.4% | 29.9% | 1.4% |

Four candles is one hour at 15m and sixteen hours at 4h. A fixed 0.5% is
therefore a rare, meaningful move on one timeframe and noise on another — which
would make the timeframe comparison at the heart of Question 1 meaningless. It
would also have discarded a quarter of the 4h data as unresolvable ties.

The ATR barrier holds roughly 35/35/30 on every timeframe. Both versions are
computed and reported; the fixed variant is retained as a sensitivity check.

### First touch, resolved chronologically

The label is decided by whichever barrier is reached *first*, scanning bars
forward one at a time. Asking "was the upper barrier reached at all?" would use
knowledge of the whole window.

### Same-candle ties are excluded, not guessed

If one candle's high breaches the upper barrier and its low breaches the lower
one, OHLCV cannot say which came first. Those samples are labelled HOLD and
flagged for exclusion. Guessing would inject a bias whose direction we could
not predict.

### The decision bar is excluded from its own window

Bar `t`'s label looks at bars `t+1` through `t+4`. A model that could see bar
`t`'s own high and low would trade on information that did not exist when the
decision was made.

---

## 6. Experimental design

Every arm trades the **same signal universe**, with exactly one thing changed:

| Arm | Pipeline |
|---|---|
| **A** | Rules only |
| **B** | A's signals, taken only when the ML model agrees above a probability threshold |
| **C** | B's inputs, with an LLM judge making the final call |
| Control — *always agree* | A's signals routed through the judge harness with no judgement |
| Control — *deterministic* | Trade only when strategy and model agree (four lines of arithmetic) |

### Why the controls exist

**Always-agree** proves the comparison machinery adds no bias. If routing
signals through the harness changes them, every System C result is suspect.

**Deterministic** is the honest benchmark for the LLM. A judge that vetoes
trades will change performance whether or not it understands anything —
filtering alone reshapes the return distribution and shrinks the sample. So
"System C beat System A" is not evidence of reasoning. The question is whether
the LLM beats a rule a junior developer could write in a minute.

A cross-check in the final run verifies that System B and the deterministic
judge produce **identical** results — they are the same rule reached by two
independent code paths. If they ever diverge, one is wrong.

### The filter gates entries, not every bar

Strategy signals are a *persistent state*, not a stream of independent
decisions. An early implementation required the model's approval on every held
bar; measured on this data it cut exposure from 51% to 11% while leaving the
trade count almost unchanged — it was shredding single positions into
fragments, each paying a round-trip fee, not selecting better trades.

Both modes remain in the code; `entry` is the default and a test demonstrates
the fragmentation.

---

## 7. Metrics and selection

**Selection uses one metric behind hard gates.** The composite score exists to
make the leaderboard readable; it double-counts return by construction and is
not a sound basis for choosing a winner.

Primary metric: **Sortino** on validation.

Eligibility gates — failing any one disqualifies a configuration outright:

- at least **30 trades** (below this nothing is measurable)
- maximum drawdown at most **40%**
- profit factor above **1.0**

Benchmarks are exempt from the gates (buy-and-hold takes one trade) but
excluded from being selected.

Display-score normalisation uses **percentile rank within the cohort**, not
min-max: one outlier would otherwise compress every other strategy into a
narrow band and dominate the ranking.

### Confidence intervals on everything

Every headline difference is reported with a bootstrap interval built by
resampling the trade sequence 1,000 times and recompounding. **If the interval
contains zero, the two systems are not distinguishable at this sample size.**

This is stated as a result, not a caveat. With a few hundred trades, most
differences are noise, and saying so is more defensible than quoting two point
estimates and declaring a winner.

The assumption is stated too: resampling treats trades as independent,
discarding serial correlation. It is a lower bound on uncertainty, not an
upper one.

### Regime-split reporting

The test period contains a 41% decline, so aggregate figures conflate "the
strategy works" with "the market fell". Every test result is also reported
split at the price peak into a rising and a falling phase.

---

## 8. Execution assumptions

The backtester is deliberately pessimistic, and consistently so.

| Assumption | Value |
|---|---|
| Fill timing | Signal on the close of bar `t`, filled at the **open of `t+1`** |
| Taker fee | 0.055% per side (Bybit linear perpetual) |
| Slippage | 2 bps, always against us |
| Same-bar stop and target | **Resolved as a stop** — OHLCV cannot say which came first |
| Gap through the stop | Fills at the **open**, not the stop price |
| Gap through the target | Fills at the **target** — no credit for a favourable gap |
| Re-entry after a stop | Blocked until the signal changes |
| Risk per trade | 1% of equity |
| Stop / target | 2 × ATR(14), 2:1 reward:risk |
| Daily loss limit | 3% |
| Leverage | 1× |
| Position sizing | Sized so a stop-out costs exactly 1% of equity |

### A consequence worth knowing

When the stop is tight relative to price, sizing for 1% risk implies a notional
several times equity. The 1× leverage cap truncates it, so **effective risk per
trade falls below the configured 1% while fees stay identical**. At 4h
volatility the full 1% is risked; at 15m roughly 0.4% is risked for the same
0.11% round-trip cost. This is a second, independent mechanism behind the 15m
collapse, and is pinned by a test.

---

## 9. Verification

**187 automated tests.** The ones that matter most:

- `test_indicators_do_not_change_when_future_data_is_appended` — recomputes
  every indicator on a truncated series and asserts history is unchanged.
- `test_backtest_is_invariant_to_future_data` — the same argument applied to
  the whole execution path.
- `test_labels_are_unchanged_by_data_beyond_their_window` — labels may look
  forward exactly `horizon` bars and no further.
- `test_prompt_contains_no_dates_or_absolute_prices` — the LLM leakage control.
- `test_the_live_loop_uses_the_inference_path` — reads `main.py` and fails if
  the label-filtered training set reappears in live inference.
- `test_source_contains_no_negative_shifts` — greps the indicator module for
  `.shift(-n)`, the most common way future data leaks in.
- The engine's arithmetic is checked against P&L computed **by hand**, not
  against its own output.

### Two bugs these tests caught

**Double-counted entry fees.** The engine deducted the entry fee from cash *and*
subtracted it again inside trade P&L. It showed up as a $5.50 discrepancy
against arithmetic done independently in a test. Invisible in an equity curve —
it just makes everything quietly worse per trade.

**Live inference using the training dataset.** `build_dataset` drops trailing
rows whose label window is incomplete, which is correct for training and fatal
for inference: the most recent candle can never have a label. The live loop was
running 4 candles behind, logging `ml +0 (0%)` every cycle with the ML and LLM
layers inert while the logs looked healthy.
