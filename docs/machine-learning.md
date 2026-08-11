# Machine learning

## The question the model is asked

> Within the next 4 candles, does price reach the upper barrier or the lower
> barrier first?

Three classes: LONG, SHORT, HOLD. This is triple-barrier labelling.

---

## Redefining the target

The project brief specified a fixed ±0.5% barrier. Measured on real data, that
definition does not survive contact with three timeframes:

| Barrier | 15m HOLD | 1h HOLD | 4h HOLD | Ties discarded (4h) |
|---|---|---|---|---|
| **Fixed 0.5%** | 53.7% | 18.6% | **2.8%** | **26.1%** |
| **ATR-scaled** (1.0 × ATR14) | 27.8% | 31.4% | 29.9% | 1.4% |

Four candles is one hour at 15m and sixteen hours at 4h. A fixed 0.5% is
therefore a rare, meaningful move on one timeframe and pure noise on another.
At 4h it leaves almost no HOLD class and discards **a quarter of all samples**
as unresolvable same-candle ties.

Scaling by ATR makes the barrier mean the same thing everywhere — "one unit of
current volatility" — and holds roughly 35/35/30 on every timeframe. Both are
computed; the fixed version is reported as a sensitivity check rather than
quietly dropped.

---

## Three rules the labeller enforces

**First touch is resolved chronologically.** Bars are scanned forward one at a
time and the *first* barrier reached decides the label. Asking "was the upper
barrier reached at all?" would use knowledge of the whole window.

**Same-candle ties are excluded, not guessed.** If one candle's high breaches
the upper barrier and its low breaches the lower one, OHLCV cannot say which
came first. Those samples are labelled HOLD and flagged for exclusion —
guessing would inject a bias whose direction is unpredictable.

**The decision bar is excluded from its own window.** Bar `t`'s label reads
bars `t+1` to `t+4`. A model seeing bar `t`'s own high and low would be trading
on information that did not exist at decision time.

---

## Features

18 features, all **scale-free**:

| Group | Features |
|---|---|
| Momentum | `rsi`, `stochrsi_k`, `stochrsi_d`, `macd_hist_pct` |
| Volatility | `atr_pct`, `realised_vol`, `bb_width`, `bb_pct_b` |
| Trend | `adx`, `di_plus`, `di_minus`, `ema_fast_slow_spread_pct`, `dist_from_trend_pct` |
| Volume | `volume_ratio` |
| Position | `vwap_dist_pct`, `donchian_pos` |
| Returns | `ret_1`, `ret_4` |

**Raw price levels are excluded deliberately.** A model given absolute BTC
price learns that "price is 60,000" implies late 2024 — memorising the calendar
rather than market structure, and collapsing the moment prices leave the
training range. Every feature here is a ratio, a bounded oscillator, or a
distance expressed as a fraction of price, so a vector from 2020 and one from
2026 are directly comparable.

`test_features_exclude_absolute_price` enforces this.

---

## Training protocol

| Control | Implementation |
|---|---|
| No shuffling | Chronological splits only |
| Embargo | 4 bars removed at the start of validation and test |
| Scaling | `StandardScaler` inside the pipeline, fitted on training folds only |
| Class imbalance | `class_weight="balanced"` on both real models |
| Baseline | `DummyClassifier(strategy="most_frequent")`, always trained |

### Why the dummy baseline is mandatory

On an imbalanced three-class target, 70% accuracy can be **worse** than always
predicting the majority class. Without the baseline, an accuracy figure is an
uninterpretable number. `uplift_vs_dummy` is the only column in the comparison
table that means anything on its own.

---

## Results

| Model | Val accuracy | Val balanced accuracy | F1 macro | Uplift vs dummy | Overfit gap |
|---|---|---|---|---|---|
| **Random Forest** | 0.393 | **0.397** | 0.391 | **+0.064** | 0.133 |
| Logistic Regression | 0.372 | 0.374 | 0.372 | +0.040 | 0.022 |
| Dummy | 0.358 | 0.333 | 0.176 | 0.000 | 0.000 |

The Random Forest beat the baseline by **6.4 points of balanced accuracy** — a
real, if modest, edge. Logistic Regression managed +4.0 with a far smaller
overfit gap.

Top features: `volume_ratio` (0.19), `bb_pct_b` (0.087), `atr_pct` (0.061),
`realised_vol` (0.060). **Volatility and volume regime dominate the classic
oscillators** — RSI ranks ninth.

> ⚠️ The final test run refits on train + validation and logs a "validation"
> balanced accuracy of 0.526. Validation is inside the training set at that
> point, so this is measured on data the model just trained on and is **not** a
> generalisation estimate. **Quote 0.397.**

---

## The filter — System B

    Take System A's signal only if the model predicts the same direction with
    probability at least `threshold`. Otherwise stand aside.

Deliberately narrow. If the model generated its own entries, A and B would be
trading different markets and the comparison would answer nothing.

### Entry gating, not per-bar gating

Strategy signals are a persistent *state*, not independent decisions. An early
implementation required approval on every held bar. Measured on this data it
cut exposure from 51% to 11% while leaving the trade count nearly unchanged —
it was shredding single positions into fragments, each paying a round-trip fee,
not selecting better trades.

`entry` mode asks the question the research is actually about: *given that the
rules want to open a trade here, does the model agree it is worth taking?* Once
open, the strategy's own exit logic governs. A vetoed entry is not retried
until the signal changes, otherwise the filter merely delays entry until the
model wavers.

### Two consequences to keep in view

The filter can only ever **remove** trades, so System B always has a smaller
sample and wider intervals than System A. And removing trades mechanically
changes win rate and drawdown *even if the model has no skill at all* — which
is precisely why the comparison needs a control arm rather than two point
estimates.

---

## What the results showed

| | Validation | Test |
|---|---|---|
| Difference (B − A) | **−35.1%** | **+12.0%** |
| 95% CI | [−73.5%, −0.2%] | [−15.8%, +37.6%] |
| P(B better) | 2.4% | 79.9% |

**The sign flipped between splits.** On one it was significantly harmful; on
the other, helpful but not distinguishable from noise. A study using a single
split would have been confidently wrong either way.

A horizon sweep (4/8/12/16 bars) confirms the effect is not simply a
label-horizon mismatch: longer horizons recover most of the damage but never
beat System A on validation. Note the tension — the model predicts *short*
horizons best, while short-horizon predictions are least useful to a strategy
holding a median of 5 bars.

Horizon 4 was kept as the pre-registered choice. Selecting the horizon that
flatters ML is exactly the error this project exists to avoid.
