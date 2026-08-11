# Results

All figures produced by the scripts in `scripts/`, saved to `data/results/`.
Methodology and leakage controls: [`methodology.md`](methodology.md).

**Summary in one line:** of 651 configurations across four methodologies and
three timeframes, one survived selection; it then lost money out of sample
while beating buy-and-hold by 31 points; neither the ML filter nor the LLM
judge produced a statistically distinguishable improvement.

---

## Question 1 — which methodology and timeframe?

### Validation leaderboard (default parameters)

| Rank | Strategy | Methodology | TF | Trades | Return | Sharpe | Sortino | Max DD | Eligible |
|---|---|---|---|---|---|---|---|---|---|
| 1 | *buy_and_hold* | *benchmark* | *15m* | *1* | *+155.6%* | *1.46* | *1.45* | *32.5%* | — |
| 4 | **ema_rsi_trend** | trend | **4h** | 233 | +22.6% | 1.20 | 1.13 | 10.4% | ✅ |
| 5 | donchian_volume_breakout | breakout | 4h | 85 | +5.8% | 0.42 | 0.28 | 12.7% | ✅ |
| 6 | vwap_stretch_reversion | mean reversion | 4h | 19 | +3.6% | 1.41 | 0.23 | 1.3% | ❌ 19 trades |
| 7 | stochrsi_macd_momentum | momentum | 4h | 9 | −0.8% | −0.27 | −0.03 | 3.1% | ❌ 9 trades |
| 8 | bollinger_squeeze_breakout | breakout | 4h | 61 | −2.6% | −0.18 | −0.09 | 13.7% | ❌ PF 0.93 |

**Only 2 of 21 strategy/timeframe combinations passed the eligibility gates.**

### Fees destroy short timeframes

Median across all non-benchmark strategies:

| Timeframe | Median trades | Median fees paid | Median return | Eligible configs |
|---|---|---|---|---|
| **15m** | 1,015 | **65% of capital** | −72% | **0** |
| 1h | 243 | 19% | −22% | 0 |
| 4h | 61 | 3% | −1% | 2 |

Worst case: `ema_rsi_trend @ 15m` took **4,297 trades**, paid **79.7% of
starting capital in fees**, and finished at $48 from $10,000. Not a bug — the
arithmetic is 4,297 × 2 × 0.055%.

Across all 651 configurations in the parameter search, **15m produced zero
eligible results.**

### Parameter search: is the winner a plateau or a spike?

651 configurations, 48 eligible. Sortino distribution within each strategy's
own grid:

| Strategy @ TF | Configs | Best | Median | Spread | % positive |
|---|---|---|---|---|---|
| **ema_rsi_trend @ 4h** | 40 | **1.67** | 0.89 | 0.78 | **65%** |
| donchian_volume_breakout @ 4h | 15 | 0.39 | −0.07 | 0.46 | 40% |
| vwap_stretch_reversion @ 4h | 40 | 0.34 | 0.00 | 0.34 | 55% |
| macd_adx_trend @ 4h | 24 | 0.28 | 0.00 | 0.28 | 50% |
| stochrsi_macd_momentum @ 4h | 40 | 0.11 | −0.04 | 0.15 | 13% |
| ema_rsi_trend @ 15m | 40 | −4.01 | −9.30 | **5.29** | **0%** |

Two-thirds of the parameter neighbourhood at 4h is profitable and the winner
sits 0.78 Sortino above its median — a **plateau**, not a spike. Compare 15m,
where the "best" configuration is 5.29 above its median and still deeply
negative: there the winner is simply the least-bad draw from a bad
distribution.

### Answer

**Trend-following, at 4h, and only that.** Of four methodologies, only trend
survived selection, and only on the longest timeframe tested. Momentum peaked
at Sortino 0.11 across every timeframe and parameter set. Mean reversion and
breakout produced marginal results at 4h and nothing elsewhere.

Winning configuration: `ema_rsi_trend @ 4h`, EMA 9/100, RSI 21, ADX filter
disabled (`adx_min=0` — the filter contributed nothing).

---

## Question 2 — did machine learning help?

### Model comparison (validation, clean)

| Model | Val accuracy | **Val balanced accuracy** | F1 macro | **Uplift vs dummy** | Overfit gap |
|---|---|---|---|---|---|
| **Random Forest** | 0.393 | **0.397** | 0.391 | **+0.064** | 0.133 |
| Logistic Regression | 0.372 | 0.374 | 0.372 | +0.040 | 0.022 |
| Dummy (majority class) | 0.358 | 0.333 | 0.176 | 0.000 | 0.000 |

`uplift_vs_dummy` is the only column meaningful on its own. On an imbalanced
three-class target, raw accuracy can be *worse* than always predicting the
majority class.

**The Random Forest learned something real: +6.4 points of balanced accuracy
over baseline.** Whether that helps trading is a separate question.

> ⚠️ The final run refits on train + validation and logs a "validation"
> accuracy of 0.526. That is measured on data the model was just trained on and
> is **not** a generalisation estimate. Quote 0.397.

Top features by importance: `volume_ratio` (0.19), `bb_pct_b` (0.087),
`atr_pct` (0.061), `realised_vol` (0.060). **Volatility and volume regime
dominate the classic oscillators.**

### Effect on trading

| | Validation | Test |
|---|---|---|
| Difference (B − A) | **−35.1%** | **+12.0%** |
| 95% CI | [−73.5%, −0.2%] | [−15.8%, +37.6%] |
| P(B better) | 2.4% | 79.9% |
| Verdict | Significantly **worse** | Not distinguishable |

**The sign flipped between splits.** Had the study used one split and declared
a verdict, it would have been confidently wrong either way.

A horizon sweep rules out the obvious objection — that a 4-bar label mismatches
a trend strategy:

| Label horizon | RF uplift | System B return | B Sharpe |
|---|---|---|---|
| 4 bars | 0.064 | 0.2–7.7% | 0.05–0.70 |
| 8 | 0.049 | 6.2% | 0.61 |
| 12 | 0.042 | 19.6% | 1.66 |
| 16 | 0.046 | 22.6% | 1.56 |

Longer horizons recover most of the damage but never beat System A's 34.6% on
validation. Note the tension: the model predicts *short* horizons best, but
short-horizon predictions are least useful to a strategy whose median hold is
5 bars. Horizon 4 was kept as the pre-registered choice — selecting the horizon
that flatters ML is exactly the error this project is built to avoid.

### Answer

**Point estimate yes, statistically no.** The model beat its baseline on
classification and improved trading on the test set, but the interval contains
zero, and the effect reversed between splits.

---

## Question 3 — did the LLM judge help?

### Test-set performance

| System | Trades | Return | Sharpe | Sortino | Max DD | Win rate | PF | Exposure |
|---|---|---|---|---|---|---|---|---|
| **A** — rules only | 129 | −10.5% | −0.91 | −0.72 | 14.5% | 27.1% | 0.80 | 45.9% |
| **B** — + ML filter | 64 | **+1.6%** | 0.22 | 0.13 | 11.2% | 35.9% | 1.05 | 22.9% |
| **C** — + LLM judge | 52 | −0.3% | −0.01 | −0.00 | **7.7%** | 34.6% | 0.98 | 19.0% |
| Control — always agree | 127 | −9.8% | −0.85 | −0.67 | 14.2% | 27.6% | 0.81 | 45.0% |
| **Buy and hold** | 1 | **−41.0%** | −0.96 | −0.92 | **53.5%** | — | — | 99.4% |

Differences vs System A, 95% bootstrap intervals:

| System | Difference | CI | P(better) | Verdict |
|---|---|---|---|---|
| B — + ML | +12.0% | [−15.8%, +37.6%] | 79.9% | not distinguishable |
| C — + LLM | +9.7% | [−18.1%, +34.8%] | 78.4% | not distinguishable |
| Control — always agree | +1.1% | [−26.1%, +31.3%] | 52.9% | not distinguishable |

### Judge behaviour

| Judge | Decisions | Agreed with strategy | Chose HOLD | Matched ML | Mean confidence |
|---|---|---|---|---|---|
| Always agree (control) | 127 | 100.0% | 0.0% | 60.6% | 100 |
| **LLM judge** | 127 | **40.9%** | **59.1%** | 64.6% | 66 |

The LLM was **neither a rubber stamp nor a coin flip**. It vetoed nearly 60% of
proposals and reported moderate confidence. It genuinely deliberated.

### Answer

**No.** System C returned −0.3% against System B's +1.6%, and **System B *is*
the deterministic four-line rule** — verified identical by an automated
cross-check. The LLM matched what a simple condition already achieved, at the
cost of an API dependency, latency and non-determinism.

Its difference from System A is statistically indistinguishable from the ML
filter's. Nothing in the data supports a claim that language-model reasoning
added value here.

---

## The headline finding: validation → test

| | Validation | Test |
|---|---|---|
| Total return | **+34.6%** | **−10.5%** |
| Sharpe | **1.92** | **−0.91** |
| Sortino | 1.67 | −0.72 |
| Max drawdown | 7.6% | 14.5% |
| Trades | 157 | 129 |

A Sharpe of 1.92 became −0.91 the moment the data had not been selected on.
This gap is what parameter selection bought on validation and could not deliver
out of sample. It is the most honest number in the project.

---

## Two findings that survive the collapse

### 1. It lost money and still worked

BTC fell 41% over the test period with a 53.5% drawdown. System A lost 10.5%
with a 14.5% drawdown — **31 points better than buy-and-hold, at roughly one
quarter of the risk.** System C lost 0.3% with a 7.7% drawdown, one seventh of
buy-and-hold's.

### 2. Risk fell monotonically with each layer

| | A | B | C |
|---|---|---|---|
| Max drawdown | 14.5% | 11.2% | **7.7%** |
| Exposure | 45.9% | 22.9% | **19.0%** |
| Fees paid | 5.9% | 3.3% | **2.5%** |

Each layer traded less, risked less and cost less. The intervals overlap, so
this is **suggestive, not proven** — but the ordering is consistent across
three independent measures.

---

## Regime split — did it only work because the market moved one way?

Test period split at the price peak (2025-10-06):

| System | Phase | Trades | Return | Sharpe | Max DD |
|---|---|---|---|---|---|
| A | rising | 28 | +4.5% | **1.68** | 5.3% |
| A | falling | 101 | −10.5% | **−1.71** | 14.5% |
| B | rising | 18 | +3.5% | 1.50 | 3.9% |
| B | falling | 46 | +1.6% | −0.24 | 11.2% |
| C | rising | 9 | +0.5% | 0.33 | 2.6% |
| C | falling | 43 | −0.3% | −0.10 | 7.7% |
| Buy & hold | rising | 1 | +15.1% | 1.92 | 12.8% |
| Buy & hold | falling | 0 | −41.0% | −1.59 | 53.5% |

This explains the whole result. **The trend strategy makes money in uptrends
and bleeds in downtrends — despite being able to short.** It took 101 trades in
the falling phase and lost on them.

The filters fixed the symptom, not the cause: System C took just **9 trades**
in the rising phase, converting a trend-follower into something that barely
participates. Excellent for drawdown, fatal for returns.

---

## Live paper trading

The system runs continuously on the 4h timeframe, deciding six times a day.
Fills are simulated locally against real BTC prices using the same fee,
slippage and stop rules as the backtester.

Over hours or days this is **a demonstration that the pipeline works end to
end, not evidence that the strategy works.** At six decisions a day, a
two-day window produces a handful of trades — far too few to measure anything.

---

## What would change the conclusions

- **A longer test period.** Thirteen months and 129 trades cannot resolve
  effects of the size observed here.
- **Walk-forward validation.** Averaging selection across multiple rolling
  folds would remove the bull-validation / bear-test asymmetry.
- **A regime filter.** The strategy's failure is specific and diagnosable: it
  loses in downtrends. Detecting the regime and standing aside is a more
  promising direction than any of the layers tested.
- **A different ML target.** The label predicts a 4-bar barrier touch; the
  strategy holds for a median of 5 bars and is trend-following. These are not
  obviously the same question.
