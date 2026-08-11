# Limitations

Every result in this project is bounded by the following. They are listed
plainly because a study that hides its weaknesses is not a study.

---

## Statistical power

**This is the dominant limitation.** 129 trades over 13 months cannot resolve
effects of the size observed. Every difference measured — ML filter, LLM judge,
both controls — has a bootstrap interval containing zero.

The honest reading of Questions 2 and 3 is **"not distinguishable at this
sample size"**, not "no effect". A larger study might find one.

The bootstrap itself assumes trades are independent, discarding serial
correlation. It is a *lower* bound on uncertainty.

---

## Selection bias

651 configurations were evaluated and the best was chosen. The winner is
therefore partly lucky by construction. This is why validation-to-test
degradation is reported prominently rather than hidden, and why the selection
distribution — best versus median — is reported alongside every search.

Sharpe fell from 1.92 on validation to −0.91 on test. That gap is the size of
the problem, measured.

---

## The validation / test regime asymmetry

Validation (2024-01 → 2025-06) is a bull market: +120%. Test (2025-07 →
2026-08) is a bear market: −41%.

Selecting on a bull-only validation set structurally favours long-biased
strategies, which are then tested in a falling market. This inflates the
apparent degradation and confounds "the strategy overfitted" with "the regime
changed".

Mitigated by reporting all test results split by regime, but not eliminated.
**Walk-forward validation across multiple rolling folds would fix it properly**
and was cut for time.

---

## Backtest realism

| Not modelled | Effect |
|---|---|
| **Funding payments** | Perpetuals pay/receive funding every 8h. Ignored entirely. For a strategy holding ~45% of the time, this is a real and unmeasured cost |
| Partial fills | Every order fills completely at one price |
| Queue position | No order-book modelling |
| Liquidation mechanics | At 1× with a 1% risk cap, not reachable — but not modelled either |
| Exchange downtime | Gaps left unfilled; the backtest assumes trading was possible |
| Fee tiers | Flat 0.055% taker. Volume discounts and maker rebates ignored |

Fills are assumed at the open of the next bar plus slippage. Where the model is
ambiguous it resolves against us, but it remains a model.

---

## Live paper trading

**The live system demonstrates that the pipeline works end to end. It is not
evidence that the strategy works.**

At six decisions a day on 4h candles, a two-day window produces a handful of
trades. No conclusion of any kind can be drawn from it.

Fills are simulated locally against real prices. No order reaches a venue, so
there is no execution risk, no rejection handling and no real slippage
measurement.

---

## The venue constraint

The project was designed for Bybit Demo Trading. Under MiCA, Bybit EU offers no
perpetual futures and restricts its API to registered broker integrations;
Bybit's global testnet geo-redirects EEA users to the EU site.

Execution therefore runs through a local simulator. The Bybit V5 client is
written and tested and remains selectable, but **has never placed a live order**
— its correctness beyond authentication is unverified against a real venue.

---

## Data

- History starts **2020-03-25**, when Bybit listed the contract — 12 days after
  the COVID crash low, which is therefore absent.
- One instrument (BTCUSDT), one exchange. Nothing here generalises to other
  assets or venues without retesting.
- Three timeframes. 5m and 1d were excluded by design.

---

## Machine learning

- **Probabilities are uncalibrated.** Random Forest confidence scores are not
  true probabilities, so a 0.35 threshold does not mean "35% likely". The
  threshold is an ordinal filter, not a probability statement. Calibration was
  cut for time.
- **The target may not match the strategy.** The label predicts a 4-bar barrier
  touch; the strategy is trend-following with a median 5-bar hold. A horizon
  sweep is reported, but the pre-registered horizon was kept.
- **The overfit gap is visible**: Random Forest scores 0.530 on train against
  0.397 on validation.
- Two model families only. XGBoost and calibration were scoped out.

---

## The LLM judge

- **127 decisions.** Far too few to characterise behaviour reliably.
- **One model, one prompt.** No ablation over prompt structure, and no test of
  whether the judge is actually using the ML input or ignoring it.
- **Non-deterministic in principle.** `temperature=0` and caching make runs
  reproducible, but the underlying model can change under the same name.
- **The leakage control is a mitigation, not a proof.** Removing dates and
  prices makes recall much harder, not impossible.

---

## Scope cut for time

Built in three days. The following were designed but not implemented:

- Walk-forward validation
- Market-regime detection and dynamic strategy selection
- Fear & Greed index, funding rates, open interest as features
- Leverage experiments (2×, 3×)
- WhatsApp alerts (no Twilio account available)
- Probability calibration
- XGBoost

---

## What this project does *not* claim

It does not claim to have found a profitable trading strategy. It found one
that **lost 10.5% out of sample**.

It does not claim machine learning improves trading. It measured an effect that
**reversed sign between splits** and never reached significance.

It does not claim LLMs make good traders. It measured one that **did not beat
four lines of arithmetic**.

What it does claim is that these questions were asked in a way that could have
produced the opposite answer, and that the negative results are as reliable as
the method allows.

**This is a research and engineering project. It is not investment advice.**
