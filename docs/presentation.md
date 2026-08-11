# Presentation guide

A slide-by-slide outline, the numbers for each, and answers to the questions
you will actually be asked.

**The narrative in one sentence:** *I built a system to test whether ML and
LLMs improve Bitcoin trading, designed it so it could tell me "no", and it
did.*

Lead with that. Negative results delivered confidently are far harder to
challenge than a rising equity curve.

---

## Slide plan (~12 minutes)

### 1. The question (1 min)

Three questions, in order. Say up front that the answers are **"only trend at
4h", "not measurably", and "no"** — then spend the talk showing why those
answers are trustworthy.

Naming the conclusion first reframes the audience from "did it work?" to "is
this rigorous?", which is the ground you want to fight on.

### 2. Data and scale (1 min)

- BTCUSDT perpetuals, 2020-03 → 2026-08
- 223,670 candles at 15m · 55,918 at 1h · 13,980 at 4h
- **Zero gaps, zero duplicates, zero NaNs**
- Regime table: 2020 +339%, 2022 −64%, 2024 +120%, 2026 −27%

### 3. Experimental design (1.5 min) — **the most important slide**

The A / B / C diagram plus both controls. Make one point:

> Every arm trades the same signals through the same code. Only the deciding
> object changes. The controls exist because a judge that vetoes trades changes
> performance whether or not it understands anything.

### 4. Leakage controls (1.5 min)

Pick three, not eleven:

- **Ichimoku's Chikou Span is the close shifted backwards.** Reading it at `t`
  reads price at `t+26`. Excluded entirely.
- **A test recomputes every indicator on truncated data** and asserts history
  is unchanged. Look-ahead bias cannot enter without turning the build red.
- **The LLM has memorised BTC history.** No dates, no prices in the prompt —
  enforced by test. Leakage through model weights, invisible to pandas.

### 5. Question 1 — the fee finding (1.5 min)

| Timeframe | Median trades | Median fees | Eligible |
|---|---|---|---|
| **15m** | 1,015 | **65% of capital** | **0** |
| 1h | 243 | 19% | 0 |
| 4h | 61 | 3% | 2 |

Worst case: 4,297 trades, **79.7% of capital in fees**, $10,000 → $48.

Then the plateau-vs-spike table: at 4h, 65% of the parameter neighbourhood is
profitable and the winner sits 0.78 above the median. At 15m the winner is
5.29 above its median and still deeply negative — the least-bad draw from a bad
distribution.

### 6. Question 2 — ML (1.5 min)

RF beat the dummy baseline by **+6.4 points** of balanced accuracy. Real, if
modest.

Then the flip:

| | Validation | Test |
|---|---|---|
| B − A | **−35.1%** | **+12.0%** |
| CI | [−73.5%, −0.2%] | [−15.8%, +37.6%] |

*"The sign reversed between splits. If I had run one split and declared a
verdict, I would have been confidently wrong either way."*

### 7. Question 3 — the LLM (1.5 min)

| | Agreed | HOLD | Confidence |
|---|---|---|---|
| LLM judge | **40.9%** | **59.1%** | 66 |

Not a rubber stamp, not a coin flip. It deliberated — and returned −0.3%
against System B's +1.6%, **where System B is the deterministic four-line
rule**.

*"The LLM matched a condition a junior developer writes in a minute, at the
cost of an API dependency and latency."*

### 8. The out-of-sample collapse (1.5 min) — **your strongest slide**

| | Validation | Test |
|---|---|---|
| Return | +34.6% | **−10.5%** |
| Sharpe | **1.92** | **−0.91** |

*"That gap is what parameter selection bought me on validation and could not
deliver out of sample. It is the most honest number in the project — and I
could only produce it because the test set was locked from day one."*

### 9. But it still worked (1 min)

| | Return | Max DD |
|---|---|---|
| System A | −10.5% | 14.5% |
| System C | −0.3% | **7.7%** |
| **Buy and hold** | **−41.0%** | **53.5%** |

Then the regime split: A made +4.5% (Sharpe 1.68) rising, −10.5% (Sharpe −1.71)
falling. **A specific, diagnosable weakness — not a shrug.**

### 10. Live demo (1.5 min)

Dashboard tour: live decisions with LLM reasoning, leaderboard, method tab.
Show a decision that produced **no** trade and explain why that is logged.

### 11. Engineering (1 min)

- ~12,000 lines, **187 tests**
- Four independent safety layers; production host absent from the codebase
- The MiCA story: a regulatory change removed the venue mid-build; execution
  swapped behind an interface with zero changes to strategy, model or judge
- Two bugs the tests caught (below)

### 12. Limitations and next steps (1 min)

Name three before anyone asks: sample size, the bull-validation /
bear-test asymmetry, funding costs ignored.

Next: walk-forward validation, a regime filter, calibrated probabilities.

---

## The two bug stories

Both are worth 30 seconds each — they demonstrate that the verification was
real, not decorative.

**Double-counted entry fees.** The engine deducted the entry fee from cash
*and* again inside trade P&L. Found as a $5.50 discrepancy against arithmetic
done by hand in a test. Invisible in an equity curve — it just makes every
trade quietly worse.

**Live inference used the training dataset.** `build_dataset` drops rows whose
label window is incomplete — correct for training, fatal for inference, since
the newest candle can never have a label. The live loop ran 4 candles behind,
logging `ml +0 (0%)` while the ML and LLM layers sat inert and the logs looked
healthy.

---

## Questions you will be asked

**"Why did it lose money?"**
> BTC fell 41% in the test period. The system lost 10.5% with a 14.5%
> drawdown — 31 points better than holding, at a quarter of the risk. The
> regime split shows exactly where it fails: downtrends. It's a trend
> strategy, and it can't tell an uptrend from a downtrend fast enough.

**"So the ML was useless?"**
> The model beat its baseline by 6.4 points — it learned something. It just
> isn't something that helps this strategy. And the trading effect reversed
> sign between validation and test, which is itself the finding: single-split
> conclusions about ML in trading are unreliable.

**"Isn't the LLM result just a bad prompt?"**
> Possibly. I can't rule it out with 127 decisions and one prompt design. What
> I can say is that it wasn't degenerate — 40.9% agreement, 59.1% HOLD,
> moderate confidence. It deliberated. It just didn't beat arithmetic.

**"Why not more data / deep learning?"**
> 13,580 samples at 4h. An LSTM would overfit and add no explanatory value.
> The constraint is sample size, not model capacity — which is also why every
> result has a confidence interval.

**"How do you know there's no look-ahead bias?"**
> Eleven documented controls, several enforced by tests that fail the build.
> The strongest is truncation invariance: recompute everything on a shortened
> series and assert history is unchanged. And the honest answer is that the
> out-of-sample collapse is itself evidence — a leaking backtest doesn't
> produce a Sharpe of −0.91.

**"Would you trade this?"**
> No. It lost money out of sample, and the sample is too small to conclude
> anything about the layers. What I'd do next is a regime filter, because the
> failure mode is specific rather than general.

**"Why a simulator instead of the exchange?"**
> MiCA. Bybit EU offers no perpetuals and restricts API access to registered
> brokers; the global testnet geo-redirects EEA users. Dropping shorts would
> have invalidated every result in a falling market, so execution moved behind
> the same interface. The Bybit client is written, tested and still selectable.

**"What's the actual contribution?"**
> A system that could have told me ML and LLMs help, and instead told me they
> don't — with the controls to make that believable. The negative result is
> the contribution.

---

## Do not say

- ❌ "The AI trading bot" — it's a research system
- ❌ "It could be profitable with more tuning" — that's the overfitting you
  just demonstrated
- ❌ "The LLM understands the market"
- ❌ Any accuracy number without the dummy baseline beside it
- ❌ **0.526** — that's the refit model scored on its own training data. Quote
  **0.397**

## Do say

- ✅ "Not distinguishable at this sample size"
- ✅ "The interval contains zero"
- ✅ "This is what overfitting looks like, measured"
- ✅ "The control arm exists because filtering changes performance regardless of
  skill"

---

## Demo checklist

```bash
# Terminal 1 — trader (leave running well before the presentation)
nohup python main.py --system C > logs/live.log 2>&1 &

# Terminal 2 — dashboard
streamlit run dashboard/app.py

# Terminal 3 — for the test suite, live
pytest -q
```

- [ ] Trader running for hours beforehand, so the dashboard has decisions
- [ ] Dashboard open on the **Experiment** tab
- [ ] `pytest -q` ready — 187 passing in ~3 seconds is a strong live moment
- [ ] `data/cache/llm_decisions.json` present, so the LLM arm replays without
      network
- [ ] Screenshots of every tab as a fallback
