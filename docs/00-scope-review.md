# Scope Review & Architecture Decision — BTC AI Trading Research System

**Status:** Planning. No code written. Awaiting approval.
**Deadline:** ~3 days (presentation).
**Reviewer verdict:** The research design is sound. The scope is roughly 3× too large. Below is what to keep, what to cut, and the five methodology problems that would invalidate the results if left unfixed.

---

## 1. The headline problem: scope vs. 3 days

Your full spec is a 3–4 week project. Everything in "Advanced" and "Stretch" must be assumed cut, and one MVP item (live Bybit paper trading) must be reframed.

**Reframe #1 — live paper trading is plumbing, not an experiment.**
Over ~2 days of runtime on a 1h timeframe you get ~48 candles and likely 0–3 trades. That is not evidence of anything. Its value is *demonstrating the loop works end to end*, live, on stage. Plan it as a demo, present it as a demo, and start it running as early as possible (see implementation order) so the dashboard has something in it by presentation day.

**Reframe #2 — the honest finding is probably "no significant difference."**
Comparing A vs B vs C on one out-of-sample period with tens-to-low-hundreds of trades cannot detect anything but a huge effect. If you present "the LLM improved Sharpe from 0.8 to 1.1" without a confidence interval, a sharp examiner will take it apart. If you present "the LLM changed Sharpe by +0.3, 95% bootstrap CI [-0.6, +1.0], so we cannot distinguish it from noise at this sample size" — that is a *stronger* project. Build the bootstrap CI in from the start; it is ~20 lines and it is the difference between a trading demo and an experiment.

---

## 2. Methodology problems that must be fixed

### 2.1 Ichimoku, Support/Resistance and Fibonacci leak the future — drop them

- **Ichimoku's Chikou Span is the close shifted *backwards* 26 periods.** Reading it at time `t` reads price at `t+26`. Direct look-ahead. The Senkou spans are shifted forward (safe), but the library output puts future-dated rows in your DataFrame, which is very easy to accidentally join on.
- **Support/Resistance and Fibonacci retracement levels** are normally computed from a swing high/low over a window that, in most naive implementations, includes bars after `t`.

These three are the highest leakage risk and the lowest value-per-hour in your list. **Cut them from the MVP.** Mention in `limitations.md` that they were excluded specifically because of look-ahead risk — that is a point in your favour, not against.

### 2.2 The ±0.5% / 4-candle target is not comparable across timeframes

4 candles means 1 hour at 15m and 16 hours at 4h. A 0.5% BTC move in one hour is a real move (rare → heavy HOLD class imbalance). A 0.5% move in 16 hours is noise (almost always hit in both directions → almost no HOLD, and the label becomes near-random).

**Fix:** define the barrier in volatility units — `±k × ATR(14)` with `k ≈ 1.0`, horizon 4 candles. This is self-scaling and makes 15m/1h/4h genuinely comparable. Keep the fixed ±0.5% version as a sensitivity check in the notebook and report both. Print the class balance per timeframe before training anything — if HOLD is >85% or <5%, the target is wrong for that timeframe.

### 2.3 Both barriers touched inside the same candle

You correctly flagged this. With OHLCV you **cannot** know whether the high or the low came first inside a bar. Do not guess.

**Rule:** if both barriers are breached within the same candle, label that sample `HOLD` and exclude it from training (record how many). Anywhere else — including the backtester's stop-loss/take-profit check — resolve **adversely**: assume the stop was hit first. Systematically pessimistic beats systematically optimistic.

### 2.4 Overlapping labels inflate every validation score

Sample `t` and sample `t+1` share 3 of their 4 future candles. Their labels are strongly autocorrelated, so a train/test boundary leaks information across the seam and cross-validation looks better than reality.

**Fix (cheap):** embargo — drop the 4 (or `horizon`) samples immediately either side of every split boundary. That is a `df.iloc[:-4]` on train and `df.iloc[4:]` on test. Ten minutes of work, removes a real bias, and it is a genuinely impressive thing to explain in a presentation.

### 2.5 Selection bias from the leaderboard

6 strategies × 3 timeframes × a parameter grid = hundreds of configurations. Ranking them and picking the top one *guarantees* the winner is partly lucky.

**Fix:** three-way chronological split, and the test set is touched **exactly once**, at the end.

| Split | Period | Used for |
|---|---|---|
| Train | 2020-01 → 2023-12 | ML model fitting |
| Validation | 2024-01 → 2025-06 | Strategy selection, parameter grid, ML thresholds |
| **Test** | 2025-07 → present | **Final A/B/C comparison only. Run once.** |

Then make the overfitting itself a result: plot each config's validation score against its test score. The gap is your slide on why backtests lie. Almost no bootcamp project shows this.

### 2.6 The A vs B comparison must isolate one variable

"Strategy → ML → trade" is ambiguous. If ML generates its own signals, System B and System A are trading different opportunity sets and the comparison means nothing.

**Fix — ML is a filter, not a signal source:**

- **A:** take every signal the rule-based strategy emits.
- **B:** take the *same* signals, but only when the ML model agrees and `P(class) ≥ threshold`.
- **C:** the same signals, ML prediction attached, and the LLM makes the final call.

Same universe, one variable changed per step. Note the mechanical consequence in your write-up: B and C will always have fewer trades than A, which mechanically raises win rate and widens confidence intervals. Report trade counts next to every metric.

### 2.7 The LLM has read the training data

An LLM has memorised a great deal of Bitcoin price history. If your prompt contains a date, or an absolute price like `$118,320`, it may partially recall what happened next. That is look-ahead bias via model weights, and it is the leakage vector nobody in the room will have thought of.

**Fix:** the judge prompt gets **no timestamps and no absolute prices.** Express everything relatively — RSI value, `% distance from EMA200`, `ATR as % of price`, MACD histogram sign, ML class + probability, current position, proposed stop and target as `%` distances. State this decision explicitly in `docs/llm-judge.md`. It is one of the strongest slides in the deck.

### 2.8 The composite Strategy Score

Your proposed weights double-count return (30% Total Return + 25% Sharpe, and Sharpe already contains return) and will rank-flip depending on normalisation.

**Fix:** use the composite for the *leaderboard display only*. **Select** on a single primary metric — **Sortino on validation** — behind hard eligibility gates:

- `n_trades ≥ 30` on the validation period (below this, nothing is measurable)
- `max_drawdown ≤ 40%`
- `profit_factor > 1.0`

Anything failing a gate is disqualified regardless of score. This kills the "3 lucky trades, 900% return" winner. For the display score, normalise by **percentile rank within the cohort**, not min-max — one outlier destroys min-max scaling.

### 2.9 Execution realism

- Signal is computed on the **close** of bar `t`; entry fills at the **open** of bar `t+1`, plus slippage. Never fill at the signal bar's close.
- Fees: Bybit linear perp taker ≈ **0.055%** per side → **~0.11% round trip**. On a 15m strategy targeting 0.5%, fees consume ~20%+ of gross edge. There is a real chance 15m is unprofitable *purely* because of costs — that is a legitimate, publishable finding, so make sure the backtester can show it.
- Slippage: fixed 1–2 bps plus half-spread is fine for the MVP. Make it a config value and run one sensitivity check at 3× slippage.
- Funding: perps pay/receive funding every 8h. Ignoring it is acceptable for the MVP — list it in `limitations.md`.

---

## 3. Answers to your six questions

### Q1 — Starting capital: **yes, standardise at 10,000 USDT**

A Bybit demo account is initialised with **50,000 USDT** (plus 50,000 USDC, 1 BTC, 1 ETH), and you can only top up via `POST /v5/account/demo-apply-money` when equity drops below 10,000 USDT.

Use `10_000` as `INITIAL_CAPITAL` in every backtest — fixed capital is required for a fair comparison, since position sizing is a % of equity and different starting balances produce different compounding paths. Then **report every metric in percentage terms** (return %, drawdown %, P&L %) so the backtest and the 50k demo account remain directly comparable. Make it one config key, not a literal scattered through the code.

### Q2 — Historical period: **2020-01 → present, with a small dev slice**

Not too large. 15m from 2020 is ~230k rows — trivial for pandas, well under a second per indicator pass. The cost is not data volume, it is the **parameter grid × the bar-by-bar backtest loop**.

- **Development:** work against **2024 only** so each iteration is seconds, not minutes.
- **Final run:** full 2020→present with the split in §2.5.
- **Engineering note:** write the backtest loop over **NumPy arrays**, not `df.iloc[i]`. That is a 20–50× speedup and is the difference between a 40-minute grid search and a 90-second one. This single decision de-risks day 2.
- Fetch from **Bybit's own kline endpoint** (same venue as execution, 1000 candles/request, ~230 requests for the 15m series) and **cache to Parquet**. Never re-download.

### Q3 — Models: **yes, Logistic Regression + Random Forest is enough**

They demonstrate the two things that matter: a linear, fully interpretable baseline and a non-linear ensemble with feature importances. Together with proper chronological validation, that is a complete supervised-learning story.

Three additions, all cheap and all worth more than XGBoost would be:

1. **A `DummyClassifier(strategy="most_frequent")` baseline.** Without it you cannot claim your model learned anything — on an imbalanced 3-class target, 70% accuracy may be *worse* than always predicting HOLD. This is the single most common failure in bootcamp ML projects.
2. **`class_weight="balanced"`** on both models.
3. **Probability calibration awareness.** Random Forest probabilities are not calibrated, so a `p > 0.6` threshold does not mean 60% confidence. Either wrap in `CalibratedClassifierCV` (10 minutes) or state the caveat.

XGBoost is a 20-minute add if everything else is done. Deep learning: no — you don't have the data volume or the time, and it would weaken the project by adding an unjustified component.

### Q4 — WhatsApp: **keep it Advanced #1, but scope it down to outbound-only**

Split it in two, because the two halves have wildly different costs:

- **Outbound alerts** (trade opened/closed + reasoning): Twilio WhatsApp sandbox, one API call, no webhook, **~1 hour**. High demo value — a message arriving on your phone mid-presentation is memorable.
- **Inbound conversational agent** (status, why did you buy, pause/resume): needs a public webhook (ngrok), a receiving server, session state, and an LLM tool-calling layer. **~4–6 hours** and it is the classic thing that breaks live on stage.

**Do outbound on day 3.** Do inbound only if the dashboard and the A/B/C comparison are both finished and verified. Note: Meta's WhatsApp Cloud API needs business verification you will not get in 3 days — Twilio's sandbox (join-code, works in minutes) is the only realistic route.

### Q5 — LLM experiment: **one provider, and don't backtest it the naive way**

**One LLM is correct.** Comparing providers answers "which vendor is better," which is a procurement question, not a research question. Your research question is "does an LLM judgement layer add value over a deterministic one" — that needs one LLM and good controls, not many LLMs.

**The cost problem is smaller than you think, if you call the LLM at the right moments.** Do not call it per candle. Call it **only at candles where System B would have traded** — that is typically 100–400 events over the test period, not 230,000.

Practical setup:

- Cheap, fast model tier; `temperature=0` for reproducibility.
- **Cache on a hash of the prompt payload.** Re-runs of the backtest then cost nothing and are instant.
- **Persist every prompt + response to SQLite.** The backtest becomes fully replayable offline — which also means your live demo cannot fail because of an API outage.
- Async with ~10 concurrent requests: a few hundred calls finish in a couple of minutes.
- Realistic cost at this volume: cents to low single-digit dollars.
- Structured output via a Pydantic schema (`decision`, `confidence`, `reason`, `risk`) with one retry on validation failure. Never regex free text.

**The control that makes this scientifically meaningful.** An LLM judge that vetoes some trades will change performance whether or not it has any insight — filtering alone changes the return distribution. So add a **deterministic judge** as a fourth arm: same inputs, trade only when strategy signal and ML prediction agree above threshold. Then:

- If **C ≈ deterministic judge**, the LLM added nothing beyond a rule you could have written in 4 lines. That is a real, honest, defensible finding.
- If **C > deterministic judge**, you have evidence of something more interesting, and the logged reasoning lets you inspect *why*.

Also log the LLM's **agreement rate** with the strategy signal. If it agrees 98% of the time it is a rubber stamp; if it agrees 50% of the time it is effectively a coin flip. That number alone is a great slide.

### Q6 — Safety: **yes, enforce it structurally, not by convention**

The strongest safeguard is structural: **the production URL never appears anywhere in the codebase.** If `api.bybit.com` is not in the source, it cannot be reached by a typo or a stray env var.

Layered controls:

1. `BYBIT_BASE_URL` is a **module-level constant** `https://api-demo.bybit.com`, not an env var. Env vars are for secrets, not for safety-critical switches.
2. Startup assertion: `TRADING_MODE` must equal `demo` or the process raises and exits before any client is constructed.
3. A hard assert inside the order-placement function that re-checks the base URL immediately before every send — cheap, and it survives refactors.
4. Generate the API keys **from the Bybit demo account**, which are separate credentials that do not authenticate against production at all. Two independent layers must both fail for real money to be at risk.
5. `.env` in `.gitignore` from the first commit; `.env.example` with placeholder values only; add a pre-commit grep for key-shaped strings if there is time.
6. Log `TRADING MODE: DEMO` loudly at startup, and display it as a badge in the Streamlit header — good for the demo, good for you.
7. Never implement a `live` code path "for later." No dead code that could be switched on.

---

## 4. Must / Should / Nice

### Must Have — this is the project

- Bybit OHLCV loader, 15m/1h/4h, 2020→present, Parquet cache
- ~10 hand-written indicators (see §5), all verified causal
- 4 strategies across 4 methodologies (trend / momentum / mean-reversion / breakout)
- Vectorised backtest engine: fees, slippage, `t+1` open fills, ATR stop, R:R take-profit, long + short
- Metrics module + bootstrap confidence intervals + eligibility gates
- Strategy × timeframe leaderboard
- Triple-barrier ATR-scaled 3-class target with the same-candle rule
- Chronological train/val/test with embargo; scaler fit on train only
- Dummy + Logistic Regression + Random Forest, with confusion matrices
- LLM judge: structured output, no dates/absolute prices, SQLite-cached
- **A vs B vs C vs deterministic-judge comparison on the untouched test set**
- SQLite persistence
- Streamlit dashboard
- Bybit demo executor with the §Q6 safeguards, running live
- README + methodology + limitations docs

### Should Have

- 5th and 6th strategies
- Small grid search (≤ 60 configs per strategy) on validation only
- WhatsApp outbound alerts
- Probability calibration
- Validation-vs-test overfitting scatter plot
- Unit tests on the backtester (a known synthetic price series with a hand-computed expected P&L) and on the labeller

### Nice to Have — expect to cut all of these

Market regime detection · Fear & Greed · funding rate / open interest features · leverage experiments · XGBoost · walk-forward validation · Optuna · WhatsApp inbound commands · dynamic strategy selection · multi-LLM comparison · news sentiment

---

## 5. Two engineering decisions that de-risk day 1

**Write the indicators yourself — do not use pandas-ta or TA-Lib.**
pandas-ta has NumPy 2 incompatibilities and the upstream project is at risk of archival; TA-Lib needs a C library build that regularly eats an afternoon on macOS. EMA, SMA, RSI, MACD, Bollinger, ATR, ADX, VWAP, StochRSI and OBV are about 150 lines of pandas total. You get zero dependency risk, you can prove every indicator is causal, and — for a bootcamp defence — being able to explain your own RSI implementation is worth more than importing one. Validate them once against a couple of TradingView values.

**Run the trading loop as a separate process from Streamlit.**
`main.py` polls, decides, executes and writes to SQLite. Streamlit only ever reads. Putting the loop inside Streamlit means it restarts on every widget interaction. Enable SQLite WAL mode so the reader never blocks the writer.

---

## 6. Implementation order

The one deviation from your proposed order: **the live paper trader moves earlier**, because it needs wall-clock time to accumulate anything worth showing.

| Block | Work | Est. |
|---|---|---|
| **Day 1 AM** | Repo skeleton, config, data loader + Parquet cache, indicators + causality tests | 3h |
| **Day 1 PM** | Strategy base class + 4 strategies; backtest engine (NumPy loop); metrics + bootstrap CI | 5h |
| **Day 1 late** | Leaderboard across strategies × timeframes → **System A result exists** | 1h |
| **Day 2 AM** | Feature engineering, triple-barrier labeller, splits + embargo, Dummy/LogReg/RF, ML-as-filter → **System B result** | 4h |
| **Day 2 midday** | SQLite schema + repository; Bybit demo executor + safeguards; **start the live paper trader running** | 3h |
| **Day 2 PM** | LLM judge: schema, prompt, cache, event-triggered backtest; deterministic judge control → **System C result** | 3h |
| **Day 2 late** | **Final test-set run, once.** Freeze results to the DB. | 1h |
| **Day 3 AM** | Streamlit dashboard (leaderboard, equity curves, A/B/C, ML panel, LLM reasoning, live trades) | 4h |
| **Day 3 midday** | WhatsApp outbound alerts; grid search if time | 2h |
| **Day 3 PM** | Docs, README, presentation narrative, dry run | 3h |

Checkpoint rule: **if System A is not producing a leaderboard by the end of day 1, cut two strategies and one timeframe immediately.**

---

## 7. Where this will actually go wrong

Ranked by expected pain:

1. **The backtest engine.** Long *and* short, with ATR stops and R:R targets, correct `t+1` fills, and correct same-candle stop/target resolution — this is the single hardest correctness problem in the project, and every downstream number depends on it. Budget more than you think. Write the synthetic-series unit test *before* trusting any result. A silently wrong backtester produces beautiful, meaningless slides.
2. **The labeller.** Off-by-one errors in the future window are invisible and catastrophic — a leaked label produces 95% accuracy and a fantasy equity curve. If ML accuracy comes out suspiciously high, assume leakage first, not brilliance.
3. **Grid search runtime.** Mitigated entirely by the NumPy-loop decision above. Cap the grid; a 60-config grid that finishes is worth more than a 5,000-config grid that doesn't.
4. **Bybit demo API auth.** V5 signing (HMAC over `timestamp + api_key + recv_window + body`) is fiddly and fails opaquely. Budget an hour, test with a balance query before attempting an order.
5. **Streamlit polish.** Consumes unlimited time. Timebox to 4 hours; it needs to be *legible*, not beautiful.
6. **WhatsApp inbound.** Highest ratio of demo-risk to research value. Outbound only.

---

## 8. Cut order, if you fall behind

Cut strictly in this order, no negotiation:

1. WhatsApp inbound → then outbound entirely
2. Grid search (ship hand-picked parameters; state it as a limitation)
3. Strategies 5 and 6
4. The 15m timeframe (highest data volume, most fee-sensitive, least likely to win)
5. Dashboard polish → static matplotlib charts
6. Live Bybit trading → executor tested against the demo API but not left running

**Never cut:** the untouched test set, the leakage controls, the bootstrap CIs, the Dummy baseline, or the deterministic-judge control. Those four are what make it an experiment instead of a demo, and they cost roughly 90 minutes in total.

---

## 9. Open questions for you

1. **Presentation slot** — how long, and is a live demo expected, or is a recorded run acceptable as a fallback?
2. **OpenAI API key** — do you already have one with credit? (Blocks Day 2 PM.)
3. **Bybit demo account** — created, with API keys generated from the demo environment?
4. **Twilio account** — do you have one? If not, WhatsApp drops below the dashboard in priority.
5. **Bootcamp rubric** — is there a marking scheme? If it awards points for specific techniques (e.g. unsupervised learning), that changes what's worth building. Market-regime detection via KMeans is ~45 minutes and would tick that box if it's on the rubric.
6. **Timeframe for live trading** — I'd suggest 1h: enough candles to show activity in 24h, low enough noise and fee drag to be defensible.

---

## 10. Final architecture

Only two changes from your proposed tree: `optimization/` is folded into `backtesting/` (one file, `grid.py`, and it may get cut), and `agents/` gains `deterministic_judge.py` as the experimental control.

```
bitcoin-ai-trader/
├── config/settings.py          # every tunable, one place
├── src/
│   ├── data/loader.py          # Bybit klines + Parquet cache
│   ├── data/bybit_client.py    # demo-only, signed V5 client
│   ├── indicators/indicators.py# hand-written, causal
│   ├── strategies/             # base.py + trend/momentum/mean_reversion/breakout
│   ├── backtesting/engine.py   # NumPy loop, fees, slippage, SL/TP, long+short
│   ├── backtesting/metrics.py  # metrics + bootstrap CIs + gates
│   ├── backtesting/grid.py     # optional
│   ├── models/features.py      # feature matrix
│   ├── models/labels.py        # triple-barrier labeller
│   ├── models/train.py         # splits, embargo, dummy/logreg/rf
│   ├── models/predict.py
│   ├── risk/manager.py         # sizing, ATR stop, R:R, daily loss limit
│   ├── agents/trading_judge.py # LLM, structured output, cached
│   ├── agents/deterministic_judge.py  # control arm
│   ├── exchange/executor.py    # demo-only, hard-asserted
│   ├── messaging/whatsapp.py   # outbound only
│   └── database/repository.py  # SQLite, WAL
├── dashboard/app.py            # read-only
├── notebooks/                  # exploration, ML experiments, results
├── tests/                      # backtester + labeller correctness
├── docs/
├── main.py                     # live loop, separate process
└── .env.example
```

**Data flow (unchanged from your design — the separation of concerns is correct):**

```
OHLCV → Indicators → Strategy Engine → ML Filter → Judge (LLM | deterministic)
      → DETERMINISTIC RISK MANAGER → Demo Execution → SQLite → Dashboard / WhatsApp
```

Keeping risk management deterministic and downstream of the LLM is the right call and worth stating explicitly in the presentation: the LLM chooses *direction*, never *exposure*.

---

**Awaiting approval before any implementation begins.**
