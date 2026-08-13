# Bitcoin AI Trading Research System

An AI-engineering research project that asks three questions and answers them
honestly.

1. Which **trading methodology, indicators and timeframe** produce the best
   risk-adjusted BTC day-trading performance?
2. Does a **machine-learning filter** improve it?
3. Does an **LLM trading judge** improve it further — or does it just reproduce
   what a four-line rule already does?

> **The system cannot trade real money.** All execution is paper trading. See
> [Safety](#safety).

---

## The answers

**Q1 — Trend-following at 4h, and only that.** Of 651 configurations across
four methodologies and three timeframes, one family survived selection. At 15
minutes, transaction costs consumed a median **65% of capital** and not one
configuration passed the eligibility gates.

**Q2 — Not measurably.** The Random Forest beat its baseline by **+6.4 points**
of balanced accuracy, but its effect on trading **reversed sign** between
validation (−35.1%) and test (+12.0%), and never reached significance.

**Q3 — No.** The LLM judge returned −0.3% against the deterministic rule's
+1.6%. It was not a rubber stamp — it vetoed 59% of proposals at moderate
confidence — but it did not beat four lines of arithmetic.

**And the headline:**

| | Validation | Test |
|---|---|---|
| Return | +34.6% | **−10.5%** |
| Sharpe | **1.92** | **−0.91** |

A Sharpe of 1.92 became −0.91 the moment the data had not been selected on.
That gap is what parameter selection bought and could not deliver — the most
honest number in the project.

Yet over the same period **BTC fell 41.0% with a 53.5% drawdown**. The system
lost 10.5% with a 14.5% drawdown; with the LLM layer, 0.3% with a 7.7%
drawdown. It lost money and still substantially outperformed holding.

Full results: [`docs/results.md`](docs/results.md)

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + notebooks and pytest
cp .env.example .env                  # add OPENAI_API_KEY for System C

python scripts/fetch_data.py  # downloads and caches BTCUSDT 15m/1h/4h
pytest -q                     # 190 tests, all offline
```

`requirements.txt` holds only what the app needs at runtime; that is the file
the deployed dashboard installs. `requirements-dev.txt` adds Jupyter,
matplotlib and pytest on top of it.

### Reproduce the study

```bash
python scripts/run_baseline.py    # System A leaderboard          (~2s)
python scripts/run_optimizer.py   # 651 configurations            (~23s)
python scripts/run_ml.py          # models + System A vs B        (~10s)
python scripts/run_llm.py         # all arms on validation        (~60s)
python scripts/run_final.py       # THE out-of-sample test        (~60s)
```

`run_final.py` reads the test period, which nothing else in the project can.
**Run it once.** Re-running after changing anything turns the test set into a
second validation set.

### Run it live

```bash
python scripts/check_setup.py                        # preflight
nohup python main.py --system C > logs/live.log 2>&1 &   # trader
streamlit run dashboard/app.py                       # dashboard
```

---

## Experimental design

Every arm trades the **same signal universe**, with exactly one thing changed:

| Arm | Pipeline |
|---|---|
| **A** | Indicators → strategy rules → trade |
| **B** | A's signals, only when the ML model agrees above a threshold |
| **C** | B's inputs, with an LLM judge making the final call |
| Control — *always agree* | Approves everything; must reproduce A exactly |
| Control — *deterministic* | Trade only when strategy and model agree |

The controls are the point. A judge that vetoes trades changes performance
whether or not it understands anything — filtering alone reshapes the return
distribution. Without a deterministic benchmark, "the LLM helped" is not a
claim you can defend.

**Every difference is reported with a bootstrap confidence interval. If the
interval contains zero, the systems are not distinguishable at this sample
size** — which for these questions is a legitimate answer.

---

## How leakage was prevented

| Risk | Control |
|---|---|
| Indicators peeking forward | Hand-written; a test recomputes each on truncated data and asserts history is unchanged |
| **Ichimoku Chikou Span** | Excluded — it is the close shifted *backwards*, so reading it at `t` reads `t+26` |
| Fibonacci / swing S-R | Excluded — hindsight-derived. Replaced by Donchian channels of the *previous* N bars |
| Shuffled time series | Never shuffled; chronological splits only |
| Overlapping labels at split seams | 4-bar embargo at every boundary |
| Scaler fitted on all data | Fitted inside the pipeline, train only |
| Test set used for tuning | `get_split("test")` raises unless explicitly unlocked |
| Acting on an unclosed candle | The loader drops the still-forming final candle |
| **The LLM having memorised BTC history** | No dates, no absolute prices in the prompt — enforced by test |
| Same-candle barrier ties | Labelled HOLD and excluded, never guessed |
| Live inference using training data | Uses the feature builder, not the label-filtered dataset — enforced by test |

The subtle one is the LLM. Told BTC is at $118,320 on 3 March, a model that has
read the internet may partially recall what happened next — look-ahead bias
travelling through model weights, invisible to every pandas-level control.
Every value the judge sees is relative: RSI, % from EMA, ATR as % of price.

Detail: [`docs/methodology.md`](docs/methodology.md)

---

## Safety

Four independent layers, any one of which prevents a real-money trade:

1. **The production trading host is absent from the codebase.** A test enforces
   it. It cannot be reached by a typo.
2. **The paper host is a module constant**, not an environment variable — a
   malformed `.env` cannot redirect order flow.
3. **`assert_paper_mode()` runs in the client constructor.**
4. **Every request re-checks the base URL** immediately before sending.

Market data is fetched by a read-only client with no signing code, no
credentials and no POST method — verified by test. `.env` is gitignored from
the first commit.

Execution runs through a local simulator that fills against **real** BTC prices
using the backtester's exact fee, slippage and stop rules. A Bybit V5 client
exists and is switchable with `--broker bybit`; Bybit EU offers no perpetual
futures under MiCA, which is why the simulator is the default. See
[`docs/architecture.md`](docs/architecture.md#the-venue-problem).

---

## Project layout

```
config/settings.py         Every tunable, one place
src/
  data/                    Read-only market client, loader, Parquet cache
  indicators/              18 hand-written, provably causal indicators
  strategies/              Trend / momentum / mean-reversion / breakout + benchmark
  backtesting/             Engine, metrics + bootstrap CIs, optimiser, split discipline
  models/                  Triple-barrier labels, features, training, ML filter
  agents/                  LLM judge, deterministic controls, comparison harness
  risk/                    Deterministic sizing and limits
  exchange/                Simulated broker + Bybit client, one interface
  database/                SQLite persistence
dashboard/                 Streamlit, read-only
scripts/                   fetch_data · run_baseline · run_optimizer · run_ml
                           run_llm · run_final · check_setup
tests/                     187 tests: causality, correctness, safety
docs/                      Methodology, results, limitations, presentation guide
main.py                    Live loop, separate process
```

### Why no `pandas-ta` or TA-Lib

`pandas-ta` has NumPy 2.x incompatibilities and an uncertain future; TA-Lib
needs a C toolchain build. More importantly, every indicator is a potential
source of look-ahead bias, and a from-scratch implementation can be **proved**
causal. ~150 lines of pandas buys zero dependency risk and full auditability.

---

## Documentation

| Document | Contents |
|---|---|
| [methodology.md](docs/methodology.md) | Experimental design, splits, leakage controls, execution assumptions |
| [results.md](docs/results.md) | Every result, with confidence intervals and regime splits |
| [strategies.md](docs/strategies.md) | The seven strategies and why each was built that way |
| [machine-learning.md](docs/machine-learning.md) | Target design, features, models, the filter |
| [llm-judge.md](docs/llm-judge.md) | Prompt design, structured output, caching, controls |
| [architecture.md](docs/architecture.md) | System design, safety layers, the venue problem |
| [limitations.md](docs/limitations.md) | Everything this project does not establish |
| [presentation.md](docs/presentation.md) | Slide plan and anticipated questions |
| [00-scope-review.md](docs/00-scope-review.md) | The original scope review |

---

## What this project does not claim

It does not claim to have found a profitable strategy — it found one that lost
10.5% out of sample. It does not claim ML improves trading — the effect
reversed sign between splits. It does not claim LLMs make good traders — one
failed to beat four lines of arithmetic.

It claims that these questions were asked in a way that could have produced the
opposite answer, and that the negative results are as reliable as the method
allows.

**Research and engineering project. Not investment advice.**
