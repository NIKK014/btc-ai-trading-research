# Bitcoin AI Trading Research System

An AI-engineering research project that asks three questions and tries to answer them honestly:

1. Which combination of **trading methodology, technical indicators and timeframe** produces the best risk-adjusted BTC day-trading performance?
2. Does adding a **machine-learning filter** improve it?
3. Does adding an **LLM trading judge** improve it further — or does it just match a rule you could write in four lines?

The winning system then paper-trades BTCUSDT perpetuals on **Bybit Demo Trading**.

> **This project cannot trade real money.** All execution is restricted to Bybit Demo Trading. See [Safety](#safety).

---

## Status

| Phase | State |
|---|---|
| Scope review & architecture | Done — [`docs/00-scope-review.md`](docs/00-scope-review.md) |
| Data loader + Parquet cache | Done |
| Indicator library + causality tests | Done |
| Strategies | Next |
| Backtest engine | Pending |
| ML models | Pending |
| LLM judge | Pending |
| Dashboard, live demo trading | Pending |

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in your keys
python scripts/fetch_data.py    # downloads and caches BTCUSDT 15m/1h/4h
pytest -q                       # 36 tests, all offline
```

`fetch_data.py` downloads from 2020-01-01 to now and caches to `data/raw/*.parquet`.
Re-run with `--refresh` to pull in newly closed candles; it never re-downloads history it already has.

---

## Experimental design

The point of the project is the comparison, not the profit. Three systems trade the **same signal universe**, changing exactly one thing at a time:

| System | Pipeline |
|---|---|
| **A — Traditional** | Indicators → strategy rules → trade |
| **B — ML enhanced** | A's signals, taken only when the ML model agrees above a probability threshold |
| **C — AI judge** | B's inputs, with an LLM making the final LONG/SHORT/HOLD call |
| **Control** | B's inputs, with a *deterministic* judge — trade only when strategy and ML agree |

The control arm matters: an LLM that vetoes trades will change performance whether or not it has any insight. Without a deterministic judge to compare against, "the LLM helped" is not a claim you can defend.

### How data leakage is prevented

| Risk | Control |
|---|---|
| Indicators peeking forward | Every indicator hand-written and proven causal by automated test |
| Ichimoku Chikou Span (close shifted *backwards*) | Indicator excluded entirely |
| Fibonacci / swing-based support-resistance | Excluded; replaced by Donchian channels of the *previous* N bars |
| Shuffled time series | Never shuffled — chronological splits only |
| Overlapping labels leaking across split seams | 4-bar embargo either side of every boundary |
| Scaler fitted on the full dataset | Fitted on train only |
| Test set used for tuning | Test period touched exactly once, at the end |
| Acting on an unclosed candle | Loader drops the still-forming final candle |
| **The LLM having memorised BTC history** | Prompt contains no timestamps and no absolute prices — only relative, scale-free values |

That last one is the subtle one. An LLM asked about BTC at "$118,320 on 3 March" may partly recall what happened next. Every value the judge sees is relative: RSI, % distance from EMA, ATR as % of price, ML probability.

### Chronological splits

| Split | Period | Purpose |
|---|---|---|
| Train | 2020-01 → 2023-12 | ML model fitting |
| Validation | 2024-01 → 2025-06 | Strategy selection, parameter search, thresholds |
| **Test** | 2025-07 → present | **Final comparison. Run once.** |

---

## Safety

Four independent layers, any one of which prevents real-money execution:

1. **The production trading host does not exist in this codebase.** It cannot be reached by a typo. A test enforces this.
2. **Order placement is hardwired to a module-level constant** (`BYBIT_DEMO_TRADE_URL`), not an environment variable — a malformed `.env` cannot redirect order flow.
3. **`TRADING_MODE` must equal `demo`** or the process raises `UnsafeConfigurationError` and exits before any client is constructed.
4. **Demo API keys are separate credentials** that do not authenticate against production at all.

Market data is fetched by a read-only client that has no signing code, no credentials and no POST method — verified by test. `.env` is gitignored from the first commit.

---

## Project layout

```
config/settings.py          Every tunable, one place
src/
  data/public_client.py     Read-only market data (no auth capability)
  data/loader.py            Fetch, Parquet cache, integrity validation
  indicators/indicators.py  Hand-written, provably causal
  strategies/               Trend / momentum / mean-reversion / breakout
  backtesting/              Engine (fees, slippage, SL/TP) + metrics
  models/                   Features, triple-barrier labels, training
  risk/manager.py           Deterministic sizing and limits
  agents/                   LLM judge + deterministic control
  exchange/                 Demo-only executor
  database/                 SQLite persistence
dashboard/app.py            Streamlit (read-only)
scripts/fetch_data.py       Data acquisition CLI
tests/                      Causality, correctness, safety
docs/                       Methodology and results
```

### Why no `pandas-ta` or TA-Lib

`pandas-ta` has NumPy 2.x incompatibilities and an uncertain maintenance future; TA-Lib needs a C toolchain build. Every indicator here is ~10 lines of pandas, has zero dependency risk, and — more importantly — can be *proved* causal. `tests/test_indicators.py` recomputes every indicator on a truncated series and asserts that historical values are unchanged, which makes look-ahead bias impossible to introduce without a red build.

### Data conventions

Candle timestamps are **UTC open times**. A `12:00` candle on the 1h timeframe covers 12:00–12:59 and is only complete at 13:00. Gaps from exchange downtime are left unfilled — interpolating candles would invent price action that never happened.

---

## Limitations

Recorded honestly, and expanded as the project progresses:

- Backtests ignore perpetual funding payments.
- Live paper trading runs for hours, not months — it demonstrates that the loop works, it is not evidence that the strategy works.
- Sample sizes are small enough that most differences between systems will fall inside their confidence intervals. Metrics are reported with bootstrap CIs for exactly this reason.
- Selecting the best of many configurations guarantees the winner is partly lucky; the validation-vs-test degradation is reported rather than hidden.

**This is a research and engineering project. It is not investment advice, and it does not claim to be a profitable trading system.**
