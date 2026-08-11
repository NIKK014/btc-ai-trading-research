# Architecture

```
OHLCV (Bybit public API)
  └─> Indicators           hand-written, provably causal
       └─> Strategy engine  Q1: which methodology and timeframe?
            └─> ML filter   Q2: does the model agree?
                 └─> Judge  Q3: LLM, or deterministic control
                      └─> DETERMINISTIC RISK MANAGER
                           └─> Executor ──> Broker (simulated | Bybit)
                                └─> SQLite ──> Streamlit dashboard
```

**The judge chooses direction. It never chooses exposure.** Position size, stop
placement, reward:risk and the daily loss limit are arithmetic, computed
downstream of every decision layer. Nothing that can hallucinate is allowed to
size a position.

---

## Module map

| Path | Responsibility |
|---|---|
| `config/settings.py` | Every tunable, in one place. Nothing downstream hardcodes a fee, period or threshold |
| `src/data/public_client.py` | Read-only market data. No signing code, no credentials, no POST method |
| `src/data/loader.py` | Paginated fetch, Parquet cache, integrity validation |
| `src/indicators/indicators.py` | 18 indicators in pandas. No `pandas-ta`, no TA-Lib |
| `src/strategies/` | Base class + trend / momentum / mean-reversion / breakout + benchmark |
| `src/backtesting/engine.py` | Event-driven backtest: fees, slippage, stops, long + short |
| `src/backtesting/metrics.py` | Metrics, eligibility gates, bootstrap intervals |
| `src/backtesting/optimizer.py` | Capped parameter search with selection-bias reporting |
| `src/backtesting/runner.py` | Split discipline — the only source of date ranges |
| `src/models/labels.py` | Triple-barrier labelling |
| `src/models/features.py` | Scale-free feature matrix |
| `src/models/train.py` | Chronological splits, embargo, Dummy / LogReg / RF |
| `src/models/predict.py` | The ML filter, plus shared entry-gating |
| `src/agents/schema.py` | Pydantic input/output contracts for the judge |
| `src/agents/trading_judge.py` | LLM judge: prompt, cache, concurrency |
| `src/agents/deterministic_judge.py` | Control arms |
| `src/agents/harness.py` | Runs any judge through one identical code path |
| `src/risk/manager.py` | Position sizing and the checks that refuse a trade |
| `src/exchange/bybit_client.py` | Signed V5 client, hardwired to paper endpoints |
| `src/exchange/simulated_broker.py` | Local broker, real prices, backtester-identical rules |
| `src/exchange/executor.py` | Turns approved decisions into orders; logs everything |
| `src/database/repository.py` | SQLite persistence |
| `dashboard/` | Read-only Streamlit app |
| `main.py` | Live loop, separate process |

---

## Design decisions

### Indicators are hand-written

`pandas-ta` has NumPy 2.x incompatibilities and an uncertain maintenance
future; TA-Lib needs a C toolchain build that regularly fails on macOS. More
importantly, every indicator is a potential source of look-ahead bias, and a
from-scratch implementation can be **proved** causal: a test recomputes each
one on a truncated series and asserts historical values are unchanged.

Cost: ~150 lines of pandas. Benefit: zero dependency risk, full auditability,
and the ability to explain every calculation under questioning.

### The backtest loop is explicit

Written over NumPy arrays rather than `df.iloc`, which is a 20–50× speedup —
651 configurations evaluate in 23 seconds. The loop itself is deliberately
readable rather than clever: every number in the study flows through it, and a
silently wrong engine produces beautiful, meaningless results.

### The trading loop is a separate process

`main.py` writes to SQLite; the dashboard only reads. Putting the loop inside
Streamlit would restart it on every widget interaction. SQLite runs in WAL
mode, negotiated at connection time with a fallback for filesystems that do not
support it, so a reader never blocks the writer.

### Execution is behind an interface

`BybitPaperClient` and `SimulatedBroker` expose the same surface — a test
asserts it — so `PaperExecutor` cannot tell them apart. This was not
speculative generality: it is what allowed the project to survive losing
exchange access mid-build (see below).

### Every decision is persisted

Including the ones that produced no trade. A log containing only trades cannot
answer "why was it flat all afternoon?", which is the first question anyone
asks watching a live demo.

---

## Safety: why real-money trading is impossible

Four independent layers. Any one of them alone prevents a real-money trade.

1. **The production trading host does not exist in this codebase.** It cannot
   be reached by a typo. A test scans every module and fails if it appears
   outside the read-only market-data client.
2. **The paper host is a module-level constant, not an environment variable.**
   A malformed `.env` cannot redirect order flow.
3. **`assert_paper_mode()` runs in the client constructor.** The client cannot
   be built unless `TRADING_MODE` is a paper mode.
4. **Every request re-checks the base URL immediately before sending.** Cheap,
   and it survives refactors.

Plus: the market-data client has no signing code, no credentials and no POST
method — verified by test — so it cannot place an order even if called with
credentials. And `.env` is gitignored from the first commit.

This was demonstrated in practice: a mainnet API key was rejected with a 401
against the paper endpoint. The safeguard working, with a real error message
rather than a claim in a README.

---

## The venue problem

Midway through the build, the intended execution venue became unavailable.

**Bybit EU offers no perpetual futures.** Under MiCA, EEA users are moved to
`bybit.eu`, which lists spot and margin only pending a MiFID II licence. Its
API is additionally restricted to registered API-broker integrations, so an
ordinary EEA account cannot trade programmatically at all. Bybit's global
testnet geo-redirects EEA users to the EU site, closing that route too.

Dropping short trades was not an option: the test period is a 41% decline, and
a long-only system would have invalidated every backtest result.

**The architecture absorbed it.** Execution moved to a second implementation of
the same interface. The strategy, model, judge, risk manager and executor are
unchanged; `--broker bybit` still selects the exchange client, which remains
written and tested.

The simulator fills against **real** BTC prices from the public feed, using the
same fee, slippage and stop rules as the backtester — including replaying
candle highs and lows to settle stops breached *between* polls, since price
moves continuously while the loop sleeps. Its P&L is therefore directly
comparable to the research results, which testnet's thin and divergent order
book would not have been.

What it is not: no order reaches a venue, and no queue position, partial fill
or liquidation is modelled.

---

## Technology choices

| Choice | Reason |
|---|---|
| pandas + NumPy | Sufficient at this data scale; 223k rows process in milliseconds |
| scikit-learn | Dummy / LogReg / RF cover the supervised story without inviting overfitting |
| Pydantic | Validates LLM output instead of regex-parsing prose |
| SQLite | Six tables that never change shape; an ORM would add a dependency and indirection |
| Streamlit | Fastest path to a presentable read-only dashboard |
| Parquet | Columnar cache; 223k candles in 10 MB |
| **No Docker** | Not covered in the bootcamp, and unnecessary for a single-machine project |
| **No RAG / vector DB** | Nothing in the problem calls for retrieval. Adding one to claim the buzzword would be dishonest |
| **No deep learning** | 13,580 samples on 4h. An LSTM would overfit and add no explanatory value |
