# Publishing the dashboard

Puts the dashboard on a public URL via Streamlit Community Cloud (free). About
fifteen minutes, most of it waiting for the first build.

The published copy is **read-only research**. It has no API keys, no trading
mode and no exchange connection, so it cannot place an order of any kind. The
paper trader keeps running on your machine and is not part of the deployment.

---

## 0. Commit what you have

Nothing below works until the work is in Git.

```bash
cd ~/Documents/Ironhack/Final_Project
git status                    # expect ~35 modified/new files
git add -A
git commit -m "Simplify dashboard to four tabs, make it deployable"
```

## 1. Create the GitHub repository

There is no remote configured yet, so this project exists only on your laptop.

Create an **empty** repo at <https://github.com/new> — name it
`btc-ai-trading-research`, no README, no .gitignore, no licence. Then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/btc-ai-trading-research.git
git branch -M main
git push -u origin main
```

If it asks for a password, GitHub wants a personal access token, not your
account password: <https://github.com/settings/tokens> → generate a classic
token with `repo` scope, paste that.

### Before you push — confirm no secrets are going up

```bash
git ls-files | grep -E "\.env$|\.key$|\.pem$"     # must print NOTHING
```

`.gitignore` already excludes `.env`, `*.key`, `*.pem`, the price cache and the
trading database. `.env.example` is committed on purpose: it lists the variable
names with no values.

## 2. Deploy

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `YOUR_USERNAME/btc-ai-trading-research`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
4. **Deploy**.

Leave *Secrets* empty. The dashboard reads saved CSVs and the public Bybit
price API; it never calls OpenAI and never authenticates to anything.

First build takes 3–5 minutes while it installs pandas, scikit-learn and
plotly. Later pushes to `main` redeploy automatically in under a minute.

## 3. Check it

Open the URL and confirm:

- [ ] Four tabs: **Live**, **Results**, **Machine learning**, **AI Judge**
- [ ] Green banner: *"this published copy has no trading mode configured"*
- [ ] **Live** shows a Bitcoin candlestick chart and a note explaining that the
      trader runs locally
- [ ] **Results** shows the out-of-sample table — System B +1.6%, buy-and-hold
      −41.0%
- [ ] **AI Judge** shows the 127 logged decisions

If the Live chart is empty, Bybit's public API was slow or is geoblocked from
the container. The other three tabs are files on disk and are unaffected —
which is why the research does not depend on that request.

---

## What the deployed copy does and does not have

| | Local (your Mac) | Published |
|---|---|---|
| Research results (CSV) | yes | yes |
| LLM decision log | yes | yes |
| BTC chart | parquet cache | fetched live from Bybit |
| Live positions and equity | yes | no — no database |
| Paper trading loop | running | never runs |
| API keys | in `.env` | none |

The split is deliberate. The price cache is 14 MB of raw candles and the
trading database changes every few minutes; neither belongs in version control.
`has_local_trading_data()` in `dashboard/data_access.py` detects which copy is
running by looking for the parquet cache, and the live panels replace
themselves with an explanation rather than rendering empty charts.

## If the build fails

Read the log in the Streamlit Cloud panel — the error is almost always in the
first traceback.

- **`ModuleNotFoundError`** — the package is missing from `requirements.txt`.
- **Build times out or the app restarts repeatedly** — usually memory. The free
  tier gives about 1 GB; this app stays well inside it because it reads saved
  results instead of retraining anything.
- **`FileNotFoundError` on a parquet file** — something is calling `load_ohlcv`
  without checking for the cache first. That path downloads six years of
  candles and will hang the page; guard it with `cache_path(...).exists()` the
  way `load_recent_prices` does.

## Verified before writing this

The deployment was rehearsed by copying only Git-tracked files into a clean
directory — no parquet, no database, no `.env` — and rendering the app there
with Streamlit's `AppTest`. It came up with no exception, all four tabs
present, and the live panels correctly replaced by their explanations.
