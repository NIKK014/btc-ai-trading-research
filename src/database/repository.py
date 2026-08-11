"""SQLite persistence.

Deliberately plain SQL over a small schema. An ORM would add a dependency and
a layer of indirection for six tables that never change shape.

Concurrency
-----------
The live trading loop writes; the Streamlit dashboard reads. WAL mode is
enabled so a reader never blocks the writer and the dashboard cannot stall the
trader mid-order.

Everything is stored
--------------------
Every decision - including the ones that resulted in no trade - is recorded
with the full inputs that produced it. Logging only the trades would make it
impossible to answer "why did it stay flat all Tuesday?", which is exactly the
question that gets asked in a presentation.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from config.settings import PATHS
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id       TEXT,
    symbol         TEXT    NOT NULL,
    timeframe      TEXT    NOT NULL,
    strategy       TEXT    NOT NULL,
    system         TEXT    NOT NULL,
    direction      INTEGER NOT NULL,
    entry_time     TEXT    NOT NULL,
    exit_time      TEXT,
    entry_price    REAL    NOT NULL,
    exit_price     REAL,
    size           REAL    NOT NULL,
    stop_price     REAL,
    target_price   REAL,
    fees           REAL    DEFAULT 0,
    pnl            REAL,
    return_pct     REAL,
    exit_reason    TEXT,
    mode           TEXT    NOT NULL DEFAULT 'demo',
    created_at     TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    timeframe         TEXT    NOT NULL,
    strategy          TEXT    NOT NULL,
    strategy_signal   INTEGER NOT NULL,
    ml_prediction     INTEGER,
    ml_confidence     REAL,
    judge_decision    TEXT,
    judge_confidence  INTEGER,
    judge_reason      TEXT,
    risk_assessment   TEXT,
    final_action      INTEGER NOT NULL,
    blocked_reason    TEXT,
    indicators        TEXT,
    model             TEXT,
    created_at        TEXT    DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    equity        REAL    NOT NULL,
    unrealised    REAL    DEFAULT 0,
    position      INTEGER DEFAULT 0,
    price         REAL,
    UNIQUE(timestamp)
);

CREATE TABLE IF NOT EXISTS backtests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT    DEFAULT CURRENT_TIMESTAMP,
    system        TEXT    NOT NULL,
    strategy      TEXT    NOT NULL,
    timeframe     TEXT    NOT NULL,
    split         TEXT    NOT NULL,
    params        TEXT,
    metrics       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS system_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_entry   ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_history(timestamp);
"""


class Repository:
    """All database access for the project."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else PATHS.database
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._journal_mode: Optional[str] = None
        self._initialise()

    @staticmethod
    def _negotiate_journal_mode(connection: sqlite3.Connection) -> str:
        """Use WAL where the filesystem supports it, otherwise fall back."""
        try:
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() == "wal":
                return "wal"
        except sqlite3.OperationalError:
            pass
        connection.execute("PRAGMA journal_mode=DELETE")
        logger.warning(
            "This filesystem does not support WAL; using DELETE journal mode. "
            "The dashboard may briefly block the trading loop under heavy reads."
        )
        return "delete"

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            # WAL lets the dashboard read while the trading loop writes.
            # Some network and container-mounted filesystems do not support the
            # shared-memory locking WAL needs, so fall back rather than fail:
            # a slower journal mode is far better than a trader that will not
            # start.
            if self._journal_mode is None:
                self._journal_mode = self._negotiate_journal_mode(connection)
            elif self._journal_mode == "wal":
                connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
        logger.info("Database ready at %s", self.path)

    # -- writes ------------------------------------------------------------

    def record_trade(self, trade: Dict[str, Any]) -> int:
        """Insert an opened trade and return its row id."""
        columns = [
            "order_id", "symbol", "timeframe", "strategy", "system", "direction",
            "entry_time", "entry_price", "size", "stop_price", "target_price",
            "fees", "mode",
        ]
        values = [trade.get(column) for column in columns]
        with self.connect() as connection:
            cursor = connection.execute(
                f"INSERT INTO trades ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})",
                values,
            )
            return int(cursor.lastrowid)

    def close_trade(
        self,
        trade_id: int,
        exit_time: str,
        exit_price: float,
        pnl: float,
        return_pct: float,
        exit_reason: str,
        fees: float = 0.0,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE trades SET exit_time=?, exit_price=?, pnl=?, return_pct=?, "
                "exit_reason=?, fees=fees+? WHERE id=?",
                (exit_time, exit_price, pnl, return_pct, exit_reason, fees, trade_id),
            )

    def record_decision(self, decision: Dict[str, Any]) -> int:
        """Log a decision, including ones that led to no trade."""
        payload = dict(decision)
        indicators = payload.get("indicators")
        if isinstance(indicators, dict):
            payload["indicators"] = json.dumps(
                {k: (None if pd.isna(v) else float(v)) for k, v in indicators.items()}
            )

        columns = [
            "timestamp", "symbol", "timeframe", "strategy", "strategy_signal",
            "ml_prediction", "ml_confidence", "judge_decision", "judge_confidence",
            "judge_reason", "risk_assessment", "final_action", "blocked_reason",
            "indicators", "model",
        ]
        with self.connect() as connection:
            cursor = connection.execute(
                f"INSERT INTO decisions ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})",
                [payload.get(column) for column in columns],
            )
            return int(cursor.lastrowid)

    def record_equity(
        self,
        timestamp: str,
        equity: float,
        unrealised: float = 0.0,
        position: int = 0,
        price: Optional[float] = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO portfolio_history "
                "(timestamp, equity, unrealised, position, price) VALUES (?, ?, ?, ?, ?)",
                (timestamp, equity, unrealised, position, price),
            )

    def record_backtest(
        self,
        system: str,
        strategy: str,
        timeframe: str,
        split: str,
        metrics: Dict[str, Any],
        params: str = "",
    ) -> int:
        clean = {
            key: (None if isinstance(value, float) and pd.isna(value) else value)
            for key, value in metrics.items()
        }
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO backtests (system, strategy, timeframe, split, params, metrics) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (system, strategy, timeframe, split, params, json.dumps(clean, default=str)),
            )
            return int(cursor.lastrowid)

    def set_state(self, key: str, value: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO system_state (key, value, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=CURRENT_TIMESTAMP",
                (key, json.dumps(value)),
            )

    # -- reads -------------------------------------------------------------

    def get_state(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM system_state WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    def open_trade(self) -> Optional[Dict[str, Any]]:
        """The currently open trade, if any."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM trades WHERE exit_time IS NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def trades(self, limit: Optional[int] = None) -> pd.DataFrame:
        query = "SELECT * FROM trades ORDER BY entry_time DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        return self._query(query)

    def decisions(self, limit: Optional[int] = None) -> pd.DataFrame:
        query = "SELECT * FROM decisions ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        return self._query(query)

    def equity_curve(self) -> pd.DataFrame:
        frame = self._query("SELECT * FROM portfolio_history ORDER BY timestamp")
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.set_index("timestamp")
        return frame

    def backtest_results(self) -> pd.DataFrame:
        frame = self._query("SELECT * FROM backtests ORDER BY run_at DESC")
        if not frame.empty:
            expanded = pd.json_normalize(frame["metrics"].map(json.loads))
            frame = pd.concat([frame.drop(columns=["metrics"]), expanded], axis=1)
        return frame

    def summary(self) -> Dict[str, Any]:
        """Headline numbers for the dashboard, in one query round trip."""
        trades = self.trades()
        closed = trades[trades["exit_time"].notna()] if not trades.empty else trades

        return {
            "total_trades": int(len(trades)),
            "closed_trades": int(len(closed)),
            "open_trades": int(len(trades) - len(closed)) if not trades.empty else 0,
            "total_pnl": float(closed["pnl"].sum()) if not closed.empty else 0.0,
            "win_rate": float((closed["pnl"] > 0).mean()) if not closed.empty else float("nan"),
            "decisions_logged": int(len(self.decisions())),
        }

    def _query(self, sql: str) -> pd.DataFrame:
        with self.connect() as connection:
            return pd.read_sql_query(sql, connection)
