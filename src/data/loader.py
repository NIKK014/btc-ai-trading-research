"""Historical OHLCV acquisition, caching and validation.

Design notes
------------
*Timestamps are candle OPEN times, in UTC.* A candle indexed ``12:00`` on the
1h timeframe covers ``12:00:00`` to ``12:59:59`` and is only complete at
``13:00``. Acting on an incomplete candle is a subtle but fatal source of
look-ahead bias, so :func:`fetch_ohlcv` drops the still-forming final candle
by default.

*Pagination runs backwards.* Bybit returns the newest ``limit`` candles at or
before ``end``, newest first. Walking backwards from "now" is deterministic;
forward pagination depends on undocumented behaviour when the requested range
exceeds the page size.

*Data is cached to Parquet* and updated incrementally, so an experiment
re-run never re-downloads history it already has.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from config.settings import DATA, DataConfig
from src.data.public_client import BybitPublicClient
from src.utils.logging_setup import get_logger
from src.utils.timeframes import bybit_interval, interval_ms, validate_timeframe

logger = get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "turnover"]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _to_ms(timestamp: str | pd.Timestamp) -> int:
    """Convert a date-like value to a UTC epoch in milliseconds."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)


def _rows_to_frame(rows: list[list[str]]) -> pd.DataFrame:
    """Convert Bybit's raw string rows into a typed, indexed DataFrame."""
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))

    frame = pd.DataFrame(rows, columns=["start_ms", *OHLCV_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame["start_ms"].astype("int64"), unit="ms", utc=True)
    frame = frame.drop(columns=["start_ms"]).set_index("timestamp")
    return frame[OHLCV_COLUMNS].astype("float64").sort_index()


def fetch_ohlcv(
    timeframe: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    *,
    symbol: str = DATA.symbol,
    category: str = DATA.category,
    drop_unclosed: bool = True,
    config: DataConfig = DATA,
    client: BybitPublicClient | None = None,
) -> pd.DataFrame:
    """Download OHLCV candles for ``timeframe`` between ``start`` and ``end``.

    Args:
        timeframe: Project label such as ``"15m"``, ``"1h"`` or ``"4h"``.
        start: Inclusive lower bound on candle open time.
        end: Exclusive upper bound. Defaults to now.
        drop_unclosed: Discard the final candle if it has not closed yet.

    Returns:
        DataFrame indexed by UTC candle open time with float OHLCV columns,
        sorted ascending and free of duplicates.
    """
    validate_timeframe(timeframe)
    client = client or BybitPublicClient()

    interval = bybit_interval(timeframe)
    step_ms = interval_ms(timeframe)
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end is not None else int(time.time() * 1000)

    chunks: list[pd.DataFrame] = []
    cursor = end_ms
    requests_made = 0

    logger.info(
        "Fetching %s %s from %s to %s",
        symbol,
        timeframe,
        pd.Timestamp(start_ms, unit="ms", tz="UTC"),
        pd.Timestamp(end_ms, unit="ms", tz="UTC"),
    )

    while cursor > start_ms:
        rows = client.get_kline(
            symbol=symbol,
            interval=interval,
            category=category,
            start_ms=start_ms,
            end_ms=cursor,
            limit=config.request_limit,
        )
        requests_made += 1
        if not rows:
            break

        chunk = _rows_to_frame(rows)
        chunks.append(chunk)

        oldest_ms = int(chunk.index[0].timestamp() * 1000)
        if oldest_ms <= start_ms:
            break
        # Step strictly before the oldest candle we just received.
        next_cursor = oldest_ms - step_ms
        if next_cursor >= cursor:  # defensive: never loop forever
            logger.warning("Pagination stalled at %s; stopping.", chunk.index[0])
            break
        cursor = next_cursor

        if requests_made % 25 == 0:
            logger.info("  ... %d requests, back to %s", requests_made, chunk.index[0])
        time.sleep(config.request_sleep)

    if not chunks:
        logger.warning("No data returned for %s %s", symbol, timeframe)
        return _rows_to_frame([])

    frame = pd.concat(chunks).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.loc[
        (frame.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC"))
        & (frame.index < pd.Timestamp(end_ms, unit="ms", tz="UTC"))
    ]

    if drop_unclosed:
        frame = drop_unclosed_candle(frame, timeframe)

    logger.info(
        "Fetched %d %s candles (%s -> %s) in %d requests",
        len(frame),
        timeframe,
        frame.index.min(),
        frame.index.max(),
        requests_made,
    )
    return frame


def drop_unclosed_candle(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Remove the final candle if its interval has not elapsed yet.

    The last row returned by the exchange is usually still forming. Its high,
    low and close will change, so using it is look-ahead bias in live trading
    and noise in backtests.
    """
    if frame.empty:
        return frame
    now = pd.Timestamp.now(tz="UTC")
    close_time = frame.index[-1] + pd.Timedelta(minutes=interval_ms(timeframe) / 60_000)
    if close_time > now:
        return frame.iloc[:-1]
    return frame


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def cache_path(timeframe: str, symbol: str = DATA.symbol, config: DataConfig = DATA) -> Path:
    """Parquet cache location for one symbol/timeframe series."""
    return config.cache_dir / f"{symbol}_{timeframe}.parquet"


def load_ohlcv(
    timeframe: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    symbol: str = DATA.symbol,
    refresh: bool = False,
    config: DataConfig = DATA,
) -> pd.DataFrame:
    """Return cached OHLCV, downloading only what is missing.

    This is the function every downstream module should call. It never
    re-downloads history that is already on disk.

    Args:
        timeframe: ``"15m"``, ``"1h"`` or ``"4h"``.
        start: Optional slice lower bound applied after loading.
        end: Optional slice upper bound applied after loading.
        refresh: Extend the cache forward with any newly closed candles.
    """
    validate_timeframe(timeframe)
    path = cache_path(timeframe, symbol, config)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        frame = pd.read_parquet(path)
        logger.info("Loaded %d cached %s candles from %s", len(frame), timeframe, path.name)
        if refresh:
            frame = _extend_cache(frame, timeframe, symbol, config, path)
    else:
        logger.info("No cache for %s %s; downloading full history.", symbol, timeframe)
        frame = fetch_ohlcv(timeframe, config.history_start, symbol=symbol, config=config)
        _write_cache(frame, path)

    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        frame = frame.loc[frame.index < pd.Timestamp(end, tz="UTC")]
    return frame.copy()


def _extend_cache(
    frame: pd.DataFrame,
    timeframe: str,
    symbol: str,
    config: DataConfig,
    path: Path,
) -> pd.DataFrame:
    """Append newly closed candles to an existing cached series."""
    if frame.empty:
        return frame
    last = frame.index[-1]
    fresh = fetch_ohlcv(
        timeframe,
        start=last,
        symbol=symbol,
        config=config,
    )
    if fresh.empty:
        return frame
    combined = pd.concat([frame, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    added = len(combined) - len(frame)
    if added:
        logger.info("Cache extended with %d new %s candles.", added, timeframe)
        _write_cache(combined, path)
    return combined


def _write_cache(frame: pd.DataFrame, path: Path) -> None:
    """Persist a series to Parquet."""
    if frame.empty:
        logger.warning("Refusing to write empty cache to %s", path)
        return
    frame.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.info("Wrote %d rows to %s", len(frame), path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ohlcv(frame: pd.DataFrame, timeframe: str) -> dict[str, object]:
    """Run integrity checks and return a report.

    Checks for duplicate or non-monotonic timestamps, missing candles, NaNs,
    non-positive prices and OHLC relationships that cannot be true. Logs a
    warning for each problem found rather than raising, so a run is never
    silently blocked by a single bad candle.
    """
    expected = pd.Timedelta(minutes=interval_ms(timeframe) / 60_000)
    deltas = frame.index.to_series().diff().dropna()
    gaps = deltas[deltas > expected]

    ohlc_invalid = int(
        (
            (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["high"] < frame["low"])
        ).sum()
    )

    report: dict[str, object] = {
        "timeframe": timeframe,
        "rows": len(frame),
        "start": frame.index.min() if len(frame) else None,
        "end": frame.index.max() if len(frame) else None,
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "monotonic": bool(frame.index.is_monotonic_increasing),
        "gaps": int(len(gaps)),
        "largest_gap": gaps.max() if len(gaps) else pd.Timedelta(0),
        "missing_candles": int((gaps / expected - 1).sum()) if len(gaps) else 0,
        "nan_values": int(frame.isna().sum().sum()),
        "non_positive_prices": int((frame[["open", "high", "low", "close"]] <= 0).sum().sum()),
        "invalid_ohlc_rows": ohlc_invalid,
    }

    for key in ("duplicate_timestamps", "nan_values", "non_positive_prices", "invalid_ohlc_rows"):
        if report[key]:
            logger.warning("%s: %s = %s", timeframe, key, report[key])
    if not report["monotonic"]:
        logger.warning("%s: index is not monotonically increasing", timeframe)
    if report["gaps"]:
        logger.info(
            "%s: %d gaps, %d missing candles, largest gap %s "
            "(exchange downtime is normal and is left unfilled)",
            timeframe,
            report["gaps"],
            report["missing_candles"],
            report["largest_gap"],
        )
    return report
