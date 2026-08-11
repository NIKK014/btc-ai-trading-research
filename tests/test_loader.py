"""Data loader tests.

These run entirely offline against a fake exchange client, so the pagination,
de-duplication and unclosed-candle logic is verified without touching the
network. The fake mimics Bybit's actual contract: newest-first rows, at most
``limit`` per call, bounded by ``end``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from src.data.loader import (
    _rows_to_frame,
    drop_unclosed_candle,
    fetch_ohlcv,
    validate_ohlcv,
)


class FakeBybitClient:
    """In-memory stand-in for :class:`BybitPublicClient`."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.call_count = 0

    def get_kline(self, symbol, interval, category, start_ms, end_ms, limit):
        self.call_count += 1
        timestamps = (self.frame.index.astype("int64") // 1_000_000).to_numpy()
        mask = timestamps <= end_ms
        if start_ms is not None:
            mask &= timestamps >= start_ms
        window = self.frame.loc[mask]
        # Bybit returns the newest `limit` rows in the window, newest first.
        window = window.iloc[-limit:]
        rows = [
            [str(int(ts.timestamp() * 1000)), *(str(v) for v in row)]
            for ts, row in zip(window.index, window.to_numpy())
        ]
        return list(reversed(rows))


@pytest.fixture
def source() -> pd.DataFrame:
    return make_ohlcv(periods=2_500, freq="1h", seed=3)


def test_fetch_paginates_backwards_over_the_full_range(source):
    """More history than one page must still come back complete and in order."""
    client = FakeBybitClient(source)
    result = fetch_ohlcv(
        "1h",
        start=source.index[0],
        end=source.index[-1] + pd.Timedelta(hours=1),
        drop_unclosed=False,
        client=client,
    )

    assert client.call_count >= 3, "2500 rows at 1000/page should need 3+ requests"
    assert len(result) == len(source)
    assert result.index.is_monotonic_increasing
    assert not result.index.duplicated().any()
    np.testing.assert_allclose(result["close"].to_numpy(), source["close"].to_numpy())


def test_fetch_respects_the_requested_window(source):
    client = FakeBybitClient(source)
    start, end = source.index[500], source.index[900]
    result = fetch_ohlcv("1h", start=start, end=end, drop_unclosed=False, client=client)

    assert result.index.min() == start
    assert result.index.max() < end
    assert len(result) == 400


def test_fetch_returns_typed_float_columns(source):
    client = FakeBybitClient(source)
    result = fetch_ohlcv(
        "1h",
        start=source.index[0],
        end=source.index[-1],
        drop_unclosed=False,
        client=client,
    )
    assert (result.dtypes == "float64").all()
    assert str(result.index.tz) == "UTC"


def test_rows_to_frame_handles_an_empty_response():
    result = _rows_to_frame([])
    assert result.empty
    assert list(result.columns) == ["open", "high", "low", "close", "volume", "turnover"]


def test_unclosed_candle_is_dropped():
    """The final candle is still forming, so its high/low/close will change."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    index = pd.date_range(end=now, periods=5, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=index,
    )
    result = drop_unclosed_candle(frame, "1h")
    assert len(result) == 4
    assert result.index[-1] == index[-2]


def test_closed_candles_are_kept():
    index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=3), periods=5, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=index,
    )
    assert len(drop_unclosed_candle(frame, "1h")) == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_passes_on_a_clean_series(source):
    report = validate_ohlcv(source, "1h")
    assert report["rows"] == len(source)
    assert report["monotonic"] is True
    assert report["duplicate_timestamps"] == 0
    assert report["gaps"] == 0
    assert report["nan_values"] == 0
    assert report["invalid_ohlc_rows"] == 0


def test_validation_detects_missing_candles(source):
    with_gap = source.drop(source.index[100:110])
    report = validate_ohlcv(with_gap, "1h")
    assert report["gaps"] == 1
    assert report["missing_candles"] == 10


def test_validation_detects_impossible_ohlc(source):
    broken = source.copy()
    broken.iloc[50, broken.columns.get_loc("high")] = broken["low"].iloc[50] - 1.0
    report = validate_ohlcv(broken, "1h")
    assert report["invalid_ohlc_rows"] >= 1
