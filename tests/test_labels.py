"""Triple-barrier labelling tests.

A leaked label is the most dangerous bug in the project: it produces excellent
accuracy and a fantasy equity curve, and nothing downstream looks wrong. These
tests pin the label definition to hand-computed answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import LabelConfig
from src.models.labels import (
    HOLD,
    LONG,
    SHORT,
    class_balance,
    triple_barrier_labels,
    usable_mask,
)

FIXED = LabelConfig(mode="fixed", fixed_pct=0.01, horizon_bars=4)


def bars(highs, lows, closes) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "close": np.asarray(closes, dtype=float),
            "volume": np.ones(len(closes)),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Label definition
# ---------------------------------------------------------------------------


def test_upside_barrier_touched_first_gives_long():
    """Close 100, 1% barriers at 101 / 99. Bar t+2 highs to 101.5."""
    frame = bars(
        highs=[100, 100.5, 101.5, 100, 100, 100],
        lows=[100, 99.5, 100.5, 100, 100, 100],
        closes=[100, 100, 101, 100, 100, 100],
    )
    labels = triple_barrier_labels(frame, FIXED)

    assert labels["label"].iloc[0] == LONG
    assert labels["bars_to_touch"].iloc[0] == 2


def test_downside_barrier_touched_first_gives_short():
    frame = bars(
        highs=[100, 100.5, 100.5, 100, 100, 100],
        lows=[100, 99.5, 98.5, 100, 100, 100],
        closes=[100, 100, 99, 100, 100, 100],
    )
    labels = triple_barrier_labels(frame, FIXED)

    assert labels["label"].iloc[0] == SHORT
    assert labels["bars_to_touch"].iloc[0] == 2


def test_neither_barrier_touched_gives_hold():
    frame = bars(
        highs=[100, 100.5, 100.4, 100.3, 100.2, 100],
        lows=[100, 99.6, 99.7, 99.8, 99.9, 100],
        closes=[100] * 6,
    )
    labels = triple_barrier_labels(frame, FIXED)

    assert labels["label"].iloc[0] == HOLD
    assert np.isnan(labels["bars_to_touch"].iloc[0])


def test_chronological_first_touch_wins():
    """Down at t+1 then up at t+2 must be SHORT, not LONG.

    A naive implementation that asks "was the upper barrier ever touched?"
    would return LONG here and would be using knowledge of the whole window.
    """
    frame = bars(
        highs=[100, 100.2, 102.0, 100, 100, 100],
        lows=[100, 98.5, 101.0, 100, 100, 100],
        closes=[100, 99, 101.5, 100, 100, 100],
    )
    labels = triple_barrier_labels(frame, FIXED)

    assert labels["label"].iloc[0] == SHORT
    assert labels["bars_to_touch"].iloc[0] == 1


def test_same_candle_tie_is_flagged_and_excluded():
    """One candle breaching both barriers has no knowable ordering."""
    frame = bars(
        highs=[100, 102.0, 100, 100, 100, 100],
        lows=[100, 98.0, 100, 100, 100, 100],
        closes=[100] * 6,
    )
    labels = triple_barrier_labels(frame, FIXED)

    assert labels["ambiguous"].iloc[0]
    assert labels["label"].iloc[0] == HOLD, "must not guess a direction"
    assert not usable_mask(labels, FIXED).iloc[0], "must be excluded from training"


def test_the_decision_bar_is_not_part_of_its_own_window():
    """Bar t's own high must not decide bar t's label."""
    frame = bars(
        highs=[105, 100.1, 100.1, 100.1, 100.1, 100],
        lows=[95, 99.9, 99.9, 99.9, 99.9, 100],
        closes=[100] * 6,
    )
    labels = triple_barrier_labels(frame, FIXED)
    assert labels["label"].iloc[0] == HOLD
    assert not labels["ambiguous"].iloc[0]


def test_trailing_bars_have_incomplete_windows():
    frame = bars(highs=[100] * 10, lows=[100] * 10, closes=[100] * 10)
    labels = triple_barrier_labels(frame, FIXED)

    assert labels["incomplete"].sum() == FIXED.horizon_bars
    assert labels["incomplete"].iloc[-1]
    assert not labels["incomplete"].iloc[0]
    assert usable_mask(labels, FIXED).sum() == len(frame) - FIXED.horizon_bars


def test_horizon_bounds_the_search():
    """A touch beyond the horizon must not be seen."""
    frame = bars(
        highs=[100, 100.1, 100.1, 100.1, 100.1, 105, 100, 100, 100, 100],
        lows=[100, 99.9, 99.9, 99.9, 99.9, 99.9, 100, 100, 100, 100],
        closes=[100] * 10,
    )
    labels = triple_barrier_labels(frame, FIXED)
    assert labels["label"].iloc[0] == HOLD, "bar 5 is outside a 4-bar horizon"


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_labels_are_unchanged_by_data_beyond_their_window(ohlcv):
    """Labels may look forward by exactly ``horizon`` bars and no further."""
    horizon = 4
    config = LabelConfig(mode="atr", horizon_bars=horizon)
    cut = 300

    full = triple_barrier_labels(ohlcv, config)
    truncated = triple_barrier_labels(ohlcv.iloc[:cut], config)

    # Rows whose window fits inside the truncated frame must match exactly.
    comparable = cut - horizon
    pd.testing.assert_series_equal(
        full["label"].iloc[:comparable],
        truncated["label"].iloc[:comparable],
        obj="labels used data beyond their horizon",
    )


# ---------------------------------------------------------------------------
# Barrier scaling - the timeframe comparability argument
# ---------------------------------------------------------------------------


def test_atr_barriers_produce_comparable_balance_across_timeframes(ohlcv):
    """The reason the barrier is ATR-scaled rather than a fixed percentage.

    The same nominal barrier must mean the same thing on every timeframe, or
    the timeframe comparison at the heart of the project is meaningless.
    """
    hourly = ohlcv
    four_hourly = ohlcv.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    config = LabelConfig(mode="atr", atr_multiple=1.0, horizon_bars=4)
    balance_1h = class_balance(triple_barrier_labels(hourly, config), config)
    balance_4h = class_balance(triple_barrier_labels(four_hourly, config), config)

    assert abs(balance_1h["hold_share"] - balance_4h["hold_share"]) < 0.15
    for balance in (balance_1h, balance_4h):
        assert 0.05 < balance["hold_share"] < 0.85, "barrier is unusable at this timeframe"


def test_class_balance_reports_the_expected_shares(ohlcv):
    labels = triple_barrier_labels(ohlcv)
    balance = class_balance(labels)

    total = balance["long_share"] + balance["short_share"] + balance["hold_share"]
    assert total == pytest.approx(1.0)
    assert balance["samples"] > 0


def test_unknown_barrier_mode_is_rejected(ohlcv):
    with pytest.raises(ValueError, match="Unknown barrier mode"):
        triple_barrier_labels(ohlcv, LabelConfig(mode="nonsense"))
