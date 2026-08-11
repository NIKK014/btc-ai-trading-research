"""Indicator correctness and - critically - causality tests.

The causality tests are the most important tests in the project. A single
indicator that peeks one bar into the future produces a beautiful backtest
and a worthless result, and the failure is completely invisible in the
equity curve. These tests make that class of bug impossible to introduce
without a red build.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from src.indicators import indicators as ind
from src.indicators.indicators import (
    ML_FEATURE_COLUMNS,
    IndicatorSpec,
    add_indicators,
    adx,
    atr,
    bollinger_bands,
    donchian_high,
    donchian_low,
    ema,
    obv,
    rsi,
    session_vwap,
    sma,
    true_range,
    volume_ratio,
)

# ---------------------------------------------------------------------------
# Causality - the tests that matter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [120, 300, 450])
def test_indicators_do_not_change_when_future_data_is_appended(ohlcv, cut):
    """The headline causality test.

    Compute every indicator on the full series, then recompute on only the
    first ``cut`` bars. If any indicator used future information, the
    historical values would differ between the two runs.
    """
    full = add_indicators(ohlcv)
    truncated = add_indicators(ohlcv.iloc[:cut])

    pd.testing.assert_frame_equal(
        full.iloc[:cut],
        truncated,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
        obj="indicator values changed when future bars were appended",
    )


def test_single_indicator_functions_are_causal(ohlcv):
    """Same check, applied function by function so a failure names the culprit."""
    cut = 250
    head = ohlcv.iloc[:cut]

    checks = {
        "sma": (lambda f: sma(f["close"], 20)),
        "ema": (lambda f: ema(f["close"], 21)),
        "rsi": (lambda f: rsi(f["close"], 14)),
        "atr": (lambda f: atr(f["high"], f["low"], f["close"], 14)),
        "true_range": (lambda f: true_range(f["high"], f["low"], f["close"])),
        "adx": (lambda f: adx(f["high"], f["low"], f["close"], 14)["adx"]),
        "bb_pct_b": (lambda f: bollinger_bands(f["close"], 20)["bb_pct_b"]),
        "session_vwap": session_vwap,
        "obv": (lambda f: obv(f["close"], f["volume"])),
        "volume_ratio": (lambda f: volume_ratio(f["volume"], 20)),
        "donchian_high": (lambda f: donchian_high(f["high"], 20)),
        "donchian_low": (lambda f: donchian_low(f["low"], 20)),
    }

    for name, fn in checks.items():
        pd.testing.assert_series_equal(
            fn(ohlcv).iloc[:cut],
            fn(head),
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
            obj=f"{name} is not causal",
        )


def test_source_contains_no_negative_shifts():
    """Guard against the most common way future data leaks in.

    ``shift(-n)`` moves data backwards in time and is how a label is built,
    never how a feature is built. It must not appear in the indicator module.
    """
    source = inspect.getsource(ind)
    offenders = re.findall(r"\.shift\(\s*-\s*\d+", source)
    assert not offenders, f"negative shift found in indicators module: {offenders}"


def test_donchian_channel_excludes_the_current_bar():
    """A breakout level that includes the current bar can never be broken."""
    high = pd.Series([10.0, 11.0, 12.0, 13.0, 99.0])
    result = donchian_high(high, period=3)
    # At the final bar the channel must reflect the three preceding bars
    # (11, 12, 13 -> 13), not the current bar's spike to 99.
    assert result.iloc[4] == 13.0
    assert high.iloc[4] > result.iloc[4], "current bar should be able to break the channel"

    low = pd.Series([10.0, 9.0, 8.0, 7.0, 0.5])
    assert donchian_low(low, period=3).iloc[4] == 7.0


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_sma_matches_hand_calculation():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, 3)
    assert result.iloc[:2].isna().all(), "warm-up period must be NaN, not partial"
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_follows_the_recursive_definition():
    series = pd.Series(np.arange(1.0, 21.0))
    period = 5
    result = ema(series, period)
    alpha = 2.0 / (period + 1)
    expected = result.iloc[-2] + alpha * (series.iloc[-1] - result.iloc[-2])
    assert result.iloc[-1] == pytest.approx(expected)


def test_rsi_is_bounded_and_saturates_on_a_one_way_market():
    rising = pd.Series(np.linspace(100, 200, 60))
    falling = pd.Series(np.linspace(200, 100, 60))

    assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(falling, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)

    noisy = rsi(make_ohlcv()["close"], 14).dropna()
    assert noisy.between(0.0, 100.0).all()


def test_true_range_accounts_for_gaps():
    """True Range must exceed the bar's own span when price gaps."""
    high = pd.Series([100.0, 130.0])
    low = pd.Series([95.0, 125.0])
    close = pd.Series([98.0, 128.0])

    result = true_range(high, low, close)
    # Bar 2 spans only 5, but gapped 32 from the previous close of 98.
    assert result.iloc[1] == pytest.approx(32.0)


def test_atr_of_a_constant_range_series_equals_that_range():
    n = 100
    frame = pd.DataFrame(
        {
            "high": np.full(n, 102.0),
            "low": np.full(n, 98.0),
            "close": np.full(n, 100.0),
        }
    )
    result = atr(frame["high"], frame["low"], frame["close"], 14)
    assert result.iloc[-1] == pytest.approx(4.0)


def test_bollinger_percent_b_is_half_at_the_middle_band(ohlcv):
    bands = bollinger_bands(ohlcv["close"], 20).dropna()
    at_middle = 100.0 * (bands["bb_mid"] - bands["bb_lower"]) / (
        bands["bb_upper"] - bands["bb_lower"]
    )
    assert at_middle.round(6).eq(50.0).all()
    assert bands["bb_upper"].ge(bands["bb_mid"]).all()
    assert bands["bb_lower"].le(bands["bb_mid"]).all()


def test_adx_and_di_stay_within_bounds(ohlcv):
    result = adx(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).dropna()
    assert not result.empty
    for column in ("adx", "di_plus", "di_minus"):
        assert result[column].between(0.0, 100.0).all(), f"{column} out of bounds"


def test_obv_signs_volume_by_price_direction():
    close = pd.Series([100.0, 101.0, 100.0, 100.0])
    volume = pd.Series([10.0, 20.0, 30.0, 40.0])
    result = obv(close, volume)
    # First bar has no prior close -> 0; then +20, then -30, then flat.
    assert result.tolist() == pytest.approx([0.0, 20.0, -10.0, -10.0])


def test_session_vwap_resets_each_day():
    """VWAP must restart at the session boundary, not run cumulatively."""
    index = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "high": np.r_[np.full(24, 100.0), np.full(24, 200.0)],
            "low": np.r_[np.full(24, 100.0), np.full(24, 200.0)],
            "close": np.r_[np.full(24, 100.0), np.full(24, 200.0)],
            "volume": np.ones(48),
        },
        index=index,
    )
    result = session_vwap(frame)
    assert result.iloc[23] == pytest.approx(100.0)
    # If VWAP leaked across the session boundary the second day would start
    # somewhere near 100, not at 200.
    assert result.iloc[24] == pytest.approx(200.0)


def test_volume_ratio_is_scale_free():
    """Doubling every volume must leave the ratio unchanged."""
    volume = pd.Series(np.random.default_rng(1).lognormal(size=100))
    baseline = volume_ratio(volume, 20).dropna()
    scaled = volume_ratio(volume * 1000.0, 20).dropna()
    pd.testing.assert_series_equal(baseline, scaled, rtol=1e-12)


# ---------------------------------------------------------------------------
# Feature set integration
# ---------------------------------------------------------------------------


def test_add_indicators_produces_the_declared_ml_features(ohlcv):
    result = add_indicators(ohlcv, dropna=True)
    missing = set(ML_FEATURE_COLUMNS) - set(result.columns)
    assert not missing, f"declared ML features are missing: {sorted(missing)}"
    assert not result.empty, "warm-up consumed the entire series"
    assert np.isfinite(result[list(ML_FEATURE_COLUMNS)].to_numpy()).all(), (
        "ML features contain inf or NaN after dropna"
    )


def test_add_indicators_preserves_the_original_ohlcv(ohlcv):
    result = add_indicators(ohlcv)
    pd.testing.assert_frame_equal(result[ohlcv.columns.tolist()], ohlcv)
    assert result.index.equals(ohlcv.index)


def test_add_indicators_rejects_incomplete_input(ohlcv):
    with pytest.raises(ValueError, match="missing columns"):
        add_indicators(ohlcv.drop(columns=["volume"]))


def test_indicator_spec_periods_are_respected(ohlcv):
    spec = IndicatorSpec(ema_fast=5, ema_mid=13, ema_slow=34, ema_trend=100)
    result = add_indicators(ohlcv, spec)
    for period in (5, 13, 34, 100):
        assert f"ema_{period}" in result.columns
