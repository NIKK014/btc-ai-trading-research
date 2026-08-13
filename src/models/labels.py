"""Triple-barrier labelling.

For each bar: within the next h candles, does price reach the upper or lower
barrier first? Three classes - LONG, SHORT, HOLD.

Four decisions that matter:

- Barriers scale with ATR rather than a fixed percentage. Four candles is one
  hour at 15m and sixteen at 4h, so a fixed threshold means something different
  on each timeframe and breaks the timeframe comparison.
- First touch is resolved by scanning forward one bar at a time.
- If a single candle breaches both barriers the order is unknowable, so the
  sample is labelled HOLD and dropped rather than guessed.
- Bar t's label reads bars t+1..t+h. Its own high and low are excluded.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config.settings import LABELS, LabelConfig
from src.indicators.indicators import atr as compute_atr
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

LONG = 1
SHORT = -1
HOLD = 0
CLASS_NAMES = {1: "LONG", -1: "SHORT", 0: "HOLD"}


def barrier_levels(
    frame: pd.DataFrame,
    config: LabelConfig = LABELS,
) -> Tuple[pd.Series, pd.Series]:
    """Upper and lower barrier price for each bar.

    Both are computed from information available at the bar's close: the close
    itself and the current ATR.
    """
    close = frame["close"]

    if config.mode == "atr":
        atr = (
            frame["atr"]
            if "atr" in frame.columns
            else compute_atr(frame["high"], frame["low"], close, config.atr_period)
        )
        offset = atr * config.atr_multiple
    elif config.mode == "fixed":
        offset = close * config.fixed_pct
    else:
        raise ValueError(f"Unknown barrier mode {config.mode!r}")

    return close + offset, close - offset


def triple_barrier_labels(
    frame: pd.DataFrame,
    config: LabelConfig = LABELS,
) -> pd.DataFrame:
    """Label every bar by which barrier its future window touches first.

    Args:
        frame: OHLCV, optionally with an ``atr`` column.
        config: Horizon, barrier mode and multiples.

    Returns:
        A frame indexed like ``frame`` with ``label`` (1 LONG, -1 SHORT,
        0 HOLD), ``ambiguous`` where both barriers fell in one candle,
        ``incomplete`` for the trailing bars, ``bars_to_touch``, and the two
        barrier levels for inspection.
    """
    horizon = config.horizon_bars
    if horizon < 1:
        raise ValueError("horizon_bars must be at least 1")

    upper, lower = barrier_levels(frame, config)
    n = len(frame)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    upper_values = upper.to_numpy(dtype=np.float64)
    lower_values = lower.to_numpy(dtype=np.float64)

    never = horizon + 1
    up_touch = np.full(n, never, dtype=np.int16)
    down_touch = np.full(n, never, dtype=np.int16)

    # Scan the window one bar at a time so the *first* touch is what counts.
    for step in range(1, horizon + 1):
        future_high = np.full(n, np.nan)
        future_low = np.full(n, np.nan)
        future_high[: n - step] = high[step:]
        future_low[: n - step] = low[step:]

        hit_up = np.nan_to_num(future_high, nan=-np.inf) >= upper_values
        hit_down = np.nan_to_num(future_low, nan=np.inf) <= lower_values

        up_touch = np.where((up_touch == never) & hit_up, step, up_touch)
        down_touch = np.where((down_touch == never) & hit_down, step, down_touch)

    labels = np.full(n, HOLD, dtype=np.int8)
    labels[up_touch < down_touch] = LONG
    labels[down_touch < up_touch] = SHORT

    # Both barriers breached inside one candle: order unknowable.
    ambiguous = (up_touch == down_touch) & (up_touch <= horizon)

    bars_to_touch = np.minimum(up_touch, down_touch).astype("float64")
    bars_to_touch[bars_to_touch == never] = np.nan

    # The final `horizon` bars have an incomplete future window.
    incomplete = np.zeros(n, dtype=bool)
    incomplete[max(n - horizon, 0) :] = True

    result = pd.DataFrame(
        {
            "label": labels,
            "ambiguous": ambiguous,
            "incomplete": incomplete,
            "bars_to_touch": bars_to_touch,
            "upper_barrier": upper_values,
            "lower_barrier": lower_values,
        },
        index=frame.index,
    )
    result.loc[result["ambiguous"], "label"] = HOLD
    return result


def usable_mask(labels: pd.DataFrame, config: LabelConfig = LABELS) -> pd.Series:
    """Rows that may be used for training.

    Excludes ambiguous same-candle ties (when configured) and the trailing bars
    whose future window runs off the end of the data.
    """
    mask = ~labels["incomplete"]
    if config.drop_ambiguous:
        mask &= ~labels["ambiguous"]
    return mask


def class_balance(labels: pd.DataFrame, config: LabelConfig = LABELS) -> Dict[str, float]:
    """Class proportions and diagnostics for one labelled series.

    Always inspect this before training. If HOLD dominates above ~85% the
    barrier is too wide for the timeframe and the model will learn to predict
    nothing; below ~5% it is too narrow and the label is close to a coin flip.
    Either way the target needs rescaling, not a better model.
    """
    usable = labels.loc[usable_mask(labels, config), "label"]
    if usable.empty:
        return {}

    shares = usable.value_counts(normalize=True)
    return {
        "samples": int(len(usable)),
        "long_share": float(shares.get(LONG, 0.0)),
        "short_share": float(shares.get(SHORT, 0.0)),
        "hold_share": float(shares.get(HOLD, 0.0)),
        "ambiguous_share": float(labels["ambiguous"].mean()),
        "median_bars_to_touch": float(labels["bars_to_touch"].median()),
        "majority_class_share": float(shares.max()),
    }


def describe_balance(balance: Dict[str, float], timeframe: str) -> str:
    """One-line readable summary, with a warning when the target is unusable."""
    if not balance:
        return f"{timeframe}: no usable labels"

    line = (
        f"{timeframe:>4}: {balance['samples']:>7,} samples | "
        f"LONG {balance['long_share']:5.1%}  SHORT {balance['short_share']:5.1%}  "
        f"HOLD {balance['hold_share']:5.1%} | "
        f"ambiguous {balance['ambiguous_share']:5.1%} | "
        f"median touch {balance['median_bars_to_touch']:.1f} bars"
    )
    majority = balance["majority_class_share"]
    if majority > 0.85:
        line += "   <- barrier too WIDE for this timeframe"
    elif balance["hold_share"] < 0.05:
        line += "   <- barrier too NARROW for this timeframe"
    return line
