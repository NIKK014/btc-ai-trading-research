"""Trading strategies, grouped by methodology.

The experiment compares methodologies rather than individual indicators:

    trend           ema_rsi_trend, macd_adx_trend
    momentum        stochrsi_macd_momentum
    mean_reversion  rsi_bollinger_reversion, vwap_stretch_reversion
    breakout        donchian_volume_breakout, bollinger_squeeze_breakout
    benchmark       buy_and_hold

The same indicator appears in several families with opposite meaning - RSI
below 30 confirms a downtrend in trend.py and signals a buy in
mean_reversion.py. That is the point: the test is whether the surrounding
methodology generates edge, not the indicator.
"""

from src.strategies.base import (
    Signal,
    Strategy,
    StrategyParams,
    build,
    combine,
    registry,
    state_machine,
)

__all__ = [
    "Signal",
    "Strategy",
    "StrategyParams",
    "build",
    "combine",
    "registry",
    "state_machine",
]
