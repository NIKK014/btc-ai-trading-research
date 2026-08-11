"""Trading strategies, organised by methodology.

The experiment compares *methodologies* rather than individual indicators, so
each strategy belongs to one of four families:

======================  ==========================================  ==================
Methodology             Premise                                     Strategies
======================  ==========================================  ==================
trend                   Moves persist                               ema_rsi_trend,
                                                                    macd_adx_trend
momentum                Rate of change carries information          stochrsi_macd_momentum
mean_reversion          Stretched price snaps back to a reference   rsi_bollinger_reversion,
                                                                    vwap_stretch_reversion
breakout                Ranges resolve violently                    donchian_volume_breakout,
                                                                    bollinger_squeeze_breakout
======================  ==========================================  ==================

Plus ``buy_and_hold``, the benchmark every result is judged against.

The same indicator can appear in several families with opposite meaning - RSI
below 30 confirms a downtrend in ``trend.py`` and signals a buy in
``mean_reversion.py``. That is the point: the experiment tests whether the
surrounding methodology, not the indicator, is what generates edge.
"""

from src.strategies.base import (
    Signal,
    Strategy,
    StrategyParams,
    build,
    combine,
    hold_until_flip,
    registry,
    state_machine,
)

__all__ = [
    "Signal",
    "Strategy",
    "StrategyParams",
    "build",
    "combine",
    "hold_until_flip",
    "registry",
    "state_machine",
]
