"""Streamlit dashboard.

Read-only by design. The live trading loop runs as a separate process and owns
all writes; this app only reads SQLite and the research CSVs. Putting the loop
inside Streamlit would restart it on every widget interaction.

    streamlit run dashboard/app.py

Built for a presentation: every panel degrades to an explanatory placeholder
when its data is missing, so a half-finished pipeline never produces a
traceback on stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from config.settings import BACKTEST, DATA, TRADING_MODE  # noqa: E402
from dashboard.data_access import (  # noqa: E402
    available_results,
    label_balance_table,
    load_decisions,
    load_equity_history,
    load_live_summary,
    load_recent_prices,
    has_local_trading_data,
    load_live_price,
    load_results_csv,
    load_trades,
    load_tuned_params,
    open_position_summary,
)

st.set_page_config(
    page_title="BTC AI Trading Research",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN, RED, BLUE, GREY = "#1f9d55", "#d64545", "#3b82f6", "#8a8f98"
DIRECTION_LABEL = {1: "LONG", -1: "SHORT", 0: "FLAT"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def placeholder(message: str, command: str = "") -> None:
    """Explain what is missing and how to produce it."""
    st.info(message + (f"\n\n```bash\n{command}\n```" if command else ""))


def pct(value, digits: int = 1) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}%}"


def num(value, digits: int = 2) -> str:
    return "-" if pd.isna(value) else f"{value:,.{digits}f}"


def signed_colour(value: float) -> str:
    return GREEN if value > 0 else (RED if value < 0 else GREY)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_header() -> None:
    summary = load_live_summary()
    broker = summary.get("broker_state") or {}
    equity = float(broker.get("equity", BACKTEST.initial_capital))
    start = BACKTEST.initial_capital
    direction = int(broker.get("direction", 0))

    st.title("Bitcoin AI Trading Research System")
    st.caption(
        "Does a machine-learning filter improve a rule-based strategy? "
        "Does an LLM judge improve it further? Every claim reported with a confidence interval."
    )

    banner, mode = st.columns([4, 1])
    with banner:
        # An unset TRADING_MODE is the safe state, not a broken one: nothing can
        # trade until it is set to a paper mode explicitly. The deployed copy
        # never sets it, so it says so rather than printing an empty backtick.
        st.success(
            f"**PAPER TRADING ONLY** - mode `{TRADING_MODE}`. "
            "This system cannot execute a trade with real funds."
            if TRADING_MODE
            else "**PAPER TRADING ONLY** - this published copy has no trading mode "
            "configured, so it cannot place an order of any kind.",
            icon="🔒",
        )
    with mode:
        st.metric("Symbol", DATA.symbol)

    # On the deployed copy there is no trading database, so these would all read
    # back the untouched starting capital - technically true, quietly misleading.
    if not has_local_trading_data():
        live = load_live_price()
        columns = st.columns(3)
        if live:
            columns[0].metric("BTC now", f"{live:,.1f}")
        else:
            # No venue reachable from this host; fall back to the snapshot's
            # last close and say so, rather than showing a bare dash.
            recent = load_recent_prices()
            last = float(recent["close"].iloc[-1]) if not recent.empty else None
            columns[0].metric(
                "BTC last close",
                f"{last:,.1f}" if last else "-",
                "snapshot, not live",
                delta_color="off",
            )
        columns[1].metric("Best system, out-of-sample", "+1.6%", "Sharpe 0.22")
        columns[2].metric("Buy and hold, same period", "-41.0%", "Sharpe -0.96")
        return

    position = open_position_summary()

    columns = st.columns(6)
    # Equity here is realised only. The open position's mark-to-market sits in
    # its own metric so the two are never silently added together.
    columns[0].metric("Equity (realised)", f"{equity:,.2f}", f"{equity - start:+,.2f}")
    columns[1].metric("Return", pct(equity / start - 1, 2) if start else "-")
    columns[2].metric("Position", DIRECTION_LABEL.get(direction, "FLAT"))

    if position:
        columns[3].metric(
            "Unrealised P&L",
            f"{position['unrealised_pnl']:+,.2f}",
            pct(position["unrealised_pct"], 2),
        )
        columns[4].metric(
            "BTC now",
            f"{position['mark_price']:,.1f}",
            f"entry {position['entry_price']:,.1f}",
            delta_color="off",
        )
    else:
        columns[3].metric("Unrealised P&L", "-")
        live = load_live_price()
        columns[4].metric("BTC now", f"{live:,.1f}" if live else "-")

    columns[5].metric("Decisions logged", int(summary.get("decisions_logged", 0)))

    if position:
        stop, target = position.get("stop_price"), position.get("target_price")
        distance = ""
        if stop and target and position["mark_price"]:
            mark = position["mark_price"]
            distance = (
                f" &nbsp;·&nbsp; stop {stop:,.0f} ({abs(mark / stop - 1):.2%} away)"
                f" &nbsp;·&nbsp; target {target:,.0f} ({abs(mark / target - 1):.2%} away)"
            )
        st.markdown(
            f"**Open {DIRECTION_LABEL[position['direction']]}** {position['size']:.4f} BTC "
            f"from {position['entry_time']:%Y-%m-%d %H:%M} UTC{distance}"
            + ("  &nbsp;·&nbsp; *price feed unavailable, marked at last close*" if position["stale"] else "")
        )


# ---------------------------------------------------------------------------
# Live tab
# ---------------------------------------------------------------------------


def render_live() -> None:
    st.subheader("Live paper trading")
    st.caption(
        "Fills are simulated locally against real BTC prices, using the same fee, "
        "slippage and stop rules as the backtester. No order reaches an exchange."
    )

    local = has_local_trading_data()
    prices = load_recent_prices()
    decisions = load_decisions() if local else pd.DataFrame()

    if prices.empty:
        if local:
            placeholder("No price history cached yet.", "python scripts/fetch_data.py")
        else:
            placeholder(
                "The live price feed is not responding right now. The research results "
                "in the other tabs are files, not live requests, so they are unaffected."
            )
        return

    figure = go.Figure(
        go.Candlestick(
            x=prices.index,
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            name=DATA.symbol,
        )
    )

    trades = load_trades()
    if not trades.empty:
        opens = trades.dropna(subset=["entry_time"])
        figure.add_trace(
            go.Scatter(
                x=opens["entry_time"],
                y=opens["entry_price"],
                mode="markers",
                marker=dict(
                    symbol=["triangle-up" if d > 0 else "triangle-down" for d in opens["direction"]],
                    size=14,
                    color=[GREEN if d > 0 else RED for d in opens["direction"]],
                    line=dict(width=1, color="white"),
                ),
                name="Entries",
            )
        )

    figure.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)

    source = prices.attrs.get("source")
    if source == "coinbase":
        st.caption(
            "Chart: Coinbase BTC-USD spot. Bybit refuses requests from datacenter "
            "IPs, so the published copy cannot reach the venue the research used. "
            "The two track each other closely, but this chart is context only - "
            "every backtest on this site was computed from Bybit BTCUSDT data."
        )
    elif source == "snapshot":
        st.caption(
            f"Chart: Bybit BTCUSDT, snapshot to {prices.index[-1]:%d %b %Y %H:%M} UTC - "
            "**not live**. Both price APIs refuse requests from the datacenter this "
            "app runs in, so it ships with a slice of real candles rather than a "
            "blank panel. The research results in the other tabs are unaffected."
        )

    if not has_local_trading_data():
        st.info(
            "**This is the published copy of the dashboard.** The chart above is live "
            "Bitcoin data, but the paper-trading loop runs on a local machine and its "
            "database is not part of this deployment, so there are no positions or "
            "decisions to show here. Everything the trader was built to test is in the "
            "**Results**, **Machine learning** and **AI Judge** tabs - those are the "
            "real, unmodified research outputs.",
            icon="ℹ️",
        )
        return

    equity_history = load_equity_history()
    if len(equity_history) > 1:
        st.markdown("**Live equity**")
        st.line_chart(equity_history["equity"], height=200)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Recent decisions**")
        st.caption(
            "Every decision is logged, including the ones that produced no trade - "
            "otherwise 'why was it flat all afternoon?' is unanswerable."
        )
        if decisions.empty:
            placeholder(
                "No decisions logged yet. The 4h loop decides six times a day.",
                "python main.py --system C",
            )
        else:
            view = decisions.head(15).copy()
            view["signal"] = view["strategy_signal"].map(DIRECTION_LABEL)
            view["ml"] = view["ml_prediction"].map(DIRECTION_LABEL)
            view["action"] = view["final_action"].map(DIRECTION_LABEL)
            view["confidence"] = view["ml_confidence"].map(lambda v: pct(v, 0))
            st.dataframe(
                view[
                    ["timestamp", "signal", "ml", "confidence", "judge_decision", "action", "blocked_reason"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.markdown("**Latest AI reasoning**")
        judged = decisions[decisions["judge_reason"].notna()] if not decisions.empty else pd.DataFrame()
        if judged.empty:
            st.caption("No LLM decision recorded yet.")
        else:
            for _, row in judged.head(3).iterrows():
                st.markdown(
                    f"**{row['judge_decision']}** &nbsp; confidence {int(row['judge_confidence'])} "
                    f"&nbsp; risk {row['risk_assessment']}"
                )
                st.caption(f"{row['timestamp']}  -  {row['judge_reason']}")
                st.divider()

    st.markdown("**Trades**")
    if trades.empty:
        st.caption("No trades yet.")
    else:
        view = trades.copy()
        view["side"] = view["direction"].map(DIRECTION_LABEL)

        # An unclosed trade has empty exit fields. Left raw, those blanks read
        # as "zero P&L" rather than "still running" - the first thing anyone
        # asks about. Label the status and mark the open position to market.
        is_open = view["exit_time"].isna()
        view["status"] = ["OPEN" if o else "closed" for o in is_open]

        last_price = float(prices["close"].iloc[-1]) if not prices.empty else None
        if last_price is not None:
            unrealised = view["direction"] * view["size"] * (last_price - view["entry_price"])
            view.loc[is_open, "pnl"] = unrealised[is_open]
            view.loc[is_open, "exit_price"] = last_price

        view["pnl"] = [
            f"{v:+,.2f} (unrealised)" if o else (f"{v:+,.2f}" if pd.notna(v) else "-")
            for v, o in zip(view["pnl"], is_open)
        ]
        view["exit_time"] = [
            "still open" if o else (f"{t:%Y-%m-%d %H:%M}" if pd.notna(t) else "-")
            for t, o in zip(view["exit_time"], is_open)
        ]
        view["exit_price"] = [
            f"{v:,.1f} (mark)" if o else (f"{v:,.1f}" if pd.notna(v) else "-")
            for v, o in zip(view["exit_price"], is_open)
        ]
        view["exit_reason"] = view["exit_reason"].fillna("-")
        view["entry_time"] = view["entry_time"].map(
            lambda t: f"{t:%Y-%m-%d %H:%M}" if pd.notna(t) else "-"
        )

        columns = [
            c
            for c in ["status", "entry_time", "exit_time", "side", "entry_price",
                      "exit_price", "size", "pnl", "exit_reason"]
            if c in view.columns
        ]
        st.dataframe(view[columns].head(20), use_container_width=True, hide_index=True)
        if is_open.any():
            st.caption(
                "An open trade shows its mark-to-market value at the latest close, not a "
                "realised result. `entry_time` is wall-clock execution time; the decision "
                "that caused it is keyed to the candle's open time, which is why they differ."
            )


# ---------------------------------------------------------------------------
# Machine learning tab
# ---------------------------------------------------------------------------


def render_ml() -> None:
    st.subheader("Question 2 - does machine learning help?")
    models = load_results_csv("ml_models_4h.csv")

    st.markdown("**The target had to be redefined**")
    st.caption(
        "A fixed 0.5% barrier means something different on every timeframe, which would make "
        "the timeframe comparison meaningless. Scaling the barrier by ATR fixes it."
    )
    balance = label_balance_table()
    if balance.empty and not has_local_trading_data():
        placeholder(
            "This table is recomputed from six years of raw candles, which are too "
            "large to publish with the app. The model results below are saved outputs "
            "and are unaffected."
        )
    elif balance.empty:
        placeholder("No cached price data.", "python scripts/fetch_data.py")
    else:
        view = balance.copy()
        for column in ("LONG", "SHORT", "HOLD", "discarded_ties"):
            view[column] = view[column].map(lambda v: pct(v))
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption(
            "Under the fixed barrier, 4h has almost no HOLD class and a quarter of samples are "
            "discarded as unresolvable same-candle ties. The ATR barrier holds roughly "
            "35/35/30 on every timeframe."
        )

    st.markdown("---")
    st.markdown("**Model comparison**")
    if models.empty:
        placeholder("No ML results yet.", "python scripts/run_ml.py")
        return

    st.caption(
        "`uplift_vs_dummy` is the only column meaningful on its own. On an imbalanced "
        "three-class target, raw accuracy can be worse than always predicting the majority "
        "class - without the baseline, an accuracy figure is uninterpretable."
    )
    display = models.copy()
    for column in [c for c in display.columns if c != "model"]:
        display[column] = display[column].map(lambda v: num(v, 3))
    st.dataframe(display, use_container_width=True, hide_index=True)

    best = models.iloc[0]
    left, right = st.columns(2)
    left.metric("Best model", best["model"])
    left.metric("Balanced accuracy", num(best["val_balanced_acc"], 3))
    right.metric("Uplift over baseline", f"{best.get('uplift_vs_dummy', 0):+.3f}")
    right.metric("Overfit gap (train - validation)", num(best.get("overfit_gap"), 3))

    if float(best.get("uplift_vs_dummy", 0)) > 0:
        st.success(
            f"The model learned something real: {float(best['uplift_vs_dummy']):+.3f} balanced "
            "accuracy over a majority-class baseline. Whether that helps *trading* is a "
            "separate question, answered in the Results tab."
        )
    else:
        st.warning("No model beat the baseline. Reported as-is.")


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------


def render_final_test() -> None:
    """The headline result: the untouched out-of-sample period."""
    final = load_results_csv("final_test.csv")
    if final.empty:
        placeholder(
            "The out-of-sample test has not been run yet. It can only be run once.",
            "python scripts/run_final.py",
        )
        return

    st.subheader("The out-of-sample result")
    st.caption(
        "The test period was touched exactly once, after every strategy, parameter and "
        "threshold had been fixed on validation. These are the only numbers in the project "
        "that were not selected for."
    )

    degradation = load_results_csv("final_degradation.csv")
    if not degradation.empty:
        st.markdown("**How much of the validation result survived?**")
        columns = st.columns(4)
        for column, metric, formatter in zip(
            columns,
            ["total_return", "sharpe", "sortino", "max_drawdown"],
            [pct, lambda v: num(v), lambda v: num(v), pct],
        ):
            row = degradation[degradation["metric"] == metric]
            if row.empty:
                continue
            validation_value = float(row["validation"].iloc[0])
            test_value = float(row["test"].iloc[0])
            column.metric(
                metric.replace("_", " ").title(),
                formatter(test_value),
                f"{formatter(validation_value)} on validation",
                delta_color="off",
            )
        st.caption(
            "The gap between these is what parameter selection bought on validation and "
            "could not deliver out of sample. It is the most honest number in the project."
        )

    st.markdown("---")
    st.markdown("**Test-set performance**")
    display = final.copy()
    for column in ("total_return", "max_drawdown", "win_rate", "exposure"):
        if column in display:
            display[column] = display[column].map(lambda v: pct(v))
    for column in ("sharpe", "sortino", "profit_factor"):
        if column in display:
            display[column] = display[column].map(lambda v: num(v))
    show = [
        c for c in ["system", "n_trades", "total_return", "sharpe", "sortino",
                    "max_drawdown", "win_rate", "profit_factor", "exposure"]
        if c in display.columns
    ]
    st.dataframe(display[show], use_container_width=True, hide_index=True)

    figure = go.Figure(
        go.Bar(
            x=final["system"],
            y=final["total_return"] * 100,
            marker_color=[signed_colour(v) for v in final["total_return"]],
            text=[pct(v) for v in final["total_return"]],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=300, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Test return (%)"
    )
    st.plotly_chart(figure, use_container_width=True)

    regime = load_results_csv("final_test_by_regime.csv")
    if not regime.empty:
        st.markdown("---")
        st.markdown("**Did it only work because the market moved one way?**")
        st.caption(
            "The test period contains a large drawdown, so aggregate figures conflate "
            "strategy skill with market direction. Split at the price peak."
        )
        view = regime.copy()
        for column in ("total_return", "max_drawdown", "win_rate"):
            if column in view:
                view[column] = view[column].map(lambda v: pct(v))
        for column in ("sharpe", "sortino", "profit_factor"):
            if column in view:
                view[column] = view[column].map(lambda v: num(v))
        st.dataframe(view, use_container_width=True, hide_index=True, height=330)


# ---------------------------------------------------------------------------
# AI Judge tab
# ---------------------------------------------------------------------------


def render_judge() -> None:
    st.subheader("The LLM trading judge")
    st.caption(
        "The judge decides direction. It never decides exposure - position size, "
        "stop placement and the daily loss limit are computed downstream by a "
        "deterministic risk manager."
    )

    decisions = load_results_csv("decisions_C_rules_ml_llm_4h.csv")
    if decisions.empty:
        placeholder(
            "No judged decisions exported yet. This replays from cache and makes "
            "zero API calls.",
            "python scripts/show_judge.py",
        )
        return

    approved = decisions["decision"] == decisions["strategy_signal"]
    held = decisions["decision"] == "HOLD"

    columns = st.columns(4)
    columns[0].metric("Decisions", len(decisions))
    columns[1].metric("Approved", pct(approved.mean(), 0))
    columns[2].metric("Vetoed (HOLD)", pct(held.mean(), 0))
    columns[3].metric("Mean confidence", num(decisions["confidence"].mean(), 0))

    left, right = st.columns(2)
    with left:
        counts = decisions["decision"].value_counts()
        figure = go.Figure(
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[
                    GREEN if d == "LONG" else (RED if d == "SHORT" else GREY)
                    for d in counts.index
                ],
                text=counts.values,
                textposition="outside",
            )
        )
        figure.update_layout(
            height=260, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Decisions"
        )
        st.plotly_chart(figure, use_container_width=True)

    with right:
        figure = go.Figure(
            go.Histogram(x=decisions["confidence"], nbinsx=20, marker_color=BLUE)
        )
        figure.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Stated confidence",
            yaxis_title="Decisions",
        )
        st.plotly_chart(figure, use_container_width=True)

    st.markdown("---")
    st.markdown("**Where the judge overruled agreement**")
    st.caption(
        "The strategy and the model both wanted the trade, and the judge stood "
        "aside anyway. These are the only decisions where the LLM did something a "
        "deterministic agreement rule would not have."
    )
    overruled = decisions[
        (decisions["decision"] == "HOLD")
        & (decisions["ml_prediction"] == decisions["strategy_signal"])
    ]
    if overruled.empty:
        st.caption("None - the judge never overruled a full agreement.")
    else:
        for _, row in overruled.head(5).iterrows():
            st.markdown(
                f"**{row['timestamp'][:16]}** &nbsp;·&nbsp; both said "
                f"**{row['strategy_signal']}** &nbsp;·&nbsp; judge said **HOLD** "
                f"@ {int(row['confidence'])}"
            )
            st.caption(row["reason"])

    st.markdown("---")
    st.markdown("**Every decision**")
    view = decisions.copy()
    view["ml_confidence"] = view["ml_confidence"].map(lambda v: pct(v, 0))
    show = [
        c
        for c in ["timestamp", "strategy_signal", "ml_prediction", "ml_confidence",
                  "decision", "confidence", "risk_assessment", "reason"]
        if c in view.columns
    ]
    st.dataframe(view[show], use_container_width=True, hide_index=True, height=420)

    st.caption(
        "Every prompt and response is cached on a hash of its payload, so this "
        "replays with the network unplugged and the backtest is fully reproducible."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    with st.sidebar:
        st.markdown("### Pipeline status")
        for step, done in available_results().items():
            st.markdown(f"{'✅' if done else '⬜️'} {step}")

        st.markdown("---")
        tuned = load_tuned_params().get("ema_rsi_trend")
        if tuned:
            st.markdown("### Winning configuration")
            st.markdown(f"**{tuned['timeframe']}** · {int(tuned['n_trades'])} trades")
            st.caption(tuned["params"])

        st.markdown("---")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Live panels refresh automatically every 20 seconds.")

    render_header()
    live, results, ml, judge = st.tabs(
        ["Live", "Results", "Machine learning", "AI Judge"]
    )
    with live:
        render_live()
    with results:
        render_final_test()
    with ml:
        render_ml()
    with judge:
        render_judge()


if __name__ == "__main__":
    main()
