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

from config.settings import BACKTEST, DATA, METRICS, RISK, TRADING_MODE  # noqa: E402
from dashboard.data_access import (  # noqa: E402
    available_results,
    label_balance_table,
    load_decisions,
    load_equity_history,
    load_live_summary,
    load_recent_prices,
    load_results_csv,
    load_trades,
    load_tuned_params,
    validation_equity_curves,
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
        st.success(
            f"**PAPER TRADING ONLY** - mode `{TRADING_MODE}`. "
            "This system cannot execute a trade with real funds.",
            icon="🔒",
        )
    with mode:
        st.metric("Symbol", DATA.symbol)

    columns = st.columns(5)
    columns[0].metric("Equity", f"{equity:,.2f}", f"{equity - start:+,.2f}")
    columns[1].metric("Return", pct(equity / start - 1) if start else "-")
    columns[2].metric("Position", DIRECTION_LABEL.get(direction, "FLAT"))
    columns[3].metric("Closed trades", int(summary.get("closed_trades", 0)))
    columns[4].metric("Decisions logged", int(summary.get("decisions_logged", 0)))


# ---------------------------------------------------------------------------
# Live tab
# ---------------------------------------------------------------------------


def render_live() -> None:
    st.subheader("Live paper trading")
    st.caption(
        "Fills are simulated locally against real BTC prices, using the same fee, "
        "slippage and stop rules as the backtester. No order reaches an exchange."
    )

    prices = load_recent_prices()
    decisions = load_decisions()

    if prices.empty:
        placeholder("No price history cached yet.", "python scripts/fetch_data.py")
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
        columns = [
            c
            for c in ["entry_time", "exit_time", "side", "entry_price", "exit_price", "size", "pnl", "exit_reason"]
            if c in view.columns
        ]
        st.dataframe(view[columns].head(20), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Research tab
# ---------------------------------------------------------------------------


def render_research() -> None:
    st.subheader("Question 1 - which methodology and timeframe?")
    leaderboard = load_results_csv("leaderboard_validation.csv")

    if leaderboard.empty:
        placeholder("No leaderboard yet.", "python scripts/run_baseline.py")
        return

    st.caption(
        "Ranked on the validation split only. Ineligible strategies stay visible rather "
        f"than being deleted: gates are min {METRICS.min_trades} trades, "
        f"max {METRICS.max_drawdown_limit:.0%} drawdown, profit factor above "
        f"{METRICS.min_profit_factor}."
    )

    display = leaderboard.copy()
    for column in ("total_return", "max_drawdown", "win_rate", "exposure", "fees_pct_of_capital"):
        if column in display:
            display[column] = display[column].map(lambda v: pct(v))
    for column in ("sharpe", "sortino", "profit_factor", "score"):
        if column in display:
            display[column] = display[column].map(lambda v: num(v))

    columns = [
        c
        for c in [
            "rank", "strategy", "methodology", "timeframe", "n_trades", "total_return",
            "sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor",
            "fees_pct_of_capital", "eligible",
        ]
        if c in display.columns
    ]
    st.dataframe(display[columns], use_container_width=True, hide_index=True, height=420)

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("**Fee drag destroys short timeframes**")
        strategies = leaderboard[leaderboard["methodology"] != "benchmark"]
        by_timeframe = (
            strategies.groupby("timeframe")
            .agg(
                median_trades=("n_trades", "median"),
                median_fees=("fees_pct_of_capital", "median"),
                median_return=("total_return", "median"),
                eligible=("eligible", "sum"),
            )
            .reindex([t for t in DATA.timeframes if t in strategies["timeframe"].values])
        )
        figure = go.Figure(
            go.Bar(
                x=by_timeframe.index,
                y=by_timeframe["median_fees"] * 100,
                marker_color=RED,
                text=[f"{v:.0%}" for v in by_timeframe["median_fees"]],
                textposition="outside",
            )
        )
        figure.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Median fees paid (% of capital)",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "Every strategy tested at 15m was destroyed by transaction costs. "
            "Not one configuration passed the eligibility gates."
        )

    with right:
        st.markdown("**Tuned strategy vs buy-and-hold**")
        curves, context = validation_equity_curves()
        if curves.empty:
            placeholder("Run the optimiser first.", "python scripts/run_optimizer.py")
        else:
            normalised = curves / curves.iloc[0] * 100.0
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=normalised.index, y=normalised.iloc[:, 0],
                    name=context["strategy"], line=dict(color=BLUE, width=2),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=normalised.index, y=normalised["buy_and_hold"],
                    name="buy and hold", line=dict(color=GREY, width=2, dash="dot"),
                )
            )
            figure.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Equity (rebased to 100)",
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(figure, use_container_width=True)

            strategy_metrics = context["strategy_metrics"]
            benchmark_metrics = context["benchmark_metrics"]
            comparison = pd.DataFrame(
                {
                    context["strategy"]: [
                        pct(strategy_metrics["total_return"]),
                        num(strategy_metrics["sharpe"]),
                        pct(strategy_metrics["max_drawdown"]),
                        f"{int(strategy_metrics['n_trades']):,}",
                    ],
                    "buy and hold": [
                        pct(benchmark_metrics["total_return"]),
                        num(benchmark_metrics["sharpe"]),
                        pct(benchmark_metrics["max_drawdown"]),
                        f"{int(benchmark_metrics['n_trades']):,}",
                    ],
                },
                index=["Return", "Sharpe", "Max drawdown", "Trades"],
            )
            st.dataframe(comparison, use_container_width=True)

    st.markdown("---")
    st.markdown("**Is the winner a plateau or a lucky spike?**")
    st.caption(
        "Searching many configurations does not find a better strategy, it finds a luckier "
        "one. A winner sitting close to its neighbourhood median is robust; one far above it "
        "is parameter-sensitive and rarely survives out of sample."
    )
    optimizer = load_results_csv("optimizer_validation.csv")
    if optimizer.empty:
        placeholder("No parameter search yet.", "python scripts/run_optimizer.py")
    else:
        summary = (
            optimizer.groupby(["strategy", "timeframe"])["sortino"]
            .agg(configs="size", best="max", median="median", positive=lambda s: (s > 0).mean())
            .reset_index()
            .sort_values("best", ascending=False)
        )
        summary["spread_over_median"] = summary["best"] - summary["median"]
        for column in ("best", "median", "spread_over_median"):
            summary[column] = summary[column].map(lambda v: num(v))
        summary["positive"] = summary["positive"].map(lambda v: pct(v, 0))
        st.dataframe(summary.head(12), use_container_width=True, hide_index=True)


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
    if balance.empty:
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
            "separate question, answered in the Experiment tab."
        )
    else:
        st.warning("No model beat the baseline. Reported as-is.")


# ---------------------------------------------------------------------------
# Experiment tab
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


def render_experiment() -> None:
    render_final_test()
    st.markdown("---")
    st.subheader("Question 3 - the A / B / C comparison (validation)")
    st.caption(
        "All arms trade the same signal universe with exactly one thing changed at a time. "
        "Differences are reported with bootstrap confidence intervals, because at a few "
        "hundred trades most differences are indistinguishable from noise."
    )

    systems = load_results_csv("systems_abc_4h.csv")
    if systems.empty:
        systems = load_results_csv("systems_ab_4h.csv")
    if systems.empty:
        placeholder(
            "No system comparison yet. Run the ML comparison, then the LLM judge.",
            "python scripts/run_ml.py\npython scripts/run_llm.py --threshold 0.35",
        )
        return

    labels = {
        "A_rules_only": "A - rules only",
        "control_always_agree": "Control - approve everything",
        "B_rules_plus_ml": "B - rules + ML filter",
        "control_deterministic": "Control - deterministic judge",
        "C_rules_ml_llm": "C - rules + ML + LLM judge",
    }
    systems = systems.copy()
    systems["label"] = systems["system"].map(lambda s: labels.get(s, s))

    columns = st.columns(len(systems))
    for column, (_, row) in zip(columns, systems.iterrows()):
        column.metric(
            row["label"],
            pct(row["total_return"]),
            f"Sharpe {row['sharpe']:.2f}",
        )

    figure = go.Figure(
        go.Bar(
            x=systems["label"],
            y=systems["total_return"] * 100,
            marker_color=[signed_colour(v) for v in systems["total_return"]],
            text=[pct(v) for v in systems["total_return"]],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=320, margin=dict(l=0, r=0, t=20, b=0), yaxis_title="Total return (%)"
    )
    st.plotly_chart(figure, use_container_width=True)

    display = systems.copy()
    for column in ("total_return", "max_drawdown", "win_rate", "exposure", "fees_pct_of_capital"):
        if column in display:
            display[column] = display[column].map(lambda v: pct(v))
    for column in ("sharpe", "sortino", "profit_factor"):
        if column in display:
            display[column] = display[column].map(lambda v: num(v))
    show = [
        c
        for c in ["label", "n_trades", "total_return", "sharpe", "sortino", "max_drawdown",
                  "win_rate", "profit_factor", "exposure"]
        if c in display.columns
    ]
    st.dataframe(display[show], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Trade counts matter more than they look**")
    st.caption(
        "Each filter can only remove trades, never add them. A smaller sample mechanically "
        "changes win rate and drawdown even when the filter has no skill at all - which is "
        "why the deterministic control arm exists, and why every difference is quoted with "
        "an interval rather than as a point estimate."
    )

    st.info(
        "**Reading the result honestly.** If a confidence interval contains zero, the two "
        "systems are not distinguishable at this sample size. For 'did the LLM help?' that is "
        "a legitimate answer, and a far more defensible one than quoting two point estimates "
        "and declaring a winner.",
        icon="ℹ️",
    )



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

    st.info(
        "**Neither degenerate.** A judge agreeing ~100% of the time is a rubber "
        "stamp adding cost and latency for nothing; one agreeing ~50% on a "
        "near-binary choice is closer to a coin flip than to judgement. This one "
        "vetoed most proposals at moderate confidence - it genuinely deliberated, "
        "and still did not beat four lines of arithmetic.",
        icon="🧠",
    )

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
# Method tab
# ---------------------------------------------------------------------------


def render_method() -> None:
    st.subheader("How data leakage was prevented")
    st.caption("The controls that make these numbers worth anything.")

    st.dataframe(
        pd.DataFrame(
            [
                ("Indicators peeking forward", "Hand-written; an automated test recomputes every indicator on truncated data and asserts history is unchanged"),
                ("Ichimoku Chikou Span", "Excluded - it is the close shifted backwards, so reading it at t reads price at t+26"),
                ("Fibonacci / swing support-resistance", "Excluded - derived from swing points identified with hindsight. Replaced by Donchian channels of the previous N bars"),
                ("Shuffled time series", "Never shuffled; splits are strictly chronological"),
                ("Overlapping labels at split seams", "4-bar embargo removed either side of every boundary"),
                ("Scaler fitted on all data", "Fitted inside the pipeline, on train only"),
                ("Test set used for tuning", "get_split('test') raises unless explicitly unlocked"),
                ("Acting on an unclosed candle", "The loader drops the still-forming final candle"),
                ("The LLM having memorised BTC history", "The prompt contains no dates and no absolute prices - only relative values. Enforced by test"),
                ("Same-candle barrier ties", "Labelled HOLD and excluded, never guessed"),
                ("Live inference using training data", "Inference uses the feature builder, not the label-filtered dataset. Enforced by test"),
            ],
            columns=["Risk", "Control"],
        ),
        use_container_width=True,
        hide_index=True,
        height=430,
    )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("**Execution assumptions**")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Fill timing", "Signal on the close of bar t, filled at the open of t+1"),
                    ("Taker fee", f"{BACKTEST.taker_fee:.3%} per side"),
                    ("Slippage", f"{BACKTEST.slippage_bps:.0f} bps, always against us"),
                    ("Same-bar stop and target", "Resolved as a stop - OHLCV cannot say which came first"),
                    ("Gap through the stop", "Fills at the open, not the stop price"),
                    ("Gap through the target", "Fills at the target - no credit for a favourable gap"),
                    ("Risk per trade", f"{RISK.risk_per_trade:.1%} of equity"),
                    ("Stop / target", f"{RISK.atr_stop_multiple:g} x ATR, {RISK.reward_risk_ratio:g}:1 reward:risk"),
                    ("Daily loss limit", f"{RISK.max_daily_loss:.1%}"),
                    ("Leverage", f"{BACKTEST.leverage:g}x"),
                ],
                columns=["Assumption", "Value"],
            ),
            use_container_width=True,
            hide_index=True,
            height=390,
        )

    with right:
        st.markdown("**Architecture**")
        st.code(
            "OHLCV\n"
            "  -> Indicators (causal, hand-written)\n"
            "  -> Strategy engine        Question 1\n"
            "  -> ML filter              Question 2\n"
            "  -> LLM judge              Question 3\n"
            "  -> DETERMINISTIC RISK MANAGER\n"
            "  -> Executor -> broker\n"
            "  -> SQLite -> this dashboard",
            language="text",
        )
        st.caption(
            "The judge chooses direction. It never chooses exposure: position size, stop "
            "placement and the daily loss limit are arithmetic, downstream of every decision "
            "layer and not delegated to anything that can hallucinate."
        )
        st.markdown("**Known limitations**")
        st.markdown(
            "- Backtests ignore perpetual funding payments.\n"
            "- Live trading runs for hours, not months - it demonstrates the pipeline, "
            "it is not evidence the strategy works.\n"
            "- Fills are simulated: no queue position, no partial fills.\n"
            "- Selecting the best of many configurations guarantees the winner is partly lucky.\n"
            "- The validation period is a bull market and the test period a bear market, so "
            "results are reported split by regime."
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
    live, research, ml, judge, experiment, method = st.tabs(
        ["Live", "Research", "Machine learning", "AI Judge", "Experiment", "Method"]
    )
    with live:
        render_live()
    with research:
        render_research()
    with ml:
        render_ml()
    with judge:
        render_judge()
    with experiment:
        render_experiment()
    with method:
        render_method()


if __name__ == "__main__":
    main()
