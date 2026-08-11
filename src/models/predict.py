"""The ML filter - System B.

System B must trade the **same opportunity set** as System A, with exactly one
thing changed: each signal now needs the model's agreement to be acted on. If
the model were allowed to generate its own entries, A and B would be trading
different markets and the comparison would answer nothing.

So the rule is deliberately narrow:

    Take System A's signal only if the model predicts the same direction with
    probability at least ``threshold``. Otherwise stand aside.

Two consequences to keep in view when reading the results. The filter can only
ever *remove* trades, so System B will always have a smaller sample and wider
confidence intervals than System A. And removing trades mechanically changes
win rate and drawdown even if the model has no skill at all - which is exactly
why the comparison needs the deterministic-agreement control rather than a
side-by-side of two point estimates.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import ML, MLConfig
from src.models.train import TrainedModel
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def predictions_frame(
    model: TrainedModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Model prediction and per-class probabilities, aligned to ``features``.

    Returns columns ``prediction``, ``prob_long``, ``prob_short``,
    ``prob_hold`` and ``confidence`` (the probability assigned to the predicted
    class).
    """
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)

    frame = pd.DataFrame(index=features.index)
    frame["prediction"] = predicted
    for value, name in ((1, "prob_long"), (-1, "prob_short"), (0, "prob_hold")):
        frame[name] = probabilities[value] if value in probabilities.columns else 0.0

    lookup = {1: "prob_long", -1: "prob_short", 0: "prob_hold"}
    frame["confidence"] = [
        frame.at[timestamp, lookup[int(value)]]
        for timestamp, value in predicted.items()
    ]
    return frame


def approval_mask(
    signals: pd.Series,
    predictions: pd.DataFrame,
    threshold: float = ML.confidence_threshold,
    *,
    require_agreement: bool = True,
) -> pd.Series:
    """Per-bar boolean: does the model endorse the strategy's direction here?"""
    aligned = predictions.reindex(signals.index)
    direction_probability = pd.Series(0.0, index=signals.index)
    direction_probability[signals == 1] = aligned.loc[signals == 1, "prob_long"]
    direction_probability[signals == -1] = aligned.loc[signals == -1, "prob_short"]

    approved = direction_probability >= threshold
    if require_agreement:
        approved &= aligned["prediction"] == signals
    return approved.fillna(False)


def apply_ml_filter(
    signals: pd.Series,
    predictions: pd.DataFrame,
    threshold: float = ML.confidence_threshold,
    *,
    require_agreement: bool = True,
    mode: str = "entry",
) -> pd.Series:
    """Gate strategy signals on model agreement.

    Args:
        signals: System A's desired direction per bar.
        predictions: Output of :func:`predictions_frame`.
        threshold: Minimum probability the model must assign to the strategy's
            direction.
        require_agreement: If False, only the probability threshold applies -
            an ablation separating "the model agreed" from "the model was
            confident".
        mode: ``"entry"`` gates only the decision to open a position and then
            lets the strategy manage it; ``"per_bar"`` requires the model's
            approval on every single bar the position is held.

    Why ``entry`` is the default
    ----------------------------
    Strategy signals are a *persistent state*, not a stream of independent
    decisions. Requiring the model's approval on every bar does not select
    better trades - it shreds one good position into a dozen fragments,
    paying a round-trip fee on each and destroying the trend-following premise
    the strategy depends on. Measured on this project's data, ``per_bar``
    filtering cut exposure from 51% to 11% while leaving the trade count
    almost unchanged: it was chopping positions up, not filtering them out.

    ``entry`` mode asks the question the research is actually about: *given
    that the rules want to open a trade here, does the model agree it is worth
    taking?* Once a position is open, the strategy's own exit logic governs.

    Returns:
        The filtered signal series, aligned to ``signals``.
    """
    if mode not in {"entry", "per_bar"}:
        raise ValueError(f"Unknown filter mode {mode!r}")

    approved = approval_mask(signals, predictions, threshold, require_agreement=require_agreement)

    if mode == "per_bar":
        return signals.where(approved, 0).astype("int8").rename("signal")

    return gate_entries(signals, approved)


def gate_entries(signals: pd.Series, approved: pd.Series) -> pd.Series:
    """Apply an approval mask to entry decisions only.

    Shared by the ML filter and every judge arm, so Systems B and C gate
    trades through identical machinery and any difference between them comes
    from the decision itself rather than from how it was applied.

    Once an entry is approved the position runs until the strategy stands
    down. A vetoed entry is not retried for the remainder of that signal run,
    otherwise the filter would merely delay entry until the approver happened
    to waver.
    """
    approved = approved.reindex(signals.index).fillna(False)
    raw = signals.to_numpy(dtype=np.int8)
    ok = approved.to_numpy(dtype=bool)
    out = np.zeros(len(raw), dtype=np.int8)

    state = 0
    #: Set when an entry was vetoed, so the same signal run is not re-tried on
    #: every subsequent bar until it happens to be approved.
    vetoed = 0
    for i in range(len(raw)):
        desired = raw[i]
        if desired == 0:
            state = 0
            vetoed = 0
        elif desired == state:
            pass  # already in the position the rules want; let it run
        else:
            if desired != vetoed and ok[i]:
                state = desired
                vetoed = 0
            else:
                if desired != vetoed:
                    vetoed = desired
                state = 0
        out[i] = state

    return pd.Series(out, index=signals.index, dtype="int8", name="signal")


def filter_diagnostics(original: pd.Series, filtered: pd.Series) -> Dict[str, float]:
    """How much the filter removed, and from which side.

    A filter that vetoes almost nothing cannot explain a performance
    difference; one that vetoes almost everything has produced a sample too
    small to measure. Both are worth catching before interpreting the results.
    """
    active = original != 0
    if not active.any():
        return {}

    kept = (filtered != 0) & active
    return {
        "signal_bars": int(active.sum()),
        "kept_bars": int(kept.sum()),
        "veto_rate": float(1.0 - kept.sum() / active.sum()),
        "long_veto_rate": float(
            1.0 - ((filtered == 1).sum() / max((original == 1).sum(), 1))
        ),
        "short_veto_rate": float(
            1.0 - ((filtered == -1).sum() / max((original == -1).sum(), 1))
        ),
    }


def deterministic_agreement_filter(
    signals: pd.Series,
    predictions: pd.DataFrame,
    threshold: float = ML.confidence_threshold,
) -> pd.Series:
    """The control arm for the LLM experiment.

    A four-line rule: trade only when the strategy and the model agree, above
    a confidence threshold. The LLM judge receives the same information and is
    free to do something more sophisticated - so if System C does not beat
    this, the honest conclusion is that the LLM added nothing a simple rule
    could not.
    """
    return apply_ml_filter(signals, predictions, threshold, require_agreement=True)


def sweep_threshold(
    signals: pd.Series,
    predictions: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Trade count and veto rate across candidate thresholds.

    Tuned on validation only. The point is to choose a threshold that filters
    meaningfully while leaving enough trades to measure - not to find whichever
    threshold happens to maximise validation return, which is just overfitting
    with extra steps.
    """
    candidates = thresholds or [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    rows = []
    for threshold in candidates:
        filtered = apply_ml_filter(signals, predictions, threshold)
        diagnostics = filter_diagnostics(signals, filtered)
        if diagnostics:
            rows.append({"threshold": threshold, **diagnostics})
    return pd.DataFrame(rows)
