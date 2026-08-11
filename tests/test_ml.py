"""Training-pipeline and ML-filter tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import SplitConfig
from src.models.features import build_dataset, correlated_pairs
from src.models.predict import apply_ml_filter, filter_diagnostics, predictions_frame
from src.models.train import chronological_splits, comparison_table, train_models


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def index_over(start: str, end: str, freq: str = "4h") -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, tz="UTC")


def test_splits_are_chronological_and_disjoint():
    index = index_over("2020-01-01", "2026-01-01")
    splits = chronological_splits(index)

    assert len(splits.train) and len(splits.validation) and len(splits.test)
    assert splits.train.max() < splits.validation.min()
    assert splits.validation.max() < splits.test.min()
    assert splits.train.intersection(splits.validation).empty
    assert splits.validation.intersection(splits.test).empty
    assert splits.train.intersection(splits.test).empty


def test_embargo_removes_bars_at_each_seam():
    """Adjacent samples share most of their label window, so the seam leaks."""
    index = index_over("2020-01-01", "2026-01-01")
    with_gap = chronological_splits(index, SplitConfig(embargo_bars=4))
    without_gap = chronological_splits(index, SplitConfig(embargo_bars=0))

    assert len(with_gap.validation) == len(without_gap.validation) - 4
    assert len(with_gap.test) == len(without_gap.test) - 4
    assert with_gap.validation.min() > without_gap.validation.min()


def test_split_boundaries_follow_the_configured_dates():
    index = index_over("2020-01-01", "2026-01-01")
    config = SplitConfig(train_end="2022-06-30", validation_end="2024-06-30", embargo_bars=0)
    splits = chronological_splits(index, config)

    assert splits.train.max() <= pd.Timestamp("2022-06-30", tz="UTC")
    assert splits.validation.min() > pd.Timestamp("2022-06-30", tz="UTC")
    assert splits.validation.max() <= pd.Timestamp("2024-06-30", tz="UTC")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_bundle():
    """Train once on a synthetic multi-year series and reuse across tests."""
    from conftest import make_ohlcv

    ohlcv = make_ohlcv(periods=9_000, freq="4h", seed=11)
    ohlcv.index = pd.date_range("2020-06-01", periods=len(ohlcv), freq="4h", tz="UTC")

    features, target, _ = build_dataset(ohlcv)
    splits = chronological_splits(features.index)
    models = train_models(features, target, splits)
    return features, target, splits, models


def test_every_model_trains_including_the_baseline(trained_bundle):
    _, _, _, models = trained_bundle
    assert set(models) == {"dummy", "logistic_regression", "random_forest"}


def test_dummy_baseline_is_reported_so_accuracy_is_interpretable(trained_bundle):
    """Without this row, "39% accuracy" is an uninterpretable number."""
    _, _, _, models = trained_bundle
    table = comparison_table(models)

    assert "uplift_vs_dummy" in table.columns
    dummy = table[table["model"] == "dummy"].iloc[0]
    assert dummy["uplift_vs_dummy"] == pytest.approx(0.0)
    assert dummy["val_balanced_acc"] == pytest.approx(1 / 3, abs=0.02), (
        "a most-frequent classifier scores 1/3 balanced accuracy on 3 classes"
    )


def test_scaling_happens_inside_the_pipeline_so_it_cannot_leak(trained_bundle):
    """The scaler must be fitted within cross-validation boundaries, never on
    the full dataset."""
    _, _, _, models = trained_bundle
    pipeline = models["logistic_regression"].pipeline
    assert "scaler" in pipeline.named_steps

    scaler = pipeline.named_steps["scaler"]
    assert hasattr(scaler, "mean_"), "scaler was never fitted"


def test_models_predict_only_valid_classes(trained_bundle):
    features, _, splits, models = trained_bundle
    subset = features.loc[splits.validation].head(200)

    for model in models.values():
        predictions = model.predict(subset)
        assert set(predictions.unique()) <= {-1, 0, 1}


def test_probabilities_are_a_valid_distribution(trained_bundle):
    features, _, splits, models = trained_bundle
    subset = features.loc[splits.validation].head(200)

    frame = predictions_frame(models["random_forest"], subset)
    totals = frame[["prob_long", "prob_short", "prob_hold"]].sum(axis=1)
    np.testing.assert_allclose(totals.to_numpy(), 1.0, rtol=1e-6)
    assert frame["confidence"].between(0.0, 1.0).all()


def test_empty_split_raises_a_clear_error(trained_bundle):
    features, target, _, _ = trained_bundle
    from src.models.train import Splits

    empty = Splits(train=features.index[:0], validation=features.index, test=features.index[:0])
    with pytest.raises(ValueError, match="Empty split"):
        train_models(features, target, empty)


# ---------------------------------------------------------------------------
# The ML filter
# ---------------------------------------------------------------------------


def signals_and_predictions(signal_values, approve):
    index = pd.date_range("2024-01-01", periods=len(signal_values), freq="4h", tz="UTC")
    signals = pd.Series(signal_values, index=index, dtype="int8")
    predictions = pd.DataFrame(
        {
            "prediction": signals.where(pd.Series(approve, index=index), 0),
            "prob_long": [0.9 if a else 0.1 for a in approve],
            "prob_short": [0.9 if a else 0.1 for a in approve],
            "prob_hold": 0.1,
        },
        index=index,
    )
    return signals, predictions


def test_entry_mode_lets_an_approved_position_run():
    """The default mode gates the entry decision, then gets out of the way."""
    signals, predictions = signals_and_predictions(
        [1, 1, 1, 1, 1], [True, False, False, False, False]
    )
    filtered = apply_ml_filter(signals, predictions, 0.5, mode="entry")
    assert filtered.tolist() == [1, 1, 1, 1, 1]


def test_per_bar_mode_fragments_the_same_position():
    """Demonstrates why per-bar filtering is not the default: it shreds a
    single position into fragments, each paying a round-trip fee."""
    signals, predictions = signals_and_predictions(
        [1, 1, 1, 1, 1], [True, False, False, False, True]
    )
    filtered = apply_ml_filter(signals, predictions, 0.5, mode="per_bar")
    assert filtered.tolist() == [1, 0, 0, 0, 1]


def test_a_vetoed_entry_is_not_retried_on_every_bar():
    """Otherwise the filter merely delays entry until the model wavers."""
    signals, predictions = signals_and_predictions(
        [1, 1, 1, 1, 1], [False, False, False, True, True]
    )
    filtered = apply_ml_filter(signals, predictions, 0.5, mode="entry")
    assert filtered.tolist() == [0, 0, 0, 0, 0], "veto holds for the whole signal run"


def test_the_filter_resets_when_the_signal_does():
    signals, predictions = signals_and_predictions(
        [1, 1, 0, 1, 1], [False, False, False, True, True]
    )
    filtered = apply_ml_filter(signals, predictions, 0.5, mode="entry")
    assert filtered.tolist() == [0, 0, 0, 1, 1]


def test_the_filter_can_only_remove_signals_never_invent_them():
    """System B must trade a subset of System A's opportunities, or the
    comparison is not measuring what it claims to."""
    rng = np.random.default_rng(5)
    values = rng.choice([-1, 0, 1], size=300)
    approve = rng.random(300) > 0.5
    signals, predictions = signals_and_predictions(values, approve)

    filtered = apply_ml_filter(signals, predictions, 0.5, mode="entry")
    active = filtered != 0
    assert (filtered[active] == signals[active]).all(), "direction was altered"
    assert (signals[active] != 0).all(), "a signal was invented where there was none"


def test_filter_diagnostics_report_the_veto_rate():
    signals, predictions = signals_and_predictions([1, 1, -1, -1], [True, True, False, False])
    filtered = apply_ml_filter(signals, predictions, 0.5, mode="entry")
    diagnostics = filter_diagnostics(signals, filtered)

    assert diagnostics["signal_bars"] == 4
    assert diagnostics["short_veto_rate"] == pytest.approx(1.0)
    assert diagnostics["long_veto_rate"] == pytest.approx(0.0)


def test_unknown_filter_mode_is_rejected():
    signals, predictions = signals_and_predictions([1], [True])
    with pytest.raises(ValueError, match="Unknown filter mode"):
        apply_ml_filter(signals, predictions, 0.5, mode="sideways")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_dataset_rows_are_aligned_and_complete(ohlcv):
    features, target, labels = build_dataset(ohlcv)

    assert features.index.equals(target.index)
    assert not features.isna().any().any()
    assert set(target.unique()) <= {-1, 0, 1}
    assert len(features) < len(ohlcv), "warm-up and incomplete windows must be dropped"


def test_features_exclude_absolute_price(ohlcv):
    """A model given raw price learns the calendar, not the market."""
    features, _, _ = build_dataset(ohlcv)
    for column in ("open", "high", "low", "close", "vwap", "bb_upper", "ema_200"):
        assert column not in features.columns


def test_correlated_pairs_finds_duplicated_information():
    frame = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 2 + 1, "c": np.random.default_rng(0).normal(size=100)})
    pairs = correlated_pairs(frame, threshold=0.95)
    assert any({p["a"], p["b"]} == {"a", "b"} for p in pairs)


def test_live_prediction_must_not_use_the_training_dataset(ohlcv):
    """The bug this test exists to prevent.

    ``build_dataset`` drops the trailing rows whose *label* window is
    incomplete - correct for training, fatal for live inference, because the
    most recent candle can never have a label and would therefore never
    receive a prediction. The live loop would then run permanently with
    ml_prediction=0, silently disabling the ML and LLM layers while appearing
    to work.
    """
    from src.models.features import build_dataset, build_features

    latest = ohlcv.index[-1]
    training, _, _ = build_dataset(ohlcv)
    inference = build_features(ohlcv).dropna()

    assert latest not in training.index, "training set correctly excludes unlabelled rows"
    assert latest in inference.index, "inference must cover the most recent candle"
    assert len(inference) > len(training)


def test_the_live_loop_uses_the_inference_path():
    """Guard against a future refactor reintroducing the bug."""
    import inspect
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
    prediction_block = source[source.index("# ML prediction for the latest closed candle."):]
    prediction_block = prediction_block[: prediction_block.index("# Decide.")]

    assert "build_features(" in prediction_block
    assert "build_dataset(" not in prediction_block, (
        "live inference must not use the label-filtered training dataset"
    )
