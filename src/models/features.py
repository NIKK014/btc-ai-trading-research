"""Feature matrix construction.

Only scale-free features are used. That constraint is doing real work: a model
given raw BTC price learns that "price is 60,000" implies late 2024, which is
memorisation of the calendar rather than of market structure, and it collapses
the moment prices leave the training range. Every feature here is a ratio, a
bounded oscillator, or a distance expressed as a fraction of price, so a
feature vector from 2020 and one from 2026 are directly comparable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import LABELS, LabelConfig
from src.indicators.indicators import ML_FEATURE_COLUMNS, IndicatorSpec, add_indicators
from src.models.labels import triple_barrier_labels, usable_mask
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_features(
    ohlcv: pd.DataFrame,
    spec: Optional[IndicatorSpec] = None,
    columns: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Indicator-derived feature matrix, warm-up rows removed."""
    enriched = add_indicators(ohlcv, spec)
    selected = list(columns or ML_FEATURE_COLUMNS)
    features = enriched[selected].replace([np.inf, -np.inf], np.nan)
    return features


def build_dataset(
    ohlcv: pd.DataFrame,
    spec: Optional[IndicatorSpec] = None,
    label_config: LabelConfig = LABELS,
    columns: Optional[Tuple[str, ...]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Aligned feature matrix, target vector and label diagnostics.

    Rows are dropped where features are still warming up, where the label
    window is incomplete, or where the same-candle tie makes the label
    unknowable. Everything is returned on a common index so nothing downstream
    has to re-align and risk an off-by-one.

    Returns:
        ``(X, y, labels)`` - features, target, and the full label frame
        including the rows that were excluded, for reporting.
    """
    enriched = add_indicators(ohlcv, spec)
    features = enriched[list(columns or ML_FEATURE_COLUMNS)].replace(
        [np.inf, -np.inf], np.nan
    )
    labels = triple_barrier_labels(enriched, label_config)

    valid = usable_mask(labels, label_config) & features.notna().all(axis=1)
    logger.info(
        "Dataset: %d of %d bars usable (%.1f%%)",
        int(valid.sum()),
        len(ohlcv),
        100.0 * valid.mean(),
    )
    return features.loc[valid], labels.loc[valid, "label"], labels


def feature_report(features: pd.DataFrame) -> pd.DataFrame:
    """Distribution summary, used to sanity-check the matrix before training."""
    described = features.describe().T[["mean", "std", "min", "max"]]
    described["nan_share"] = features.isna().mean()
    described["zero_variance"] = features.std() < 1e-12
    return described.round(4)


def correlated_pairs(features: pd.DataFrame, threshold: float = 0.95) -> List[Dict[str, float]]:
    """Feature pairs above a correlation threshold.

    Near-duplicate features do not break tree models but they do split
    importance between them, which makes the "which indicators mattered"
    analysis misleading. Worth reporting even when nothing is dropped.
    """
    matrix = features.corr().abs()
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    pairs = [
        {"a": a, "b": b, "correlation": float(value)}
        for a, row in upper.iterrows()
        for b, value in row.items()
        if pd.notna(value) and value >= threshold
    ]
    return sorted(pairs, key=lambda p: -p["correlation"])
