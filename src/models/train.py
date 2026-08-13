"""Model training with chronological validation.

Leakage controls, all of which are load-bearing:

* **No shuffling.** Splits are chronological. Randomly shuffling a financial
  time series lets the model interpolate between bars minutes apart and
  produces accuracy figures that are pure fiction.
* **Embargo at every boundary.** Labels look ``horizon`` bars forward, so a
  sample just before a split boundary shares most of its future window with
  samples just after it. A buffer of ``embargo_bars`` is removed at each seam.
* **Scaler fitted on train only.** Fitting on everything leaks the test
  period's mean and variance into training.
* **A dummy baseline is always trained.** On an imbalanced three-class target,
  70% accuracy can be *worse* than always predicting the majority class.
  Without the baseline, a model's accuracy is an uninterpretable number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import ML, SPLIT, MLConfig, SplitConfig
from src.models.labels import CLASS_NAMES
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

CLASS_ORDER = [-1, 0, 1]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@dataclass
class Splits:
    """Chronologically separated train / validation / test index sets."""

    train: pd.Index
    validation: pd.Index
    test: pd.Index

    def describe(self) -> str:
        parts = []
        for name in ("train", "validation", "test"):
            index = getattr(self, name)
            if len(index):
                parts.append(f"{name}: {len(index):,} rows  {index[0].date()} -> {index[-1].date()}")
            else:
                parts.append(f"{name}: empty")
        return "\n".join(parts)


def chronological_splits(
    index: pd.Index,
    config: SplitConfig = SPLIT,
) -> Splits:
    """Split an index by date, removing an embargo buffer at each seam.

    The embargo is applied to the *start* of the validation and test blocks,
    which is the side that would otherwise inherit overlapping label windows
    from the block before it.
    """
    train_end = pd.Timestamp(config.train_end, tz="UTC")
    validation_end = pd.Timestamp(config.validation_end, tz="UTC")
    gap = config.embargo_bars

    train = index[index <= train_end]
    validation = index[(index > train_end) & (index <= validation_end)]
    test = index[index > validation_end]

    if gap:
        validation = validation[gap:]
        test = test[gap:]

    return Splits(train=train, validation=validation, test=test)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_models(config: MLConfig = ML) -> Dict[str, Pipeline]:
    """The model zoo: a baseline, a linear model and an ensemble.

    Kept deliberately small. Two real models plus a baseline is enough to
    demonstrate supervised learning properly; a third would add runtime and
    another chance to overfit the validation set without adding insight.
    """
    return {
        "dummy": Pipeline(
            [("model", DummyClassifier(strategy="most_frequent"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=config.logreg_max_iter,
                        class_weight=config.class_weight,
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=config.rf_n_estimators,
                        max_depth=config.rf_max_depth,
                        min_samples_leaf=config.rf_min_samples_leaf,
                        class_weight=config.class_weight,
                        random_state=config.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


@dataclass
class TrainedModel:
    """A fitted pipeline plus everything needed to report on it."""

    name: str
    pipeline: Pipeline
    metrics: Dict[str, Any] = field(default_factory=dict)
    confusion: Optional[pd.DataFrame] = None
    importances: Optional[pd.Series] = None

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.pipeline.predict(features), index=features.index, name="prediction"
        ).astype("int8")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        proba = self.pipeline.predict_proba(features)
        classes = self.pipeline.named_steps["model"].classes_
        return pd.DataFrame(proba, index=features.index, columns=[int(c) for c in classes])


def evaluate(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: str = "",
) -> Dict[str, Any]:
    """Classification metrics appropriate for an imbalanced three-class target.

    ``balanced_accuracy`` is the headline: plain accuracy on a target that is
    70% HOLD rewards a model for predicting HOLD forever.
    """
    return {
        "split": label,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_hold_share": float((y_pred == 0).mean()),
        "predicted_long_share": float((y_pred == 1).mean()),
        "predicted_short_share": float((y_pred == -1).mean()),
    }


def confusion_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Labelled confusion matrix, rows actual and columns predicted."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    names = [CLASS_NAMES[c] for c in CLASS_ORDER]
    return pd.DataFrame(matrix, index=[f"actual_{n}" for n in names], columns=[f"pred_{n}" for n in names])


def feature_importances(pipeline: Pipeline, columns: List[str]) -> Optional[pd.Series]:
    """Importances for tree models, absolute mean coefficients for linear ones."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).mean(axis=0)
    else:
        return None
    return pd.Series(values, index=columns).sort_values(ascending=False)


def train_models(
    features: pd.DataFrame,
    target: pd.Series,
    splits: Splits,
    config: MLConfig = ML,
) -> Dict[str, TrainedModel]:
    """Fit every model on train and score it on train and validation.

    The test split is not touched here. It is scored once, at the end of the
    project, by the final comparison script.
    """
    train_index = features.index.intersection(splits.train)
    validation_index = features.index.intersection(splits.validation)

    if len(train_index) == 0 or len(validation_index) == 0:
        raise ValueError(
            f"Empty split (train={len(train_index)}, validation={len(validation_index)}). "
            "Check that the data range covers the configured split dates."
        )

    x_train, y_train = features.loc[train_index], target.loc[train_index]
    x_validation, y_validation = features.loc[validation_index], target.loc[validation_index]

    logger.info(
        "Training on %d rows, validating on %d rows, %d features",
        len(x_train),
        len(x_validation),
        features.shape[1],
    )

    trained: Dict[str, TrainedModel] = {}
    for name, pipeline in build_models(config).items():
        pipeline.fit(x_train, y_train)

        train_metrics = evaluate(y_train, pd.Series(pipeline.predict(x_train), index=train_index), "train")
        validation_predictions = pd.Series(pipeline.predict(x_validation), index=validation_index)
        validation_metrics = evaluate(y_validation, validation_predictions, "validation")

        trained[name] = TrainedModel(
            name=name,
            pipeline=pipeline,
            metrics={"train": train_metrics, "validation": validation_metrics},
            confusion=confusion_frame(y_validation, validation_predictions),
            importances=feature_importances(pipeline, list(features.columns)),
        )
        logger.info(
            "%-20s train balanced acc %.3f | validation balanced acc %.3f",
            name,
            train_metrics["balanced_accuracy"],
            validation_metrics["balanced_accuracy"],
        )

    return trained


def comparison_table(trained: Dict[str, TrainedModel]) -> pd.DataFrame:
    """Model scores side by side, with the dummy's uplift made explicit.

    ``uplift_vs_dummy`` is the only column that means anything on its own. A
    positive value says the model learned something; zero or negative says it
    did not, however impressive the raw accuracy looks.
    """
    rows = []
    for name, model in trained.items():
        validation = model.metrics["validation"]
        rows.append(
            {
                "model": name,
                "train_balanced_acc": model.metrics["train"]["balanced_accuracy"],
                "val_accuracy": validation["accuracy"],
                "val_balanced_acc": validation["balanced_accuracy"],
                "val_f1_macro": validation["f1_macro"],
                "pred_hold_share": validation["predicted_hold_share"],
            }
        )

    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["model"] == "dummy", "val_balanced_acc"]
    if not baseline.empty:
        frame["uplift_vs_dummy"] = frame["val_balanced_acc"] - float(baseline.iloc[0])
    frame["overfit_gap"] = frame["train_balanced_acc"] - frame["val_balanced_acc"]
    return frame.sort_values("val_balanced_acc", ascending=False).reset_index(drop=True)
