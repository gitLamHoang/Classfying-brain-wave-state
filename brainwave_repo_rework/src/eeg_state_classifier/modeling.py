"""Training and evaluation utilities for EEG state classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class TrainResult:
    """Artifacts returned after training."""

    model: Pipeline
    metrics: dict[str, Any]
    confusion_matrix: np.ndarray
    labels: list[str]


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build the baseline scaler + SVM classifier pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)),
        ]
    )


def _cv_splits(y: pd.Series, max_splits: int = 5) -> int:
    counts = y.value_counts()
    if counts.empty:
        raise ValueError("Cannot train on an empty target vector")
    return max(2, min(max_splits, int(counts.min())))


def train_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    tune: bool = True,
) -> TrainResult:
    """Train and evaluate an SVM EEG-state classifier.

    The function uses a stratified split so both classes are represented in
    train and test sets when possible. When ``tune=True``, it performs a small
    grid search over SVM regularization and kernel parameters.
    """

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of rows")
    if y.nunique() < 2:
        raise ValueError("At least two classes are required for classification")

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = build_pipeline(random_state=random_state)
    best_params: dict[str, Any] = {}
    cv_score = None

    if tune and y_train.value_counts().min() >= 2:
        cv = StratifiedKFold(
            n_splits=_cv_splits(y_train),
            shuffle=True,
            random_state=random_state,
        )
        param_grid = {
            "svc__C": [0.1, 1.0, 10.0],
            "svc__gamma": ["scale", "auto"],
            "svc__kernel": ["rbf", "linear"],
        }
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = float(search.best_score_)
    else:
        model = pipeline.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    metrics: dict[str, Any] = {
        "train_accuracy": float(model.score(x_train, y_train)),
        "test_accuracy": float(model.score(x_test, y_test)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "labels": labels,
        "best_params": best_params,
        "cv_accuracy": cv_score,
        "classification_report": report,
    }
    return TrainResult(model=model, metrics=metrics, confusion_matrix=cm, labels=labels)


def evaluate_classifier(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Evaluate a saved classifier on a labeled feature table."""

    y_pred = model.predict(x)
    labels = sorted(y.unique().tolist())
    return {
        "accuracy": float(model.score(x, y)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_model(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def save_confusion_matrix_plot(cm: np.ndarray, labels: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("EEG State Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
