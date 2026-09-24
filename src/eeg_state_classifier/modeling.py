"""Training and evaluation utilities for EEG state classifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
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
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _check_class_coverage(y: pd.Series, train: np.ndarray, test: np.ndarray, context: str) -> None:
    expected = set(y.unique())
    if set(y.iloc[train].unique()) != expected or set(y.iloc[test].unique()) != expected:
        raise ValueError(
            f"{context} must contain every class on both sides. Collect more independent "
            "participants or predefine a suitable split; do not select a seed using test scores."
        )


def _check_groups(groups: pd.Series, train: np.ndarray, test: np.ndarray) -> dict[str, int]:
    train_groups = set(groups.iloc[train])
    test_groups = set(groups.iloc[test])
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError("Participant overlap detected in a validation split")
    return {
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": len(overlap),
    }


def train_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    tune: bool = True,
    groups: pd.Series | None = None,
) -> TrainResult:
    """Fit using a fixed held-out split and training-only hyperparameter selection.

    Provide participant IDs as ``groups`` for independent participant evaluation.
    Without groups, row splitting is only a synthetic pipeline smoke test: windows
    from one recording or participant are not independent observations.
    """
    if len(x) != len(y) or not x.index.equals(y.index):
        raise ValueError("x and y must contain the same rows in the same index order")
    if x.empty or x.columns.duplicated().any():
        raise ValueError("Features must be nonempty and have unique column names")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
        raise ValueError("All features must be numeric; remove identifiers and metadata")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Features must contain only finite values")
    if y.isna().any() or y.astype(str).str.strip().eq("").any():
        raise ValueError("Labels must not be missing or blank")
    if y.nunique() < 2:
        raise ValueError("At least two classes are required for classification")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    positions = np.arange(len(x))
    if groups is not None:
        if len(groups) != len(x) or not groups.index.equals(x.index):
            raise ValueError("groups must match feature rows in the same index order")
        if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError("Participant IDs must not be missing or blank")
        if groups.name is not None and groups.name in x.columns:
            raise ValueError("The participant group column must not be included in features")
        if groups.nunique() < 2:
            raise ValueError("At least two independent participants are required")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train, test = next(splitter.split(x, y, groups))
        validation: dict[str, Any] = {
            "strategy": "participant_disjoint",
            **_check_groups(groups, train, test),
        }
    else:
        train, test = train_test_split(
            positions, test_size=test_size, random_state=random_state, stratify=y
        )
        validation = {
            "strategy": "row_split_synthetic_smoke_only",
            "warning": "Row splitting does not establish generalization to new participants.",
        }
    _check_class_coverage(y, train, test, "Holdout")
    x_train, x_test = x.iloc[train], x.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]
    pipeline = build_pipeline(random_state=random_state)
    best_params: dict[str, Any] = {}
    cv_score = None
    cv_indices: list[tuple[np.ndarray, np.ndarray]] = []
    fold_audit = []

    if tune:
        if groups is not None:
            training_groups = groups.iloc[train]
            groups_per_class = (
                pd.DataFrame({"label": y_train.to_numpy(), "group": training_groups.to_numpy()})
                .groupby("label")["group"]
                .nunique()
            )
            n_splits = min(5, int(groups_per_class.min()))
            if n_splits < 2:
                raise ValueError("Tuning requires each class in at least two training participants")
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_indices = list(cv.split(x_train, y_train, training_groups))
        else:
            n_splits = min(5, int(y_train.value_counts().min()))
            if n_splits < 2:
                raise ValueError("Tuning requires at least two training rows per class")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_indices = list(cv.split(x_train, y_train))
        for fold, (fit, valid) in enumerate(cv_indices, start=1):
            _check_class_coverage(y_train, fit, valid, f"Tuning fold {fold}")
            audit = {"fold": fold, "n_train": len(fit), "n_validation": len(valid)}
            if groups is not None:
                audit.update(_check_groups(training_groups, fit, valid))
            fold_audit.append(audit)
        search = GridSearchCV(
            pipeline,
            param_grid={
                "svc__C": [0.1, 1.0, 10.0],
                "svc__gamma": ["scale", "auto"],
                "svc__kernel": ["rbf", "linear"],
            },
            cv=cv_indices,
            scoring="accuracy",
            n_jobs=1,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = float(search.best_score_)
    else:
        model = pipeline.fit(x_train, y_train)

    membership = {
        "train_positions": train.tolist(),
        "test_positions": test.tolist(),
        "cv": [[fit.tolist(), valid.tolist()] for fit, valid in cv_indices],
    }
    validation.update(
        {
            "seed": random_state,
            "requested_test_fraction": test_size,
            "tuning_folds": fold_audit,
            "split_sha256": hashlib.sha256(
                json.dumps(membership, sort_keys=True).encode()
            ).hexdigest(),
            "probabilities_calibrated": False,
        }
    )
    y_pred = model.predict(x_test)
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "validation": validation,
        "feature_columns": x.columns.tolist(),
        "train_accuracy": float(model.score(x_train, y_train)),
        "test_accuracy": float(model.score(x_test, y_test)),
        "n_train": len(x_train),
        "n_test": len(x_test),
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
