"""Train an EEG state classifier from feature CSV files.

Examples
--------
Train from one labeled feature table:
    python scripts/train_model.py --features-csv data/sample/eeg_features_sample.csv

Train from the original two-file format:
    python scripts/train_model.py --awake-csv testmo.csv --sleepy-csv testnham.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from eeg_state_classifier.io import load_training_table, save_json
from eeg_state_classifier.modeling import save_confusion_matrix_plot, save_model, train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EEG-state SVM classifier.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--features-csv", type=Path, help="CSV containing features and a label column."
    )
    input_group.add_argument(
        "--awake-csv",
        type=Path,
        help="Feature CSV for the awake/alert class. Use together with --sleepy-csv.",
    )
    parser.add_argument("--sleepy-csv", type=Path, help="Feature CSV for the sleepy/drowsy class.")
    parser.add_argument(
        "--label-column", default="label", help="Name of label column in --features-csv."
    )
    parser.add_argument(
        "--group-column", help="Participant ID column, excluded from model features."
    )
    parser.add_argument("--model-out", type=Path, default=Path("models/svm_eeg_state.joblib"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-tune", action="store_true", help="Disable GridSearchCV hyperparameter tuning."
    )
    return parser.parse_args()


def load_training_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    if args.features_csv:
        return load_training_table(args.features_csv, args.label_column, args.group_column)

    if args.group_column:
        raise ValueError("--group-column requires a single labeled --features-csv table")
    if args.sleepy_csv is None:
        raise ValueError("--sleepy-csv is required when --awake-csv is used")

    awake = pd.read_csv(args.awake_csv)
    sleepy = pd.read_csv(args.sleepy_csv)
    x = pd.concat([awake, sleepy], ignore_index=True)
    y = pd.Series(["awake"] * len(awake) + ["sleepy"] * len(sleepy), name="label")
    return x, y, None


def main() -> None:
    args = parse_args()
    x, y, groups = load_training_data(args)

    result = train_classifier(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        tune=not args.no_tune,
        groups=groups,
    )

    sources = [args.features_csv] if args.features_csv else [args.awake_csv, args.sleepy_csv]
    result.metrics["input_sha256"] = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    ]
    result.metrics["group_column"] = args.group_column
    save_model(result.model, args.model_out)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(result.metrics, args.reports_dir / "metrics.json")
    pd.DataFrame(result.confusion_matrix, index=result.labels, columns=result.labels).to_csv(
        args.reports_dir / "confusion_matrix.csv"
    )
    save_confusion_matrix_plot(
        result.confusion_matrix,
        result.labels,
        args.reports_dir / "figures" / "confusion_matrix.png",
    )

    print(f"Saved model to {args.model_out}")
    print(f"Validation: {result.metrics['validation']['strategy']}")
    print(f"Test accuracy: {result.metrics['test_accuracy']:.3f}")
    print(f"Reports saved to {args.reports_dir}")


if __name__ == "__main__":
    main()
