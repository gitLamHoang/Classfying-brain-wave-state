"""Predict EEG state from a raw text recording using sliding windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eeg_state_classifier.features import FEATURE_COLUMNS, extract_features, make_windows
from eeg_state_classifier.io import read_raw_txt
from eeg_state_classifier.modeling import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding-window EEG-state prediction.")
    parser.add_argument("--raw-txt", type=Path, required=True, help="Raw EEG text file, one sample per line.")
    parser.add_argument("--model", type=Path, required=True, help="Trained .joblib model.")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/predictions.csv"))
    parser.add_argument("--sampling-rate", type=float, default=512.0)
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--no-filter", action="store_true", help="Disable 0.5-40 Hz band-pass filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_raw_txt(args.raw_txt)
    model = load_model(args.model)

    rows: list[dict[str, object]] = []
    for window in make_windows(
        samples,
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    ):
        features = extract_features(
            window.values,
            sampling_rate=args.sampling_rate,
            apply_filter=not args.no_filter,
        )
        x = pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        prediction = model.predict(x)[0]
        row: dict[str, object] = {
            "start_seconds": window.start_sample / args.sampling_rate,
            "end_seconds": window.end_sample / args.sampling_rate,
            "prediction": prediction,
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x)[0]
            for class_name, probability in zip(model.classes_, probabilities, strict=False):
                row[f"probability_{class_name}"] = float(probability)

        rows.append(row)

    if not rows:
        raise ValueError("Recording is shorter than the requested prediction window.")

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(output.tail(min(10, len(output))).to_string(index=False))
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
