"""Generate a small synthetic EEG-like dataset for demos and tests.

This is not a scientific dataset. It exists so recruiters and reviewers can run
this repository end-to-end without access to private participant recordings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_state_classifier.features import FEATURE_COLUMNS, extract_features


def synthetic_eeg(
    label: str, seconds: float, sampling_rate: int, rng: np.random.Generator
) -> np.ndarray:
    t = np.arange(int(seconds * sampling_rate)) / sampling_rate
    noise = rng.normal(0, 0.35, size=len(t))

    if label == "awake":
        # More alpha/beta activity.
        signal = 1.2 * np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 18 * t)
    elif label == "sleepy":
        # More theta/delta activity.
        signal = 1.4 * np.sin(2 * np.pi * 6 * t) + 1.0 * np.sin(2 * np.pi * 2.5 * t)
    else:
        raise ValueError(f"Unknown label: {label}")

    drift = 0.05 * np.sin(2 * np.pi * 0.2 * t)
    return signal + noise + drift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic sample data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--sampling-rate", type=int, default=512)
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument(
        "--participants", type=int, help="Generate this many synthetic participant IDs."
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 1 or (args.participants is not None and args.participants < 2):
        raise ValueError("Use positive samples per class and at least two synthetic participants")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    rows = []
    for participant in range(args.participants or 1):
        gain = rng.uniform(0.8, 1.2) if args.participants else 1.0
        for label in ["awake", "sleepy"]:
            for _ in range(args.samples_per_class):
                signal = gain * synthetic_eeg(label, args.window_seconds, args.sampling_rate, rng)
                features = extract_features(
                    signal, sampling_rate=args.sampling_rate, apply_filter=True
                )
                metadata = (
                    {"participant_id": f"synthetic-{participant:02d}"} if args.participants else {}
                )
                rows.append({**features, "label": label, **metadata})

    columns = [*FEATURE_COLUMNS, "label"] + (["participant_id"] if args.participants else [])
    df = pd.DataFrame(rows, columns=columns)
    filename = "eeg_features_grouped.csv" if args.participants else "eeg_features_sample.csv"
    df.to_csv(args.output_dir / filename, index=False)

    # One longer recording for the prediction demo.
    awake_recording = synthetic_eeg("awake", seconds=30, sampling_rate=args.sampling_rate, rng=rng)
    np.savetxt(args.output_dir / "eeg_raw_awake_sample.txt", awake_recording, fmt="%.8f")

    sleepy_recording = synthetic_eeg(
        "sleepy", seconds=30, sampling_rate=args.sampling_rate, rng=rng
    )
    np.savetxt(args.output_dir / "eeg_raw_sleepy_sample.txt", sleepy_recording, fmt="%.8f")

    print(f"Saved sample feature table and raw recordings to {args.output_dir}")


if __name__ == "__main__":
    main()
