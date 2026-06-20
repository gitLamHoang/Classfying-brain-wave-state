"""Input/output helpers for EEG experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_raw_txt(path: str | Path) -> np.ndarray:
    """Read one EEG sample per line from a text file.

    Empty lines are ignored. Non-numeric rows raise a clear error so data
    collection issues are caught early.
    """

    samples: list[float] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                samples.append(float(value))
            except ValueError as exc:
                raise ValueError(f"Non-numeric value at {path}:{line_number}: {value!r}") from exc
    if not samples:
        raise ValueError(f"No EEG samples found in {path}")
    return np.asarray(samples, dtype=float)


def load_feature_table(path: str | Path, label_column: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    """Load a feature table and split it into X/y."""

    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"Expected label column {label_column!r} in {path}")
    x = df.drop(columns=[label_column])
    y = df[label_column]
    return x, y


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
