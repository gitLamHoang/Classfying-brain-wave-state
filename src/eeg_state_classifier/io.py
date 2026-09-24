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


def load_training_table(
    path: str | Path, label_column: str = "label", group_column: str | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Load features, labels, and optional participant metadata separately."""
    if group_column == label_column:
        raise ValueError("Group and label columns must be different")
    # Preserve IDs such as 001 and 01 instead of merging them through integer parsing.
    df = pd.read_csv(path, dtype={group_column: "string"} if group_column else None)
    required = [label_column] + ([group_column] if group_column else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df.drop(columns=required), df[label_column], df[group_column] if group_column else None


def load_feature_table(
    path: str | Path, label_column: str = "label"
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the legacy ungrouped synthetic feature format."""
    x, y, _ = load_training_table(path, label_column)
    return x, y


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
