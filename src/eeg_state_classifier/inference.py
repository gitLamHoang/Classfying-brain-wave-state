"""Schema-checked batch inference for trusted public-EEG benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCHEMA_VERSION = "physionet-eyes-model-v1"
METADATA_COLUMNS = {"participant_id", "run", "label", "epoch_index", "start_seconds"}


def predict_table(artifact: dict[str, Any], table: pd.DataFrame) -> pd.DataFrame:
    """Align declared features, validate inputs, and emit labels without fake probabilities."""
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark artifact schema")
    columns = artifact.get("feature_columns")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(c, str) and c.startswith("feat_") for c in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("Artifact must declare unique feat_ predictor columns")
    if table.empty or table.columns.duplicated().any():
        raise ValueError("Prediction input must be nonempty and have unique columns")
    missing = set(columns) - set(table.columns)
    unknown = set(table.columns) - set(columns) - METADATA_COLUMNS
    if missing or unknown:
        raise ValueError(
            f"Feature schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    features = table.loc[:, columns]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("Predictor values must be numeric")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Predictor values must be finite")
    model = artifact.get("model")
    if model is None or not callable(getattr(model, "predict", None)):
        raise ValueError("Artifact has no fitted prediction model")
    if getattr(model, "n_features_in_", len(columns)) != len(columns):
        raise ValueError("Model and artifact feature counts disagree")
    learned_names = getattr(model, "feature_names_in_", None)
    if learned_names is not None and list(learned_names) != columns:
        raise ValueError("Model and artifact feature orders disagree")
    label_names = artifact.get("label_names")
    if not isinstance(label_names, list) or len(set(label_names)) != 2:
        raise ValueError("Artifact must declare the two eye-condition classes")
    if set(label_names) != {"eyes_closed", "eyes_open"}:
        raise ValueError("This model is only for eyes-open/eyes-closed baseline classification")
    predictions = model.predict(features)
    if len(predictions) != len(table) or not set(predictions).issubset(set(label_names)):
        raise ValueError("Predictions do not match the declared label contract")
    metadata = [c for c in table.columns if c in METADATA_COLUMNS and c != "label"]
    output = table.loc[:, metadata].copy()
    output["predicted_state"] = predictions
    return output


def predict_file(model_path: Path, features_path: Path, output_path: Path) -> dict[str, Any]:
    """Read a trusted joblib, validate a CSV, and save predictions plus a provenance sidecar."""
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError("Prediction outputs already exist; choose a new output path")
    # joblib is executable serialization: only use artifacts you created or otherwise trust.
    artifact = joblib.load(model_path)
    table = pd.read_csv(features_path, dtype={"participant_id": "string"})
    started = time.perf_counter()
    result = predict_table(artifact, table)
    elapsed = time.perf_counter() - started
    metadata = {
        "task": "eyes_open_vs_eyes_closed",
        "rows": len(result),
        "prediction_seconds": elapsed,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "config_sha256": artifact.get("config_sha256"),
        "selected_candidate": artifact.get("selected_candidate"),
        "probabilities_calibrated": False,
        "note": "Offline research predictions, not drowsiness detection or medical decisions.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
