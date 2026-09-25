"""Frozen, participant-disjoint evaluation of real public eyes-open/closed EEG.

The held-out participant data never enters learned preprocessing, hyperparameter selection,
or model fitting. This benchmark is an offline baseline-condition experiment, not
a clinical, drowsiness, or prospective deployment validation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FROZEN_CONFIG_SHA256 = "854a8b0031370bd42681562eb6ea32a93f9248a462f9b272e37c4f7f4a90b6b3"
OFFICIAL_CHECKSUM_MANIFEST_SHA256 = (
    "7f5d16957d8ee7bce86cc7ccba0e5994f63f33781607eb3f838392d49311a208"
)
PREPARATION_SOURCE_FILES = ("src/eeg_state_classifier/physionet.py", "scripts/prepare_physionet.py")
METADATA_COLUMNS = ["participant_id", "run", "label", "epoch_index", "start_seconds"]
MODEL_SCHEMA_VERSION = "physionet-eyes-model-v1"
FULL_STATUS = "FINAL_FROZEN_PROTOCOL_FULL_109_PARTICIPANTS"
SUBSET_STATUS = "NONFINAL_SUBSET_DIAGNOSTIC_DO_NOT_CITE_AS_FULL_BENCHMARK"
LOGGER = logging.getLogger(__name__)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _read_inputs(
    features_path: Path,
    manifest_path: Path,
    config_path: Path,
    allow_subset: bool,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    input_directories = {
        features_path.parent,
        manifest_path.parent,
        features_path.resolve().parent,
        manifest_path.resolve().parent,
    }
    if any((directory / "preparation_failure.json").exists() for directory in input_directories):
        raise ValueError("Rejecting inputs beside a preparation_failure.json marker")
    config_bytes = config_path.read_bytes()
    config_hash = _sha256(config_bytes)
    if config_hash != FROZEN_CONFIG_SHA256:
        raise ValueError("Config does not match the frozen protocol SHA256; do not retune holdout")
    config = json.loads(config_bytes)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("data_kind") != "real_public_eeg":
        raise ValueError("Manifest data_kind must be real_public_eeg")
    if manifest.get("config_sha256") != config_hash:
        raise ValueError("Manifest config_sha256 does not match frozen config")
    features_bytes = features_path.read_bytes()
    features_hash = _sha256(features_bytes)
    if manifest.get("features_sha256") != features_hash:
        raise ValueError("Features SHA256 does not match manifest")

    columns = manifest.get("feature_columns")
    if (
        not isinstance(columns, list)
        or len(columns) != config["expected_eeg_channels"] * 8
        or not all(isinstance(name, str) and name.startswith("feat_") for name in columns)
        or len(set(columns)) != len(columns)
        or set(columns) & set(METADATA_COLUMNS)
    ):
        raise ValueError("Feature schema requires exactly 512 unique feat_* columns; no metadata")
    expected_columns = [
        f"feat_{channel}_{kind}_{band}"
        for channel in config["expected_channel_names"]
        for kind in ("log10", "relative")
        for band in config["bands_hz"]
    ]
    if columns != expected_columns:
        raise ValueError("Feature names/order do not match the frozen channel/band schema")
    header = next(csv.reader(io.StringIO(features_bytes.decode("utf-8"))), [])
    if header != METADATA_COLUMNS + columns:
        raise ValueError("CSV columns must exactly match metadata and ordered manifest features")
    frame = pd.read_csv(io.BytesIO(features_bytes), dtype={"participant_id": str, "label": str})
    if frame.empty or frame.isna().any().any():
        raise ValueError("Feature table must be nonempty and contain no missing values")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
        raise ValueError("All features must be numeric")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("All features must contain only finite values")
    for column in ("run", "epoch_index", "start_seconds"):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Metadata {column} must be numeric")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Metadata {column} must contain finite values")
        if column != "start_seconds" and not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Metadata {column} must contain integer values")
    if (frame["epoch_index"] < 0).any():
        raise ValueError("epoch_index must be nonnegative")
    expected_start = (
        config["discard_start_seconds"] + frame["epoch_index"] * config["stride_seconds"]
    )
    if not np.allclose(frame["start_seconds"], expected_start, rtol=0, atol=1e-8):
        raise ValueError("start_seconds must match the frozen epoch grid")
    if frame.duplicated(["participant_id", "run", "epoch_index"]).any():
        raise ValueError("Duplicate participant/run/epoch rows are not allowed")
    run_labels = {int(run): label for run, label in config["runs"].items()}
    if not frame["run"].isin(run_labels).all():
        raise ValueError("Unexpected run in feature table")
    if not frame["run"].map(run_labels).eq(frame["label"]).all():
        raise ValueError("Labels must match the frozen baseline run-to-condition mapping")

    participants = sorted(frame["participant_id"].unique().tolist())
    declared_participants = manifest.get("participant_ids")
    if (
        not isinstance(declared_participants, list)
        or not all(isinstance(p, str) for p in declared_participants)
        or len(declared_participants) != len(set(declared_participants))
        or sorted(declared_participants) != participants
    ):
        raise ValueError("Actual participant coverage does not match manifest participant_ids")
    expected = [f"S{p:03d}" for p in range(config["subject_first"], config["subject_last"] + 1)]
    if not set(participants).issubset(expected):
        raise ValueError("Participant identifiers must be canonical S001..S109")
    if participants != expected and not allow_subset:
        raise ValueError("Full 109-participant coverage required; --allow-subset is nonfinal only")
    labels = set(run_labels.values())
    for participant, rows in frame.groupby("participant_id"):
        if set(rows["label"]) != labels:
            raise ValueError(f"Every participant must have both classes; failed: {participant}")
    for key in ("n_rows", "row_count", "n_epochs", "windows_count"):
        if key in manifest and manifest[key] != len(frame):
            raise ValueError(f"Manifest {key} does not match feature table")
    if "n_features" in manifest and manifest["n_features"] != len(columns):
        raise ValueError("Manifest n_features does not match feature schema")
    if "participant_count" in manifest and manifest["participant_count"] != len(participants):
        raise ValueError("Manifest participant_count does not match participant coverage")
    if "recordings_count" in manifest and manifest["recordings_count"] != 2 * len(participants):
        raise ValueError("Manifest recordings_count does not match participant/run coverage")
    _validate_preparation_evidence(frame, manifest, config, participants, expected)
    hashes = {
        "config_sha256": config_hash,
        "features_sha256": features_hash,
        "manifest_sha256": _sha256(manifest_bytes),
    }
    return frame, manifest, config, hashes


def _validate_preparation_evidence(
    frame: pd.DataFrame,
    manifest: dict,
    config: dict,
    participants: list[str],
    all_participants: list[str],
) -> None:
    """Reconcile the feature table with the pinned preparation provenance and QC."""
    checksum_manifest = manifest.get("checksum_manifest")
    if (
        not isinstance(checksum_manifest, dict)
        or checksum_manifest.get("sha256") != OFFICIAL_CHECKSUM_MANIFEST_SHA256
        or checksum_manifest.get("url") != config["source_base_url"] + "SHA256SUMS.txt"
    ):
        raise ValueError("Missing or incorrect official checksum-manifest provenance")
    expected_sources = {
        f"{participant}/{participant}R{int(run):02d}.edf": (participant, int(run), label)
        for participant in participants
        for run, label in config["runs"].items()
    }
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(expected_sources):
        raise ValueError("Source coverage must contain exactly each canonical participant/run EDF")
    if not all(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in source_hashes.values()
    ):
        raise ValueError("Every source EDF requires a valid SHA256 digest")
    if manifest.get("expected_participant_ids") != participants:
        raise ValueError(
            "Requested participant IDs must match actual complete participant coverage"
        )
    if manifest.get("full_protocol") is not (participants == all_participants):
        raise ValueError("Manifest full_protocol flag disagrees with participant coverage")
    required_counts = {
        "n_rows": len(frame),
        "windows_count": len(frame),
        "n_features": len(manifest["feature_columns"]),
        "participant_count": len(participants),
        "recordings_count": len(expected_sources),
    }
    for name, expected_count in required_counts.items():
        if type(manifest.get(name)) is not int or manifest[name] != expected_count:
            raise ValueError(f"Manifest {name} must match the verified feature table")
    if manifest.get("label_counts") != frame["label"].value_counts().to_dict():
        raise ValueError("Manifest label_counts must match the feature table")

    root = Path(__file__).resolve().parents[2]
    source_evidence = manifest.get("preprocessing_source_sha256")
    if not isinstance(source_evidence, dict) or set(source_evidence) != set(
        PREPARATION_SOURCE_FILES
    ):
        raise ValueError("Manifest must record both preparation source file SHA256 digests")
    for relative_path in PREPARATION_SOURCE_FILES:
        path = root / relative_path
        if not path.is_file() or source_evidence[relative_path] != _sha256(path.read_bytes()):
            raise ValueError(
                f"Preparation source SHA256 differs from current file: {relative_path}"
            )
    lock_path = root / "uv.lock"
    if not lock_path.is_file() or manifest.get("dependency_lock_sha256") != _sha256(
        lock_path.read_bytes()
    ):
        raise ValueError("Preparation dependency lock SHA256 differs from current uv.lock")

    count_keys = (
        "candidate_windows",
        "retained_windows",
        "rejected_flat_windows",
        "rejected_nonfinite_windows",
        "rejected_invalid_power_windows",
    )
    qc = manifest.get("qc")
    if not isinstance(qc, dict) or not isinstance(qc.get("recordings"), list):
        raise TypeError("Manifest requires recording-level quality-control evidence")
    if len(qc["recordings"]) != len(expected_sources):
        raise ValueError("QC recording coverage must match all source recordings")
    totals = dict.fromkeys(count_keys, 0)
    seen_sources = set()
    for record in qc["recordings"]:
        if not isinstance(record, dict):
            raise TypeError("Malformed QC recording")
        source_path = record.get("source_path")
        if source_path not in expected_sources or source_path in seen_sources:
            raise ValueError("QC recording coverage contains duplicate or unexpected source paths")
        seen_sources.add(source_path)
        participant, run, label = expected_sources[source_path]
        if (record.get("participant_id"), record.get("run"), record.get("label")) != (
            participant,
            run,
            label,
        ) or record.get("source_sha256") != source_hashes[source_path]:
            raise ValueError("QC source identity/hash does not match source manifest")
        if any(type(record.get(key)) is not int or record[key] < 0 for key in count_keys):
            raise ValueError("QC window counts must be nonnegative integers")
        retained = frame.loc[(frame["participant_id"] == participant) & (frame["run"] == run)]
        if record["retained_windows"] != len(retained):
            raise ValueError("QC retained windows disagree with feature-table recording rows")
        rejected = sum(record[key] for key in count_keys if key.startswith("rejected_"))
        if record["candidate_windows"] != record["retained_windows"] + rejected:
            raise ValueError("QC candidate count must equal retained plus rejected windows")
        if retained["epoch_index"].max() >= record["candidate_windows"]:
            raise ValueError("Feature epoch index exceeds the recording's QC candidate count")
        if record.get("exclusion_reason") is not None or record.get("nonfinite_sample_count") != 0:
            raise ValueError("A retained recording cannot have a QC exclusion or nonfinite samples")
        samples = record.get("samples")
        if type(samples) is not int or samples <= 0:
            raise ValueError("QC recording sample count must be a positive integer")
        fs = config["sampling_rate_hz"]
        first = round(config["discard_start_seconds"] * fs)
        window = round(config["window_seconds"] * fs)
        stride = round(config["stride_seconds"] * fs)
        expected_candidates = max(0, (samples - first - window) // stride + 1)
        if record["candidate_windows"] != expected_candidates:
            raise ValueError(
                "QC candidate count does not match frozen timing and recording samples"
            )
        for key in count_keys:
            totals[key] += record[key]
    for key, total in totals.items():
        if type(qc.get(key)) is not int or qc[key] != total:
            raise ValueError(f"QC aggregate {key} disagrees with recording-level counts")
    if totals["retained_windows"] != len(frame):
        raise ValueError("QC total retained windows does not match feature table")


def _build_candidates(config: dict) -> dict[str, Pipeline]:
    """The candidate list is fixed before any labels reach model selection."""
    seed = config["split_seed"]
    candidates = {
        "dummy": Pipeline([("classifier", DummyClassifier(**config["candidates"]["dummy"]))]),
        "random_forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        **config["candidates"]["random_forest"],
                        random_state=config["random_forest_seed"],
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }
    for c in config["candidates"]["logistic"]["C"]:
        candidates[f"logistic_C_{c:g}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=c, max_iter=5000, random_state=seed)),
            ]
        )
    for c in config["candidates"]["svm_rbf"]["C"]:
        candidates[f"svm_rbf_C_{c:g}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=c,
                        kernel="rbf",
                        gamma=config["candidates"]["svm_rbf"]["gamma"],
                        random_state=seed,
                    ),
                ),
            ]
        )
    return dict(sorted(candidates.items()))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "n_windows": len(y_true),
    }


def participant_cluster_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    replicates: int = 2000,
    seed: int = 20260924,
) -> dict:
    """Percentile intervals, resampling whole held-out participants with replacement.

    Each draw retains every window of a sampled participant. These intervals use
    pooled-window metrics and do not represent independent-window uncertainty.
    """
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    if not (len(y_true) == len(y_pred) == len(groups)) or len(y_true) == 0:
        raise ValueError("Bootstrap inputs must be nonempty and aligned")
    if replicates < 2 or len(set(labels)) != 2:
        raise ValueError("Bootstrap requires at least two replicates and two unique labels")
    if not set(y_true).issubset(labels) or not set(y_pred).issubset(labels):
        raise ValueError("Bootstrap labels do not match observations")
    participants = np.unique(groups)
    matrices = []
    for participant in participants:
        mask = groups == participant
        if set(y_true[mask]) != set(labels):
            raise ValueError("Every bootstrap participant must have both classes")
        matrices.append(confusion_matrix(y_true[mask], y_pred[mask], labels=labels))
    matrices = np.asarray(matrices, dtype=float)
    sampled = np.random.default_rng(seed).integers(
        0, len(participants), size=(replicates, len(participants))
    )
    pooled = matrices[sampled].sum(axis=1)
    correct = np.diagonal(pooled, axis1=1, axis2=2)
    actual = pooled.sum(axis=2)
    predicted = pooled.sum(axis=1)
    metrics = {
        "accuracy": correct.sum(axis=1) / pooled.sum(axis=(1, 2)),
        "balanced_accuracy": (correct / actual).mean(axis=1),
        "macro_f1": (2 * correct / (actual + predicted)).mean(axis=1),
    }
    return {
        "method": "participant_cluster_percentile",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "n_participants": len(participants),
        "aggregation": "pooled_windows_after_whole_participant_resampling",
        "intervals": {
            name: dict(zip(("low", "high"), map(float, np.percentile(values, [2.5, 97.5]))))
            for name, values in metrics.items()
        },
    }


def _partition_audit(frame: pd.DataFrame, train: np.ndarray, valid: np.ndarray) -> dict:
    training = sorted(frame.iloc[train]["participant_id"].unique().tolist())
    validation = sorted(frame.iloc[valid]["participant_id"].unique().tolist())
    if set(training) & set(validation):
        raise ValueError("Participant overlap in evaluation split")
    expected_labels = set(frame["label"])
    if any(set(frame.iloc[index]["label"]) != expected_labels for index in (train, valid)):
        raise ValueError("Every evaluation partition must contain both classes")
    return {
        "train_participant_ids": training,
        "validation_participant_ids": validation,
        "n_train_windows": len(train),
        "n_validation_windows": len(valid),
        "participant_overlap": [],
        "train_row_positions_sha256": _json_hash(train.tolist()),
        "validation_row_positions_sha256": _json_hash(valid.tolist()),
    }


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "mne"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _source_snapshot() -> dict:
    root = Path(__file__).resolve().parents[2]
    source_paths = sorted(
        {
            *root.glob("src/**/*.py"),
            *root.glob("scripts/*.py"),
            *root.glob("configs/*.json"),
            root / "pyproject.toml",
            root / "uv.lock",
        }
    )
    snapshot = {
        "source_sha256": {
            str(path.relative_to(root)): _sha256(path.read_bytes())
            for path in source_paths
            if path.is_file()
        }
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.splitlines()
        snapshot.update({"git_revision": revision, "git_dirty": bool(status), "git_status": status})
    except (OSError, subprocess.SubprocessError):
        snapshot.update({"git_revision": None, "git_dirty": None, "git_status": None})
    return snapshot


def run_benchmark(
    features_path: str | Path,
    manifest_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    allow_subset: bool = False,
) -> dict:
    """Run exactly one frozen selection/holdout evaluation in a new output folder.

    Returns the JSON-compatible report; the versioned model artifact is saved to
    ``model.joblib``. Existing output directories are always rejected. Subsets,
    even explicitly allowed ones, cannot produce a final full-benchmark status.
    """
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    source_snapshot = _source_snapshot()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output directory: {output_dir}")
    frame, manifest, config, hashes = _read_inputs(
        Path(features_path), Path(manifest_path), Path(config_path), allow_subset
    )
    columns = manifest["feature_columns"]
    labels = sorted(config["runs"].values())
    x, y, groups = frame[columns], frame["label"], frame["participant_id"]
    participants = sorted(groups.unique().tolist())
    status = SUBSET_STATUS if allow_subset else FULL_STATUS
    seed = config["split_seed"]
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=config["test_fraction"], random_state=seed).split(
            x, y, groups
        )
    )
    holdout_audit = _partition_audit(frame, train, test)
    if len(holdout_audit["train_participant_ids"]) < config["inner_folds"]:
        raise ValueError("At least five training participants required by the fixed inner folds")
    x_train, y_train, groups_train = x.iloc[train], y.iloc[train], groups.iloc[train]
    folds = list(
        StratifiedGroupKFold(
            n_splits=config["inner_cv"]["n_splits"],
            shuffle=config["inner_cv"]["shuffle"],
            random_state=config["inner_cv"]["random_state"],
        ).split(x_train, y_train, groups_train)
    )
    fold_audits = [
        {"fold": n, **_partition_audit(frame, train[fit], train[valid])}
        for n, (fit, valid) in enumerate(folds, 1)
    ]
    candidates = _build_candidates(config)
    # Atomically reserve a never-overwritten run directory before expensive fitting.
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_started = time.perf_counter()
    selection = []
    for name, candidate in candidates.items():
        LOGGER.info("Evaluating candidate %s using training participants only", name)
        candidate_started = time.perf_counter()
        scores = []
        fold_seconds = []
        for fit, valid in folds:
            fold_started = time.perf_counter()
            estimator = clone(candidate)
            estimator.fit(x_train.iloc[fit], y_train.iloc[fit])
            prediction = estimator.predict(x_train.iloc[valid])
            scores.append(float(balanced_accuracy_score(y_train.iloc[valid], prediction)))
            fold_seconds.append(time.perf_counter() - fold_started)
        selection.append(
            {
                "candidate": name,
                "fold_balanced_accuracy": scores,
                "mean_balanced_accuracy": float(np.mean(scores)),
                "std_balanced_accuracy": float(np.std(scores)),
                "fold_seconds": fold_seconds,
                "seconds": time.perf_counter() - candidate_started,
            }
        )
        LOGGER.info("Candidate %s mean inner-CV balanced accuracy: %.6f", name, np.mean(scores))
    # No holdout transform, prediction, score, or fit has occurred at this point.
    selected = min(selection, key=lambda row: (-row["mean_balanced_accuracy"], row["candidate"]))
    selected_name = selected["candidate"]
    LOGGER.info("Selected %s before opening final holdout", selected_name)
    selection_seconds = time.perf_counter() - selection_started
    selection_record = {
        "validation_status": status,
        "selection_metric": "balanced_accuracy",
        "tie_break": "alphabetical candidate name",
        "selected_candidate": selected_name,
        "candidate_results": selection,
        "inner_folds": fold_audits,
        "holdout_not_evaluated_yet": True,
        **hashes,
    }
    with (output_dir / "selection_before_holdout.json").open("x") as handle:
        json.dump(selection_record, handle, indent=2, allow_nan=False)
        handle.write("\n")

    fit_started = time.perf_counter()
    model = clone(candidates[selected_name]).fit(x_train, y_train)
    dummy = clone(candidates["dummy"]).fit(x_train, y_train)
    final_fit_seconds = time.perf_counter() - fit_started
    training_matrix_hash = _sha256(x_train.to_numpy(dtype="<f8").tobytes(order="C"))
    training_rows_hash = _sha256(frame.iloc[train].to_csv(index=False).encode())
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model": model,
        "feature_columns": columns,
        "label_names": labels,
        "data_kind": "real_public_eeg",
        "validation_status": status,
        "selected_candidate": selected_name,
        "training_participant_ids": holdout_audit["train_participant_ids"],
        "training_rows_sha256": training_rows_hash,
        "training_matrix_sha256": training_matrix_hash,
        "versions": _versions(),
        "source_snapshot": source_snapshot,
        **hashes,
    }
    with (output_dir / "model.joblib").open("xb") as handle:
        joblib.dump(artifact, handle)

    holdout_started = time.perf_counter()
    x_test = x.iloc[test]
    y_test = y.iloc[test].to_numpy()
    groups_test = groups.iloc[test].to_numpy()
    predictions = {"selected": model.predict(x_test), "dummy": dummy.predict(x_test)}
    final_holdout = {}
    participant_metrics = []
    for name, prediction in predictions.items():
        final_holdout[name] = {
            "candidate": selected_name if name == "selected" else "dummy",
            **_metrics(y_test, prediction, labels),
            "confidence_intervals": participant_cluster_bootstrap(
                y_test,
                prediction,
                groups_test,
                labels,
                replicates=config["bootstrap_replicates"],
                seed=config["bootstrap_seed"],
            ),
        }
        for participant in sorted(set(groups_test)):
            mask = groups_test == participant
            scores = _metrics(y_test[mask], prediction[mask], labels)
            participant_metrics.append(
                {
                    "participant_id": participant,
                    "model": name,
                    **{
                        key: value
                        for key, value in scores.items()
                        if key not in {"confusion_matrix", "confusion_matrix_labels"}
                    },
                }
            )
        final_holdout[name]["participant_macro_balanced_accuracy"] = float(
            np.mean(
                [row["balanced_accuracy"] for row in participant_metrics if row["model"] == name]
            )
        )
    holdout_seconds = time.perf_counter() - holdout_started
    predictions_frame = frame.iloc[test][METADATA_COLUMNS].copy()
    predictions_frame.insert(0, "source_row_position", test)
    for name, prediction in predictions.items():
        predictions_frame[f"prediction_{name}"] = prediction
    predictions_frame.to_csv(output_dir / "test_predictions.csv", index=False, mode="x")
    pd.DataFrame(participant_metrics).to_csv(
        output_dir / "participant_metrics.csv", index=False, mode="x"
    )
    report = {
        "schema_version": "physionet-eyes-benchmark-v1",
        "validation_status": status,
        "allow_subset": allow_subset,
        "started_at_utc": started_at,
        "data_kind": "real_public_eeg",
        "task": config["task"],
        "dataset": config["dataset"],
        "dataset_version": config["dataset_version"],
        "n_participants": len(participants),
        "participant_ids": participants,
        "n_windows": len(frame),
        "n_features": len(columns),
        "feature_columns": columns,
        "label_names": labels,
        "split": {
            "strategy": "GroupShuffleSplit",
            "seed": seed,
            "test_fraction": config["test_fraction"],
            **holdout_audit,
            "split_sha256": _json_hash({"train": train.tolist(), "test": test.tolist()}),
        },
        "selection": selection_record,
        "final_holdout": final_holdout,
        "per_participant_metrics": participant_metrics,
        "timings_seconds": {
            "inner_selection": selection_seconds,
            "final_fit": final_fit_seconds,
            "holdout_evaluation_and_bootstrap": holdout_seconds,
            "total": time.perf_counter() - started,
        },
        "versions": artifact["versions"],
        "source_snapshot": source_snapshot,
        "training_rows_sha256": training_rows_hash,
        "training_matrix_sha256": training_matrix_hash,
        "model_sha256": _sha256((output_dir / "model.joblib").read_bytes()),
        "limitations": [
            "Eyes-open versus eyes-closed baseline classification, not drowsiness or diagnosis.",
            "Run identity and fixed recording order are confounded with condition.",
            "Participant-disjoint same-dataset holdout is not external validation.",
            "Zero-phase filtering requires future samples; this is offline inference.",
            "Confidence intervals resample held-out participants and do not measure dataset shift.",
        ],
        **hashes,
    }
    with (output_dir / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report
