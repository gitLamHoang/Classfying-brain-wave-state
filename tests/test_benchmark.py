"""Network-free tests of the public benchmark's leakage and integrity boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from eeg_state_classifier import benchmark

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "physionet_eyes_protocol.json"


def _save_input(directory, frame, columns, participants=None):
    # Synthetic contract fixtures: tests never load or fit real public EEG.
    directory.mkdir(parents=True, exist_ok=True)
    feature_path = directory / "features.csv"
    frame.to_csv(feature_path, index=False)
    config = json.loads(CONFIG_PATH.read_text())
    root = CONFIG_PATH.parents[1]
    actual_participants = sorted(frame.participant_id.unique().tolist())
    sources = {}
    qc_records = []
    for (participant, run), rows in frame.groupby(["participant_id", "run"]):
        relative_path = f"{participant}/{participant}R{run:02d}.edf"
        sources[relative_path] = hashlib.sha256(relative_path.encode()).hexdigest()
        candidates = int(rows["epoch_index"].max()) + 1
        qc_records.append(
            {
                "participant_id": participant,
                "run": int(run),
                "label": rows.iloc[0]["label"],
                "source_path": relative_path,
                "source_sha256": sources[relative_path],
                "candidate_windows": candidates,
                "retained_windows": len(rows),
                "rejected_flat_windows": candidates - len(rows),
                "rejected_nonfinite_windows": 0,
                "rejected_invalid_power_windows": 0,
                "nonfinite_sample_count": 0,
                "exclusion_reason": None,
                "samples": 160 * (2 + 4 * candidates),
            }
        )
    qc = {
        key: sum(record[key] for record in qc_records)
        for key in (
            "candidate_windows",
            "retained_windows",
            "rejected_flat_windows",
            "rejected_nonfinite_windows",
            "rejected_invalid_power_windows",
        )
    }
    qc["recordings"] = qc_records
    manifest = {
        "data_kind": "real_public_eeg",
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "feature_columns": columns,
        "participant_ids": (
            sorted(frame.participant_id.unique().tolist()) if participants is None else participants
        ),
        "n_rows": len(frame),
        "n_features": len(columns),
        "windows_count": len(frame),
        "participant_count": len(actual_participants),
        "expected_participant_ids": actual_participants,
        "full_protocol": actual_participants == [f"S{p:03d}" for p in range(1, 110)],
        "recordings_count": len(sources),
        "source_hashes": sources,
        "checksum_manifest": {
            "url": config["source_base_url"] + "SHA256SUMS.txt",
            "sha256": benchmark.OFFICIAL_CHECKSUM_MANIFEST_SHA256,
        },
        "preprocessing_source_sha256": {
            relative_path: hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
            for relative_path in benchmark.PREPARATION_SOURCE_FILES
        },
        "dependency_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "label_counts": frame["label"].value_counts().to_dict(),
        "qc": qc,
    }
    manifest_path = directory / "features_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return feature_path, manifest_path


@pytest.fixture
def dataset(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    columns = [
        f"feat_{channel}_{kind}_{band}"
        for channel in config["expected_channel_names"]
        for kind in ("log10", "relative")
        for band in config["bands_hz"]
    ]
    rows = []
    for participant in range(1, 15):
        for run, label in ((1, "eyes_open"), (2, "eyes_closed")):
            for epoch in range(2):
                rows.append(
                    {
                        "participant_id": f"S{participant:03d}",
                        "run": run,
                        "label": label,
                        "epoch_index": epoch,
                        "start_seconds": 2.0 + 4.0 * epoch,
                    }
                )
    metadata = pd.DataFrame(rows)
    rng = np.random.default_rng(53)
    x = rng.normal(0, 0.1, (len(rows), len(columns)))
    x += (metadata["label"] == "eyes_open").to_numpy()[:, None] * 2
    frame = pd.concat([metadata, pd.DataFrame(x, columns=columns)], axis=1)
    paths = _save_input(tmp_path / "original", frame, columns)
    return frame, columns, paths


def _read(paths, allow_subset=True):
    return benchmark._read_inputs(*paths, CONFIG_PATH, allow_subset)


@pytest.fixture
def fast_candidates(monkeypatch):
    original = benchmark._build_candidates

    def build(config):
        candidates = original(config)
        return {name: candidates[name] for name in ("dummy", "logistic_C_0.1")}

    monkeypatch.setattr(benchmark, "_build_candidates", build)


def test_frozen_candidate_set():
    candidates = benchmark._build_candidates(json.loads(CONFIG_PATH.read_text()))
    assert list(candidates) == [
        "dummy",
        "logistic_C_0.1",
        "logistic_C_1",
        "logistic_C_10",
        "random_forest",
        "svm_rbf_C_0.1",
        "svm_rbf_C_1",
        "svm_rbf_C_10",
    ]
    forest = candidates["random_forest"]["classifier"]
    assert forest.n_estimators == 200 and forest.min_samples_leaf == 2
    assert forest.random_state == 20260924
    assert "scaler" not in candidates["dummy"].named_steps
    assert "scaler" not in candidates["random_forest"].named_steps


def test_grouped_evaluation_artifacts_and_immutable_output(dataset, tmp_path, fast_candidates):
    frame, columns, paths = dataset
    output = tmp_path / "run"
    report = benchmark.run_benchmark(*paths, CONFIG_PATH, output, allow_subset=True)
    assert report["validation_status"] == benchmark.SUBSET_STATUS
    split = report["split"]
    assert not set(split["train_participant_ids"]) & set(split["validation_participant_ids"])
    assert len(split["validation_participant_ids"]) == 3
    validation_memberships = []
    for fold in report["selection"]["inner_folds"]:
        train, valid = set(fold["train_participant_ids"]), set(fold["validation_participant_ids"])
        assert not train & valid
        assert train | valid == set(split["train_participant_ids"])
        assert not (train | valid) & set(split["validation_participant_ids"])
        assert len(fold["train_row_positions_sha256"]) == 64
        validation_memberships += list(valid)
    assert sorted(validation_memberships) == split["train_participant_ids"]
    artifact = joblib.load(output / "model.joblib")
    assert artifact["feature_columns"] == columns
    assert artifact["schema_version"] == "physionet-eyes-model-v1"
    assert artifact["label_names"] == ["eyes_closed", "eyes_open"]
    assert len(artifact["training_matrix_sha256"]) == 64
    assert artifact["model"].n_features_in_ == 512
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert len(predictions) == split["n_validation_windows"]
    assert set(predictions.participant_id) == set(split["validation_participant_ids"])
    pd.testing.assert_series_equal(
        predictions["label"],
        frame.iloc[predictions["source_row_position"]]["label"].reset_index(drop=True),
    )
    selected = report["final_holdout"]["selected"]
    assert selected["confidence_intervals"]["replicates"] == 2000
    assert selected["participant_macro_balanced_accuracy"] == 1.0
    assert report["final_holdout"]["dummy"]["balanced_accuracy"] == 0.5
    assert report["source_snapshot"]["source_sha256"]["uv.lock"]
    saved = json.loads((output / "report.json").read_text())
    assert saved == report
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        benchmark.run_benchmark(*paths, CONFIG_PATH, output, allow_subset=True)


def test_holdout_values_cannot_change_selection_scaling_or_fitted_model(
    dataset, tmp_path, fast_candidates
):
    frame, columns, paths = dataset
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=20260924).split(
            frame[columns], frame["label"], frame["participant_id"]
        )
    )
    first = benchmark.run_benchmark(*paths, CONFIG_PATH, tmp_path / "run_a", allow_subset=True)
    perturbed = frame.copy()
    perturbed.loc[test, columns] += 10000.0
    changed_paths = _save_input(tmp_path / "perturbed", perturbed, columns)
    second = benchmark.run_benchmark(
        *changed_paths, CONFIG_PATH, tmp_path / "run_b", allow_subset=True
    )
    a = joblib.load(tmp_path / "run_a" / "model.joblib")
    b = joblib.load(tmp_path / "run_b" / "model.joblib")
    assert a["selected_candidate"] == b["selected_candidate"] == "logistic_C_0.1"
    np.testing.assert_allclose(b["model"]["scaler"].mean_, frame.iloc[train][columns].mean())
    np.testing.assert_array_equal(a["model"]["scaler"].mean_, b["model"]["scaler"].mean_)
    np.testing.assert_array_equal(a["model"]["classifier"].coef_, b["model"]["classifier"].coef_)
    assert a["training_rows_sha256"] == b["training_rows_sha256"]
    assert a["training_matrix_sha256"] == b["training_matrix_sha256"]
    assert first["split"]["split_sha256"] == second["split"]["split_sha256"]
    for original, altered in zip(
        first["selection"]["candidate_results"], second["selection"]["candidate_results"]
    ):
        assert original["fold_balanced_accuracy"] == altered["fold_balanced_accuracy"]


def test_subset_requires_explicit_nonfinal_opt_in(dataset):
    with pytest.raises(ValueError, match="Full 109-participant coverage"):
        _read(dataset[2], allow_subset=False)


def test_every_inner_scaler_is_fitted_only_to_its_training_fold(
    dataset, tmp_path, monkeypatch, fast_candidates
):
    frame, _, paths = dataset
    fit_rows = []
    original_fit = StandardScaler.fit

    def audit_fit(self, x, *args, **kwargs):
        fit_rows.append(x.index.to_numpy())
        return original_fit(self, x, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", audit_fit)
    report = benchmark.run_benchmark(*paths, CONFIG_PATH, tmp_path / "audited", allow_subset=True)
    assert len(fit_rows) == 6  # Five inner-fold fits, then one final training fit.
    heldout = set(report["split"]["validation_participant_ids"])
    for indices, fold in zip(fit_rows[:5], report["selection"]["inner_folds"]):
        actual = set(frame.iloc[indices]["participant_id"])
        assert actual == set(fold["train_participant_ids"])
        assert not actual & set(fold["validation_participant_ids"])
        assert not actual & heldout
    assert set(frame.iloc[fit_rows[-1]]["participant_id"]) == set(
        report["split"]["train_participant_ids"]
    )


def test_exact_cv_ties_select_alphabetical_candidate(dataset, tmp_path, monkeypatch):
    original = benchmark._build_candidates

    def candidates(config):
        base = original(config)
        return {
            "z_logistic": base["logistic_C_0.1"],
            "a_logistic": base["logistic_C_0.1"],
            "dummy": base["dummy"],
        }

    monkeypatch.setattr(benchmark, "_build_candidates", candidates)
    report = benchmark.run_benchmark(*dataset[2], CONFIG_PATH, tmp_path / "tied", allow_subset=True)
    assert report["selection"]["selected_candidate"] == "a_logistic"


def test_full_participant_coverage_accepted(tmp_path, dataset):
    frame, columns, _ = dataset
    copies = []
    for participant in range(1, 110):
        rows = frame.iloc[:4].copy()
        rows["participant_id"] = f"S{participant:03d}"
        copies.append(rows)
    full = pd.concat(copies, ignore_index=True)
    paths = _save_input(tmp_path / "full", full, columns)
    result, _, _, _ = _read(paths, allow_subset=False)
    assert result["participant_id"].nunique() == 109


def test_modified_feature_bytes_rejected(dataset):
    _, _, (features, manifest) = dataset
    with features.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="SHA256"):
        _read((features, manifest))


def test_modified_protocol_rejected(dataset, tmp_path):
    path = tmp_path / "changed_config.json"
    path.write_bytes(CONFIG_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen protocol SHA256"):
        benchmark._read_inputs(*dataset[2], path, True)


def test_manifest_participant_coverage_cannot_lie(dataset, tmp_path):
    frame, columns, _ = dataset
    paths = _save_input(tmp_path / "bad", frame, columns, participants=["S001"])
    with pytest.raises(ValueError, match="Actual participant coverage"):
        _read(paths)


def test_metadata_cannot_be_declared_as_features(dataset):
    _, _, (_, manifest_path) = dataset
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_columns"][0] = "participant_id"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no metadata"):
        _read(dataset[2])


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), "invalid"])
def test_nonnumeric_or_nonfinite_features_rejected(dataset, tmp_path, bad):
    frame, columns, _ = dataset
    frame = frame.copy()
    if isinstance(bad, str):
        frame[columns[0]] = frame[columns[0]].astype(object)
    frame.loc[0, columns[0]] = bad
    paths = _save_input(tmp_path / "bad", frame, columns)
    with pytest.raises(ValueError, match="finite|missing|numeric"):
        _read(paths)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("extra_metadata", "CSV columns"),
        ("unknown_feature", "frozen channel/band schema"),
        ("wrong_run_label", "run-to-condition"),
        ("duplicate_epoch", "Duplicate"),
        ("bad_epoch_start", "frozen epoch grid"),
        ("missing_class", "both classes"),
    ],
)
def test_malformed_schemas_and_epoch_metadata_rejected(dataset, tmp_path, mutation, message):
    frame, columns, _ = dataset
    frame, columns = frame.copy(), columns.copy()
    if mutation == "extra_metadata":
        frame["recording_id"] = "S001R01"
    elif mutation == "unknown_feature":
        frame = frame.rename(columns={columns[0]: "feat_unrecognized"})
        columns[0] = "feat_unrecognized"
    elif mutation == "wrong_run_label":
        frame.loc[0, "label"] = "eyes_closed"
    elif mutation == "duplicate_epoch":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "bad_epoch_start":
        frame.loc[0, "start_seconds"] = 0.0
    elif mutation == "missing_class":
        frame = frame.loc[~((frame["participant_id"] == "S001") & (frame["run"] == 1))]
    paths = _save_input(tmp_path / "bad", frame, columns)
    with pytest.raises(ValueError, match=message):
        _read(paths)


def test_cluster_bootstrap_reproducible_and_resamples_complete_participants():
    labels = ["eyes_closed", "eyes_open"]
    # Each cluster is either wholly correct or wholly incorrect, giving wide intervals.
    y = np.array(labels * 2)
    pred = np.array(labels + labels[::-1])
    groups = np.array(["S001", "S001", "S002", "S002"])
    first = benchmark.participant_cluster_bootstrap(y, pred, groups, labels)
    second = benchmark.participant_cluster_bootstrap(y, pred, groups, labels)
    assert first == second
    assert first["n_participants"] == 2
    for interval in first["intervals"].values():
        assert interval == {"low": 0.0, "high": 1.0}
    perfect = benchmark.participant_cluster_bootstrap(y, y, groups, labels)
    assert perfect["intervals"]["balanced_accuracy"] == {"low": 1.0, "high": 1.0}


def test_bootstrap_rejects_unaligned_or_one_class_participant():
    with pytest.raises(ValueError, match="aligned"):
        benchmark.participant_cluster_bootstrap(
            np.array(["a", "b"]), np.array(["a"]), np.array(["S001"]), ["a", "b"]
        )
    with pytest.raises(ValueError, match="both classes"):
        benchmark.participant_cluster_bootstrap(
            np.array(["a", "b"]), np.array(["a", "b"]), np.array(["S001", "S002"]), ["a", "b"]
        )


@pytest.mark.parametrize("location", ["features", "manifest"])
def test_preparation_failure_marker_rejects_even_valid_inputs(dataset, tmp_path, location):
    features_path, manifest_path = dataset[2]
    if location == "manifest":
        separate = tmp_path / "separate"
        separate.mkdir()
        moved = separate / manifest_path.name
        moved.write_bytes(manifest_path.read_bytes())
        manifest_path = moved
    marker_dir = features_path.parent if location == "features" else manifest_path.parent
    (marker_dir / "preparation_failure.json").write_text('{"status": "failed"}')
    with pytest.raises(ValueError, match="preparation_failure.json"):
        _read((features_path, manifest_path))


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_source", "Source coverage"),
        ("extra_source", "Source coverage"),
        ("invalid_source_digest", "valid SHA256"),
        ("untrusted_checksum_manifest", "official checksum-manifest"),
        ("untrusted_checksum_url", "official checksum-manifest"),
        ("missing_qc_record", "QC recording coverage"),
        ("duplicate_qc_record", "duplicate or unexpected"),
        ("wrong_qc_identity", "QC source identity/hash"),
        ("wrong_qc_digest", "QC source identity/hash"),
        ("wrong_retained_count", "QC retained windows"),
        ("wrong_rejected_count", "retained plus rejected"),
        ("wrong_qc_total", "QC aggregate"),
        ("wrong_qc_samples", "frozen timing"),
        ("nonfinite_retained_recording", "nonfinite samples"),
        ("preparation_source_changed", "Preparation source SHA256"),
        ("dependency_lock_changed", "dependency lock SHA256"),
        ("missing_preparation_sources", "both preparation source"),
        ("incorrect_full_protocol", "full_protocol flag"),
        ("missing_requested_participant", "Requested participant IDs"),
        ("incorrect_label_counts", "label_counts"),
    ],
)
def test_preparation_provenance_and_qc_tampering_rejected(dataset, mutation, message):
    manifest_path = dataset[2][1]
    manifest = json.loads(manifest_path.read_text())
    first_source = next(iter(manifest["source_hashes"]))
    first_qc = manifest["qc"]["recordings"][0]
    if mutation == "missing_source":
        manifest["source_hashes"].pop(first_source)
    elif mutation == "extra_source":
        manifest["source_hashes"]["S110/S110R01.edf"] = "a" * 64
    elif mutation == "invalid_source_digest":
        manifest["source_hashes"][first_source] = "not-a-digest"
    elif mutation == "untrusted_checksum_manifest":
        manifest["checksum_manifest"]["sha256"] = "a" * 64
    elif mutation == "untrusted_checksum_url":
        manifest["checksum_manifest"]["url"] = "https://example.com/SHA256SUMS.txt"
    elif mutation == "missing_qc_record":
        manifest["qc"]["recordings"].pop()
    elif mutation == "duplicate_qc_record":
        manifest["qc"]["recordings"][-1] = first_qc
    elif mutation == "wrong_qc_identity":
        first_qc["participant_id"] = "S109"
    elif mutation == "wrong_qc_digest":
        first_qc["source_sha256"] = "a" * 64
    elif mutation == "wrong_retained_count":
        first_qc["retained_windows"] += 1
    elif mutation == "wrong_rejected_count":
        first_qc["rejected_flat_windows"] += 1
    elif mutation == "wrong_qc_total":
        manifest["qc"]["retained_windows"] += 1
    elif mutation == "wrong_qc_samples":
        first_qc["samples"] += 640
    elif mutation == "nonfinite_retained_recording":
        first_qc["nonfinite_sample_count"] = 1
    elif mutation == "preparation_source_changed":
        manifest["preprocessing_source_sha256"][benchmark.PREPARATION_SOURCE_FILES[0]] = "a" * 64
    elif mutation == "dependency_lock_changed":
        manifest["dependency_lock_sha256"] = "a" * 64
    elif mutation == "missing_preparation_sources":
        manifest.pop("preprocessing_source_sha256")
    elif mutation == "incorrect_full_protocol":
        manifest["full_protocol"] = True
    elif mutation == "missing_requested_participant":
        manifest["expected_participant_ids"].pop()
    elif mutation == "incorrect_label_counts":
        manifest["label_counts"]["eyes_closed"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        _read(dataset[2])


def test_consistent_fixed_qc_exclusion_is_accepted(dataset, tmp_path):
    frame, columns, _ = dataset
    # Remove epoch zero, preserving epoch one and recording the excluded window in QC.
    frame = frame.drop(index=0)
    paths = _save_input(tmp_path / "qc_excluded", frame, columns)
    accepted, manifest, _, _ = _read(paths)
    assert len(accepted) == len(frame)
    assert manifest["qc"]["rejected_flat_windows"] == 1
