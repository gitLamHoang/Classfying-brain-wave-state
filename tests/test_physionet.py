from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")

from eeg_state_classifier import physionet


@pytest.fixture
def protocol() -> dict:
    path = Path(__file__).parents[1] / "configs" / "physionet_eyes_protocol.json"
    return json.loads(path.read_text())


def make_raw(protocol: dict, seconds: float = 14, frequency: float = 10) -> mne.io.RawArray:
    fs = protocol["sampling_rate_hz"]
    time = np.arange(round(seconds * fs)) / fs
    # Nonidentical channel amplitudes permit a meaningful order-invariance check.
    values = np.array(
        [
            (10 + channel) * 1e-6 * np.sin(2 * np.pi * frequency * time)
            + 2e-6 * np.sin(2 * np.pi * 3 * time)
            for channel in range(64)
        ]
    )
    info = mne.create_info(protocol["expected_channel_names"], fs, ch_types="eeg")
    return mne.io.RawArray(values, info, verbose="ERROR")


def extract(raw: mne.io.RawArray, protocol: dict):
    return physionet.extract_recording_features(
        raw,
        protocol,
        participant_id="S001",
        run=1,
    )


def test_real_signal_schema_timing_and_spectral_information(protocol):
    rows, channels, qc = extract(make_raw(protocol, seconds=15), protocol)
    assert channels == protocol["expected_channel_names"]
    assert len(rows) == 3
    assert qc["candidate_windows"] == qc["retained_windows"] == 3
    assert qc["discarded_tail_samples"] == 160
    assert qc["discarded_start_samples"] == 320
    assert [row["start_seconds"] for row in rows] == [2.0, 6.0, 10.0]
    assert [row["epoch_index"] for row in rows] == [0, 1, 2]
    columns = [name for name in rows[0] if name.startswith("feat_")]
    assert len(columns) == 512
    assert np.isfinite([[row[name] for name in columns] for row in rows]).all()
    assert rows[1]["feat_af3_relative_alpha"] > 0.9
    assert rows[1]["feat_af3_log10_alpha"] > rows[1]["feat_af3_log10_beta"]
    assert rows[1]["feat_af3_log10_alpha"] < rows[1]["feat_tp8_log10_alpha"]


def test_channel_order_case_and_trailing_dots_are_normalized(protocol):
    original = make_raw(protocol)
    reordered = original.copy().reorder_channels(list(reversed(original.ch_names)))
    reordered.rename_channels({name: name.upper() + "." for name in reordered.ch_names})
    expected_rows, expected_channels, _ = extract(original, protocol)
    actual_rows, actual_channels, _ = extract(reordered, protocol)
    assert actual_channels == expected_channels
    pd.testing.assert_frame_equal(pd.DataFrame(actual_rows), pd.DataFrame(expected_rows))


def test_scale_changes_absolute_power_but_not_relative_power(protocol):
    raw = make_raw(protocol)
    scaled = raw.copy()
    scaled._data *= 10
    rows, _, _ = extract(raw, protocol)
    scaled_rows, _, _ = extract(scaled, protocol)
    assert scaled_rows[1]["feat_af3_log10_alpha"] - rows[1][
        "feat_af3_log10_alpha"
    ] == pytest.approx(2)
    assert scaled_rows[1]["feat_af3_relative_alpha"] == pytest.approx(
        rows[1]["feat_af3_relative_alpha"]
    )


def test_flat_epoch_is_rejected_before_filter_can_add_edge_ringing(protocol):
    raw = make_raw(protocol)
    raw._data[0, 320:960] = 0
    rows, _, qc = extract(raw, protocol)
    assert qc["rejected_flat_windows"] == 1
    assert qc["candidate_windows"] == 3
    assert qc["retained_windows"] == 2
    assert [row["epoch_index"] for row in rows] == [1, 2]


def test_nonfinite_recording_rejects_all_windows_without_imputation(protocol):
    raw = make_raw(protocol)
    # A NaN even in the discarded prefix invalidates per-recording filtering.
    raw._data[0, 0] = np.nan
    rows, _, qc = extract(raw, protocol)
    assert rows == []
    assert qc["nonfinite_sample_count"] == 1
    assert qc["rejected_nonfinite_windows"] == 3
    assert qc["exclusion_reason"] == "nonfinite_recording_no_imputation"


def test_short_recording_has_explicit_exclusion(protocol):
    rows, _, qc = extract(make_raw(protocol, seconds=3), protocol)
    assert rows == []
    assert qc["candidate_windows"] == 0
    assert qc["exclusion_reason"] == "recording_too_short"


def test_wrong_sampling_rate_is_fatal(protocol):
    raw = make_raw(protocol).resample(128, verbose="ERROR")
    with pytest.raises(ValueError, match="expected 160 Hz"):
        extract(raw, protocol)


def test_missing_or_unknown_channel_is_fatal(protocol):
    raw = make_raw(protocol)
    with pytest.raises(ValueError, match="64 distinct"):
        extract(raw.copy().drop_channels([raw.ch_names[0]]), protocol)
    raw.rename_channels({raw.ch_names[0]: "UNEXPECTED"})
    with pytest.raises(ValueError, match="frozen protocol"):
        extract(raw, protocol)


def test_duplicate_normalized_channels_are_fatal(protocol):
    raw = make_raw(protocol)
    raw.rename_channels({"af4": "AF3."})
    with pytest.raises(ValueError, match="64 distinct"):
        extract(raw, protocol)


def test_welch_matches_frozen_half_open_integration_and_floor(protocol):
    raw = make_raw(protocol)
    epoch = raw.get_data()[:, :640]
    actual = physionet._spectral_features(epoch, protocol)
    frequencies, psd = physionet.signal.welch(epoch, fs=160, axis=-1, **protocol["welch"])
    mask = (frequencies >= 8) & (frequencies < 13)
    alpha = np.trapezoid(psd[0, mask], frequencies[mask])
    total_mask = (frequencies >= 1) & (frequencies < 40)
    total = np.trapezoid(psd[0, total_mask], frequencies[total_mask])
    assert actual[2] == pytest.approx(np.log10(max(alpha, 1e-20)))
    assert actual[6] == pytest.approx(alpha / max(total, 1e-20))


class FakeResponse(io.BytesIO):
    def geturl(self):
        return "https://physionet.org/test.edf"


def test_download_checks_hash_then_atomically_reuses_cache(tmp_path, monkeypatch):
    payload = b"an authentic test payload"
    expected = hashlib.sha256(payload).hexdigest()
    calls = []

    def fake_open(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse(payload)

    monkeypatch.setattr(physionet, "urlopen", fake_open)
    destination = tmp_path / "nested" / "source.edf"
    assert (
        physionet.download_verified("https://physionet.org/test.edf", destination, expected)
        == destination
    )
    physionet.download_verified("https://physionet.org/test.edf", destination, expected)
    assert len(calls) == 1
    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob("*.part"))
    destination.write_bytes(b"corrupted")
    with pytest.raises(physionet.IntegrityError, match="Cached SHA256 mismatch"):
        physionet.download_verified("https://physionet.org/test.edf", destination, expected)
    assert len(calls) == 1


def test_bad_download_never_publishes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(physionet, "urlopen", lambda *args, **kwargs: FakeResponse(b"corrupt"))
    destination = tmp_path / "source.edf"
    with pytest.raises(physionet.IntegrityError, match="Downloaded SHA256 mismatch"):
        physionet.download_verified("https://physionet.org/test.edf", destination, "0" * 64)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_transient_download_failure_retries(tmp_path, monkeypatch):
    payload = b"valid"
    calls = []

    def fake_open(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise URLError("temporarily offline")
        return FakeResponse(payload)

    monkeypatch.setattr(physionet, "urlopen", fake_open)
    monkeypatch.setattr(physionet.time, "sleep", lambda seconds: None)
    destination = tmp_path / "source.edf"
    physionet.download_verified(
        "https://physionet.org/test.edf",
        destination,
        hashlib.sha256(payload).hexdigest(),
    )
    assert len(calls) == 2
    assert destination.read_bytes() == payload


def test_checksum_manifest_rejects_unsafe_and_conflicting_paths():
    with pytest.raises(ValueError, match="Unsafe"):
        physionet.parse_checksums("0" * 64 + " ../../escape")
    with pytest.raises(physionet.IntegrityError, match="Conflicting"):
        physionet.parse_checksums("0" * 64 + " S001/a.edf\n" + "1" * 64 + " S001/a.edf")
    assert physionet.parse_checksums("0" * 64 + " *S001/a.edf") == {"S001/a.edf": "0" * 64}


def fake_dataset(tmp_path, monkeypatch, protocol, *, flat_run=None):
    raw_dir = tmp_path / "raw"
    source_hashes = {}
    for subject in [1, 2]:
        for run in [1, 2]:
            path = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            target = raw_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.encode())
            source_hashes[path] = physionet.sha256_file(target)
    checksum_path = raw_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "\n".join(f"{digest} {path}" for path, digest in source_hashes.items())
    )
    monkeypatch.setattr(
        physionet,
        "download_dataset",
        lambda *args, **kwargs: (source_hashes, checksum_path),
    )

    def read_fake(path):
        raw = make_raw(protocol)
        if path.name == flat_run:
            raw._data[:] = 0
        return raw

    monkeypatch.setattr(physionet, "_read_edf", read_fake)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(protocol))
    return config_path, raw_dir, tmp_path / "processed", source_hashes


def test_prepare_preserves_subjects_provenance_and_consistent_schema(
    tmp_path, monkeypatch, protocol
):
    config_path, raw_dir, output_dir, hashes = fake_dataset(tmp_path, monkeypatch, protocol)
    manifest = physionet.prepare_dataset(config_path, raw_dir, output_dir, subjects=[2, 1])
    table = pd.read_csv(output_dir / "features.csv")
    assert manifest["data_kind"] == "real_public_eeg"
    assert manifest["source_hashes"] == hashes
    assert manifest["participant_ids"] == ["S001", "S002"]
    assert manifest["expected_participant_ids"] == manifest["participant_ids"]
    assert manifest["full_protocol"] is False
    assert manifest["n_rows"] == manifest["windows_count"] == len(table) == 12
    assert manifest["recordings_count"] == 4
    assert manifest["n_features"] == 512
    assert manifest["config_sha256"] == physionet.sha256_file(config_path)
    repository_root = Path(__file__).parents[1]
    assert manifest["preprocessing_source_sha256"] == {
        path: physionet.sha256_file(repository_root / path)
        for path in physionet.PREPROCESSING_SOURCE_PATHS
    }
    assert manifest["dependency_lock_sha256"] == physionet.sha256_file(repository_root / "uv.lock")
    assert manifest["features_sha256"] == physionet.sha256_file(output_dir / "features.csv")
    assert set(table["label"]) == {"eyes_open", "eyes_closed"}
    assert set(table["run"]) == {1, 2}
    assert list(table.columns) == physionet.METADATA_COLUMNS + manifest["feature_columns"]
    assert json.loads((output_dir / "features_manifest.json").read_text()) == manifest
    assert not (output_dir / "preparation_failure.json").exists()


def test_prepare_never_silently_loses_a_participant_class(tmp_path, monkeypatch, protocol):
    config_path, raw_dir, output_dir, _ = fake_dataset(
        tmp_path,
        monkeypatch,
        protocol,
        flat_run="S002R02.edf",
    )
    with pytest.raises(ValueError, match="no usable windows"):
        physionet.prepare_dataset(config_path, raw_dir, output_dir, subjects=[1, 2])
    assert not (output_dir / "features.csv").exists()
    assert not (output_dir / "features_manifest.json").exists()
    failure = json.loads((output_dir / "preparation_failure.json").read_text())
    assert failure["expected_participant_ids"] == ["S001", "S002"]
    assert failure["qc_recordings_completed"][-1]["rejected_flat_windows"] == 3
    assert set(failure["preprocessing_source_sha256"]) == set(physionet.PREPROCESSING_SOURCE_PATHS)
    assert len(failure["dependency_lock_sha256"]) == 64


@pytest.mark.parametrize("artifact", physionet.PREPARATION_ARTIFACTS)
def test_preparation_refuses_existing_artifacts_without_mutation(
    tmp_path,
    monkeypatch,
    protocol,
    artifact,
):
    config_path, raw_dir, output_dir, _ = fake_dataset(tmp_path, monkeypatch, protocol)
    output_dir.mkdir()
    previous = b"preserve this prior success or failure unchanged"
    (output_dir / artifact).write_bytes(previous)

    def forbidden_network(*args, **kwargs):
        pytest.fail("Preparation must refuse stale outputs before accessing sources")

    monkeypatch.setattr(physionet, "download_dataset", forbidden_network)
    with pytest.raises(FileExistsError, match="fresh output directory"):
        physionet.prepare_dataset(config_path, raw_dir, output_dir, subjects=[1, 2])
    assert list(output_dir.iterdir()) == [output_dir / artifact]
    assert (output_dir / artifact).read_bytes() == previous


def test_preparation_captures_source_provenance_at_start(tmp_path, monkeypatch, protocol):
    config_path, raw_dir, output_dir, _ = fake_dataset(tmp_path, monkeypatch, protocol)
    expected = {path: "a" * 64 for path in physionet.PREPROCESSING_SOURCE_PATHS}
    source_state = {"hashes": expected.copy(), "lock": "b" * 64}
    monkeypatch.setattr(
        physionet,
        "_preparation_provenance",
        lambda: (source_state["hashes"].copy(), source_state["lock"]),
    )
    original_read = physionet._read_edf

    def read_and_change_source_state(path):
        source_state["hashes"] = {key: "c" * 64 for key in expected}
        source_state["lock"] = "d" * 64
        return original_read(path)

    monkeypatch.setattr(physionet, "_read_edf", read_and_change_source_state)
    manifest = physionet.prepare_dataset(config_path, raw_dir, output_dir, subjects=[1, 2])
    assert manifest["preprocessing_source_sha256"] == expected
    assert manifest["dependency_lock_sha256"] == "b" * 64


def test_prepare_rechecks_downloaded_file_before_edf_use(tmp_path, monkeypatch, protocol):
    config_path, raw_dir, output_dir, _ = fake_dataset(tmp_path, monkeypatch, protocol)
    (raw_dir / "S001/S001R01.edf").write_bytes(b"tampered after download")
    with pytest.raises(physionet.IntegrityError, match="Source changed"):
        physionet.prepare_dataset(config_path, raw_dir, output_dir, subjects=[1, 2])


@pytest.mark.parametrize("subjects", [[], [0], [110], [1, 1], [True], ["1"]])
def test_invalid_subject_selection_fails_before_network(tmp_path, protocol, subjects):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="unique integer"):
        physionet.prepare_dataset(path, tmp_path / "raw", tmp_path / "processed", subjects)
