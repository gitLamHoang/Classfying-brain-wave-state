"""Verified PhysioNet baseline ingestion and offline multichannel EEG features.

This is an eyes-open/eyes-closed benchmark, not a drowsiness detector. All
filtering occurs within an individual recording, before its windows are made;
it is zero-phase/offline processing and cannot support online latency claims.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy import signal

LOGGER = logging.getLogger(__name__)
METADATA_COLUMNS = ["participant_id", "run", "label", "epoch_index", "start_seconds"]
CHECKSUM_NAME = "SHA256SUMS.txt"
PREPARATION_ARTIFACTS = ("features.csv", "features_manifest.json", "preparation_failure.json")
PREPROCESSING_SOURCE_PATHS = (
    "src/eeg_state_classifier/physionet.py",
    "scripts/prepare_physionet.py",
)


class IntegrityError(ValueError):
    """An existing or newly downloaded file does not match its official digest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preparation_provenance() -> tuple[dict[str, str], str]:
    """Bind features to the producing source checkout and dependency lock."""
    repository_root = Path(__file__).resolve().parents[2]
    source_hashes = {
        relative_path: sha256_file(repository_root / relative_path)
        for relative_path in PREPROCESSING_SOURCE_PATHS
    }
    return source_hashes, sha256_file(repository_root / "uv.lock")


def parse_checksums(text: str) -> dict[str, str]:
    """Parse the official SHA256SUMS format without accepting unsafe paths."""
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is None:
            raise ValueError(f"Malformed checksum entry on line {line_number}")
        relative_path = parts[1].lstrip("*")
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in relative_path:
            raise ValueError(f"Unsafe checksum path: {relative_path!r}")
        relative_path = str(path)
        digest = parts[0].lower()
        if relative_path in checksums and checksums[relative_path] != digest:
            raise IntegrityError(f"Conflicting checksums for {relative_path}")
        checksums[relative_path] = digest
    if not checksums:
        raise ValueError("Checksum manifest is empty")
    return checksums


def download_verified(
    url: str,
    destination: Path,
    expected_sha256: str | None,
    *,
    timeout_seconds: float = 60.0,
    attempts: int = 3,
) -> Path:
    """Cache an HTTPS object atomically; never accept a corrupt cached object.

    The checksum list itself is authenticated by HTTPS and saved with its own
    digest in the output manifest. Each EDF must additionally match that list.
    Network failures are retried; integrity failures are immediately fatal.
    """
    destination = Path(destination)
    if urlparse(url).scheme != "https":
        raise ValueError("Only HTTPS source URLs are accepted")
    if attempts < 1 or timeout_seconds <= 0:
        raise ValueError("attempts and timeout_seconds must be positive")
    if expected_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Expected checksum must be a lowercase SHA256 digest")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Cache destination is not a file: {destination}")
        if expected_sha256 and sha256_file(destination) != expected_sha256:
            raise IntegrityError(f"Cached SHA256 mismatch: {destination}; remove and retry")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        temporary_path: Path | None = None
        try:
            request = Request(
                url, headers={"User-Agent": "eeg-state-classifier/physionet-benchmark"}
            )
            with urlopen(request, timeout=timeout_seconds) as response:
                if urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Refusing a redirect from HTTPS to an insecure scheme")
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{destination.name}.",
                    suffix=".part",
                    dir=destination.parent,
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        digest.update(chunk)
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if expected_sha256 and digest.hexdigest() != expected_sha256:
                    raise IntegrityError(f"Downloaded SHA256 mismatch: {url}")
                os.replace(temporary_path, destination)
                return destination
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
            LOGGER.warning("Retrying %s after HTTP %s", url, error.code)
        except (URLError, TimeoutError, OSError) as error:
            if attempt + 1 == attempts:
                raise
            LOGGER.warning("Retrying %s after %s", url, error)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Unable to download {url}")  # pragma: no cover


def _validate_config(config: dict[str, Any]) -> None:
    """Validate preprocessing parameters before touching the data."""
    fs = config["sampling_rate_hz"]
    if not 0 < config["filter_low_hz"] < config["filter_high_hz"] < fs / 2:
        raise ValueError("Filter cutoffs must lie between zero and Nyquist")
    if config["expected_eeg_channels"] != 64:
        raise ValueError("The PhysioNet benchmark requires 64 EEG channels")
    channels = config["expected_channel_names"]
    if len(channels) != 64 or len(set(channels)) != 64 or channels != sorted(channels):
        raise ValueError("The protocol must specify 64 distinct sorted channel names")
    if any(name != normalize_channel_name(name) for name in channels):
        raise ValueError("Protocol channel names must already be normalized")
    if config["window_seconds"] < 2 or config["stride_seconds"] != config["window_seconds"]:
        raise ValueError("Require nonoverlapping windows of at least two seconds")
    if config["discard_start_seconds"] < 0 or config["filter_order"] < 1:
        raise ValueError("Invalid discard interval or filter order")
    if config["runs"] != {"1": "eyes_open", "2": "eyes_closed"}:
        raise ValueError("Baseline runs must be 1=eyes_open, 2=eyes_closed")
    if list(config["bands_hz"]) != ["delta", "theta", "alpha", "beta"]:
        raise ValueError("Require the ordered delta, theta, alpha, beta bands")
    for band, bounds in config["bands_hz"].items():
        low, high = bounds
        if not config["filter_low_hz"] <= low < high <= config["filter_high_hz"]:
            raise ValueError(f"Invalid band bounds for {band}")
    for key in ["window_seconds", "stride_seconds", "discard_start_seconds"]:
        if not float(config[key] * fs).is_integer():
            raise ValueError(f"{key} must resolve to an integer sample count")
    welch = config["welch"]
    if (
        welch["nperseg"] != 2 * fs
        or not 0 <= welch["noverlap"] < welch["nperseg"]
        or welch["nfft"] < welch["nperseg"]
    ):
        raise ValueError("Invalid two-second Welch specification")
    if (
        not np.isfinite(config["power_floor_volts_squared"])
        or config["power_floor_volts_squared"] <= 0
    ):
        raise ValueError("The spectral power floor must be finite and positive")
    base_url = config["source_base_url"]
    if urlparse(base_url).scheme != "https" or not base_url.endswith("/"):
        raise ValueError("source_base_url must be an HTTPS directory URL")


def download_dataset(
    config: dict[str, Any],
    raw_dir: Path,
    subjects: list[int],
    *,
    workers: int = 6,
    mirror_base_url: str | None = None,
) -> tuple[dict[str, str], Path]:
    """Download and verify every requested recording against official hashes."""
    raw_dir = Path(raw_dir)
    checksum_path = download_verified(
        urljoin(config["source_base_url"], CHECKSUM_NAME),
        raw_dir / CHECKSUM_NAME,
        None,
    )
    checksums = parse_checksums(checksum_path.read_text(encoding="utf-8"))
    download_base_url = mirror_base_url or config["source_base_url"]
    if urlparse(download_base_url).scheme != "https" or not download_base_url.endswith("/"):
        raise ValueError("Mirror must be an HTTPS directory URL")
    paths = [
        f"S{subject:03d}/S{subject:03d}R{int(run):02d}.edf"
        for subject in subjects
        for run in config["runs"]
    ]
    missing = [path for path in paths if path not in checksums]
    if missing:
        raise IntegrityError(f"Official checksum list is missing requested files: {missing}")

    def fetch(relative_path: str) -> str:
        download_verified(
            urljoin(download_base_url, relative_path),
            raw_dir / relative_path,
            checksums[relative_path],
        )
        return relative_path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, path in enumerate(pool.map(fetch, paths), start=1):
            LOGGER.info("Verified source %d/%d: %s", index, len(paths), path)
    return {path: checksums[path] for path in paths}, checksum_path


def normalize_channel_name(name: str) -> str:
    """Normalize EDF labels such as Fc5., FC5, and Fc5 to the same channel."""
    normalized = name.strip().rstrip(".").strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9]*", normalized) is None:
        raise ValueError(f"Unexpected EEG channel label: {name!r}")
    return normalized


def feature_columns(channel_names: list[str], bands: dict[str, Any]) -> list[str]:
    return [
        f"feat_{channel.lower()}_{kind}_{band}"
        for channel in channel_names
        for kind in ["log10", "relative"]
        for band in bands
    ]


def _read_edf(path: Path) -> Any:
    try:
        import mne
    except ImportError as error:
        raise RuntimeError("Install the benchmark extra: pip install -e '.[benchmark]'") from error
    return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")


def _spectral_features(epoch: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    fs = config["sampling_rate_hz"]
    frequencies, psd = signal.welch(epoch, fs=fs, axis=-1, **config["welch"])

    def power(low: float, high: float) -> np.ndarray:
        mask = (frequencies >= low) & (frequencies < high)
        if mask.sum() < 2:
            raise ValueError("Each band needs at least two Welch frequency bins")
        return np.trapezoid(psd[:, mask], frequencies[mask], axis=-1)

    absolute = np.column_stack([power(*bounds) for bounds in config["bands_hz"].values()])
    total = power(config["filter_low_hz"], config["filter_high_hz"])
    if (total <= 0).any() or not np.isfinite(total).all():
        raise ValueError("Invalid total spectral power")
    floor = config["power_floor_volts_squared"]
    values = np.concatenate(
        [np.log10(np.maximum(absolute, floor)), absolute / np.maximum(total[:, None], floor)],
        axis=1,
    ).ravel()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite spectral features")
    return values


def extract_recording_features(
    raw: Any,
    config: dict[str, Any],
    *,
    participant_id: str,
    run: int,
    expected_channels: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Validate a recording and return retained windows plus explicit QC counts.

    Flat means exactly zero peak-to-peak voltage in any raw epoch channel; this
    fixed rule is label independent. A nonfinite sample rejects the entire
    recording because per-recording filtering cannot proceed without imputation.
    Rejection never quietly removes a participant: prepare_dataset subsequently
    requires usable windows for every requested participant/run.
    """
    fs = float(raw.info["sfreq"])
    if not np.isclose(fs, config["sampling_rate_hz"], rtol=0, atol=1e-9):
        raise ValueError(f"{participant_id} run {run}: expected 160 Hz; found {fs}")
    channels = [normalize_channel_name(name) for name in raw.ch_names]
    if len(channels) != config["expected_eeg_channels"] or len(set(channels)) != len(channels):
        raise ValueError(f"{participant_id} run {run}: require 64 distinct EEG channels")
    if set(raw.get_channel_types()) != {"eeg"}:
        raise ValueError(f"{participant_id} run {run}: unexpected non-EEG channel type")
    ordered_channels = sorted(channels)
    if ordered_channels != config["expected_channel_names"]:
        raise ValueError(f"{participant_id} run {run}: channel schema differs from frozen protocol")
    if expected_channels is not None and ordered_channels != expected_channels:
        raise ValueError(f"{participant_id} run {run}: inconsistent channel schema")
    order = [channels.index(name) for name in ordered_channels]
    values = raw.get_data(picks=order)
    if values.ndim != 2 or values.shape[0] != len(channels):
        raise ValueError(f"{participant_id} run {run}: malformed EEG array")
    start = int(config["discard_start_seconds"] * fs)
    size = int(config["window_seconds"] * fs)
    stride = int(config["stride_seconds"] * fs)
    starts = list(range(start, values.shape[1] - size + 1, stride))
    qc: dict[str, Any] = {
        "participant_id": participant_id,
        "run": run,
        "label": config["runs"][str(run)],
        "samples": values.shape[1],
        "duration_seconds": values.shape[1] / fs,
        "candidate_windows": len(starts),
        "retained_windows": 0,
        "rejected_flat_windows": 0,
        "rejected_nonfinite_windows": 0,
        "rejected_invalid_power_windows": 0,
        "discarded_start_samples": min(start, values.shape[1]),
        "discarded_tail_samples": max(
            0, values.shape[1] - (starts[-1] + size if starts else start)
        ),
        "nonfinite_sample_count": int((~np.isfinite(values)).sum()),
        "exclusion_reason": None,
    }
    if not starts:
        qc["exclusion_reason"] = "recording_too_short"
        return [], ordered_channels, qc
    if qc["nonfinite_sample_count"]:
        qc["rejected_nonfinite_windows"] = len(starts)
        qc["exclusion_reason"] = "nonfinite_recording_no_imputation"
        return [], ordered_channels, qc
    sos = signal.butter(
        config["filter_order"],
        [config["filter_low_hz"], config["filter_high_hz"]],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, values, axis=-1)
    columns = feature_columns(ordered_channels, config["bands_hz"])
    rows = []
    for epoch_index, first_sample in enumerate(starts):
        original_epoch = values[:, first_sample : first_sample + size]
        if (np.ptp(original_epoch, axis=-1) == 0).any():
            qc["rejected_flat_windows"] += 1
            continue
        epoch = filtered[:, first_sample : first_sample + size]
        if not np.isfinite(epoch).all():
            qc["rejected_nonfinite_windows"] += 1
            continue
        try:
            features = _spectral_features(epoch, config)
        except ValueError:
            qc["rejected_invalid_power_windows"] += 1
            continue
        row = {
            "participant_id": participant_id,
            "run": run,
            "label": config["runs"][str(run)],
            "epoch_index": epoch_index,
            "start_seconds": first_sample / fs,
        }
        row.update(zip(columns, features, strict=True))
        rows.append(row)
    qc["retained_windows"] = len(rows)
    if not rows:
        qc["exclusion_reason"] = "all_windows_failed_qc"
    return rows, ordered_channels, qc


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".json.part",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_dataset(
    config_path: Path,
    raw_dir: Path,
    output_dir: Path,
    subjects: list[int] | None = None,
    *,
    mirror_base_url: str | None = None,
) -> dict[str, Any]:
    """Create a reproducible feature table; subsets are marked as such in metadata.

    Every requested EDF must be available, checksum verified, correctly shaped,
    and represented by at least one retained window. Any failure is fatal and
    writes preparation_failure.json; no incomplete benchmark is published.
    Existing success/failure artifacts are immutable: use a fresh output
    directory for each attempt so a failed run cannot expose stale results.
    """
    config_path, raw_dir, output_dir = map(Path, (config_path, raw_dir, output_dir))
    previous_artifacts = [name for name in PREPARATION_ARTIFACTS if (output_dir / name).exists()]
    if previous_artifacts:
        raise FileExistsError(
            f"Preparation output already contains artifacts {previous_artifacts}; "
            "use a fresh output directory"
        )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    _validate_config(config)
    all_subjects = list(range(config["subject_first"], config["subject_last"] + 1))
    selected_subjects = all_subjects if subjects is None else list(subjects)
    if (
        not selected_subjects
        or any(type(subject) is not int for subject in selected_subjects)
        or len(set(selected_subjects)) != len(selected_subjects)
        or not set(selected_subjects).issubset(all_subjects)
    ):
        raise ValueError(
            "subjects must be unique integer participant IDs within the protocol range"
        )
    selected_subjects.sort()
    requested_ids = [f"S{subject:03d}" for subject in selected_subjects]
    output_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    qc_records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    preprocessing_source_sha256: dict[str, str] = {}
    dependency_lock_sha256: str | None = None
    try:
        preprocessing_source_sha256, dependency_lock_sha256 = _preparation_provenance()
        source_hashes, checksum_path = download_dataset(
            config,
            raw_dir,
            selected_subjects,
            mirror_base_url=mirror_base_url,
        )
        rows: list[dict[str, Any]] = []
        channels: list[str] | None = None
        unusable_recordings = []
        for subject in selected_subjects:
            participant_id = f"S{subject:03d}"
            for run in map(int, config["runs"]):
                relative_path = f"{participant_id}/{participant_id}R{run:02d}.edf"
                source_path = raw_dir / relative_path
                # Recheck at point of use as well as during cache/download verification.
                if sha256_file(source_path) != source_hashes[relative_path]:
                    raise IntegrityError(f"Source changed before EDF read: {relative_path}")
                raw = _read_edf(source_path)
                try:
                    recording_rows, channels, qc = extract_recording_features(
                        raw,
                        config,
                        participant_id=participant_id,
                        run=run,
                        expected_channels=channels,
                    )
                finally:
                    raw.close()
                qc["source_path"] = relative_path
                qc["source_sha256"] = source_hashes[relative_path]
                qc_records.append(qc)
                if not recording_rows:
                    unusable_recordings.append(relative_path)
                rows.extend(recording_rows)
                LOGGER.info(
                    "Prepared %s: %d/%d retained windows",
                    relative_path,
                    qc["retained_windows"],
                    qc["candidate_windows"],
                )
        if unusable_recordings:
            raise ValueError(f"Requested recordings have no usable windows: {unusable_recordings}")
        if channels is None:
            raise ValueError("No channels were prepared")
        columns = feature_columns(channels, config["bands_hz"])
        table = pd.DataFrame(rows, columns=METADATA_COLUMNS + columns)
        if not np.isfinite(table[columns].to_numpy()).all():
            raise ValueError("Prepared table contains nonfinite features")
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir) as staging_name:
            staged_csv = Path(staging_name) / "features.csv"
            table.to_csv(staged_csv, index=False, float_format="%.17g", lineterminator="\n")
            qc_count_keys = [
                "candidate_windows",
                "retained_windows",
                "rejected_flat_windows",
                "rejected_nonfinite_windows",
                "rejected_invalid_power_windows",
            ]
            manifest = {
                "schema_version": "1.0",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_kind": "real_public_eeg",
                "task": config["task"],
                "dataset": config["dataset"],
                "dataset_version": config["dataset_version"],
                "dataset_url": config["dataset_url"],
                "protocol_version": config["protocol_version"],
                "config_sha256": config_sha256,
                "preprocessing_source_sha256": preprocessing_source_sha256,
                "dependency_lock_sha256": dependency_lock_sha256,
                "features_sha256": sha256_file(staged_csv),
                "checksum_manifest": {
                    "url": urljoin(config["source_base_url"], CHECKSUM_NAME),
                    "sha256": sha256_file(checksum_path),
                },
                "source_hashes": source_hashes,
                "source_transport": {
                    "canonical_base_url": config["source_base_url"],
                    "configured_download_base_url": mirror_base_url or config["source_base_url"],
                    "cache_policy": "Preexisting cache origin is not inferred; every EDF must match "
                    "canonical publisher SHA256, including files from official mirrors.",
                },
                "feature_columns": columns,
                "channel_names": channels,
                "participant_ids": sorted(table["participant_id"].unique().tolist()),
                "expected_participant_ids": requested_ids,
                "participant_count": len(requested_ids),
                "full_protocol": selected_subjects == all_subjects,
                "recordings_count": len(source_hashes),
                "windows_count": len(table),
                "n_rows": len(table),
                "n_features": len(columns),
                "label_counts": table["label"].value_counts().sort_index().to_dict(),
                "qc": {
                    **{key: sum(record[key] for record in qc_records) for key in qc_count_keys},
                    "flat_rule": "Any raw epoch channel has exactly zero peak-to-peak voltage",
                    "nonfinite_rule": "Reject complete recording; no interpolation or imputation",
                    "recordings": qc_records,
                },
                "preprocessing": {
                    "filter": "scipy.signal.butter SOS + sosfiltfilt, per recording, offline",
                    "filter_low_hz": config["filter_low_hz"],
                    "filter_high_hz": config["filter_high_hz"],
                    "filter_order": config["filter_order"],
                    "discard_start_seconds": config["discard_start_seconds"],
                    "window_seconds": config["window_seconds"],
                    "stride_seconds": config["stride_seconds"],
                    "bands_hz": config["bands_hz"],
                    "welch": config["welch"],
                    "psd_units": "V^2/Hz",
                    "power_units": "V^2",
                    "band_integration": config["integration"],
                    "power_floor_volts_squared": config["power_floor_volts_squared"],
                    "quality_control": config["quality_control"],
                },
                "library_versions": {
                    "python": sys.version.split()[0],
                    **{
                        package: importlib.metadata.version(package)
                        for package in ["numpy", "pandas", "scipy", "mne"]
                    },
                },
            }
            os.replace(staged_csv, output_dir / "features.csv")
            _write_json_atomic(output_dir / "features_manifest.json", manifest)
        return manifest
    except Exception as error:
        _write_json_atomic(
            output_dir / "preparation_failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "config_sha256": config_sha256,
                "preprocessing_source_sha256": preprocessing_source_sha256,
                "dependency_lock_sha256": dependency_lock_sha256,
                "expected_participant_ids": requested_ids,
                "source_hashes": source_hashes,
                "qc_recordings_completed": qc_records,
            },
        )
        raise
