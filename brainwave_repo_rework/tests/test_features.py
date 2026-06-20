from __future__ import annotations

import numpy as np

from eeg_state_classifier.features import FEATURE_COLUMNS, extract_features, make_windows


def test_extract_features_returns_expected_columns() -> None:
    sampling_rate = 512
    t = np.arange(15 * sampling_rate) / sampling_rate
    samples = np.sin(2 * np.pi * 10 * t)

    features = extract_features(samples, sampling_rate=sampling_rate)

    assert list(features.keys()) == FEATURE_COLUMNS
    assert all(np.isfinite(value) for value in features.values())
    assert features["alpha_power"] > features["delta_power"]


def test_make_windows_uses_expected_window_count() -> None:
    sampling_rate = 10
    samples = np.arange(50)
    windows = list(make_windows(samples, sampling_rate=sampling_rate, window_seconds=2, stride_seconds=1))

    assert len(windows) == 4
    assert windows[0].start_sample == 0
    assert windows[-1].end_sample == 50
