"""Signal filtering and EEG feature extraction.

The original prototype used FFT amplitudes from one 15-second EEG window.
This module keeps that idea but makes it reusable, testable, and safer:

- functions do not read files or load models as side effects;
- division-by-zero is handled for ratio features;
- Welch power spectral density is used for more stable bandpower estimates;
- output feature order is explicit so training and prediction stay consistent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy import signal

BAND_DEFINITIONS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

FEATURE_COLUMNS: list[str] = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "alpha_beta_ratio",
    "theta_beta_ratio",
    "delta_beta_ratio",
    "theta_alpha_ratio",
    "delta_alpha_ratio",
    "slow_fast_ratio",
]


@dataclass(frozen=True)
class Window:
    """A single EEG window and its timing metadata."""

    start_sample: int
    end_sample: int
    values: np.ndarray

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


def bandpass_filter(
    samples: np.ndarray,
    sampling_rate: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to a 1-D EEG signal.

    Parameters
    ----------
    samples:
        Raw EEG voltage/ADC samples.
    sampling_rate:
        Sampling frequency in Hz.
    low_hz, high_hz:
        Frequency cutoffs for the pass band.
    order:
        Butterworth filter order.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1:
        raise ValueError("samples must be a 1-D array")
    if len(values) < order * 6:
        raise ValueError("not enough samples to apply a stable band-pass filter")
    if not 0 < low_hz < high_hz < sampling_rate / 2:
        raise ValueError("cutoff frequencies must be between 0 and Nyquist frequency")

    sos = signal.butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return signal.sosfiltfilt(sos, values)


def make_windows(
    samples: np.ndarray,
    sampling_rate: float,
    window_seconds: float = 15.0,
    stride_seconds: float = 1.0,
) -> Iterator[Window]:
    """Yield fixed-length sliding windows from a 1-D EEG signal."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1:
        raise ValueError("samples must be a 1-D array")
    window_size = int(round(window_seconds * sampling_rate))
    stride_size = int(round(stride_seconds * sampling_rate))
    if window_size <= 0 or stride_size <= 0:
        raise ValueError("window and stride must be positive")
    if len(values) < window_size:
        return

    for start in range(0, len(values) - window_size + 1, stride_size):
        end = start + window_size
        yield Window(start, end, values[start:end])


def _bandpower(freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator / (denominator + eps))


def extract_features(
    samples: np.ndarray,
    sampling_rate: float = 512.0,
    apply_filter: bool = True,
) -> dict[str, float]:
    """Extract bandpower and ratio features from one EEG window.

    Returns a dictionary ordered by ``FEATURE_COLUMNS``.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1:
        raise ValueError("samples must be a 1-D array")
    if len(values) < int(sampling_rate):
        raise ValueError("provide at least 1 second of samples")

    values = signal.detrend(values)
    if apply_filter:
        values = bandpass_filter(values, sampling_rate=sampling_rate)

    nperseg = min(len(values), int(2 * sampling_rate))
    freqs, psd = signal.welch(values, fs=sampling_rate, nperseg=nperseg)

    powers = {
        band: _bandpower(freqs, psd, low, high)
        for band, (low, high) in BAND_DEFINITIONS.items()
    }

    delta = powers["delta"]
    theta = powers["theta"]
    alpha = powers["alpha"]
    beta = powers["beta"]

    features = {
        "delta_power": delta,
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "alpha_beta_ratio": _safe_ratio(alpha, beta),
        "theta_beta_ratio": _safe_ratio(theta, beta),
        "delta_beta_ratio": _safe_ratio(delta, beta),
        "theta_alpha_ratio": _safe_ratio(theta, alpha),
        "delta_alpha_ratio": _safe_ratio(delta, alpha),
        "slow_fast_ratio": _safe_ratio(delta + theta, alpha + beta),
    }
    return {column: features[column] for column in FEATURE_COLUMNS}
