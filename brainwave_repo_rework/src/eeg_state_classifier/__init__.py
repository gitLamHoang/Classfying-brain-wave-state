"""EEG state classification utilities."""

from .features import BAND_DEFINITIONS, extract_features, make_windows
from .modeling import evaluate_classifier, train_classifier

__all__ = [
    "BAND_DEFINITIONS",
    "extract_features",
    "make_windows",
    "train_classifier",
    "evaluate_classifier",
]
