"""Batch inference from a trusted real-EEG benchmark model and prepared feature CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_state_classifier.inference import predict_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trusted benchmark model.joblib")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(predict_file(args.model, args.features, args.output), indent=2))


if __name__ == "__main__":
    main()
