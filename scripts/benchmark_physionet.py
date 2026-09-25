"""Evaluate the frozen PhysioNet eyes-open/eyes-closed protocol once."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eeg_state_classifier.benchmark import run_benchmark


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Prepared features.csv")
    parser.add_argument("--manifest", type=Path, required=True, help="features_manifest.json")
    parser.add_argument("--config", type=Path, default=Path("configs/physionet_eyes_protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Diagnostic only: conspicuously nonfinal, cannot support full benchmark claims",
    )
    args = parser.parse_args()
    report = run_benchmark(
        args.features, args.manifest, args.config, args.output_dir, args.allow_subset
    )
    selected = report["final_holdout"]["selected"]
    print(f"Status: {report['validation_status']}")
    print(f"Selected by inner grouped CV: {selected['candidate']}")
    print(f"Held-out balanced accuracy: {selected['balanced_accuracy']:.4f}")
    print(f"Report: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
