"""Prepare verified real EEG features for the frozen eyes-open/closed benchmark."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eeg_state_classifier.physionet import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/physionet_eyes_protocol.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/physionet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/physionet"))
    parser.add_argument(
        "--mirror-base-url",
        help="Optional HTTPS mirror directory; all EDF hashes still use the canonical publisher list.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Optional preparation-only subset; final benchmark requires all protocol subjects.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = prepare_dataset(
        args.config,
        args.raw_dir,
        args.output_dir,
        args.subjects,
        mirror_base_url=args.mirror_base_url,
    )
    print(
        f"Prepared {manifest['n_rows']} windows × {manifest['n_features']} features from "
        f"{manifest['recordings_count']} recordings / {manifest['participant_count']} participants."
    )
    print(f"Full frozen protocol: {manifest['full_protocol']}")
    print(f"Features SHA256: {manifest['features_sha256']}")
    print(f"Saved {args.output_dir / 'features_manifest.json'}")


if __name__ == "__main__":
    main()
