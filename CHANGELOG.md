# Changelog

## 2026-09-24 — Participant isolation and reproducible review

- Added participant-disjoint holdout and grouped inner model selection, with zero-overlap and class-coverage checks before fitting.
- Added `--group-column` and separate string-preserving metadata loading; excluded participant IDs from predictors.
- Added fold audits, input hashes, split fingerprints, and explicit labels for legacy row-split smoke tests.
- Disabled unvalidated probability output; predictions remain class labels.
- Added a grouped synthetic generator path, 15 new validation/input tests (17 total), locked Python dependencies, Ruff checks, and CI with a complete generate/train/predict run.
- Verified 96 synthetic windows across 12 synthetic IDs: 9 train / 3 holdout IDs, five disjoint tuning folds, and 16 raw-recording predictions. Synthetic accuracy is 1.0 because labels are engineered into the signals; it is not evidence about real participants.
- Published evaluator-first documentation, experiment boundaries, product hypotheses, and scaling tradeoffs. Real-data validation and user interviews remain future work.
