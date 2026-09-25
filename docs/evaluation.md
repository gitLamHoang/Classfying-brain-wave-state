# Original single-channel evaluation contract

This page describes the legacy synthetic/hardware workflow. The real 64-channel public-data experiment uses a separate [frozen protocol](real_eeg_benchmark.md).

## Unit of independence

Rows are windows; participants are the unit of independence. Repeated or overlapping windows from one participant stay together, including windows from different sessions. A session ID alone is insufficient when the same person has multiple sessions. The caller is responsible for accurate IDs and labels.

`train_classifier(x, y, groups=...)` validates aligned pandas indices, finite numeric predictors, nonmissing labels and IDs, and at least two classes. The CSV loader preserves string IDs such as `001` and `01` and drops both the label and group columns from features. Remove other metadata columns before training.

## Fixed holdout, training-only selection

1. `GroupShuffleSplit` selects one deterministic holdout using seed 42 by default. `--test-size` controls the fraction of groups, so the fraction of rows may differ.
2. Both train and test must contain every class. Invalid coverage stops the run; the script does not search seeds for a favorable outcome.
3. `StratifiedGroupKFold` generates up to five tuning folds using only training participants. The maximum is bounded by the number of training groups containing each class. All folds are prechecked for class coverage and zero participant overlap. Stratification is approximate; an invalid fold stops the run rather than silently dropping a class.
4. `GridSearchCV` evaluates 12 SVM configurations inside a `StandardScaler` → `SVC` pipeline. Scaler statistics are fitted separately within each fold. Selection uses mean fold accuracy; the best configuration is refit on all training participants.
5. The holdout is scored after selection. Its values never enter scaling or hyperparameter search. No probability calibration is performed.

This is one held-out evaluation with inner cross-validation, not an estimate from repeated outer cross-validation. Fold accuracy weights windows, not participants. Unequal recording lengths can therefore dominate the metric. Real studies should predefine participant-level metrics and uncertainty estimates before opening their final holdout.

## Audit output

`reports/metrics.json` records the strategy, seed, requested group fraction, feature order, train/test row counts, group counts, overlap counts, inner fold sizes, selected parameters, classification report, input SHA-256, and split SHA-256. The split fingerprint hashes row-index membership; it is reproducible only alongside the exact input file hash and row order. Participant identifiers are not written to the metrics file.

The synthetic fixture deliberately gives every generated ID both classes. It validates the software path and audit counts, not physiological diversity. Its 1.0 accuracy must not be presented as real-participant performance.

## Regression checks

The tests include an adversarial holdout perturbation: add a large offset to all held-out feature rows, rerun training, and verify that fitted scaler statistics, selected parameters, and predictions on the unchanged input remain identical. Other tests reject missing IDs, group identifiers accidentally used as predictors, misaligned rows, missing holdout classes, insufficient training participants, and non-finite feature values.

Legacy ungrouped runs retain row-stratified splitting for a synthetic smoke test. They are visibly labeled `row_split_synthetic_smoke_only`, including the CLI output and metrics. Never mix those scores with participant-disjoint results.
