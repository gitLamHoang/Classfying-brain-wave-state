# Real EEG benchmark

This is a reproducible experiment for **baseline eyes-open versus eyes-closed EEG**.
It extends the original single-channel research prototype with real public recordings,
audited acquisition, multichannel features, a locked evaluation protocol, and batch
inference. It does not relabel eyes-closed recordings as sleep or drowsiness.

## Data and attribution

Source: [PhysioNet EEG Motor Movement/Imagery Dataset, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
Only baseline runs 1 and 2 are used, for subjects S001–S109. The dataset documents
64 EEG electrodes at 160 Hz. The actual file lengths determine the number of windows;
the code does not assume every recording has exactly one minute of samples.

The downloader verifies every EDF against the publisher's `SHA256SUMS.txt`, uses
atomic file writes, and rechecks the cache. Raw recordings, feature tables, serialized
models, and per-window predictions stay outside Git. A preparation manifest records
source URLs/hashes, normalized channels, quality exclusions, and the feature-table hash.

Dataset attribution: Schalk, G. (2009), *EEG Motor Movement/Imagery Dataset*,
version 1.0.0, PhysioNet, DOI [10.13026/C28G6P](https://doi.org/10.13026/C28G6P).
Data are available under the **Open Data Commons Attribution License v1.0**.
Original method: Schalk et al. (2004), *BCI2000: A General-Purpose Brain-Computer
Interface (BCI) System*, IEEE Transactions on Biomedical Engineering 51(6):1034–1043.
PhysioNet resource: Goldberger et al. (2000), *PhysioBank, PhysioToolkit, and PhysioNet:
Components of a New Research Resource for Complex Physiologic Signals*, Circulation
101(23):e215–e220, DOI [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215).

## Frozen experiment

The complete contract is [configs/physionet_eyes_protocol.json](../configs/physionet_eyes_protocol.json).
It was committed before model fitting. The final protocol was made more explicit
after a pre-fit review; no measured model or holdout result informed those revisions.

1. Verify acquisition and the 64-channel schema, including units and sample rate.
2. Apply a fourth-order, zero-phase 1–40 Hz SOS filter separately to each recording.
3. Discard the first two seconds; form four-second nonoverlapping windows and report
   incomplete tails. Reject raw flat-channel windows and nonfinite recordings under
   fixed rules. Missing sources and loss of either condition for a participant fail.
4. Compute 512 spectral features: four log absolute and four relative band powers per
   channel. Welch parameters, integration bounds, voltage-power floor, and ordered
   channel names are fixed in the protocol. Participant/run/time/condition metadata
   never enter the predictors.
5. Reserve 20% of participants using a seeded group split. Both baseline recordings
   and every epoch from each participant stay in one partition.
6. Compare a dummy classifier, logistic regression, RBF SVM, and random forest using
   five grouped training-only folds. Fit scaling inside each training fold.
7. Select the candidate using only mean inner balanced accuracy. Open the final
   participant holdout once, after selection, and report the selected model and dummy.
8. Report pooled balanced accuracy, accuracy, macro F1, participant metrics,
   confusion counts, and confidence intervals from 2,000 whole-participant bootstrap
   samples. Persist a feature-schema-aware inference artifact.

The holdout estimate is window-weighted. The report also includes participant-level
results; these can differ when recording lengths or accepted epoch counts differ.
Bootstrap samples retain all rows from each resampled participant and recompute the
same pooled statistic. They describe uncertainty within this cohort, not performance
on another device or population.

## Reproduce

Use Python 3.12 with the committed dependency lock. Raw data are a few hundred MB;
feature extraction and fitting run locally on CPU. See the top-level README for the
measured run and exact commands.

Use a new output directory for each benchmark. Completed result directories refuse
overwriting. The model and prediction sidecars bind results to the protocol and inputs.
The batch predictor accepts only the declared numerical features and known metadata,
aligns features to the trained order, rejects unknown/missing/nonfinite inputs, and
emits class labels rather than uncalibrated probability claims.

Only load `joblib` files you created or trust; Python serialized models can execute code.

## Interpretation and limits

- Eyes-open/closed is the recorded baseline condition, not clinical sleep staging.
- Baseline conditions were recorded in a fixed run order. Condition and run identity
  are confounded; a model may exploit recording differences as well as physiology.
- Participants are independent across partitions, but all recordings come from the
  same public dataset. This is not external, prospective, or cross-device validation.
- The zero-phase filter uses future samples inside a completed recording. This is an
  offline research workflow, not a validated streaming or real-time classifier.
- The fixed quality checks catch malformed/nonfinite/flat data, not all eye, muscle,
  motion, or electrode artifacts. No artifact-free or clinical-quality claim is made.
- The hyperparameter search is deliberately bounded. Strong performance on this task
  would not establish state-of-the-art EEG decoding or justify medical use.

The original synthetic awake/sleepy demo remains a separate software test. Its scores
are never combined with the public real-data benchmark.
