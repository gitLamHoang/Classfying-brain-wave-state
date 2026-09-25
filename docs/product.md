# Product hypothesis and engineering decisions

## Intended user and working scope

A student researcher needs to turn EEG recordings into an experiment they can inspect and reproduce. The working slice is acquire → validate → extract → select → evaluate → predict. This audience remains a hypothesis; no adoption or user interviews are claimed.

The implementation now includes a full public-cohort experiment, not only synthetic fixtures. It handles 218 source files, multichannel signal processing, explicit participant isolation, bounded model selection, uncertainty estimates and schema-checked inference without a UI.

## Architecture and tradeoffs

**Acquisition and preparation are separate from fitting.** Network retries, checksum failures and malformed EDFs fail before any model selection. Source hashes, per-recording QC counts and preparation-code hashes bind each feature table to the experiment that created it. Output reuse is rejected to prevent stale success artifacts from masquerading as a completed rerun.

**Keep metadata out of predictors.** Run labels are needed to audit the task, and participant IDs define the split, but neither enters the 512-dimensional feature matrix. Fitted scaling lives inside each training fold. Every participant's recordings and windows remain together.

**Use a bounded, inspectable comparison.** Eight predefined candidates cover a dummy classifier, linear logistic regression, nonlinear RBF SVM and random forest. This is a useful baseline comparison for the available cohort, not a search for the largest model. The configuration was committed before the first fit; selection is written to disk before the final holdout is evaluated.

**Report uncertainty at the correct unit.** Four-second windows are correlated within a participant. The confidence intervals resample whole participants and recompute the pooled metric, alongside participant-level scores. They quantify uncertainty within the held-out cohort, not generalization to new devices or populations.

**Batch first.** A CLI and a versioned artifact make input contracts and failures easy to test. A web service or UI would add operations without solving a demonstrated user need. The current zero-phase preprocessing is explicitly offline; streaming would require a different signal-processing design and a new evaluation.

## Next useful experiments

1. Observe student researchers importing data, understanding rejected inputs and tracing outputs to source recordings.
2. Evaluate a separately collected cohort with randomized condition order to address run-order confounding and dataset shift.
3. Compare artifact handling under a new predefined protocol; preserve the current result as historical evidence.
4. Measure workload and latency on target hardware before designing online inference.

The original single-channel serial collection and synthetic demo remain available as a separate workflow. Medical diagnosis, driver safety interventions and calibrated clinical risk scores are outside the demonstrated scope.
