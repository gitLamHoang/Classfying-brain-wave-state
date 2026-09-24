# Product hypothesis and engineering decisions

## Who might use this?

A student researcher building a low-cost EEG prototype needs a short, inspectable path from a recording to an experiment they can explain. The working product slice is: collect or generate → extract features → train with participant isolation → inspect an audit → predict completed windows. This intended audience is a hypothesis; no interviews, adoption, or demand are claimed.

A useful first user study would observe researchers importing a table with participant IDs, understanding a split rejection, and tracing a prediction back to the feature schema. Success would mean completing those tasks without editing library code. Record actual observations before expanding the interface.

## What exists today?

- A reusable Python signal-processing and training package with CLI entry points.
- Optional serial capture plus synthetic signals for hardware-free review.
- Participant-disjoint holdout and grouped model selection with explicit failure paths.
- Locked dependencies, CI, a runnable demo, unit tests, and compact synthetic evidence.

## Why this architecture?

A scikit-learn pipeline keeps preprocessing coupled to the model. Explicit metadata separation prevents a participant identifier from becoming a predictor. Single-process, bounded 12-configuration tuning is easy to reproduce on a laptop. Input hashes and fold audits make experiments reviewable without publishing private recordings or model binaries.

A service or distributed system would add operations before there is a demonstrated need. For larger experiments, feature extraction is separable by recording and could produce partitioned tables; training could then consume a versioned manifest. Those are future interfaces, not current throughput claims. Kernel SVM fitting will become a constraint as the number of windows grows; benchmark against linear models before scaling hardware.

## Next milestones, in order

1. Validate the workflow with student researchers and document concrete points of confusion.
2. Obtain appropriately consented data with stable participant and session IDs; document collection and labeling procedures.
3. Freeze a real-data protocol with baselines, participant-weighted metrics, and uncertainty intervals before evaluation.
4. Add artifact detection and signal review where observed data problems justify them.
5. Benchmark memory and latency on representative recordings; choose batch or online interfaces using those measurements.

Medical diagnosis, driver safety interventions, calibrated risk scores, and real-time clinical monitoring are outside the demonstrated scope.
