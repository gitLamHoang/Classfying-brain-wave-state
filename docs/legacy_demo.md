# Original single-channel workflow

This page covers only the original synthetic/hardware path. For the current real-data multichannel experiment, start with the [repository README](../README.md) and [benchmark protocol](real_eeg_benchmark.md).

[![Checks](https://github.com/gitLamHoang/Classfying-brain-wave-state/actions/workflows/ci.yml/badge.svg)](https://github.com/gitLamHoang/Classfying-brain-wave-state/actions/workflows/ci.yml)

**From a raw sensor recording to a reproducible classification experiment—with participants kept separate during evaluation.**

Small EEG prototypes can look accurate when overlapping windows from the same person appear in both training and testing. This project turns a hardware + ML prototype into an inspectable signal-processing pipeline and makes that evaluation boundary explicit.

**Status:** research prototype. Public recordings and participant IDs are synthetic. They demonstrate working software, not clinical effectiveness or performance on people. No real-participant accuracy is claimed.

[Run the demo](#run-the-demo) · [Evaluation design](evaluation.md) · [Measured synthetic run](evidence/grouped_synthetic.json) · [Product direction](product.md)

## What you can inspect in two minutes

| Question | Evidence |
|---|---|
| Does it run from raw signal to prediction? | CI generates signals, trains the grouped model, and predicts sliding windows. |
| Can one participant appear on both sides? | Outer holdout and every tuning fold check for zero participant overlap. |
| Can held-out data influence preprocessing? | Scaling stays inside the training pipeline; a test perturbs held-out features and verifies unchanged scaling and model selection. |
| Is the result reproducible? | Locked environment, fixed seed, input hash, split fingerprint, feature names, and fold counts accompany metrics. |

## Run the demo

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/gitLamHoang/Classfying-brain-wave-state.git
cd Classfying-brain-wave-state
uv sync --frozen --python 3.12 --extra dev
uv run python scripts/generate_sample_data.py \
  --output-dir data/generated --participants 12 --samples-per-class 4
uv run python scripts/train_model.py \
  --features-csv data/generated/eeg_features_grouped.csv \
  --group-column participant_id --test-size 0.25
uv run python scripts/predict_window.py \
  --raw-txt data/generated/eeg_raw_awake_sample.txt \
  --model models/svm_eeg_state.joblib
```

The verified demo creates **96 synthetic windows across 12 synthetic IDs**, holds out **3 IDs / 24 windows**, and tunes on the remaining **9 IDs / 72 windows** using five group-disjoint folds. Prediction emits **16 windows** from a 30-second generated recording. Labels are engineered into the synthetic signals, so their easy separability is not evidence of real EEG accuracy.

Generated models, recordings, and reports stay out of Git. Review local `reports/metrics.json`, `reports/confusion_matrix.csv`, and `reports/predictions.csv` after running. A compact, explicitly synthetic [run record](evidence/grouped_synthetic.json) is committed for comparison.

## How it works

```mermaid
flowchart LR
    A[Serial recording or synthetic signal] --> B[15-second windows]
    B --> C[Band-pass filter and Welch PSD]
    C --> D[10 bandpower and ratio features]
    D --> E[Participant-disjoint holdout]
    E --> F[Grouped tuning on training participants]
    F --> G[Scaler + SVM artifact]
    G --> H[Window predictions and evaluation audit]
```

The Python package uses NumPy/SciPy for signal processing, pandas for feature tables, and scikit-learn for training. Four bandpowers—delta, theta, alpha, beta—and six ratios provide an understandable baseline. Model inference uses the same ordered feature schema as training. Probabilities are not exported: this baseline has no separately validated probability calibration.

| Location | Responsibility |
|---|---|
| `src/eeg_state_classifier/features.py` | Filtering, windowing, and feature extraction |
| `src/eeg_state_classifier/modeling.py` | Split checks, grouped tuning, evaluation, persistence |
| `src/eeg_state_classifier/io.py` | Separate participant metadata from predictors |
| `scripts/` | Generate, train, predict, and optionally record serial input |
| `tests/` | Signal invariants, metadata handling, and validation isolation |

## Bring a feature table

Use numeric feature columns, a `label` column, and a stable `participant_id` shared across **all sessions from the same participant**. Pass `--group-column participant_id`; it is removed before fitting. Raw-signal prediction expects the ten canonical columns in `FEATURE_COLUMNS`; other feature schemas can use the training library separately. Missing IDs, non-finite features, missing classes in either side of a split, and invalid tuning folds fail before model fitting. See the [input contract and split policy](evaluation.md).

The original `--awake-csv` / `--sleepy-csv` format and ungrouped sample still run, but their metrics are labeled `row_split_synthetic_smoke_only`. They cannot support claims about new participants. Use `--no-tune` for a fixed baseline when there are enough participants for a holdout but too few for grouped tuning.

Only load model files you created or trust: joblib deserialization executes Python objects.

## Development

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

CI runs these checks and the complete synthetic demo on Python 3.12. The lockfile pins dependency versions; `pip install -e ".[dev]"` remains available for exploratory environments.

## Limits and next experiment

This is an offline, single-channel prototype. The zero-phase filter uses future samples inside a completed window; it is not a validated low-latency streaming classifier. SVM training is intended for small feature tables and has not been load-benchmarked. Group separation helps prevent one leakage source; it does not fix weak labels, signal artifacts, or cohort bias.

The next scientific step is a consented dataset with participant/session metadata, a predefined evaluation protocol, and simpler baseline comparisons. The next product step is testing the experiment workflow with student researchers. Neither user demand nor medical utility has been validated. [Design tradeoffs and next milestones](product.md) · [Hardware collection notes](hardware_protocol.md)

Initial work was developed with guidance from Professor Trinh Van Chien, School of Information and Communication Technology, Hanoi University of Science and Technology.
