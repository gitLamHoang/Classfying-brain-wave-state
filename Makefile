.PHONY: install test lint sample train predict

install:
	uv sync --frozen --python 3.12 --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

sample:
	uv run python scripts/generate_sample_data.py --output-dir data/generated --participants 12 --samples-per-class 4

train:
	uv run python scripts/train_model.py --features-csv data/generated/eeg_features_grouped.csv --group-column participant_id --test-size 0.25

predict:
	uv run python scripts/predict_window.py --raw-txt data/generated/eeg_raw_awake_sample.txt --model models/svm_eeg_state.joblib
