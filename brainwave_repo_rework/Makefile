.PHONY: install test lint sample train predict clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests

sample:
	python scripts/generate_sample_data.py --output-dir data/sample

train:
	python scripts/train_model.py --features-csv data/sample/eeg_features_sample.csv --model-out models/svm_eeg_state.joblib --reports-dir reports

predict:
	python scripts/predict_window.py --raw-txt data/sample/eeg_raw_awake_sample.txt --model models/svm_eeg_state.joblib --output-csv reports/sample_predictions.csv

clean:
	rm -f models/*.joblib reports/*.json reports/*.csv reports/figures/*.png
