from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_state_classifier.inference import SCHEMA_VERSION, predict_file, predict_table


@pytest.fixture
def artifact():
    x = pd.DataFrame({"feat_a": [-2.0, -1.0, 1.0, 2.0], "feat_b": [1.0, 2.0, -2.0, -1.0]})
    y = ["eyes_closed", "eyes_closed", "eyes_open", "eyes_open"]
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "feature_columns": x.columns.tolist(),
        "label_names": sorted(set(y)),
        "config_sha256": "test",
        "selected_candidate": "logistic",
    }


def test_aligns_features_and_preserves_metadata_without_target(artifact):
    a = pd.DataFrame(
        {"feat_b": [1.0], "participant_id": ["S001"], "feat_a": [-2.0], "label": ["x"]}
    )
    result = predict_table(artifact, a)
    assert result.to_dict("records") == [
        {"participant_id": "S001", "predicted_state": "eyes_closed"}
    ]
    assert result.equals(
        predict_table(artifact, a[["participant_id", "feat_a", "feat_b", "label"]])
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "invalid"])
def test_rejects_nonfinite_and_nonnumeric(artifact, bad_value):
    with pytest.raises(ValueError, match="numeric|finite"):
        predict_table(artifact, pd.DataFrame({"feat_a": [bad_value], "feat_b": [1]}))


def test_rejects_extra_predictor_and_missing_feature(artifact):
    with pytest.raises(ValueError, match="schema mismatch"):
        predict_table(artifact, pd.DataFrame({"feat_a": [1], "feat_injected": [2]}))


def test_rejects_renamed_artifact_features(artifact):
    artifact["feature_columns"] = ["feat_b", "feat_a"]
    with pytest.raises(ValueError, match="orders disagree"):
        predict_table(artifact, pd.DataFrame({"feat_a": [1], "feat_b": [2]}))


def test_file_roundtrip_and_no_overwrite(artifact, tmp_path):
    model = tmp_path / "model.joblib"
    features = tmp_path / "features.csv"
    output = tmp_path / "predictions.csv"
    joblib.dump(artifact, model)
    pd.DataFrame({"feat_a": [-2, 2], "feat_b": [1, -1]}).to_csv(features, index=False)
    metadata = predict_file(model, features, output)
    assert metadata["rows"] == 2 and len(metadata["model_sha256"]) == 64
    assert pd.read_csv(output)["predicted_state"].tolist() == ["eyes_closed", "eyes_open"]
    assert output.with_suffix(".csv.json").exists()
    with pytest.raises(FileExistsError):
        predict_file(model, features, output)


def test_rejects_other_label_task(artifact):
    artifact["label_names"] = ["awake", "sleepy"]
    with pytest.raises(ValueError, match="eye.*condition|eyes-open"):
        predict_table(artifact, pd.DataFrame({"feat_a": [1], "feat_b": [2]}))
