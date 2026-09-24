from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from eeg_state_classifier.modeling import train_classifier


@pytest.fixture
def dataset():
    rng = np.random.default_rng(17)
    groups = pd.Series(np.repeat(np.arange(12), 8), name="participant_id")
    y = pd.Series(np.tile(["awake"] * 4 + ["sleepy"] * 4, 12), name="label")
    x = pd.DataFrame(
        {
            "alpha": (y == "awake").astype(float) + rng.normal(0, 0.3, len(y)),
            "beta": rng.normal(size=len(y)),
        }
    )
    return x, y, groups


def test_grouped_holdout_and_every_tuning_fold_are_disjoint(dataset):
    x, y, groups = dataset
    result = train_classifier(x, y, groups=groups, test_size=0.25)
    audit = result.metrics["validation"]
    assert audit["strategy"] == "participant_disjoint"
    assert audit["train_groups"] == 9
    assert audit["test_groups"] == 3
    assert audit["group_overlap"] == 0
    assert len(audit["tuning_folds"]) == 5
    assert all(fold["group_overlap"] == 0 for fold in audit["tuning_folds"])
    assert not hasattr(result.model, "predict_proba")
    assert result.metrics["feature_columns"] == ["alpha", "beta"]
    assert len(audit["split_sha256"]) == 64


def test_test_participants_do_not_affect_scaling_or_model_selection(dataset):
    x, y, groups = dataset
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(x, y, groups)
    )
    original = train_classifier(x, y, groups=groups, test_size=0.25)
    perturbed = x.copy()
    perturbed.iloc[test] += 10000
    result = train_classifier(perturbed, y, groups=groups, test_size=0.25)
    np.testing.assert_allclose(result.model["scaler"].mean_, x.iloc[train].mean())
    assert result.metrics["best_params"] == original.metrics["best_params"]
    np.testing.assert_array_equal(result.model.predict(x), original.model.predict(x))
    assert (
        result.metrics["validation"]["split_sha256"]
        == original.metrics["validation"]["split_sha256"]
    )


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_missing_participant_ids_rejected(dataset, bad_id):
    x, y, groups = dataset
    groups = groups.astype(object)
    groups.iloc[0] = bad_id
    with pytest.raises(ValueError, match="Participant IDs must not"):
        train_classifier(x, y, groups=groups)


def test_group_identifier_cannot_be_a_predictor(dataset):
    x, y, groups = dataset
    with pytest.raises(ValueError, match="must not be included in features"):
        train_classifier(x.assign(participant_id=groups), y, groups=groups)


def test_group_row_order_must_match_features(dataset):
    x, y, groups = dataset
    with pytest.raises(ValueError, match="same index order"):
        train_classifier(x, y, groups=groups.iloc[::-1])


def test_single_class_participant_holdout_rejected():
    x = pd.DataFrame({"value": range(8)})
    y = pd.Series(["awake"] * 4 + ["sleepy"] * 4)
    groups = pd.Series(["a"] * 4 + ["b"] * 4)
    with pytest.raises(ValueError, match="Holdout must contain every class"):
        train_classifier(x, y, groups=groups, tune=False)


def test_tuning_requires_independent_participants_per_class():
    x = pd.DataFrame({"value": range(8)})
    y = pd.Series(["awake", "sleepy"] * 4)
    groups = pd.Series(["a"] * 4 + ["b"] * 4)
    with pytest.raises(ValueError, match="two training participants"):
        train_classifier(x, y, groups=groups)


@pytest.mark.parametrize("bad_value", [np.inf, np.nan])
def test_non_finite_features_rejected(dataset, bad_value):
    x, y, groups = dataset
    x.iloc[0, 0] = bad_value
    with pytest.raises(ValueError, match="finite values"):
        train_classifier(x, y, groups=groups)


def test_legacy_row_mode_is_labeled_as_smoke_only(dataset):
    x, y, _ = dataset
    result = train_classifier(x, y, tune=False)
    assert result.metrics["validation"]["strategy"] == "row_split_synthetic_smoke_only"
    assert result.metrics["validation"]["tuning_folds"] == []
