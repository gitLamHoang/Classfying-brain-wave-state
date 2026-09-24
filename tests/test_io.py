from __future__ import annotations

import pytest

from eeg_state_classifier.io import load_training_table


def test_group_ids_are_preserved_and_excluded_from_features(tmp_path):
    source = tmp_path / "features.csv"
    source.write_text("alpha,label,participant_id\n0.8,awake,001\n0.2,sleepy,01\n")
    x, y, groups = load_training_table(source, group_column="participant_id")
    assert x.columns.tolist() == ["alpha"]
    assert y.tolist() == ["awake", "sleepy"]
    assert groups.tolist() == ["001", "01"]


def test_missing_group_column_rejected(tmp_path):
    source = tmp_path / "features.csv"
    source.write_text("alpha,label\n0.8,awake\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_training_table(source, group_column="participant_id")


def test_group_cannot_be_label(tmp_path):
    with pytest.raises(ValueError, match="must be different"):
        load_training_table(tmp_path / "unused.csv", group_column="label")
