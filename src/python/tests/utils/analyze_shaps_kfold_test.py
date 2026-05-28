"""Tests for analyze_shaps_kfold helpers."""
import json
from pathlib import Path

from epiclass.utils.shap.analyze_shaps_kfold import _persist_important_features


def test_persist_overwrite_replaces_existing_file(tmp_path: Path) -> None:
    """overwrite=True drops any prior content and writes only new entries."""
    json_path = tmp_path / "important_features.json"
    json_path.write_text(json.dumps({"old_class": {"80": [1, 2, 3]}}))

    _persist_important_features(
        json_path=json_path,
        important_features={"new_class": {80: [9, 9]}},
        overwrite=True,
    )

    assert json.loads(json_path.read_text()) == {"new_class": {"80": [9, 9]}}


def test_persist_no_overwrite_merges_with_existing(tmp_path: Path) -> None:
    """overwrite=False keeps skipped classes and lets new entries replace matching ones."""
    json_path = tmp_path / "important_features.json"
    json_path.write_text(
        json.dumps({"keep_me": {"80": [1, 2]}, "replace_me": {"80": [3, 4]}})
    )

    _persist_important_features(
        json_path=json_path,
        important_features={"replace_me": {80: [99]}, "new_class": {80: [42]}},
        overwrite=False,
    )

    assert json.loads(json_path.read_text()) == {
        "keep_me": {"80": [1, 2]},
        "replace_me": {"80": [99]},
        "new_class": {"80": [42]},
    }


def test_persist_no_overwrite_no_existing_file_writes_fresh(tmp_path: Path) -> None:
    """overwrite=False with no prior file just writes the new entries."""
    json_path = tmp_path / "important_features.json"

    _persist_important_features(
        json_path=json_path,
        important_features={"only_class": {80: [1, 2]}},
        overwrite=False,
    )

    assert json.loads(json_path.read_text()) == {"only_class": {"80": [1, 2]}}
