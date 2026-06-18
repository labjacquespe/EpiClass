"""Tests for prediction-file naming and discovery helpers.

``build_prediction_tag`` builds the provenance suffix appended to default prediction
filenames; ``resolve_split_prediction_csvs`` discovers one prediction CSV per CV fold,
tolerating the tagged names and picking the newest when a fold was retrained.
"""
# pylint: disable=missing-function-docstring
import os
from pathlib import Path

from epiclass.core.prediction_files import (
    build_prediction_tag,
    experiment_id_from_checkpoint,
    resolve_split_prediction_csvs,
)


class TestBuildPredictionTag:
    """Provenance suffix from comet id + checkpoint stem."""

    def test_both_parts(self):
        tag = build_prediction_tag("abc123", "/run/split0/epoch=1-step=57.ckpt")
        assert tag == "abc123_epoch=1-step=57"

    def test_checkpoint_only(self):
        assert (
            build_prediction_tag(None, "logs/epoch=2-step=99.ckpt") == "epoch=2-step=99"
        )

    def test_experiment_only(self):
        assert build_prediction_tag("abc123", None) == "abc123"

    def test_empty(self):
        assert build_prediction_tag(None, None) == ""

    def test_accepts_path_object(self):
        tag = build_prediction_tag(None, Path("a/b/last.ckpt"))
        assert tag == "last"


class TestExperimentIdFromCheckpoint:
    """Recover the training comet id from the checkpoint's grandparent directory."""

    def test_comet_structure(self):
        ckpt = "/run/split0/epiclass/35d1e5aed6bc4b589ccb23325d75201f/checkpoints/epoch=1-step=57.ckpt"
        assert experiment_id_from_checkpoint(ckpt) == "35d1e5aed6bc4b589ccb23325d75201f"

    def test_csvlogger_structure_returns_none(self):
        # general_training: logs/version_N/checkpoints/... carries no comet id.
        ckpt = "/run/fold_0/logs/version_0/checkpoints/epoch=1-step=57.ckpt"
        assert experiment_id_from_checkpoint(ckpt) is None

    def test_non_checkpoint_dir_returns_none(self):
        assert experiment_id_from_checkpoint("/some/where/model.ckpt") is None

    def test_none_input(self):
        assert experiment_id_from_checkpoint(None) is None


def _touch(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ID,Predicted class\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class TestResolveSplitPredictionCsvs:
    """One prediction CSV per fold, newest tagged file winning."""

    def test_one_per_split_legacy_name(self, tmp_path: Path):
        for i in range(3):
            _touch(tmp_path / f"split{i}" / "validation_prediction.csv")
        resolved = resolve_split_prediction_csvs(tmp_path, "validation")
        assert list(resolved) == ["split0", "split1", "split2"]

    def test_tagged_name(self, tmp_path: Path):
        _touch(tmp_path / "split0" / "validation_prediction_abc_epoch=1-step=5.csv")
        resolved = resolve_split_prediction_csvs(tmp_path, "validation")
        assert set(resolved) == {"split0"}

    def test_newest_wins_on_retrain(self, tmp_path: Path):
        old = _touch(
            tmp_path / "split0" / "validation_prediction_old.csv", mtime=1_000_000
        )
        new = _touch(
            tmp_path / "split0" / "validation_prediction_new.csv", mtime=2_000_000
        )
        resolved = resolve_split_prediction_csvs(tmp_path, "validation")
        assert resolved["split0"] == new
        assert resolved["split0"] != old

    def test_fold_dir_naming(self, tmp_path: Path):
        _touch(tmp_path / "fold_0" / "validation_prediction.csv")
        _touch(tmp_path / "fold_1" / "validation_prediction.csv")
        resolved = resolve_split_prediction_csvs(tmp_path, "validation")
        assert list(resolved) == ["fold_0", "fold_1"]

    def test_set_name_filter(self, tmp_path: Path):
        _touch(tmp_path / "split0" / "validation_prediction.csv")
        _touch(tmp_path / "split0" / "test_prediction.csv")
        assert set(resolve_split_prediction_csvs(tmp_path, "test")) == {"split0"}

    def test_no_matches(self, tmp_path: Path):
        (tmp_path / "split0").mkdir()
        assert not resolve_split_prediction_csvs(tmp_path, "validation")
