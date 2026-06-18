"""Tests for LightningDenseClassifier.restore_model.

restore_model is the entry point used by every script that consumes a
trained model (predict, compute_shaps, epiatlas_training --restore,
epiatlas_training_no_valid --restore, general_training). It reads
``<model_dir>/best_checkpoint.list``, takes the last line, splits off
the leading checkpoint path (the format is "<ckpt_path> <iso_timestamp>"),
and loads the .ckpt via Lightning. The error surface is what we want
covered: missing list file, empty list, blank entry, missing ckpt file.
"""
# pylint: disable=redefined-outer-name, missing-function-docstring
from pathlib import Path

import pytest

from epiclass.core.model_checkpoint import last_checkpoint_path
from epiclass.core.model_pytorch import LightningDenseClassifier
from tests.epilap_test_data import FIXTURES_DIR

SACCER3_DIR = FIXTURES_DIR / "saccer3"
REAL_CKPT = (
    SACCER3_DIR
    / "EpiLaP"
    / "35d1e5aed6bc4b589ccb23325d75201f"
    / "checkpoints"
    / "epoch=1-step=57.ckpt"
)


@pytest.fixture
def ckpt_path() -> Path:
    """The one real ckpt shipped with the saccer3 fixtures."""
    assert REAL_CKPT.is_file(), f"Fixture ckpt missing: {REAL_CKPT}"
    return REAL_CKPT


def _write_list(model_dir: Path, lines: list[str]) -> Path:
    """Write best_checkpoint.list with the given lines and return model_dir."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best_checkpoint.list").write_text(
        "\n".join(lines) + "\n" if lines else ""
    )
    return model_dir


class TestRestoreModelHappyPath:
    """Verify a well-formed best_checkpoint.list loads the model."""

    def test_single_entry(self, tmp_path: Path, ckpt_path: Path):
        model_dir = _write_list(tmp_path, [f"{ckpt_path} 2023-05-08 16:39:54.954455"])
        model = LightningDenseClassifier.restore_model(model_dir, verbose=False)
        assert isinstance(model, LightningDenseClassifier)

    def test_picks_last_entry(self, tmp_path: Path, ckpt_path: Path):
        # First line points at a nonexistent path; restore must read the LAST
        # line, not the first.
        bad = tmp_path / "stale" / "epoch=0-step=1.ckpt"
        model_dir = _write_list(
            tmp_path,
            [
                f"{bad} 2023-01-01 00:00:00",
                f"{ckpt_path} 2023-05-08 16:39:54.954455",
            ],
        )
        model = LightningDenseClassifier.restore_model(model_dir, verbose=False)
        assert isinstance(model, LightningDenseClassifier)

    def test_entry_without_timestamp(self, tmp_path: Path, ckpt_path: Path):
        # The split on " " is maxsplit=1: a bare path with no trailing
        # timestamp should still resolve cleanly.
        model_dir = _write_list(tmp_path, [str(ckpt_path)])
        model = LightningDenseClassifier.restore_model(model_dir, verbose=False)
        assert isinstance(model, LightningDenseClassifier)


class TestRestoreModelErrors:
    """Verify the documented error surface."""

    def test_missing_list_file(self, tmp_path: Path):
        # No best_checkpoint.list at all → open() raises FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            LightningDenseClassifier.restore_model(tmp_path, verbose=False)

    def test_empty_list_file(self, tmp_path: Path):
        model_dir = _write_list(tmp_path, [])
        with pytest.raises(FileNotFoundError, match="Empty checkpoint list"):
            LightningDenseClassifier.restore_model(model_dir, verbose=False)

    def test_blank_last_entry(self, tmp_path: Path, ckpt_path: Path):
        # Last line is empty/whitespace → the split yields "" → blank-entry
        # branch. A non-blank earlier line must NOT rescue it.
        model_dir = _write_list(
            tmp_path,
            [f"{ckpt_path} 2023-05-08 16:39:54.954455", ""],
        )
        with pytest.raises(FileNotFoundError, match="blank last entry"):
            LightningDenseClassifier.restore_model(model_dir, verbose=False)

    def test_checkpoint_path_missing_on_disk(self, tmp_path: Path):
        ghost = tmp_path / "ghost" / "epoch=9-step=99.ckpt"
        model_dir = _write_list(tmp_path, [f"{ghost} 2099-01-01 00:00:00"])
        with pytest.raises(FileNotFoundError, match="does not exist"):
            LightningDenseClassifier.restore_model(model_dir, verbose=False)


class TestLastCheckpointPath:
    """last_checkpoint_path returns the path of the last list entry, or None."""

    def test_picks_last_entry_without_disk_check(self, tmp_path: Path):
        # No checkpoint file on disk: last_checkpoint_path does NOT validate existence.
        ckpt = tmp_path / "logs" / "epoch=1-step=57.ckpt"
        model_dir = _write_list(
            tmp_path,
            [
                "/stale/epoch=0-step=1.ckpt 2023-01-01 00:00:00",
                f"{ckpt} 2023-05-08 16:39:54.954455",
            ],
        )
        assert last_checkpoint_path(model_dir) == ckpt

    def test_missing_list_returns_none(self, tmp_path: Path):
        assert last_checkpoint_path(tmp_path) is None

    def test_empty_list_returns_none(self, tmp_path: Path):
        model_dir = _write_list(tmp_path, [])
        assert last_checkpoint_path(model_dir) is None

    def test_blank_last_entry_returns_none(self, tmp_path: Path):
        model_dir = _write_list(tmp_path, ["/some/epoch=1-step=57.ckpt x", ""])
        assert last_checkpoint_path(model_dir) is None
