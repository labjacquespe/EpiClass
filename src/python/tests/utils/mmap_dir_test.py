"""Tests for epiclass.utils.mmap_dir.resolve_mmap_dir.

Pins the three-way priority (explicit --mmap_dir > $SLURM_TMPDIR/mmap_cache >
a fresh temp dir) and the cleanup contract: only the temp-dir branch is ever
removed by this code, and it is removed even when the caller's block raises.
"""
from pathlib import Path

import pytest

from epiclass.utils.mmap_dir import resolve_mmap_dir


class TestResolveMmapDir:
    """resolve_mmap_dir() priority and cleanup behavior."""

    def test_explicit_dir_used_as_is(self, tmp_path: Path, monkeypatch):
        """An explicit --mmap_dir wins even when $SLURM_TMPDIR is also set."""
        monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "slurm"))
        explicit_dir = tmp_path / "explicit"

        with resolve_mmap_dir(explicit_dir) as mmap_dir:
            assert mmap_dir == explicit_dir

    def test_explicit_dir_not_cleaned_up(self, tmp_path: Path):
        """Explicit dirs are the caller's responsibility, not deleted on exit."""
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        (explicit_dir / "cache.npy").touch()

        with resolve_mmap_dir(explicit_dir):
            pass

        assert explicit_dir.exists()
        assert (explicit_dir / "cache.npy").exists()

    def test_slurm_tmpdir_used_when_no_explicit_dir(self, tmp_path: Path, monkeypatch):
        """$SLURM_TMPDIR/mmap_cache is used when set and no --mmap_dir given."""
        slurm_dir = tmp_path / "slurm_scratch"
        monkeypatch.setenv("SLURM_TMPDIR", str(slurm_dir))

        with resolve_mmap_dir(None) as mmap_dir:
            assert mmap_dir == slurm_dir / "mmap_cache"

    def test_slurm_tmpdir_not_cleaned_up(self, tmp_path: Path, monkeypatch):
        """SLURM owns cleanup of its own scratch space, not this code."""
        slurm_dir = tmp_path / "slurm_scratch"
        monkeypatch.setenv("SLURM_TMPDIR", str(slurm_dir))

        with resolve_mmap_dir(None) as mmap_dir:
            mmap_dir.mkdir(parents=True)
            (mmap_dir / "cache.npy").touch()

        assert (slurm_dir / "mmap_cache" / "cache.npy").exists()

    def test_falls_back_to_temp_dir_when_neither_set(self, monkeypatch):
        """No --mmap_dir and no $SLURM_TMPDIR: a fresh temp dir is created."""
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)

        with resolve_mmap_dir(None) as mmap_dir:
            assert mmap_dir.is_dir()
            assert mmap_dir.name.startswith("epiclass_mmap_")
            leftover = mmap_dir

        assert not leftover.exists()

    def test_temp_dir_cleaned_up_even_on_exception(self, monkeypatch):
        """Cleanup runs in a finally, so it happens even if the block raises."""
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)

        leftover = None
        with pytest.raises(ValueError):
            with resolve_mmap_dir(None) as mmap_dir:
                leftover = mmap_dir
                raise ValueError("boom")

        assert leftover is not None
        assert not leftover.exists()

    def test_empty_slurm_tmpdir_falls_back_to_temp_dir(self, monkeypatch):
        """An empty-string $SLURM_TMPDIR (unset-but-exported) is treated as unset."""
        monkeypatch.setenv("SLURM_TMPDIR", "")

        with resolve_mmap_dir(None) as mmap_dir:
            assert mmap_dir.name.startswith("epiclass_mmap_")
            leftover = mmap_dir

        assert not leftover.exists()
