"""Tests for utils/rebase_checkpoint_list.py.

The utility repairs ``best_checkpoint.list`` files after their model directory
has moved: it treats the directory holding the list as the new base, detects the
stale old base by matching path suffixes against what exists on disk, and swaps
the old base for the new one on every matching line (preserving the timestamp
tail). These tests cover that rewrite, the no-op paths, the unresolvable-path
error, backup creation, and ``--dry-run``.
"""
# pylint: disable=redefined-outer-name, missing-function-docstring
from pathlib import Path

from epiclass.utils.rebase_checkpoint_list import process_list_file

OLD_BASE = Path("/old/home/logs")
SUFFIX = Path("EpiLaP/abc123/checkpoints/epoch=1-step=57.ckpt")
SUFFIX2 = Path("EpiLaP/def456/checkpoints/epoch=2-step=99.ckpt")
TS = "2023-05-08 16:39:54.954455"


def _make_model_dir(tmp_path: Path, suffixes: list[Path]) -> list[Path]:
    """Create real ckpt files under tmp_path for each suffix; return their paths."""
    created = []
    for suffix in suffixes:
        ckpt = tmp_path / suffix
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("fake checkpoint", encoding="utf-8")
        created.append(ckpt)
    return created


def _write_list(tmp_path: Path, lines: list[str]) -> Path:
    list_file = tmp_path / "best_checkpoint.list"
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_file


def test_rewrites_stale_base_across_lines(tmp_path: Path):
    _make_model_dir(tmp_path, [SUFFIX, SUFFIX2])
    list_file = _write_list(
        tmp_path,
        [
            f"{OLD_BASE / SUFFIX} {TS}",
            f"{OLD_BASE / SUFFIX2} {TS}",
        ],
    )

    ok = process_list_file(list_file, dry_run=False, assume_yes=True, backup=False)
    assert ok

    new_lines = list_file.read_text(encoding="utf-8").splitlines()
    assert new_lines[0] == f"{tmp_path / SUFFIX} {TS}"
    assert new_lines[1] == f"{tmp_path / SUFFIX2} {TS}"
    # Rewritten paths must actually exist.
    assert (tmp_path / SUFFIX).is_file()
    assert (tmp_path / SUFFIX2).is_file()


def test_preserves_timestamp_tail_and_entry_without_timestamp(tmp_path: Path):
    _make_model_dir(tmp_path, [SUFFIX, SUFFIX2])
    list_file = _write_list(
        tmp_path,
        [
            str(OLD_BASE / SUFFIX),  # no timestamp
            f"{OLD_BASE / SUFFIX2} {TS}",
        ],
    )

    process_list_file(list_file, dry_run=False, assume_yes=True, backup=False)
    new_lines = list_file.read_text(encoding="utf-8").splitlines()
    assert new_lines[0] == str(tmp_path / SUFFIX)
    assert new_lines[1] == f"{tmp_path / SUFFIX2} {TS}"


def test_noop_when_last_entry_already_valid(tmp_path: Path):
    _make_model_dir(tmp_path, [SUFFIX])
    # Path already points at the real location -> nothing to change.
    list_file = _write_list(tmp_path, [f"{tmp_path / SUFFIX} {TS}"])
    before = list_file.read_text(encoding="utf-8")

    ok = process_list_file(list_file, dry_run=False, assume_yes=True, backup=False)
    assert ok
    assert list_file.read_text(encoding="utf-8") == before


def test_error_when_checkpoint_not_beside_list(tmp_path: Path):
    # No ckpt created on disk -> no suffix resolves -> failure, file untouched.
    list_file = _write_list(tmp_path, [f"{OLD_BASE / SUFFIX} {TS}"])
    before = list_file.read_text(encoding="utf-8")

    ok = process_list_file(list_file, dry_run=False, assume_yes=True, backup=False)
    assert not ok
    assert list_file.read_text(encoding="utf-8") == before


def test_backup_created(tmp_path: Path):
    _make_model_dir(tmp_path, [SUFFIX])
    list_file = _write_list(tmp_path, [f"{OLD_BASE / SUFFIX} {TS}"])
    original = list_file.read_text(encoding="utf-8")

    process_list_file(list_file, dry_run=False, assume_yes=True, backup=True)
    backup = list_file.with_name(list_file.name + ".bak")
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == original


def test_dry_run_writes_nothing(tmp_path: Path):
    _make_model_dir(tmp_path, [SUFFIX])
    list_file = _write_list(tmp_path, [f"{OLD_BASE / SUFFIX} {TS}"])
    before = list_file.read_text(encoding="utf-8")

    ok = process_list_file(list_file, dry_run=True, assume_yes=True, backup=True)
    assert ok
    assert list_file.read_text(encoding="utf-8") == before
    assert not list_file.with_name(list_file.name + ".bak").exists()


def test_missing_list_file(tmp_path: Path):
    ok = process_list_file(
        tmp_path / "best_checkpoint.list", dry_run=False, assume_yes=True, backup=False
    )
    assert not ok


def test_fallback_appends_surviving_checkpoint(tmp_path: Path):
    # The expected per-epoch ckpt was deleted; only last.ckpt survives in the
    # checkpoint dir. With --fallback-ckpt, a new line is appended (not a
    # rewrite) so the LAST line resolves.
    last_ckpt = SUFFIX.parent / "last.ckpt"
    _make_model_dir(tmp_path, [last_ckpt])  # only last.ckpt exists, not SUFFIX
    list_file = _write_list(tmp_path, [f"{OLD_BASE / SUFFIX} {TS}"])

    ok = process_list_file(
        list_file,
        dry_run=False,
        assume_yes=True,
        backup=False,
        fallback_name="last.ckpt",
    )
    assert ok
    new_lines = list_file.read_text(encoding="utf-8").splitlines()
    # Original stale line is preserved; a new last line points at last.ckpt.
    assert len(new_lines) == 2
    assert new_lines[0] == f"{OLD_BASE / SUFFIX} {TS}"
    assert new_lines[1].split(" ", 1)[0] == str(tmp_path / last_ckpt)
    assert (tmp_path / last_ckpt).is_file()


def test_fallback_disabled_by_default_still_errors(tmp_path: Path):
    # Only last.ckpt survives, but without fallback the missing exact ckpt is a
    # hard error and the file is left untouched.
    _make_model_dir(tmp_path, [SUFFIX.parent / "last.ckpt"])
    list_file = _write_list(tmp_path, [f"{OLD_BASE / SUFFIX} {TS}"])
    before = list_file.read_text(encoding="utf-8")

    ok = process_list_file(list_file, dry_run=False, assume_yes=True, backup=False)
    assert not ok
    assert list_file.read_text(encoding="utf-8") == before


def test_fallback_errors_when_no_survivor(tmp_path: Path):
    # Fallback requested but the named file isn't there either -> hard error.
    _make_model_dir(tmp_path, [SUFFIX.parent / "other.ckpt"])
    list_file = _write_list(tmp_path, [f"{OLD_BASE / SUFFIX} {TS}"])

    ok = process_list_file(
        list_file,
        dry_run=False,
        assume_yes=True,
        backup=False,
        fallback_name="last.ckpt",
    )
    assert not ok
