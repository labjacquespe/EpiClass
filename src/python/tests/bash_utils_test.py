"""Tests for snippets in src/bash_utils/.

Focused on the validation_prediction merging block in epiatlas_training.sh,
which must keep a single header at the top and sort+dedupe the data rows
after write_pred_table started emitting an "ID" column header.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

HEADER = "ID,True class,Predicted class,classA,classB"


def _write_split(dir_path: Path, rows: list[str]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "validation_prediction.csv").write_text(
        HEADER + "\n" + "\n".join(rows) + "\n",
        encoding="utf8",
    )


def _run_new_merge(log_dir: Path) -> Path:
    """Run the current merge snippet from epiatlas_training.sh."""
    merged = log_dir / "full-10fold-validation_prediction.csv"
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        cd "{log_dir}"
        split_files=( split*/validation_prediction.csv )
        head -n 1 "${{split_files[0]}}" >"{merged.name}"
        tail -n +2 -q "${{split_files[@]}}" | sort -u >>"{merged.name}"
        """
    )
    subprocess.run(["bash", "-c", script], check=True)
    return merged


def _run_old_merge(log_dir: Path) -> Path:
    """Run the previous `cat | sort -ru` snippet for comparison."""
    merged = log_dir / "old-full-10fold-validation_prediction.csv"
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        cd "{log_dir}"
        cat split*/validation_prediction.csv | sort -ru >"{merged.name}"
        """
    )
    subprocess.run(["bash", "-c", script], check=True)
    return merged


def _populate_splits(tmp_path: Path) -> None:
    _write_split(
        tmp_path / "split0",
        ["aaa,classA,classA,0.9,0.1", "bbb,classB,classA,0.3,0.7"],
    )
    _write_split(
        tmp_path / "split1",
        # 'aaa' row duplicated across splits — should appear only once.
        ["aaa,classA,classA,0.9,0.1", "ccc,classA,classB,0.6,0.4"],
    )
    _write_split(
        tmp_path / "split2",
        ["ddd,classB,classB,0.2,0.8"],
    )


def test_merge_validation_predictions_keeps_header_and_dedupes(tmp_path: Path) -> None:
    """Merged file: header at line 1, unique sorted data, no duplicate headers."""
    _populate_splits(tmp_path)

    merged = _run_new_merge(tmp_path)

    lines = merged.read_text(encoding="utf8").splitlines()
    assert lines[0] == HEADER, "header must stay at the top after merge"
    data = lines[1:]

    assert data == sorted(set(data))
    assert HEADER not in data

    expected_ids = {"aaa", "bbb", "ccc", "ddd"}
    assert {row.split(",", 1)[0] for row in data} == expected_ids


def test_new_merge_preserves_same_data_rows_as_old(tmp_path: Path) -> None:
    """New merge yields the same unique data rows as the old `sort -ru` block.

    The old snippet left the header at the bottom (a known bug — the TODO that
    motivated adding the "ID" index_label). What we care about preserving is
    the set of unique data rows; the new snippet just lifts the header to the
    top where pandas can read it.
    """
    _populate_splits(tmp_path)

    new_lines = set(_run_new_merge(tmp_path).read_text(encoding="utf8").splitlines())
    old_lines = set(_run_old_merge(tmp_path).read_text(encoding="utf8").splitlines())

    # Both outputs contain the single header line plus the unique data rows.
    assert new_lines == old_lines
