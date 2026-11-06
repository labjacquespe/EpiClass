"""Test functions from paper_utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from epiclass.utils.notebooks.paper.paper_utilities import (
    merge_renamed_columns,
    writable_path,
)


def test_writable_path_new_file(tmp_path: Path):
    """A file that does not exist yet can be written, str paths included."""
    path = tmp_path / "new.csv"
    assert writable_path(path) == path
    assert writable_path(str(path)) == path


def test_writable_path_refuses_existing_file(tmp_path: Path):
    """An existing file is refused, and left untouched."""
    path = tmp_path / "existing.csv"
    path.write_text("content", encoding="utf-8")
    with pytest.raises(FileExistsError):
        writable_path(path)
    assert path.read_text(encoding="utf-8") == "content"


def test_writable_path_explicit_overwrite(tmp_path: Path):
    """An existing file is only allowed when explicitly requested."""
    path = tmp_path / "existing.csv"
    path.write_text("content", encoding="utf-8")
    assert writable_path(path, overwrite=True) == path


def test_merge_renamed_columns_both_names():
    """New name values have priority, old name fills the gaps and is dropped."""
    df = pd.DataFrame({"old": [1.0, 2.0], "new": [None, 3.0]})
    merged = merge_renamed_columns(df, {"old": "new"})
    assert list(merged.columns) == ["new"]
    assert merged["new"].tolist() == [1.0, 3.0]


def test_merge_renamed_columns_only_old_name():
    """The old name is renamed when the new one is absent."""
    merged = merge_renamed_columns(pd.DataFrame({"old": [1, 2]}), {"old": "new"})
    assert list(merged.columns) == ["new"]
    assert merged["new"].tolist() == [1, 2]


def test_merge_renamed_columns_only_new_name():
    """Nothing to do when the old name is absent."""
    df = pd.DataFrame({"new": [1, 2]})
    assert merge_renamed_columns(df, {"old": "new"}).equals(df)
