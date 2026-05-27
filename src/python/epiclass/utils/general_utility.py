"""For lost little functions that don't fit anywhere else."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

# Signal-ID list files were historically suffixed `.md5` (the IDs were md5
# hashes); the canonical form is now `_ids.list`. Readers should accept both
# so old artifacts still load.
SIGNAL_ID_LIST_SUFFIXES = ("_ids.list", ".md5")


def write_hdf5_paths_to_file(
    signal_ids: Iterable[str], parent: str, suffix: str, filepath: str | Path
) -> List[Path]:
    """Write a list of signal IDs to a file, with a prefix and suffix.
    Creates files of the format: Path(prefix=parent)/{signal_id}_{suffix}.hdf5

    Returns a list of the paths written to the file.
    """
    files = []
    with open(filepath, "w", encoding="utf8") as f:
        for signal_id in signal_ids:
            path = Path(parent) / (f"{signal_id}_{suffix}.hdf5")
            f.write(f"{path}\n")
            files.append(path)
    return files


def write_signal_ids_to_file(
    signal_ids: Iterable[str], logdir: str | Path, name: str
) -> Path:
    """Write a list of signal IDs to a file. Return filepath."""
    filename = Path(logdir) / f"{name}_ids.list"
    with open(filename, "w", encoding="utf8") as f:
        for signal_id in signal_ids:
            f.write(f"{signal_id}\n")
    return filename


def find_signal_id_lists(folder: Path, prefix_glob: str = "*") -> List[Path]:
    """Find signal-ID list files in `folder` accepting either canonical suffix.

    `prefix_glob` is everything before the suffix (default `*`). Both
    `_ids.list` (current) and `.md5` (legacy) suffixes are searched; results
    are concatenated, new-style first.
    """
    paths: List[Path] = []
    for suffix in SIGNAL_ID_LIST_SUFFIXES:
        paths.extend(folder.glob(f"{prefix_glob}{suffix}"))
    return paths


def get_valid_filename(name: str):
    """Tranform a string into a valid filename."""
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "_", s)
    s = re.sub(r"(_{2,})", "_", s)
    s = s.replace("_-_", "-")
    if s[-1] == "_":
        s = s[:-1]
    return s


def read_paths(list_path: Path) -> List[Path]:
    """Read a list of paths from a file."""
    with open(list_path, "r", encoding="utf-8") as file_of_paths:
        paths = []
        for path in file_of_paths:
            path = Path(path.rstrip())
            paths.append(path)
    return paths
