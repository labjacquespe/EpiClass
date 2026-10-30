"""Regenerate ``fixtures/run_info/``: the paper's training run tree, reduced to what
``epiclass.utils.notebooks.paper.run_info`` reads.

The source is the real ``training_results/dfreeze_v2`` folder of the paper (feature set
folders, with the ``mixed/`` feature sets one level down). The fixture keeps:

- ``dfreeze_v2/``: the same folder tree, but only the SLURM log files
  (``output_job*.o`` / ``output_job*.e``), and in them only the lines run_info parses
  (experiment key, model input layer, checkpoint paths, SLURM job id). The cluster
  path prefix of the checkpoint paths is replaced by ``/CLUSTER_LOGDIR/``: run_info
  only reads the path parts next to ``checkpoints``.
- ``expected_*.tsv``: what run_info returns on the *real* tree, used by
  ``tests/utils/notebooks/paper/run_info_test.py``:
  - ``expected_task_names.tsv``: task name of every k-fold run folder (or the error).
  - ``expected_exp_keys.tsv``: ``collect_exp_keys`` result per feature set folder.
  - ``expected_input_sizes.tsv``: ``extract_input_sizes_from_output_files`` result.
  - ``expected_node_jobs.tsv``: ``extract_node_jobs_from_error_files`` result.

Before writing, the script checks that the reduced logs give the same results as the
real ones, so the fixture is lossless for run_info.

Run from ``src/python/``, then repack the fixtures (see README.md):

    python -m tests.fixtures_gen.make_run_info_fixture <path/to/training_results/dfreeze_v2>
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from epiclass.utils.notebooks.paper.run_info import (
    collect_exp_keys,
    extract_input_sizes_from_output_files,
    extract_node_jobs_from_error_files,
    parse_run_dir,
    task_name_from_run_dir,
)
from tests.epilap_test_data import FIXTURES_DIR

RUN_INFO_DIR = FIXTURES_DIR / "run_info"
TREE_NAME = "dfreeze_v2"

LOG_FILE_RE = re.compile(r"^output_job.*\.[oe]$")
KEPT_LINE_MARKERS = (
    "The current experiment key is",
    "(1): Linear",
    "/checkpoints/",
    "SLURM_JOB_ID",
)
# e.g. /lustre06/project/<id>/<user>/epilap/output/logs/<release>/...
CLUSTER_PREFIX_RE = re.compile(r"/\S*?/(?:epiclass-)?logs/")
CLUSTER_PREFIX = "/CLUSTER_LOGDIR/"


def feature_set_dirs(tree_root: Path) -> List[Path]:
    """Feature set folders: children of the tree root, and of its mixed/ folder."""
    parents = [tree_root, tree_root / "mixed"]
    return sorted(
        folder
        for parent in parents
        for folder in parent.iterdir()
        if folder.is_dir() and folder.name != "mixed"
    )


def reduce_log(text: str) -> str:
    """Keep the lines run_info parses, with the cluster path prefix replaced."""
    kept = [
        CLUSTER_PREFIX_RE.sub(CLUSTER_PREFIX, line)
        for line in text.splitlines()
        if any(marker in line for marker in KEPT_LINE_MARKERS)
    ]
    return "\n".join(kept) + "\n" if kept else ""


def copy_reduced_logs(source_root: Path, target_root: Path) -> int:
    """Copy every SLURM log of source_root, reduced, to the same place under target_root.

    Folder symlinks are recreated as relative symlinks, not followed: the real tree has
    some (e.g. hg38_100kb_all_none/<task> -> mixed/hg38_100kb_all_none/<task>), and the
    run_info readers do not all treat them the same way (os.walk follows them, rglob
    does not), so the fixture has to keep them.
    """
    n_files = 0
    for parent, dir_names, files in os.walk(source_root, followlinks=False):
        parent = Path(parent)
        if "checkpoints" in parent.parts:
            continue
        for dir_name in dir_names:
            link = parent / dir_name
            if not link.is_symlink():
                continue
            link_target = link.resolve()
            if not link_target.is_relative_to(source_root):
                raise ValueError(f"Symlink {link} points outside {source_root}.")
            target_link = target_root / link.relative_to(source_root)
            target_link.parent.mkdir(parents=True, exist_ok=True)
            target_link.symlink_to(
                os.path.relpath(
                    target_root / link_target.relative_to(source_root),
                    target_link.parent,
                ),
                target_is_directory=True,
            )
        for file_name in files:
            if not LOG_FILE_RE.match(file_name):
                continue
            target = target_root / parent.relative_to(source_root) / file_name
            target.parent.mkdir(parents=True, exist_ok=True)
            source_text = (parent / file_name).read_text(
                encoding="utf8", errors="replace"
            )
            target.write_text(reduce_log(source_text), encoding="utf8")
            n_files += 1
    return n_files


def task_names_table(tree_root: Path) -> pd.DataFrame:
    """Task name of every k-fold run folder, as SplitResultsHandler.general_split_metrics
    would see it (folder name containing "10fold", relative to its feature set folder)."""
    rows = []
    for fs_dir in feature_set_dirs(tree_root):
        for parent, _, _ in os.walk(fs_dir, followlinks=True):
            parent = Path(parent)
            if "10fold" not in parent.name or "checkpoints" in parent.parts:
                continue
            relpath = parent.relative_to(fs_dir)
            category, name_parts = parse_run_dir(relpath)
            try:
                task_name = task_name_from_run_dir(relpath)
            except ValueError:
                task_name = "ValueError"
            rows.append(
                {
                    "feature_set_dir": fs_dir.relative_to(tree_root).as_posix(),
                    "run_dir": relpath.as_posix(),
                    "category": category,
                    "name_parts": ";".join(name_parts),
                    "task_name": task_name,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["feature_set_dir", "run_dir"], ignore_index=True
    )


def exp_keys_table(tree_root: Path) -> pd.DataFrame:
    """collect_exp_keys result for every feature set folder."""
    rows = []
    for fs_dir in feature_set_dirs(tree_root):
        for run_dir, exp_info in collect_exp_keys(fs_dir).items():
            for split_name, exp_key in exp_info:
                rows.append(
                    {
                        "feature_set_dir": fs_dir.relative_to(tree_root).as_posix(),
                        "run_dir": Path(run_dir).relative_to(fs_dir).as_posix(),
                        "split": split_name,
                        "exp_key": exp_key,
                    }
                )
    return pd.DataFrame(rows).sort_values(list(rows[0]), ignore_index=True)


def per_folder_table(results: Dict[str, Dict[str, Set]], value_name: str) -> pd.DataFrame:
    """Flatten {parent: {folder: {values}}} into one row per value."""
    rows = [
        {"parent": parent, "folder": folder, value_name: value}
        for parent, folder_values in results.items()
        for folder, values in folder_values.items()
        for value in values
    ]
    return pd.DataFrame(rows).sort_values(list(rows[0]), ignore_index=True)


def log_values_tables(tree_root: Path) -> Dict[str, pd.DataFrame]:
    """Input sizes and node jobs, read with the run_info extractors (one level deep)."""
    parents = {".": tree_root, "mixed": tree_root / "mixed"}
    input_sizes = {
        name: extract_input_sizes_from_output_files(path)
        for name, path in parents.items()
    }
    node_jobs = {
        name: extract_node_jobs_from_error_files(path) for name, path in parents.items()
    }
    return {
        "input_sizes": per_folder_table(input_sizes, "input_size"),
        "node_jobs": per_folder_table(node_jobs, "slurm_job_id"),
    }


def main() -> None:
    """Rebuild fixtures/run_info/ from the given dfreeze_v2 folder."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("source", type=Path, help="training_results/dfreeze_v2 folder")
    source_root = parser.parse_args().source.resolve()

    tables = {
        "task_names": task_names_table(source_root),
        "exp_keys": exp_keys_table(source_root),
        **log_values_tables(source_root),
    }

    if RUN_INFO_DIR.exists():
        shutil.rmtree(RUN_INFO_DIR)
    target_root = RUN_INFO_DIR / TREE_NAME
    n_files = copy_reduced_logs(source_root, target_root)

    # The reduced logs must give what the real ones give (task names need no logs).
    for name, real_table in tables.items():
        if name == "task_names":
            continue
        fixture_table = (
            exp_keys_table(target_root)
            if name == "exp_keys"
            else log_values_tables(target_root)[name]
        )
        if not fixture_table.equals(real_table):
            raise AssertionError(f"Reduced logs change the {name} results.")

    for name, table in tables.items():
        table.to_csv(RUN_INFO_DIR / f"expected_{name}.tsv", sep="\t", index=False)
        print(f"expected_{name}.tsv: {len(table)} rows")
    print(f"{n_files} reduced log files under {target_root}")


if __name__ == "__main__":
    main()
