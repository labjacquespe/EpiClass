"""Information about training runs, read from their output folders and log files.

Covers the folder naming of k-fold runs (task names), and values read from the SLURM
logs: model input sizes, job IDs and comet-ml experiment keys.
"""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def extract_data_from_files(
    parent_folder: Path,
    file_pattern: str,
    search_line: str,
    extract_pattern: str,
    type_cast: type = int,
    unique: bool = True,
) -> Dict[str, Set[Any]]:
    """
    Extracts data from files matching a specific pattern within each directory of a parent folder.

    Args:
        parent_folder (Path): The directory containing subdirectories to search.
        file_pattern (str): Glob pattern for files to search within each subdirectory.
        search_line (str): Line identifier to search for data extraction.
        extract_pattern (str): Regex pattern to extract the desired data from the identified line.
        type_cast (type): Type to cast the extracted data to.

    Returns:
        Dict[str, Set[type]]: A dictionary with subdirectory names as keys and sets of extracted data as values.
    """
    data = defaultdict(set)
    for folder in parent_folder.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.rglob(file_pattern):
            with open(file, "r", encoding="utf8") as f:
                # print(f"Reading {file.name}")
                lines = [l.rstrip() for l in f if search_line in l]

            if not lines:
                # print(f"Skipping {file.name}, no relevant data found.")
                continue
            if len(lines) > 1 and unique:
                if len(set(lines)) == 1:
                    pass
                else:
                    raise ValueError(
                        f"Incorrect file reading, captured more than one unique line in {file.name}: {lines}"
                    )
            matches = [re.match(pattern=extract_pattern, string=line) for line in lines]
            try:
                extracted_data = [type_cast(match.group(1)) for match in matches]  # type: ignore
                # print(f"Extracted data: {extracted_data}")
            except AttributeError as err:
                raise ValueError(
                    f"Could not extract data from {file.name} using pattern {extract_pattern}."
                ) from err
            data[folder.name].update(extracted_data)
    return dict(data)


def extract_input_sizes_from_output_files(parent_folder: Path) -> Dict[str, Set[int]]:
    """Extracts model input sizes from output (.o) files."""
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.o",
        search_line="(1): Linear",
        extract_pattern=r".*in_features=(\d+).*",
        type_cast=int,
    )


def extract_node_jobs_from_error_files(parent_folder: Path) -> Dict[str, Set[int]]:
    """Extracts SLURM job IDs from error (.e) files."""
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.e",
        search_line="SLURM_JOB_ID",
        extract_pattern=r".*SLURM_JOB_ID\s+: (\d+)$",
        type_cast=int,
    )


def extract_experiment_keys_from_output_files(parent_folder: Path) -> Dict[str, Set[str]]:
    """Extracts experiment keys from output (.o) files."""
    search_line = "The current experiment key is"
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.o",
        search_line=search_line,
        extract_pattern=search_line + r" (\w{32})$",
        type_cast=str,
        unique=False,
    )


def extract_experiment_info(line: str, verbose: bool = False) -> Tuple[str, str] | None:
    """Extract split and experiment key from a checkpoint path line.

    Line should contain a path ending with .../<split>/<project>/<exp_key>/checkpoints/...
    where <split> is either a CV split (e.g. 'split0') or a training-mode name
    for no-validation runs (e.g. 'complete_no_valid_oversample').
    """
    if "checkpoints" not in line:
        return None

    parts = line.strip().split("/")
    try:
        i = parts.index("checkpoints")
    except ValueError:
        return None
    if i < 3:
        return None

    info = (parts[i - 3], parts[i - 1])
    if verbose:
        print(f"Extracted info from line: {info}")

    return info


def process_log_file(file_path: Path, verbose: bool = False) -> Set[Tuple[str, str]]:
    """Process a single log file and extract experiment information."""
    experiment_info = set()
    if verbose:
        print(f"Processing log file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                if result := extract_experiment_info(line, verbose=verbose):
                    experiment_info.add(result)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error processing {file_path}: {e}")

    if verbose:
        print(
            f"Extracted {len(experiment_info)} experiment info entries from {file_path}"
        )

    return experiment_info


def collect_exp_keys(
    folder: Path,
    exclusion_pattern: str = "shap|predict",
    verbose: bool = False,
) -> Dict[str, Set[Tuple[str, str]]]:
    """Collect experiment keys from log files (.o files), from a given folder.

    Requires folder to be parent of *1l_3000n folder.
    """
    experiments_keys = defaultdict(set)

    log_files = list(folder.glob("*1l_3000n/**/*.o"))
    if verbose:
        print(f"Found {len(list(log_files))} log files in {folder}")
    for file in log_files:
        if re.search(exclusion_pattern, str(file)):
            if verbose:
                print(f"Excluding file {file} due to pattern match")
            continue
        if experiment_info := process_log_file(file, verbose=verbose):
            experiments_keys[file.parent].update(experiment_info)

    if verbose:
        print(f"Collected experiment keys for {len(experiments_keys)} folders")
    return experiments_keys


def parse_run_dir(relpath: Path, k: int = 10) -> Tuple[str, List[str]]:
    """Split a k-fold run folder into its task category and other name parts.

    Args:
        relpath: Run folder, relative to its feature set folder.
            e.g. "assay_epiclass_1l_3000n/11c/10fold-oversampling"
        k: Number of folds.

    Returns:
        (category, name_parts), e.g. ("assay_epiclass", ["11c"])
    """
    category = relpath.parts[0].replace("_1l_3000n", "")
    name_parts = [
        part
        for part in relpath.parts[1:]
        if part not in [f"{k}fold", f"{k}fold-oversampling"]
    ]
    return category, name_parts


def task_name_from_run_dir(relpath: Path, k: int = 10) -> str:
    """Return the task name of a k-fold run folder, e.g. "assay_epiclass_11c".

    See parse_run_dir. Raises a ValueError if the folder has more than one other name part.
    """
    category, name_parts = parse_run_dir(relpath, k)
    if len(name_parts) > 1:
        raise ValueError(f"Too many parts in the name: {name_parts}. Path: {relpath}")
    return "_".join([category, *name_parts])


def get_split_exp_keys(
    feature_set_dir: Path, task_name: str, k: int = 10
) -> Dict[str, str]:
    """Return {split: comet-ml experiment key} of the k-fold oversampled run of a task.

    Args:
        feature_set_dir: Feature set folder, parent of the task folders.
        task_name: Task name, as given by task_name_from_run_dir.
        k: Number of folds.

    Raises:
        ValueError: If any of the k splits does not have exactly one experiment key.
    """
    split_keys = defaultdict(set)
    for run_dir, exp_info in collect_exp_keys(feature_set_dir).items():
        relpath = Path(run_dir).relative_to(feature_set_dir)
        if relpath.name != f"{k}fold-oversampling":
            continue
        if task_name_from_run_dir(relpath, k) != task_name:
            continue
        for split_name, exp_key in exp_info:
            split_keys[split_name].add(exp_key)

    if len(split_keys) != k or any(len(keys) != 1 for keys in split_keys.values()):
        raise ValueError(f"{feature_set_dir.name} {task_name}: not one key per split.")
    return {split_name: keys.pop() for split_name, keys in split_keys.items()}
