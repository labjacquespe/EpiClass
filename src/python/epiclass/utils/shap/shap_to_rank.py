"""Script to convert SHAP values files to rank values."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.utils.shap.shap_utils import extract_shap_values_and_info


def parse_arguments() -> argparse.Namespace:
    """Define CLI argument parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "parent_folder",
        type=DirectoryChecker(),
        help="Folder parent of each training split. (e.g. split0)",
    )
    return parser.parse_args()


def process_split(
    folder: Path,
) -> Tuple[List[np.ndarray], List[str], List[Tuple[str, str]]]:
    """Process a single split folder, extracting SHAP values and associated metadata.

    Converts the SHAP values to ranks (for each sample).

    Args:
        folder (Path): Path to the split folder containing SHAP values.

    Returns:
        split_ranks (List[np.ndarray]): List of SHAP rank matrices for each class.
        eval_signal_ids (List[str]): List of evaluation signal IDs.
        classes (List[Tuple[str, str]]): List of (class_idx, class_name) tuples.
    """
    split_name = folder.parent.name
    print(f"Processing {split_name}")
    shap_matrices, eval_signal_ids, classes = extract_shap_values_and_info(
        folder, verbose=False
    )

    split_ranks = []
    for shap_matrix in shap_matrices:
        ranks = np.argsort(-np.abs(shap_matrix), axis=1).argsort(axis=1)
        split_ranks.append(ranks.astype(np.int32))
        del shap_matrix

    del shap_matrices
    return split_ranks, eval_signal_ids, classes


def main(parent_folder: Path | None = None):  # type: ignore
    """Main"""
    if parent_folder is None:
        cli = parse_arguments()
        parent_folder: Path = cli.parent_folder
    print(f"Collecting SHAP values from: {parent_folder}")

    all_classes = None
    all_signal_ids: List = []
    concat_ranks: List = []

    for folder in sorted(parent_folder.glob("split*/shap")):
        if not folder.is_dir():
            continue

        split_ranks, eval_signal_ids, classes = process_split(folder)

        if all_classes is None:
            all_classes = classes
            concat_ranks = [[] for _ in range(len(classes))]
        elif not np.array_equal(all_classes, classes):
            raise ValueError("Classes differ between splits")

        all_signal_ids.extend(eval_signal_ids)

        for class_idx, rank_matrix in enumerate(split_ranks):
            concat_ranks[class_idx].append(rank_matrix)

        del split_ranks  # Free up memory after processing each split

    # Concatenate rank matrices for each class
    for class_idx in range(len(all_classes)):  # type: ignore
        concat_ranks[class_idx] = np.concatenate(concat_ranks[class_idx], axis=0)

    # Sanity check signal IDs
    if len(all_signal_ids) != len(set(all_signal_ids)):
        raise ValueError("Some evaluation signal IDs are duplicated between splits")

    # Save new combined archive
    output_folder = parent_folder / "shap_ranks"
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / "all_shap_abs_ranks.npz"

    np.savez_compressed(
        file=output_file,
        ranks=np.array(concat_ranks),
        signal_ids=np.array(all_signal_ids),
        classes=np.array(all_classes),
    )


if __name__ == "__main__":
    main()
