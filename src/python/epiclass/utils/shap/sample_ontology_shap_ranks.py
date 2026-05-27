"""Create graphs of shap values ranks for certain cell types VS other cell types.

Examine:
    - ranks on important features for an assay + cell type VS other assay + cell types
    - ranks on important features for a cell type (one output class) vs othe cell types
    - both above, but for features unique to the selection
"""
# pylint: disable=too-many-nested-blocks,too-many-branches
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.metadata import Metadata
from epiclass.utils.shap.subset_features_handling import (
    filter_feature_sets,
    process_all_subsamplings,
    read_all_feature_sets,
)
from epiclass.utils.time import time_now_str

CELL_TYPE = "harmonized_sample_ontology_intermediate"
ASSAY = "assay_epiclass"
TRACK = "track_type"


def parse_arguments() -> argparse.Namespace:
    """Define CLI argument parser."""
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument(
        "shap_ranks",
        type=Path,
        help="NPZ file containing combined SHAP ranks for all splits.",
    )
    parser.add_argument(
        "global_analysis_folder",
        type=DirectoryChecker(),
        help=("Folder which contains results of global k-fold SHAP analysis (important features). "
              "Children folders should be named after subsets of data (e.g. assay/track_type)")
    )
    parser.add_argument(
        "metadata",
        type=Path,
        help="A metadata JSON file.",
    )
    parser.add_argument(
        "output_folder",
        type=DirectoryChecker(),
        help="Directory where to save outputs.",
    )
    # fmt: on
    return parser.parse_args()


def compute_iqr(data: List[int]) -> float:
    """Calculate the interquartile range."""
    if len(data) > 0:  # type: ignore
        q75, q25 = np.percentile(a=data, q=[75, 25])  # type: ignore
        iqr = q75 - q25
    else:
        iqr = np.nan
    return iqr


rank_stats_type = Union[
    Dict[Tuple[str, str], Dict[int, Tuple[float, float]]],
    Dict[str, Dict[int, Tuple[float, float]]],
]


def calculate_rank_stats(
    all_subset_ranks: Dict[Tuple[str, str], Dict[int, List[int]]]
    | Dict[str, Dict[int, List[int]]]
) -> Tuple[rank_stats_type, rank_stats_type]:
    """
    Calculate statistics for ranks associated with each feature in each subset of samples.

    Args:
        all_subset_ranks (Dict[Tuple[str, str], Dict[int, List[int]]] OR Dict[str, Dict[int, List[int]]]):
            The ranks of each feature in each subset of samples. Can be either:
            - A dictionary with (assay, cell_type) tuples as keys and feature ranks as values, or
            - A dictionary with cell_type strings as keys and feature ranks as values.
            Feature ranks should be a dictionary with feature indices as keys and lists of ranks as values.

    Returns:
        Tuple[Dict[Tuple[str, str], Dict[int, Tuple[float, float]]], Dict[Tuple[str, str], Dict[int, Tuple[float, float]]]]:
            A tuple with two dictionaries:
            1. The average rank and standard deviation for each feature in each subset.
            2. The median rank and interquartile range for each feature in each subset.
    """
    avg_ranks = {
        key: {
            f: (float(np.mean(ranks)), float(np.std(ranks)))
            for f, ranks in subset_data.items()
        }
        for key, subset_data in all_subset_ranks.items()
    }
    med_ranks = {
        key: {
            f: (float(np.median(ranks)), float(compute_iqr(ranks)))
            for f, ranks in subset_data.items()
        }
        for key, subset_data in all_subset_ranks.items()
    }
    return avg_ranks, med_ranks  # type: ignore


def write_stats(
    output_file: Path,
    header: str,
    data: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]]
    | Dict[str, Dict[int, Tuple[float, float]]],
    features_idx: Iterable[int],
    include_assay: bool = True,
) -> None:
    """
    Write statistical data to a TSV file.

    This function can handle two types of data structures:
    1. Data with assay and cell type: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]]
    2. Data with only cell type: Dict[str, Dict[int, Tuple[float, float]]]

    Args:
        output_file (Path): The path to the output TSV file.
        header (str): A format string for the header of each feature column.
                      Should contain '{f}' which will be replaced by the feature index.
        data (Dict[Tuple[str, str], Dict[int, Tuple[float, float]]] OR
              Dict[str, Dict[int, Tuple[float, float]]]):
            The statistical data to write. Can be either:
            - A dictionary with (assay, cell_type) tuples as keys and feature stats as values, or
            - A dictionary with cell_type strings as keys and feature stats as values.
            Feature stats should be a dictionary with feature indices as keys and (stat1, stat2) tuples as values.
        features_idx (Iterable[int]): Feature indices to include in the output.
        include_assay (bool, optional): Whether to include the assay column in the output.
                                        Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If the data structure doesn't match the include_assay parameter.

    Example:
        write_stats(
            Path("output.tsv"),
            "Feature_{f}_Avg\tFeature_{f}_Std",
            {("AssayA", "CellType1"): {0: (1.0, 0.1), 1: (2.0, 0.2)}},
            [0, 1],
            include_assay=True
        )
    """
    # Determine if the data includes assay information
    has_assay = isinstance(next(iter(data.keys())), tuple)

    if has_assay != include_assay:
        raise ValueError("Data structure doesn't match the include_assay parameter.")

    # Create header
    if include_assay and has_assay:
        header_line = (
            "Assay\tCellType\t"
            + "\t".join(header.format(f=f) for f in features_idx)
            + "\n"
        )
    else:
        header_line = (
            "CellType\t" + "\t".join(header.format(f=f) for f in features_idx) + "\n"
        )

    # Create content lines
    content_lines = []
    for key, feature_stats in data.items():
        if has_assay:
            assay, cell_type = key
            line_start = f"{assay}\t{cell_type}\t"
        else:
            cell_type = key
            line_start = f"{cell_type}\t"

        feature_values = "\t".join(
            f"{feature_stats[feature_idx][0]:.2f}\t{feature_stats[feature_idx][1]:.2f}"
            for feature_idx in features_idx
        )
        content_lines.append(line_start + feature_values + "\n")

    # Write to file using a buffer
    with open(output_file, "w", encoding="utf8", buffering=1024 * 1024) as f:
        f.write(header_line)
        f.writelines(content_lines)


def main():
    """Main"""
    print(f"Main starting at {time_now_str()}")
    cli = parse_arguments()
    shap_ranks: Path = cli.shap_ranks
    global_analysis_folder: Path = cli.global_analysis_folder
    metadata: Metadata = Metadata(cli.metadata)
    output_folder: Path = cli.output_folder

    # Sanity check
    for path in [shap_ranks, global_analysis_folder, output_folder]:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

    # Collect important features
    print(f"{time_now_str()} - Loading important features.")
    important_features = read_all_feature_sets(global_analysis_folder)
    important_features = filter_feature_sets(important_features, minimum_count=8)

    complex_feature_subsets = process_all_subsamplings(
        global_analysis_folder, aggregate=True, minimum_count=8, verbose=False
    )

    # Load SHAP ranks+classes
    print(f"{time_now_str()} - Loading SHAP ranks file.")
    with np.load(shap_ranks, allow_pickle=True) as f:
        rank_data = dict(f.items())

    output_classes: List[str] = [pair[1] for pair in rank_data["classes"]]
    available_signal_ids: Set[str] = set(rank_data["signal_ids"])
    class_to_idx: Dict[str, int] = {
        pair[1]: int(pair[0]) for pair in rank_data["classes"]
    }

    # Filter metadata
    print(f"{time_now_str()} - Filtering metadata")
    for sid in list(metadata.signal_ids):
        if sid not in available_signal_ids:
            del metadata[sid]

    assays: List[str] = metadata.unique_classes(ASSAY)
    cell_types: List[str] = metadata.unique_classes(CELL_TYPE)

    # metadata sanity check
    if set(output_classes) != set(cell_types):
        raise ValueError(
            f"Output classes do not match cell types in metadata:\nSHAP npz:{output_classes}\nMetadata:{cell_types}"
        )
    if len(rank_data["ranks"]) != len(output_classes):
        raise ValueError(
            f"Number of classes in SHAP npz does not match number of classes in ranks:\nSHAP npz:{len(output_classes)}\nRanks:{len(rank_data['ranks'])}"
        )

    signal_id_sets = {
        assay: {cell_type: set() for cell_type in cell_types} for assay in assays
    }
    for sid, dset in metadata.items:
        assay = dset[ASSAY]
        cell_type = dset[CELL_TYPE]
        signal_id_sets[assay][cell_type].add(sid)

    # the following code won't handle rna or wgbs properly because the track type matters more there
    # get avg+stdev rank of each bin for samples in different subsets
    print(f"{time_now_str()} - Processing ranks for important features")
    for assay in assays:
        try:
            assay_features = important_features[assay]
        except KeyError as e:
            print(f"No important features found for {assay}")
            raise e  # each assay should have important features

        # Selecting a feature set, then collect ranks for each sample subset
        for ct, features_idx in assay_features.items():
            all_subset_ranks = {
                (a, c): {f: [] for f in features_idx} for a in assays for c in cell_types
            }

            # Calculate average ranks for all (assay, ct) subsets
            print(f"{time_now_str()} - Processing feature set for ({assay},{ct})")
            for output_class in output_classes:
                class_ranks: np.ndarray = rank_data["ranks"][class_to_idx[output_class]]

                for subset_assay in assays:
                    for subset_ct in cell_types:
                        subset_signal_ids = signal_id_sets[subset_assay][subset_ct]
                        samples_idx = [
                            i
                            for i, sid in enumerate(rank_data["signal_ids"])
                            if sid in subset_signal_ids
                        ]

                        for i in samples_idx:
                            for feature_idx in features_idx:
                                all_subset_ranks[(subset_assay, subset_ct)][
                                    feature_idx
                                ].append(class_ranks[i, feature_idx])

            avg_ranks, median_ranks = calculate_rank_stats(all_subset_ranks)

            # Save rank stats
            output_file = output_folder / f"{assay}_{ct}_feature_set_avg_ranks.tsv"
            write_stats(
                output_file=output_file,
                header="Feature_{f}_avg\tFeature_{f}_std",
                data=avg_ranks,
                features_idx=features_idx,
                include_assay=True,
            )
            output_file = output_folder / f"{assay}_{ct}_feature_set_median_ranks.tsv"
            write_stats(
                output_file=output_file,
                header="Feature_{f}_med\tFeature_{f}_iqr",
                data=median_ranks,
                features_idx=features_idx,
                include_assay=True,
            )
            print(
                f"{time_now_str()} - Saved ranks statistics for features from {assay} - {ct}"
            )

    # Do a similar analysis but with merge_samplings_[output_class].
    # This time compare output class vs other output classes
    for subset_name, features_idx in complex_feature_subsets.items():
        if "merge_samplings_" not in subset_name:
            continue

        output_class = subset_name.replace("merge_samplings_", "")
        print(f"{time_now_str()} - Processing feature set for '{subset_name}'")

        all_subset_ranks = {ct: {f: [] for f in features_idx} for ct in cell_types}
        for ct in cell_types:
            subset_signal_ids = set(metadata.ids_per_class(CELL_TYPE)[ct])
            samples_idx = [
                i
                for i, sid in enumerate(rank_data["signal_ids"])
                if sid in subset_signal_ids
            ]

            for i in samples_idx:
                for feature_idx in features_idx:
                    all_subset_ranks[ct][feature_idx].append(
                        rank_data["ranks"][class_to_idx[output_class]][i, feature_idx]
                    )

        avg_ranks, median_ranks = calculate_rank_stats(all_subset_ranks)

        # Save rank stats
        output_file = output_folder / f"{subset_name}_feature_set_avg_ranks.tsv"
        write_stats(
            output_file=output_file,
            header="Feature_{f}_avg\tFeature_{f}_std",
            data=avg_ranks,
            features_idx=features_idx,
            include_assay=False,
        )
        output_file = output_folder / f"{subset_name}_feature_set_median_ranks.tsv"
        write_stats(
            output_file=output_file,
            header="Feature_{f}_med\tFeature_{f}_iqr",
            data=median_ranks,
            features_idx=features_idx,
            include_assay=False,
        )

        print(
            f"{time_now_str()} - Saved ranks statistics for features from '{subset_name}'"
        )


if __name__ == "__main__":
    main()
