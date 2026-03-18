"""
This module provides functionalities compute a metric on given regions from bigwig files.
Uses pyBigWig.
"""
# pylint: disable=invalid-name
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import pandas as pd
import pyBigWig

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.utils.general_utility import read_paths
from epiclass.utils.time import time_now_str


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    # fmt: off
    arg_parser.add_argument(
        "bigwig_list",
        type=Path,
        help="A file with bigwig filenames (as absolute paths)"
    )
    arg_parser.add_argument(
        "bed_regions",
        type=Path,
        help="Bed of regions to consider.")
    arg_parser.add_argument(
        "metric",
        type=str,
        help="Metric to compute.",
        choices=["max", "min", "mean", "std", "coverage"],
    )
    arg_parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=4,
        help="Number of jobs to run in parallel.",
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to write the results.",
    )
    # fmt: on
    return arg_parser.parse_args()


def chunks(lst: List, n: int) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def compute_metrics(
    bw: pyBigWig.bigWigFile, regions: pd.DataFrame, metric: str
) -> List[float | None]:
    """Compute metric value in each region using pyBigWig stats.
    Returns a list of scalar metric values (or None if region is unsummarizable).
    """
    scalar_values = []
    for region in regions.itertuples(index=False):
        # pyBigWig.stats with nBins=1 (default) returns a list e.g. [0.75], or None
        stat_result = bw.stats(
            region.chr, region.start, region.end, type=metric, exact=True
        )
        if stat_result is None:
            # Region was unsummarizable (e.g., no coverage)
            scalar_values.append(None)
        else:
            # stat_result is a list like [value], extract the scalar.
            # It should always contain one element if not None and nBins=1.
            scalar_values.append(stat_result[0])
    return scalar_values


def read_regions(regions_path: Path) -> pd.DataFrame:
    """Read regions from a bed file and add a 'region_id' column."""
    regions_df = pd.read_csv(
        regions_path, sep="\t", header=None, names=["chr", "start", "end"]
    )

    regions_df["region_id"] = (
        regions_df["chr"].astype(str)
        + ":"
        + regions_df["start"].astype(str)
        + "-"
        + regions_df["end"].astype(str)
    )

    return regions_df


def compute_all_metrics(
    file_path: Path, regions: pd.DataFrame, metric: str
) -> Tuple[str, Dict[str, float | None] | None]:
    """Compute metric values in regions, returning a dictionary mapping region_id to value.

    Parameters
        file_path (Path): Path to bigwig file
        regions (pd.DataFrame): DataFrame with regions (chr, start, end)
        metric (str): Metric to compute. Must respect metrics available to pyBigWig stats.

    Return
        Tuple[filename, Dict[str, float|None]|None] :
        Tuple with filename and dictionary mapping region_id to value or None if error.
    """
    try:
        bw = pyBigWig.open(str(file_path), "r")
    except (RuntimeError, OSError) as err:
        print(f"{err}: Could not process {file_path}.", flush=True, file=sys.stderr)
        return (file_path.name, None)

    try:
        raw_values = compute_metrics(bw, regions, metric=metric)
    except RuntimeError as err:
        print(
            f"{err}: Error computing metrics for {file_path}.",
            flush=True,
            file=sys.stderr,
        )
        raw_values = None
    finally:
        bw.close()

    metric_values_for_regions = {}
    if raw_values:
        for i, region_id in enumerate(regions["region_id"]):
            metric_values_for_regions[region_id] = raw_values[i]
    else:
        metric_values_for_regions = None

    return (file_path.name, metric_values_for_regions)


def main():
    """Main. See module docstring."""
    cli = parse_arguments()

    bw_list_path: Path = cli.bigwig_list.resolve()
    bw_list_name = bw_list_path.stem

    logdir: Path = cli.output_dir.resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    # Filter names tend to have points in them
    regions_path: Path = cli.bed_regions.resolve()
    regions_name = regions_path.stem.replace(".", "_")

    if cli.n_jobs:
        max_workers = cli.n_jobs
    elif "SLURM_CPUS_PER_TASK" in os.environ:
        max_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        max_workers = 4

    metric_type: str = cli.metric

    output_filename_base = f"{metric_type}_{bw_list_name}_{regions_name}"

    log_every_n_files = 2500  # Define how many bigwigs to process before logging

    # Read lists
    regions_df = read_regions(regions_path)
    bw_files = read_paths(bw_list_path)

    # Divide bw list into chunks, and process files concurrently
    all_results = []
    # Create the partial function once, as regions and metric don't change
    compute_metrics_for_file_partial = partial(
        compute_all_metrics, regions=regions_df, metric=metric_type
    )

    print(f"Computing {metric_type} for {len(bw_files)} bigwigs...", flush=True)
    for idx, file_chunk in enumerate(chunks(bw_files, log_every_n_files)):
        with ThreadPoolExecutor(max_workers) as executor:
            chunk_results = list(
                executor.map(compute_metrics_for_file_partial, file_chunk)
            )

        all_results.extend(chunk_results)

        # Log the results of the current chunk, it's only a failsafe if job fails
        chunk_name = logdir / f"{output_filename_base}_chunk_{idx}.pkl"
        if chunk_name.exists():
            chunk_name = (
                logdir / f"{output_filename_base}_chunk_{idx}_{time_now_str()}.pkl"
            )

        with open(chunk_name, "wb") as f:
            pickle.dump(all_results, f)

    # Prepare data for DataFrame construction
    # Each item in data_for_df will be a dictionary representing a row
    print("\nConstructing final DataFrame from all results...", flush=True)
    region_ids = regions_df["region_id"].tolist()
    data_for_df = []
    for filename, region_metrics_dict in all_results:
        row_data = {"filename": filename}
        if region_metrics_dict:
            # Ensure all region_ids are present as keys, even if value is None (or NA)
            for r_id in region_ids:
                row_data[r_id] = region_metrics_dict.get(
                    r_id, pd.NA
                )  # Use pd.NA for missing
        else:
            # File processing failed, fill all region columns for this file with NA
            for r_id in region_ids:
                row_data[r_id] = pd.NA
        data_for_df.append(row_data)

    if not data_for_df:
        print("No data was processed. Exiting.", file=sys.stderr)
        return

    # Create the final DataFrame
    # Columns will be 'filename' + all region_ids
    final_df = pd.DataFrame(data_for_df)
    final_df = final_df.set_index("filename")  # Set filenames as index
    final_df = final_df.reindex(
        columns=region_ids
    )  # Ensure all region_ids are present and in same order

    output_path = logdir / f"{output_filename_base}.h5"
    print(
        f"Saving final DataFrame to {output_path}...",
        flush=True,
    )

    final_df.to_hdf(
        output_path,
        key="df",
        mode="w",
        format="fixed",
        complevel=9,
    )


if __name__ == "__main__":
    main()
