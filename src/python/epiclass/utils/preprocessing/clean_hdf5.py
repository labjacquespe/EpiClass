"""
Module: clean_hdf5

This module provides functionality for cleaning and processing genomic data stored in HDF5 files. Specifically, it allows for applying blacklist filters onto HDF5 datasets to modify certain genomic intervals. It reads BED (Browser Extensible Data) files containing these intervals, preprocesses them, determines the positions that are subject to treatment based on the blacklist, and applies changes to HDF5 datasets.

The module primarily consists of the following functions:

    load_bed(path: Path | str) -> BEDIntervals:
    Loads the contents of a BED file.

    preprocess_bed(bed: BEDIntervals) -> Dict[str, BEDIntervals]:
    Preprocesses the BED intervals.

    get_positions_to_treat(blacklist_chrom_intervals: Dict[str, BEDIntervals], bin_resolution: int) -> Dict[str, List[int]]:
    Generates the positions to treat based on the bin resolution and blacklist intervals.

    process_file(og_hdf5_path: Path, positions_to_treat: Dict[str, List[int]], output_dir: Path):
    Cleans one HDF5 file and saves a copy with blacklisted regions set to zero. This function assumes the positions to treat contain chromosome vector indices for the right resolution.

main() -> None:
The main driver function that organizes the entire process of reading, filtering and writing the HDF5 files.

This module also defines the following type aliases:

    BEDInterval = Tuple[str, int, int]: Represents a single genomic interval from a BED file.
    BEDIntervals = List[BEDInterval]: Represents a list of genomic intervals from a BED file.

Dependencies:
This module requires the h5py library for HDF5 file operations, and the epiclass package for argument parsing and directory checking utilities.

Usage:
Invoke the main function to process a list of HDF5 files according to command-line arguments:

    hdf5_list: A file containing a list of HDF5 filenames.
    bed_filter: A path to a BED file containing the blacklist positions.
    output_dir: The directory where the modified HDF5 files will be saved.
    n_jobs (optional): The number of parallel jobs to run.

Example:
$ python clean_hdf5.py hdf5_list.txt blacklist.bed output_directory -n 4

Author:
Joanny Raby
"""
from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker

BEDInterval = Tuple[str, int, int]
BEDIntervals = List[BEDInterval]


def print_h5py_structure(file: str) -> None:
    """Prints the structure of an h5py file.

    Args:
        file: The path to the h5py file.
    """

    def print_structure(name: str, obj: h5py.Group) -> None:
        """Recursive function to print the structure of the h5py file."""
        print(name, type(obj))
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                print_structure(name + "/" + key, obj[key])  # type: ignore

    with h5py.File(file, "r") as f:
        f.visititems(print_structure)


def load_bed(path: Path | str) -> BEDIntervals:
    """Loads the contents of a BED file.

    Args:
        path: The path to the BED file.

    Returns:
        A list of tuples representing the BED intervals (chromosome, start, end).
    """
    data = []
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            parts = line.strip().split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            data.append((chrom, start, end))
    return data


def preprocess_bed(bed: BEDIntervals) -> Dict[str, BEDIntervals]:
    """Preprocesses the BED intervals.

    Args:
        bed: A list of tuples representing the BED intervals (chromosome, start, end).

    Returns:
        A dictionary mapping chromosomes to sorted intervals.
    """
    chrom_intervals = {}

    for interval in bed:
        chrom = interval[0]

        if chrom not in chrom_intervals:
            chrom_intervals[chrom] = []

        chrom_intervals[chrom].append(interval)

    for intervals in chrom_intervals.values():
        intervals.sort(key=lambda interval: interval[1])

    return chrom_intervals


def get_positions_to_treat(
    blacklist_chrom_intervals: Dict[str, BEDIntervals], bin_resolution: int
) -> Dict[str, List[int]]:
    """Generate the positions to treat based on the bin resolution and blacklist intervals.

    Args:
        blacklist_chrom_intervals: A dictionary mapping chromosomes to sorted intervals.
        bin_resolution: The bin resolution.

    Returns:
        A dictionary mapping chromosomes to a list of positions to be treated.
    """
    positions = defaultdict(list)

    for chr_name, intervals in blacklist_chrom_intervals.items():
        for _, start, end in intervals:
            # convert from base pairs to data resolution
            # adjust end_index if `end` is exactly on a bin boundary
            start_index = start // bin_resolution
            end_index = (end - 1) // bin_resolution
            positions[chr_name].extend(range(start_index, end_index + 1))
        positions[chr_name] = list(set(positions[chr_name]))
    return positions


def check_file(
    og_hdf5_path: Path,
    positions_to_treat: Dict[str, List[int]],
) -> bool:
    """Checks one hdf5 file to ensure all designated positions are zero.

    Supposes the positions to treat contain chromosome vector indices for the right resolution.

    Returns: True if all positions are zero, False otherwise.
    """
    with h5py.File(og_hdf5_path, "r") as file:
        header = list(file.keys())[0]
        hdf5_data: h5py.Group = file[header]  # type: ignore
        for chrom, dataset in hdf5_data.items():
            if chrom in positions_to_treat:
                for position in positions_to_treat[chrom]:
                    if not np.isclose(dataset[position], 0):
                        return False

        file.close()
    return True


def process_file(
    og_hdf5_path: Path,
    positions_to_treat: Dict[str, List[int]],
    output_dir: Path,
):
    """Clean one hdf5 file. Save copy with regions that touch blacklisted regions to 0.

    Supposes the positions to treat contain chromosome vector indices for the right resolution.
    """
    hdf5_path = output_dir / (og_hdf5_path.stem + "_0blklst.hdf5")

    if hdf5_path.is_file():
        warnings.warn(f"{hdf5_path} already exists. Skipping.")
        return

    shutil.copy(og_hdf5_path, hdf5_path)

    with h5py.File(hdf5_path, "r+") as file:
        header = list(file.keys())[0]
        hdf5_data: h5py.Group = file[header]  # type: ignore
        for chrom, dataset in hdf5_data.items():
            if chrom in positions_to_treat:
                for position in positions_to_treat[chrom]:
                    dataset[position] = 0

        file.close()


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "bed_filter",
        type=Path,
        help="Path of the bed to of position to put to 0, like a blacklist.",
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to create the new hdf5 files.",
    )
    arg_parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel.",
    )
    arg_parser.add_argument(
        "-c",
        "--check_only",
        action="store_true",
        help="Only check if hdf5 files are already treated, without processing.",
    )
    return arg_parser.parse_args()


def main() -> None:
    """Main function."""
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    output_dir = cli.output_dir
    blacklist_path = cli.bed_filter
    n_cores = cli.n_jobs
    check_only = cli.check_only

    if n_cores < 1:
        raise ValueError("n_jobs must be >= 1")

    with open(hdf5_list_path, "r", encoding="utf8") as f:
        hdf5_files = [Path(line.strip()) for line in f if line.strip()]

    blacklist_bed = load_bed(blacklist_path)
    blacklist_chrom_intervals = preprocess_bed(blacklist_bed)
    positions_to_treat = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []

        for file in hdf5_files:
            with h5py.File(file, "r") as hdf5_file:
                bin_resolution: int = hdf5_file.attrs["bin"][0]  # type: ignore # pylint: disable=unsubscriptable-object

                # If the positions to treat for this bin resolution have not been generated yet, generate them
                try:
                    pos_to_treat = positions_to_treat[bin_resolution]
                except KeyError:
                    pos_to_treat = get_positions_to_treat(
                        blacklist_chrom_intervals, bin_resolution
                    )
                    positions_to_treat[bin_resolution] = pos_to_treat

                if check_only:
                    futures.append(executor.submit(check_file, file, pos_to_treat))
                else:
                    futures.append(
                        executor.submit(
                            process_file,
                            file,
                            pos_to_treat,
                            output_dir,
                        )
                    )

            hdf5_file.close()

        # Wait for all futures to complete
        # Any exceptions will be re-raised when calling result
        if check_only:
            for future, file in zip(futures, hdf5_files):
                if not future.result():
                    print(f"{file} has not been treated.")
                else:
                    print(f"{file} has been treated.")
        else:
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()


if __name__ == "__main__":
    main()
