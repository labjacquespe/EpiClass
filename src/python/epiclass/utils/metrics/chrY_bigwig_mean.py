"""
This module provides functionalities compute the chrY and chrX mean signal from bigwig files.
"""

# pylint: disable=invalid-name
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generator, List, Tuple

import pandas as pd
import pyBigWig

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.utils.general_utility import read_paths
from epiclass.utils.time import time_now_str


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "bigwig_list", type=Path, help="A file with bigwig filenames (as absolute paths)"
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to write the mean signal file.",
    )
    return arg_parser.parse_args()


def chunks(lst: List, n: int) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def compute_mean(file_path: Path) -> Tuple[str, int, int]:
    """Compute mean signal value in chrY and chrX

    Return
        Tuple[filename, chrY_mean, chrX_mean]
    """
    try:
        bw = pyBigWig.open(str(file_path), "r")
    except (RuntimeError, OSError) as err:
        print(f"{err}: Could not process {file_path}.", flush=True, file=sys.stderr)
        return (file_path.name, 0, 0)

    try:
        chrY_mean = bw.stats("chrY", exact=True)[0]
        chrX_mean = bw.stats("chrX", exact=True)[0]
    except RuntimeError as err:
        print(f"{err}: Could not process {file_path}.", flush=True, file=sys.stderr)
        chrY_mean = 0
        chrX_mean = 0
    finally:
        bw.close()

    return (
        file_path.name,
        chrY_mean,
        chrX_mean,
    )


def main():
    """Main. See module docstring."""
    cli = parse_arguments()

    bw_list_path = cli.bigwig_list
    logdir: Path = cli.output_dir.resolve()
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
    log_every_n_files = 1000  # Define how many bigwigs to process before logging

    bw_files = read_paths(bw_list_path)

    # Divide bw list into chunks, and process files concurrently
    all_results = []
    for idx, chunk in enumerate(chunks(bw_files, log_every_n_files)):
        with ThreadPoolExecutor(max_workers) as executor:
            chunk_result = list(executor.map(compute_mean, chunk))

        all_results.extend(chunk_result)

        # Log the results of the current chunk
        chunk_name = logdir / f"mean_chunk_{idx}.csv"
        if chunk_name.exists():
            chunk_name = logdir / f"mean_chunk_{idx}_{time_now_str()}.csv"

        pd.DataFrame(chunk_result, columns=["filename", "chrY", "chrX"]).to_csv(
            chunk_name, index=False
        )

    # Combine all results and save.
    final_name = logdir / "mean_combined.csv"
    if final_name.exists():
        final_name = logdir / f"mean_combined_{time_now_str()}.csv"
    pd.DataFrame(all_results, columns=["filename", "chrY", "chrX"]).to_csv(
        final_name, index=False
    )


if __name__ == "__main__":
    main()
