"""
This module provides functionalities to copy HDF5 files to a new directory, convert their datasets to float32 data type, and repack them to reduce their sizes.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.loaders.hdf5_loader import Hdf5Loader

# Setting up logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to write the new hdf5 files.",
    )
    return arg_parser.parse_args()


def copy_hdf5_file(file_path: Path, logdir: Path) -> Path | None:
    """
    Copies an HDF5 file to a new location, appending "_float32.hdf5" to the filename.
    """
    new_hdf5_path = logdir / (file_path.stem + "_float32.hdf5")

    if new_hdf5_path.is_file():
        logging.warning("%s already exists. Skipping.", new_hdf5_path)
        return None

    shutil.copy(file_path, new_hdf5_path)

    return new_hdf5_path


def cast_datasets_to_float32(file_path: Path) -> bool:
    """
    Casts all the datasets in an HDF5 file to float32 data type.
    """
    max_casting_error = 1e-5

    modified = False
    with h5py.File(file_path, "r+") as f:
        for _, group in f.items():
            for dataset_name, dataset in list(group.items()):
                # Cast the dataset to float32, remove the old dataset and save the new one
                if dataset.dtype == np.float64:
                    modified = True
                    attrs = dict(dataset.attrs.items())

                    og_dataset = dataset[:]
                    casted_dataset = dataset.astype(np.float32)[:]

                    # Verify the difference between the original dataset and the casted dataset
                    diff = np.abs(casted_dataset - og_dataset)
                    max_diff = np.max(diff)
                    if max_diff > max_casting_error:
                        logging.warning(
                            "Recasting max(diff)=%.5f > %.5f: %s",
                            max_diff,
                            max_casting_error,
                            file_path,
                        )
                    del group[dataset_name]

                    group.create_dataset(
                        dataset_name, data=casted_dataset, dtype=np.float32
                    )
                    group[dataset_name].attrs.update(attrs)

    return modified


def repack_hdf5_file(file_path: Path) -> None:
    """
    Repacks an HDF5 file to reduce its size. Uses the h5repack command line tool.
    """
    tmp_path = str(file_path)
    tmp_path = tmp_path + "_repacked.hdf5"
    try:
        subprocess.run(["h5repack", str(file_path), tmp_path], check=True)
        shutil.move(tmp_path, str(file_path))
    except FileNotFoundError:
        logging.error("'h5repack' command not found.")


def process_file(hdf5_file: Path, logdir: Path) -> None:
    """
    Processes an HDF5 file by copying it to a new location, casting its datasets to float32 data type, and repacking it.

    The function first attempts to copy the input file to a new location by appending "_float32.hdf5" to the filename.
    If the new file already exists, the function logs a warning and returns.

    If the new file is successfully created, the function casts all the datasets in the file to float32 data type,
    and logs any big difference between the original and casted datasets. The function then repacks the file to reduce its size.

    If any error occurs during the process, the function logs the error message and traceback, and skips the current file.

    Args:
        hdf5_file (Path): The absolute path to the input HDF5 file.
        logdir (Path): The directory where the new file will be created.

    Returns:
        None
    """
    try:
        new_filepath = copy_hdf5_file(hdf5_file, logdir)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Error: %s. Skipping file %s\n%s", e, hdf5_file, traceback.format_exc()
        )
        return

    if new_filepath:
        modified = cast_datasets_to_float32(new_filepath)
        if modified:
            logging.info(
                "Casting and verification successful. Repacking file %s", new_filepath
            )
            repack_hdf5_file(new_filepath)
        else:
            logging.info("File already existing or no casting needed. Skipping.")
            new_filepath.unlink(missing_ok=True)


def main():
    """
    Main function that parses command-line arguments and performs the operations to copy and cast HDF5 files.
    """
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    logdir = cli.output_dir.resolve()
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))

    if shutil.which("h5repack") is None:
        raise FileNotFoundError("'h5repack' command not found.")

    hdf5_files = list(Hdf5Loader.read_list(hdf5_list_path, adapt=True).values())

    with ThreadPoolExecutor(max_workers) as executor:
        executor.map(process_file, hdf5_files, [logdir] * len(hdf5_files))


if __name__ == "__main__":
    main()
