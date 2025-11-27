"""
This module provides a script to convert float64 datasets in HDF5 files to float32 and repack files to reduce size,
while preserving all groups, attributes, and file structure.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

# Setting up logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list",
        type=Path,
        help="File containing list of HDF5 file paths (absolute paths)."
    )
    arg_parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where to write the new hdf5 files.",
    )
    return arg_parser.parse_args()
    # fmt: on


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
            if not isinstance(group, h5py.Group):
                continue

            for dataset_name, dataset in list(group.items()):
                # Cast the dataset to float32, remove the old dataset and save the new one
                if not isinstance(dataset, h5py.Dataset):
                    continue
                if dataset.dtype != np.float64:
                    continue

                # Dataset needs casting
                modified = True
                attrs = dict(dataset.attrs.items())

                og_arr = dataset[...]
                casted_arr = og_arr.astype(np.float32)

                # Verify the difference between the original dataset and the casted dataset
                diff = np.abs(casted_arr - og_arr)
                diff = diff[np.isfinite(diff)]  # ignore NaN/Inf
                max_diff = np.max(diff)

                if max_diff > max_casting_error:
                    logging.warning(
                        "Biggest casting difference '%.5f' (on %s) exceeds threshold (%.5f) for: %s",
                        max_diff,
                        dataset_name,
                        max_casting_error,
                        file_path,
                    )

                # Replace dataset
                del group[dataset_name]
                group.create_dataset(
                    dataset_name,
                    data=casted_arr,
                    dtype="float32",
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,  # improves gzip compression
                    fletcher32=True,  # improves data integrity
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
    except FileNotFoundError as e:
        if "h5repack" in str(e):
            logging.error("'h5repack' command not found: %s", e)
        else:
            logging.error("FileNotFoundError during repacking: %s", e)


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
    # First, copy the file to the new location with a modified name
    try:
        new_filepath = copy_hdf5_file(hdf5_file, logdir)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(
            "Error: %s. Skipping file %s\n%s", e, hdf5_file, traceback.format_exc()
        )
        return

    if new_filepath is None:
        logging.info("File already exists. Skipping: %s", hdf5_file)
        return

    # Cast datasets to float32
    modified = cast_datasets_to_float32(new_filepath)
    if modified:
        logging.info("Casting and verification successful. Repacking: %s", new_filepath)
        repack_hdf5_file(new_filepath)
    else:
        logging.info("No casting needed. Skipping: %s", new_filepath)
        new_filepath.unlink(missing_ok=True)


def main():
    """
    Main function that parses command-line arguments and performs the operations to copy and cast HDF5 files.
    """
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    outdir = cli.output_dir.resolve()
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))

    if shutil.which("h5repack") is None:
        raise FileNotFoundError("'h5repack' command not found.")

    with open(hdf5_list_path, "r", encoding="utf8") as f:
        hdf5_files = [Path(line.strip()) for line in f if line.strip()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, hdf5_files, [outdir] * len(hdf5_files))


if __name__ == "__main__":
    main()
