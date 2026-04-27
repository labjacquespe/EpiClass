"""
This module provides a script to convert float64 datasets in HDF5 files to float32 and repack files to reduce size,
while preserving all groups, attributes, and file structure.

Note: Due to the nature of hdf5, no filesize can be gained without repacking the file, even after casting to float32.
"""
# pylint: disable=broad-exception-caught
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
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
        nargs="?",
        default=None,
        help="Directory where to write the new hdf5 files. "
             "Required unless --overwrite is used.",
    )
    arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files in-place instead of writing to output_dir.",
    )
    return arg_parser.parse_args()
    # fmt: on


def cast_datasets_to_float32(file_path: Path) -> bool:
    """
    Casts all the datasets in an HDF5 file to float32 data type.

    The function iterates through all the groups and datasets in the file, checking if each dataset is of type float64.
    If a dataset is of type float64, it is cast to float32. The original dataset is deleted and replaced with the new one.

    Returns:
        bool: True if any datasets were modified, False otherwise.

    Raises:
        ValueError: If any finite values in the original dataset become infinite or NaN in the casted dataset, indicating an overflow during the casting process.
    """
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

                # Hard fail: finite values that became inf/nan
                overflow_mask = np.isfinite(og_arr) & ~np.isfinite(casted_arr)
                if np.any(overflow_mask):
                    raise ValueError(
                        f"Casting '{dataset_name}' in {file_path} would overflow float32 "
                        f"({np.count_nonzero(overflow_mask)} values affected). "
                        f"Example: {og_arr[overflow_mask][0]!r} → {casted_arr[overflow_mask][0]!r}"
                    )

                # Replace dataset
                del group[dataset_name]
                group.create_dataset(
                    dataset_name,
                    data=casted_arr,
                    dtype="float32",
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
        # overwrite original file with repacked version
        shutil.move(tmp_path, str(file_path))
    except FileNotFoundError as e:
        if "h5repack" in str(e):
            logging.error("'h5repack' command not found: %s", e)
        else:
            logging.error("FileNotFoundError during repacking: %s", e)


def process_file(hdf5_file: Path, outdir: Path | None, overwrite: bool = False) -> None:
    """
    Processes an HDF5 file by casting its float64 datasets to float32 and repacking.

    In normal mode, copies to outdir with a "_float32.hdf5" suffix.
    In overwrite mode, works on a temp copy next to the original and replaces it on success.

    Args:
        hdf5_file: The absolute path to the input HDF5 file.
        outdir: The output directory (used only when overwrite is False).
        overwrite: If True, replace the original file in-place.
    """
    # --- Set up the working copy ---
    if overwrite:
        # Temp file in the same directory (same filesystem for safe moves)
        fd, tmp_str = tempfile.mkstemp(suffix=".hdf5", dir=hdf5_file.parent)
        os.close(fd)
        work_path = Path(tmp_str)
    else:
        work_path = outdir / (hdf5_file.stem + "_float32.hdf5")  # type: ignore
        if work_path.is_file():
            logging.warning("%s already exists. Skipping.", work_path)
            return

    try:
        shutil.copy(hdf5_file, work_path)
    except Exception as e:
        logging.error("Error copying %s: %s\n%s", hdf5_file, e, traceback.format_exc())
        if overwrite:
            work_path.unlink(missing_ok=True)
        return

    # --- Cast and repack ---
    try:
        modified = cast_datasets_to_float32(work_path)
    except Exception as e:
        logging.error("Error casting %s: %s\n%s", hdf5_file, e, traceback.format_exc())
        work_path.unlink(missing_ok=True)
        return

    if not modified:
        logging.info("No casting needed. Skipping: %s", hdf5_file)
        work_path.unlink(missing_ok=True)
        return

    logging.info("Casting and verification successful. Repacking: %s", hdf5_file)
    repack_hdf5_file(work_path)

    # --- Finalize ---
    if overwrite:
        try:
            shutil.move(str(work_path), str(hdf5_file))
            logging.info("Overwritten original: %s", hdf5_file)
        except Exception as e:
            logging.error(
                "Error replacing original %s: %s\n%s",
                hdf5_file,
                e,
                traceback.format_exc(),
            )
            work_path.unlink(missing_ok=True)


def main():
    """
    Main function that parses command-line arguments and performs the operations to copy and cast HDF5 files.
    """
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    overwrite = cli.overwrite
    outdir = cli.output_dir

    if overwrite and outdir is not None:
        logging.warning("--overwrite is set; output_dir (%s) will be ignored.", outdir)

    if not overwrite and outdir is None:
        logging.error("output_dir is required when --overwrite is not used.")
        return 1

    if outdir is not None:
        outdir = outdir.resolve()
        if not overwrite and not outdir.is_dir():
            logging.error("Output directory does not exist: %s", outdir)
            return 1

    if shutil.which("h5repack") is None:
        logging.error("'h5repack' command not found.")
        return 1

    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))

    with open(hdf5_list_path, "r", encoding="utf8") as f:
        hdf5_files = [Path(line.strip()) for line in f if line.strip()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(
            process_file,
            hdf5_files,
            [outdir] * len(hdf5_files),
            [overwrite] * len(hdf5_files),
        )

    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except Exception:
        logging.exception("Unhandled error")
        sys.exit(1)
