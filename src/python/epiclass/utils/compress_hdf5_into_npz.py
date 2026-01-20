"""Read a list of HDF5 files and compress their contents into a single NPZ file."""

import argparse
from pathlib import Path

import numpy as np

from epiclass.core.loaders.hdf5_loader import Hdf5Loader


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress multiple HDF5 files into a single NPZ file."
    )
    parser.add_argument(
        "--hdf5_list",
        required=True,
        type=Path,
        help="List of HDF5 files to be compressed.",
    )
    parser.add_argument(
        "--chromsizes",
        required=True,
        type=Path,
        help="Path to the chromosome sizes file, which dictate which chromosomes to include.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=False,
        type=Path,
        help="Output NPZ file name.",
    )
    return parser.parse_args()


def main():
    "main"
    args = parse_arguments()
    hdf5_list_path = args.hdf5_list
    chromsizes_path = args.chromsizes
    output_npz_path = args.output

    for path in [hdf5_list_path, chromsizes_path]:
        if not path.is_file():
            raise FileNotFoundError(f"The specified file does not exist: {path}")

    if output_npz_path is None:
        output_npz_path = hdf5_list_path.with_suffix(".npz")

    if not output_npz_path.parent.exists():
        print(f"Creating output directory: {output_npz_path.parent}")
        output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    if output_npz_path.is_file():
        print(f"The output file already exists: {output_npz_path}")
        print("Renaming the desired output file to avoid overwriting.")
        output_npz_path = output_npz_path.with_name(output_npz_path.stem + "_new.npz")
        print(f"New output file: {output_npz_path}")

    hdf5_loader = Hdf5Loader(
        chrom_file=chromsizes_path,
        normalization=False,
    )

    hdf5_loader.load_hdf5s(
        data_file=hdf5_list_path,
        strict=True,
        adapt=False,
        verbose=True,
    )

    # Stack signals into a single matrix
    signal_matrix = np.stack(list(hdf5_loader.signals.values()), axis=0).astype(
        np.float32
    )

    # Store IDs as array for consistency
    ids_array = np.array(list(hdf5_loader.signals.keys()))

    np.savez_compressed(output_npz_path, signals=signal_matrix, ids=ids_array)


if __name__ == "__main__":
    main()
