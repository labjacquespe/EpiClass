"""Read a list of HDF5 files and compress their contents into a single NPZ file."""

import argparse
from pathlib import Path

import numpy as np

from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader


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

    hdf5_loader = LazyHdf5Loader(
        chrom_file=chromsizes_path,
        normalization=False,
        mmap_dir=output_npz_path.parent / "mmap_cache",
    )
    hdf5_loader.register_hdf5s(
        data_file=hdf5_list_path,
        strict=True,
        verbose=True,
    )
    hdf5_loader.preload_all()

    # The mmap-backed array is already the (n_samples, signal_length) matrix
    # we need; np.savez_compressed will read it from disk in chunks rather
    # than forcing the whole thing into RAM at once.
    ids = list(hdf5_loader.file_paths.keys())
    signal_matrix = hdf5_loader.as_mmap()
    ids_array = np.array(ids)

    # Save the signals and ids to a compressed NPZ file
    np.savez_compressed(
        file=output_npz_path,
        signals=signal_matrix,
        ids=ids_array,
        allow_pickle=False,
    )


if __name__ == "__main__":
    main()
