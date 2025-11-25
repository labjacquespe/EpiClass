"""Compute PCA for some hardcoded datasets. (dataset selection done in .sh script)"""
# pylint: disable=import-error, redefined-outer-name, use-dict-literal, too-many-lines, unused-import, unused-argument, too-many-branches
from __future__ import annotations

import argparse
import os
import warnings
from importlib import metadata
from pathlib import Path
from typing import List

import numpy as np
import skops.io as skio
from sklearn.decomposition import IncrementalPCA

from epiclass.core.loaders.hdf5_loader import Hdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = argparse.ArgumentParser(
        description="Compute Incremental PCA embeddings for hdf5 files."
    )
    arg_parser.add_argument(
        "chromsize",
        type=Path,
        help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "output",
        type=Path,
        default=None,
        help="Directory to save embeddings in. Saves in home directory if not provided.",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        help="Size of batches for incremental PCA. Default 30 000.",
        default=30000,
    )
    arg_parser.add_argument(
        "--input_list",
        type=Path,
        help="List of hdf5 files to load. Absolute path is recommended. By default, all hdf5 files in SLURM_TMPDIR or /tmp are used.",
        default=None,
    )
    # fmt: on
    return arg_parser.parse_args()


def find_rows_with_same_values(arr, atol=1e-5) -> List[int]:
    """Find rows in an array with all values close to the first value."""
    problematic_rows = []
    for idx, row in enumerate(arr):
        if np.all(np.isclose(row, row[0], atol=atol)):
            problematic_rows.append(idx)
    return problematic_rows


def main():
    """Run the main function."""
    cli = parse_arguments()

    output_dir = cli.output if cli.output is not None else Path.home()

    if cli.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Initialize HDF5 loader
    chromsize_path = cli.chromsize
    hdf5_loader = Hdf5Loader(chrom_file=chromsize_path, normalization=True)

    # Handle input file paths
    if cli.input_list is not None:
        hdf5_paths_list_path = cli.input_list
        # Count files in input list for reporting
        with open(hdf5_paths_list_path, "r", encoding="utf8") as f:
            total_files = sum(1 for _ in f)
    else:
        # Find all hdf5 files
        hdf5_input_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
        all_paths = list(hdf5_input_dir.rglob("*.hdf5"))
        if not all_paths:
            raise FileNotFoundError(f"No hdf5 files found in {hdf5_input_dir}.")

        total_files = len(all_paths)
        print(f"Found {total_files} hdf5 files.")

        hdf5_paths_list_path = output_dir / f"{output_dir.name}_umap_files.list"
        with open(hdf5_paths_list_path, "w", encoding="utf8") as f:
            for path in all_paths:
                f.write(f"{path}\n")
        print(f"Saved hdf5 files list to: {hdf5_paths_list_path}")

    # Load HDF5 files
    print("Loading HDF5 files.")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cannot read file directly with")
            hdf5_dict = hdf5_loader.load_hdf5s(
                data_file=hdf5_paths_list_path,
                verbose=True,
                strict=False,
            ).signals
    except Exception as e:
        raise RuntimeError(f"Error loading HDF5 files: {str(e)}") from e

    if not hdf5_dict:
        raise ValueError("No valid data loaded from HDF5 files")

    print(f"Loaded {len(hdf5_dict)}/{total_files} files.")

    # Extract data and free memory
    file_names = list(hdf5_dict.keys())
    try:
        # Stack all arrays into a single 2D array
        data = np.array(list(hdf5_dict.values()), dtype=np.float32)
        print(f"Dataset shape: {data.shape}")
    except Exception as e:
        raise RuntimeError(f"Error converting data to numpy array: {str(e)}") from e
    finally:
        del hdf5_dict  # Free memory immediately

    if data.size == 0:
        raise ValueError("Empty dataset after conversion")

    # Find rows containing NaN or Inf values
    problematic_rows = (~np.isfinite(data)).any(axis=1)
    if np.any(problematic_rows):
        row_indices = np.where(problematic_rows)[0]
        print(f"Problematic rows (indices): {row_indices}")

        affected_files = [file_names[i] for i in row_indices]
        print(f"Filenames with issues: {affected_files}")

        raise ValueError("Dataset contains inf or NaN values")

    problematic_rows_idx = find_rows_with_same_values(data)
    if problematic_rows_idx:
        print(f"Problematic rows (indices): {problematic_rows_idx}")
        affected_files = [file_names[i] for i in problematic_rows_idx]
        print(f"Filenames with issues: {affected_files}")
        raise ValueError(
            "Dataset contains rows with all identical values. Check preprocessing steps."
        )

    N_files = len(file_names)

    # Validate batch size against dataset size
    if cli.batch_size > N_files:
        print(
            f"Warning: batch_size ({cli.batch_size}) is larger than dataset size ({N_files})"
        )
        batch_size = N_files
    else:
        batch_size = cli.batch_size

    # PCA computation
    print("Computing PCA")
    n_components = min(3, N_files)  # Ensure n_components doesn't exceed dataset size
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    try:
        X_ipca = ipca.fit_transform(data)
    except Exception as e:
        raise RuntimeError(f"PCA computation failed: {str(e)}") from e
    finally:
        del data  # Free memory immediately after PCA

    # Save results
    fit_name = f"IPCA_fit_n{N_files}.skops"
    X_name = f"X_IPCA_n{N_files}.skops"
    dump_fit = {"file_names": file_names, "ipca_fit": ipca}
    dump_transformed_data = {"file_names": file_names, "X_ipca": X_ipca}

    try:
        skio.dump(dump_fit, output_dir / fit_name)
        skio.dump(dump_transformed_data, output_dir / X_name)
    except Exception as e:
        raise RuntimeError(f"Error saving results: {str(e)}") from e

    # Save requirements
    try:
        dists = metadata.distributions()
        req_file_name = "IPCA_saved_files_requirements.txt"
        with open(output_dir / req_file_name, "w", encoding="utf8") as f:
            for dist in dists:
                name = dist.metadata["Name"]
                version = dist.version
                f.write(f"{name}=={version}\n")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Could not save requirements file: {str(e)}")

    print(f"Saved IPCA fit to: {output_dir / fit_name}")
    print(f"Saved transformed data to: {output_dir / X_name}")
    print(f"Saved requirements to: {output_dir / req_file_name}")


if __name__ == "__main__":
    main()
