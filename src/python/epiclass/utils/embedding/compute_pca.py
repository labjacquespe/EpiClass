"""Compute IncrementalPCA on HDF5 signals.

Supports two input formats:
  - Single-sample HDF5 (default): one HDF5 file per sample with per-chromosome
    datasets. Requires --chromsize. Loaded via LazyHdf5Loader into a single
    mmap-backed .npy file; IncrementalPCA reads it in batches.
  - Chunked HDF5 (--chunked): pre-concatenated multi-sample HDF5 files
    produced by hdf5_chunks_creation.py. No --chromsize needed. PCA streams
    chunk-by-chunk via partial_fit, never materializing the full matrix.
"""
# pylint: disable=duplicate-code
from __future__ import annotations

import argparse
import os
import warnings
from importlib import metadata
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import skops.io as skio
from sklearn.decomposition import IncrementalPCA

from epiclass.core.lazy.chunked_hdf5_loader import ChunkedHdf5Loader
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = argparse.ArgumentParser(
        description="Compute Incremental PCA embeddings for hdf5 files."
    )
    arg_parser.add_argument(
        "hdf5",
        type=Path,
        help="For single format: file listing HDF5 paths (or omitted to scan "
             "SLURM_TMPDIR/tmp). For chunked format: directory or file of "
             "chunk HDF5s.",
        nargs="?",
        default=None,
    )
    arg_parser.add_argument(
        "output",
        type=Path,
        default=None,
        help="Directory to save embeddings in. Saves in home directory if not provided.",
    )
    arg_parser.add_argument(
        "--chunked",
        action="store_true",
        help="Input is chunked HDF5 format (produced by hdf5_chunks_creation.py). "
             "If not set, single-sample HDF5 format is assumed.",
    )
    arg_parser.add_argument(
        "--chromsize",
        type=Path,
        default=None,
        help="A file with chrom sizes. Required for single-sample HDF5 format.",
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
        help="DEPRECATED alias for the hdf5 positional in single-sample mode.",
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


def _resolve_single_input_list(hdf5: Path | None, output_dir: Path) -> Tuple[Path, int]:
    """Pick the HDF5 list for the single-sample path; fall back to scanning."""
    if hdf5 is not None:
        with open(hdf5, "r", encoding="utf8") as f:
            total = sum(1 for _ in f)
        return hdf5, total

    scan_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    paths = list(scan_dir.rglob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No hdf5 files found in {scan_dir}.")
    total = len(paths)
    print(f"Found {total} hdf5 files in {scan_dir}.")

    list_path = output_dir / f"{output_dir.name}_pca_files.list"
    with open(list_path, "w", encoding="utf8") as f:
        for path in paths:
            f.write(f"{path}\n")
    print(f"Saved hdf5 files list to: {list_path}")
    return list_path, total


def _pca_single(
    cli: argparse.Namespace, output_dir: Path
) -> Tuple[IncrementalPCA, np.ndarray, List[str]]:
    """Single-sample path: mmap-backed full matrix, IncrementalPCA streams it."""
    if cli.chromsize is None:
        raise ValueError(
            "--chromsize is required for single-sample HDF5 format. "
            "Use --chunked if your data is in chunked format."
        )

    hdf5 = cli.input_list if cli.input_list is not None else cli.hdf5
    list_path, total_files = _resolve_single_input_list(hdf5, output_dir)

    hdf5_loader = LazyHdf5Loader(
        chrom_file=cli.chromsize,
        normalization=True,
        mmap_dir=output_dir / "mmap_cache",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot read file directly with")
        hdf5_loader.register_hdf5s(data_file=list_path, verbose=True, strict=False)
        hdf5_loader.preload_all()

    if not hdf5_loader.file_paths:
        raise ValueError("No valid data loaded from HDF5 files")
    print(f"Loaded {len(hdf5_loader.file_paths)}/{total_files} files.")

    file_names = list(hdf5_loader.file_paths.keys())
    data = hdf5_loader.as_mmap()
    print(f"Dataset shape: {data.shape}")
    _validate_data(data, file_names)

    n_components, batch_size = _resolve_pca_params(cli.batch_size, len(file_names))
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_ipca = ipca.fit_transform(data)
    return ipca, X_ipca, file_names


def _pca_chunked(cli: argparse.Namespace) -> Tuple[IncrementalPCA, np.ndarray, List[str]]:
    """Chunked path: partial_fit + transform per chunk file (true streaming)."""
    if cli.hdf5 is None:
        raise ValueError(
            "Provide a chunk directory or file as positional 'hdf5' for --chunked."
        )

    loader = ChunkedHdf5Loader()
    loader.register_chunked_hdf5s(cli.hdf5, strict=True)
    if loader.num_registered == 0:
        raise ValueError("No samples registered from chunked input.")
    print(
        f"Loaded {loader.num_registered} samples from "
        f"{len(loader.chunk_files)} chunk file(s)."
    )

    file_names = list(loader.sample_ids)
    n_components, batch_size = _resolve_pca_params(cli.batch_size, len(file_names))
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Fit pass: stream chunk by chunk.
    for chunk_path in loader.chunk_files:
        with h5py.File(chunk_path, "r") as f:
            signals = f["signals"][:]
        _validate_data(signals, ids=None, where=str(chunk_path))
        # IncrementalPCA needs at least n_components rows per partial_fit call.
        for start in range(0, len(signals), batch_size):
            block = signals[start : start + batch_size]
            if len(block) < n_components:
                continue  # accumulate; tail handled by transform
            ipca.partial_fit(block)

    # Transform pass: same stream, project, concatenate.
    transformed = []
    for chunk_path in loader.chunk_files:
        with h5py.File(chunk_path, "r") as f:
            signals = f["signals"][:]
        transformed.append(ipca.transform(signals))
    X_ipca = np.concatenate(transformed, axis=0)
    return ipca, X_ipca, file_names


def _validate_data(
    data: np.ndarray, ids: List[str] | None = None, where: str = "dataset"
) -> None:
    """Raise if data contains NaN/Inf or rows with all identical values."""
    if data.size == 0:
        raise ValueError(f"Empty {where}")

    bad = (~np.isfinite(data)).any(axis=1)
    if np.any(bad):
        idxs = np.where(bad)[0]
        msg = f"{where} contains inf or NaN values at rows {list(idxs)}"
        if ids is not None:
            msg += f" ({[ids[i] for i in idxs]})"
        raise ValueError(msg)

    flat_rows = find_rows_with_same_values(data)
    if flat_rows:
        msg = f"{where} has rows with all identical values at {flat_rows}"
        if ids is not None:
            msg += f" ({[ids[i] for i in flat_rows]})"
        raise ValueError(msg)


def _resolve_pca_params(requested_batch: int, n_samples: int) -> Tuple[int, int]:
    """Cap batch_size to n_samples; n_components = min(3, n_samples)."""
    if requested_batch <= 0:
        raise ValueError("batch_size must be positive")
    batch_size = min(requested_batch, n_samples)
    n_components = min(3, n_samples)
    return n_components, batch_size


def main():
    """Run the main function."""
    cli = parse_arguments()
    output_dir = cli.output if cli.output is not None else Path.home()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing PCA")
    if cli.chunked:
        ipca, X_ipca, file_names = _pca_chunked(cli)
    else:
        ipca, X_ipca, file_names = _pca_single(cli, output_dir)
    n = len(file_names)

    fit_name = f"IPCA_fit_n{n}.skops"
    X_name = f"X_IPCA_n{n}.skops"
    skio.dump({"file_names": file_names, "ipca_fit": ipca}, output_dir / fit_name)
    skio.dump({"file_names": file_names, "X_ipca": X_ipca}, output_dir / X_name)

    try:
        dists = metadata.distributions()
        req_file = "IPCA_saved_files_requirements.txt"
        with open(output_dir / req_file, "w", encoding="utf8") as f:
            for dist in dists:
                f.write(f"{dist.metadata['Name']}=={dist.version}\n")
        print(f"Saved requirements to: {output_dir / req_file}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: Could not save requirements file: {str(e)}")

    print(f"Saved IPCA fit to: {output_dir / fit_name}")
    print(f"Saved transformed data to: {output_dir / X_name}")


if __name__ == "__main__":
    main()
