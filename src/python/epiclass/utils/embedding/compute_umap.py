"""Compute UMAP embeddings (varying nearest-neighbors + densMap) from HDF5 signals.

Supports two input formats:
  - Single-sample HDF5 (default): loaded via LazyHdf5Loader into a single
    mmap-backed `.npy` file. Requires --chromsize. UMAP/pynndescent see a
    memory-mapped ndarray (copy-on-write) so the dataset never has to live
    fully in RAM.
  - Chunked HDF5 (--chunked): pre-concatenated multi-sample HDF5s
    (produced by hdf5_chunks_creation.py). No --chromsize needed.
    Materializes the full matrix via load_batch — UMAP needs random
    access to all rows, so streaming isn't possible here. For genuinely
    huge chunked inputs, PCA-first via compute_pca.py.

By default the script sweeps {standard, densmap} × {2D, 3D} × {15, 30, 100}
nearest-neighbor sizes — 12 embeddings total. Use ``--max_embeddings`` to cap
the sweep when testing or smoke-running.
"""
# pylint: disable=duplicate-code, too-many-branches
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import warnings
from importlib import metadata
from pathlib import Path
from typing import List, Tuple

import numpy as np
import umap
from umap.umap_ import nearest_neighbors

from epiclass.core.lazy.chunked_hdf5_loader import ChunkedHdf5Loader
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser(
        description="Compute UMAP embeddings for hdf5 signals."
    )

    # fmt: off
    arg_parser.add_argument(
        "hdf5",
        type=Path,
        nargs="?",
        default=None,
        help="For single format: file listing HDF5 paths (or omit to scan "
             "SLURM_TMPDIR/tmp). For chunked format: directory or file of "
             "chunk HDF5s.",
    )
    arg_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Directory to save embeddings in. Defaults to home directory.",
    )
    arg_parser.add_argument(
        "--chunked",
        action="store_true",
        help="Input is chunked HDF5 format. If not set, single-sample HDF5 is assumed.",
    )
    arg_parser.add_argument(
        "--chromsize",
        type=Path,
        default=None,
        help="A file with chrom sizes. Required for single-sample HDF5 format.",
    )
    arg_parser.add_argument(
        "-l", "--input_list",
        type=Path,
        default=None,
        help="DEPRECATED alias for the hdf5 positional in single-sample mode.",
    )
    arg_parser.add_argument(
        "--load_knn",
        type=Path,
        default=None,
        help="Directory containing a precomputed knn pickle file.",
    )
    arg_parser.add_argument(
        "--max_embeddings",
        type=int,
        default=None,
        help="If set, compute at most this many embeddings from the sweep.",
    )
    # fmt: on
    return arg_parser.parse_args()


def _load_single(
    cli: argparse.Namespace, output_dir: Path
) -> Tuple[np.ndarray, List[str]]:
    """Single-sample path: mmap-backed (copy-on-write so numba accepts it)."""
    if cli.chromsize is None:
        raise ValueError(
            "--chromsize is required for single-sample HDF5 format. "
            "Use --chunked if your data is in chunked format."
        )

    hdf5 = cli.input_list if cli.input_list is not None else cli.hdf5
    if hdf5 is None:
        scan_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
        paths = list(scan_dir.rglob("*.hdf5"))
        if not paths:
            raise FileNotFoundError(f"No hdf5 files found in {scan_dir}.")
        hdf5 = output_dir / f"{output_dir.name}_umap_files.list"
        with open(hdf5, "w", encoding="utf8") as f:
            for path in paths:
                f.write(f"{path}\n")
        print(f"Wrote auto-discovered hdf5 list to {hdf5}")

    loader = LazyHdf5Loader(
        chrom_file=cli.chromsize,
        normalization=True,
        mmap_dir=output_dir / "mmap_cache",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot read file directly with")
        loader.register_hdf5s(data_file=hdf5, verbose=True, strict=False)
        loader.preload_all()

    print(f"Loaded {len(loader.file_paths)} files.")
    # Copy-on-write: pynndescent's numba-jitted kernels reject read-only arrays.
    return loader.as_mmap(mmap_mode="c"), list(loader.file_paths.keys())


def _load_chunked(
    cli: argparse.Namespace,
) -> Tuple[np.ndarray, List[str]]:
    """Chunked path: materialize via load_batch (UMAP needs random access)."""
    if cli.hdf5 is None:
        raise ValueError(
            "Provide a chunk directory or file as positional 'hdf5' for --chunked."
        )
    loader = ChunkedHdf5Loader()
    loader.register_chunked_hdf5s(cli.hdf5, strict=True)
    if loader.num_registered == 0:
        raise ValueError("No samples registered from chunked input.")
    ids = list(loader.sample_ids)
    print(f"Loaded {len(ids)} samples from {len(loader.chunk_files)} chunk file(s).")
    return loader.load_batch(ids), ids


def main():
    """Run the main function."""
    cli = parse_arguments()

    load_knn_dir: Path | None = cli.load_knn
    if load_knn_dir is not None:
        if not load_knn_dir.exists():
            raise FileNotFoundError(f"Could not find {load_knn_dir}.")
        if not next(load_knn_dir.glob("precomputed_knn_*.pkl")):
            raise FileNotFoundError(
                f"No precomputed knn pickle files found in {load_knn_dir}."
            )

    if cli.output is not None:
        output_dir = cli.output
        try:
            output_dir.mkdir(exist_ok=True)
        except FileNotFoundError:
            output_dir = Path.home()
    else:
        output_dir = Path.home()

    if cli.chunked:
        data, file_names = _load_chunked(cli)
    else:
        data, file_names = _load_single(cli, output_dir)

    # UMAP parameter sweep
    nn_default = 15
    nn_bigger = 30
    nn_biggest = 100
    embedding_params = {}
    for nn_size, n_dim in itertools.product([nn_default, nn_bigger, nn_biggest], [2, 3]):
        embedding_params[f"standard_{n_dim}D_nn{nn_size}"] = {
            "n_neighbors": nn_size,
            "min_dist": 0.1,
            "n_components": n_dim,
            "low_memory": False,
        }
        embedding_params[f"densmap_{n_dim}D_nn{nn_size}"] = {
            "n_neighbors": nn_size,
            "min_dist": 0.1,
            "n_components": n_dim,
            "low_memory": False,
            "densmap": True,
        }

    nn_knn = 100
    if not load_knn_dir:
        # Compute+save knn graph
        precomputed_knn = nearest_neighbors(
            X=data,
            n_neighbors=nn_knn,
            metric="correlation",
            random_state=42,
            low_memory=False,
            metric_kwds=None,
            angular=None,
        )

        with open(output_dir / f"precomputed_knn_{nn_knn}.pkl", "wb") as f:
            pickle.dump(precomputed_knn, f)
        print(f"Saved precomputed_knn_{nn_knn}.pkl")

        # Save requirements so knn pickle is never lost in the future
        dists = metadata.distributions()
        with open(output_dir / "pickle_requirements.txt", "w", encoding="utf8") as f:
            for dist in dists:
                name = dist.metadata["Name"]
                version = dist.version
                f.write(f"{name}=={version}\n")
        print("Saved pickle_requirements.txt")
    else:
        # Load precomputed knn graph
        with open(output_dir / f"precomputed_knn_{nn_knn}.pkl", "rb") as f:
            precomputed_knn = pickle.load(f)

    # Compute+save embeddings (capped by --max_embeddings if set)
    items = list(embedding_params.items())
    if cli.max_embeddings is not None:
        items = items[: cli.max_embeddings]
    for name, params in items:
        filename = output_dir / f"embedding_{name}.pkl"
        if filename.exists():
            print(f"Embedding {name} already exists. Skipping.")
            continue

        embedding = umap.UMAP(
            **params, random_state=42, precomputed_knn=precomputed_knn
        ).fit_transform(X=data)

        with open(filename, "wb") as f:
            pickle.dump({"ids": file_names, "embedding": embedding, "params": params}, f)
            print(f"Saved embedding_{name}.pkl")


if __name__ == "__main__":
    main()
