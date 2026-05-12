"""Compute various UMAP embeddings (modify nearest neighboors + densMap or not) for some hardcoded datasets."""

# pylint: disable=import-error, redefined-outer-name, use-dict-literal, too-many-lines, unused-import, unused-argument, too-many-branches
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import umap
from umap.umap_ import nearest_neighbors

from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "--load_knn",
        type=Path,
        default=None,
        help="Directory containing precomputed knn pickle file",
    )
    arg_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Directory to save embeddings in.",
    )
    arg_parser.add_argument(
        "-l", "--hdf5_list",
        type=Path,
        default=None,
        help="List of hdf5 files to load, only the filenames will be considered.",
    )
    # fmt: on
    return arg_parser.parse_args()


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

    input_basedir = Path("/lustre06/project/6007017/rabyj/epilap/input")
    chromsize_path = input_basedir / "chromsizes" / "hg38.noy.chrom.sizes"
    for path in [input_basedir, chromsize_path]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}.")

    hdf5_loader = LazyHdf5Loader(
        chrom_file=chromsize_path,
        normalization=True,
        mmap_dir=output_dir / "mmap_cache",
    )

    # Get all paths
    hdf5_input_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))

    all_paths = list(hdf5_input_dir.rglob("*.hdf5"))
    if not all_paths:
        raise FileNotFoundError(f"No hdf5 files found in {hdf5_input_dir}.")

    # Filter hdf5 files
    if cli.hdf5_list is not None:
        with open(cli.hdf5_list, "r", encoding="utf8") as f:
            hdf5_list = f.readlines()
        hdf5_to_read = set(Path(x.strip()).name for x in hdf5_list)
        all_paths = [x for x in all_paths if x.name in hdf5_to_read]

    # Save list of hdf5 files actually read
    hdf5_paths_list_path = output_dir / f"{output_dir.name}_umap_files.list"
    with open(hdf5_paths_list_path, "w", encoding="utf8") as f:
        for path in all_paths:
            f.write(f"{path}\n")

    # Register + preload (mmap-backed; full matrix lives on disk, OS pages in
    # rows as nearest_neighbors / umap touch them).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot read file directly with")
        hdf5_loader.register_hdf5s(
            data_file=hdf5_paths_list_path,
            verbose=True,
            strict=False,
        )
        hdf5_loader.preload_all()

    print(f"Loaded {len(hdf5_loader.file_paths)}/{len(all_paths)} files.")
    file_names = list(hdf5_loader.file_paths.keys())
    data = hdf5_loader.as_mmap()

    # UMAP parameters
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

    # Compute+save embeddings
    for name, params in embedding_params.items():
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
