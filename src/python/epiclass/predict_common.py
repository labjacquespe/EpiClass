"""Shared data-loading helpers for the prediction entry points.

``predict.py`` (single model) and ``predict_CV.py`` (ensemble over all CV fold models) both
turn a list of unlabeled samples into a ``TensorDataset`` the exact same way -- the only thing
that differs is how many models consume it. Keeping the loader / dataset construction here lets
the ensemble build the data **once** and reuse it across every fold model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import analysis
from epiclass.core.data.dataset import DataSet
from epiclass.core.lazy.chunked_hdf5_loader import ChunkedHdf5Loader
from epiclass.core.lazy.lazy_data_classes import LazyUnknownData, SignalLoader
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.core.model_pytorch import LightningDenseClassifier


def add_data_arguments(arg_parser: argparse.ArgumentParser) -> None:
    """Register the input-format / data-loading CLI args shared by predict entry points."""
    # fmt: off
    arg_parser.add_argument(
        "hdf5", type=Path,
        help="For single format: file listing HDF5 paths. "
             "For chunked format: directory or file of chunk HDF5s.",
    )
    arg_parser.add_argument(
        "--chromsize", type=Path,
        help="Chromosome sizes file. Required for single-sample HDF5 format.",
    )
    arg_parser.add_argument(
        "--chunked", action="store_true",
        help="Input is chunked HDF5 format (e.g. produced by convert_to_chunked.py). "
             "If not set, single-sample HDF5 format is assumed.",
    )
    arg_parser.add_argument(
        "--mmap_dir", type=Path, default=None,
        help="Directory for the mmap cache (single format only). "
             "Defaults to <logdir>/mmap_cache. On HPC, set to $SLURM_TMPDIR.",
    )
    arg_parser.add_argument(
        "--hdf5_dir", type=Path,
        help="Override HDF5 file paths to this directory (single format). "
             "Useful when HDF5s are copied to $SLURM_TMPDIR.",
    )
    # fmt: on


def build_loader(
    cli: argparse.Namespace, mmap_default_dir: Path
) -> tuple[SignalLoader, list[str]]:
    """Return (loader, ordered sample_ids) for the selected input format.

    ``mmap_default_dir`` is the fallback mmap cache directory used for the single-sample
    format when ``--mmap_dir`` is not given.
    """
    if cli.chunked:
        loader = ChunkedHdf5Loader()
        loader.register_chunked_hdf5s(cli.hdf5, strict=True)
        return loader, loader.sample_ids

    if cli.chromsize is None:
        raise ValueError(
            "--chromsize is required for single-sample HDF5 format. "
            "Use --chunked if your data is in chunked format."
        )
    mmap_dir = cli.mmap_dir if cli.mmap_dir is not None else mmap_default_dir
    loader = LazyHdf5Loader(
        chrom_file=cli.chromsize,
        normalization=True,
        mmap_dir=mmap_dir,
    )
    # strict=True opens every HDF5 to validate it exists — necessary when the mmap
    # cache must be built, but pointless (and very slow on Lustre: thousands of
    # small-file opens) when the cache already exists, since the data is then read
    # from the .npy. On reuse, preload_all's row-count check verifies the cache
    # matches the file list instead.
    loader.register_hdf5s(
        cli.hdf5, hdf5_dir=cli.hdf5_dir, strict=not loader.mmap_exists()
    )
    loader.preload_all()
    return loader, list(loader.file_paths.keys())


def build_test_dataset(
    cli: argparse.Namespace, mmap_default_dir: Path
) -> Tuple[LazyUnknownData, TensorDataset, DataSet]:
    """Build the unlabeled test data once: (LazyUnknownData, TensorDataset, DataSet).

    The ``TensorDataset`` drives inference; the ``DataSet`` carries the sample ids used to
    label prediction-CSV rows. Raises ``ValueError`` when no samples are registered.
    """
    loader, sample_ids = build_loader(cli, mmap_default_dir)
    n = len(sample_ids)

    test_data = LazyUnknownData(
        ids=sample_ids,
        loader=loader,
        y=np.zeros(n, dtype=np.int64),
        y_str=[""] * n,
    )
    if test_data.num_examples == 0:
        raise ValueError("Trying to predict without any test data.")

    signals, labels = test_data.materialize()
    test_dataset = TensorDataset(
        torch.from_numpy(signals).float(),
        torch.from_numpy(labels).int(),
    )

    datasets = DataSet.empty_collection(data_class=LazyUnknownData)
    datasets.set_test(test_data)
    return test_data, test_dataset, datasets


def write_test_predictions(
    model: LightningDenseClassifier,
    datasets: DataSet,
    test_dataset: TensorDataset,
    path: Path,
) -> None:
    """Score ``test_dataset`` with ``model`` and write the test-prediction CSV to ``path``.

    Inference does not use an experiment logger (``logger=None``): metrics/CSVs are written to
    disk only.
    """
    analyzer = analysis.Analysis(
        model,
        datasets_info=datasets,
        logger=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=test_dataset,
    )
    analyzer.write_test_prediction(path=path)


# Re-exported so callers needing a directory-validating CLI type don't import argparseutils.
__all__ = [
    "add_data_arguments",
    "build_loader",
    "build_test_dataset",
    "write_test_predictions",
    "DirectoryChecker",
]
