"""Shared data-loading helpers for the prediction entry points.

``predict.py`` (single model) and ``predict_CV.py`` (ensemble over all CV fold models) both
turn a list of unlabeled samples into a ``TensorDataset`` the exact same way -- the only thing
that differs is how many models consume it. Keeping the loader / dataset construction here lets
the ensemble build the data **once** and reuse it across every fold model.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from lightning.pytorch import loggers as pl_loggers
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
    arg_parser.add_argument(
        "--offline", action="store_true",
        help="Log offline instead of online.",
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
    loader.register_hdf5s(cli.hdf5, hdf5_dir=cli.hdf5_dir, strict=True)
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


def setup_comet_logger(
    cli: argparse.Namespace,
    name_parts: Sequence[str],
    offline_directory: Path,
) -> pl_loggers.CometLogger:
    """Build the prediction-run Comet logger and log the run's standard provenance.

    Logs the experiment key, the SLURM job id / Cluster tag when on a cluster, and the input
    format. Callers add any run-specific tags afterwards.
    """
    comet_logger = pl_loggers.CometLogger(
        project="EpiClass",
        name="-".join(name_parts),
        offline_directory=offline_directory,  # type: ignore
        online=not cli.offline,
        auto_metric_logging=False,
    )
    exp_key = comet_logger.experiment.get_key()
    print(f"The current experiment key is {exp_key}")
    comet_logger.experiment.log_other("Experience key", f"{exp_key}")
    if "SLURM_JOB_ID" in os.environ:
        comet_logger.experiment.log_other("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        comet_logger.experiment.add_tag("Cluster")
    comet_logger.experiment.log_other(
        "input_format", "chunked" if cli.chunked else "single"
    )
    return comet_logger


def write_test_predictions(
    model: LightningDenseClassifier,
    datasets: DataSet,
    test_dataset: TensorDataset,
    logger: Optional[pl_loggers.CometLogger],
    path: Path,
) -> None:
    """Score ``test_dataset`` with ``model`` and write the test-prediction CSV to ``path``."""
    analyzer = analysis.Analysis(
        model,
        datasets_info=datasets,
        logger=logger,
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
    "setup_comet_logger",
    "write_test_predictions",
    "DirectoryChecker",
]
