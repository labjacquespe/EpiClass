"""Predict labels for unlabeled samples using a trained model.

Supports two input formats:
  - Single-sample HDF5 (default): one HDF5 file per sample with per-chromosome
    datasets. Requires --chromsize. Builds a memory-mapped .npy file before
    inference.
  - Chunked HDF5 (--chunked): pre-concatenated multi-sample HDF5 files
    produced by convert_to_chunked.py. No --chromsize needed; reads directly.

Output: a CSV with one row per sample, a 'Predicted class' column, and one
column per class containing the softmax probability (written via
analysis.Analysis.write_test_prediction).
"""
# pylint: disable=wrong-import-position, ungrouped-imports
import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

warnings.simplefilter("ignore", category=FutureWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
from lightning.pytorch import loggers as pl_loggers

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import analysis
from epiclass.core.data.dataset import DataSet
from epiclass.core.lazy.chunked_hdf5_loader import ChunkedHdf5Loader
from epiclass.core.lazy.lazy_data_classes import LazyUnknownData, SignalLoader
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    # fmt: off
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5", type=Path,
        help="For single format: file listing HDF5 paths. "
             "For chunked format: directory or file of chunk HDF5s.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for output logs.",
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
        "--model", type=DirectoryChecker(),
        help="Directory from which to load the model. Defaults to logdir.",
    )
    arg_parser.add_argument(
        "--offline", action="store_true",
        help="Log offline instead of online.",
    )
    # fmt: on
    return arg_parser.parse_args()


def _build_loader(cli: argparse.Namespace) -> tuple[SignalLoader, list[str]]:
    """Return (loader, ordered sample_ids) for the selected input format."""
    if cli.chunked:
        loader = ChunkedHdf5Loader()
        loader.register_chunked_hdf5s(cli.hdf5, strict=True)
        return loader, loader.sample_ids

    if cli.chromsize is None:
        raise ValueError(
            "--chromsize is required for single-sample HDF5 format. "
            "Use --chunked if your data is in chunked format."
        )
    mmap_dir = (
        cli.mmap_dir if cli.mmap_dir is not None else Path(cli.logdir) / "mmap_cache"
    )
    loader = LazyHdf5Loader(
        chrom_file=cli.chromsize,
        normalization=True,
        mmap_dir=mmap_dir,
    )
    loader.register_hdf5s(cli.hdf5, hdf5_dir=cli.hdf5_dir, strict=True)
    loader.preload_all()
    return loader, list(loader.file_paths.keys())


def main():
    """Main called from command line."""
    begin = time_now()
    print(f"begin {begin}")

    cli = parse_arguments()
    logdir = Path(cli.logdir)
    model_dir = Path(cli.model) if cli.model else logdir
    is_online = not cli.offline

    # --- Logger ---
    comet_logger = pl_loggers.CometLogger(
        project="EpiClass",
        name="-".join(logdir.parts[-2:]),
        offline_directory=logdir,  # type: ignore
        online=is_online,
        auto_metric_logging=False,
    )
    exp_key = comet_logger.experiment.get_key()
    print(f"The current experiment key is {exp_key}")
    comet_logger.experiment.log_other("Experience key", f"{exp_key}")

    if "SLURM_JOB_ID" in os.environ:
        comet_logger.experiment.log_other("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        comet_logger.experiment.add_tag("Cluster")

    fmt = "chunked" if cli.chunked else "single"
    comet_logger.experiment.log_other("input_format", fmt)

    # --- Load data ---
    now = time_now()
    print(f"Loading data ({fmt} format): {now}")
    loader, sample_ids = _build_loader(cli)
    n = len(sample_ids)
    print(f"Registered {n} samples  ({time_now() - now})")

    test_data = LazyUnknownData(
        ids=sample_ids,
        loader=loader,
        y=np.zeros(n, dtype=np.int64),
        y_str=[""] * n,
    )
    if test_data.num_examples == 0:
        raise ValueError("Trying to predict without any test data.")

    # Materialize into a TensorDataset; analysis.Analysis consumes it.
    signals, labels = test_data.materialize()
    test_dataset = TensorDataset(
        torch.from_numpy(signals).float(),
        torch.from_numpy(labels).int(),
    )

    datasets = DataSet.empty_collection(data_class=LazyUnknownData)
    datasets.set_test(test_data)

    # --- Restore model ---
    my_model = LightningDenseClassifier.restore_model(model_dir)
    print("Model successfully restored.")

    # --- Predictions (full softmax vector per sample) ---
    analyzer = analysis.Analysis(
        my_model,
        datasets_info=datasets,
        logger=comet_logger,
        train_dataset=None,
        val_dataset=None,
        test_dataset=test_dataset,
    )
    predict_path = logdir / f"{model_dir.stem}_test_prediction_{cli.hdf5.stem}.csv"
    analyzer.write_test_prediction(path=predict_path)

    end = time_now()
    main_time = end - begin
    print(f"end {end}")
    print(f"Main() duration: {main_time}")
    comet_logger.experiment.log_other("Main duration", main_time)
    comet_logger.experiment.add_tag("Finished")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
