"""Train the hybrid AVE on independent-sample datasets (no UUID metadata).

General-data counterpart to ave_training.py, in the same way general_training.py
is the counterpart to epiatlas_training.py. Uses GeneralFoldFactory (stratified
k-fold, no UUID/track_type/EPIRR requirement) so the AVE can train on datasets
like saccer3 whose samples are independent.

The per-fold training + reconstruction-error scoring logic is shared verbatim
with ave_training.py (do_one_experiment / score_validation_set); only the
data-loading / fold-generation differs. Like ave_training.py it logs through a
Comet logger (use --offline to avoid an API key).
"""
# pylint: disable=duplicate-code, wrong-import-position, ungrouped-imports

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
from lightning.pytorch import loggers as pl_loggers

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.data_source import EpiDataSource
from epiclass.core.lazy.general_fold_factory import GeneralFoldFactory
from epiclass.mains.ave_training import do_one_experiment
from epiclass.utils.check_dir import create_dirs
from epiclass.utils.my_logging import log_pre_training
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "category", type=str, help="The metadata category driving CV stratification.",
    )
    arg_parser.add_argument(
        "hyperparameters", type=Path, help="A json file containing model hyperparameters.",
    )
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!",
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    arg_parser.add_argument(
        "--offline",
        action="store_true",
        help="Will log data offline instead of online.",
    )
    arg_parser.add_argument(
        "--restore",
        action="store_true",
        help="Skips training, tries to restore existing models in logdir for scoring.",
    )
    arg_parser.add_argument(
        "--n_fold", type=int, default=4, help="Number of CV folds (default: 4).",
    )
    arg_parser.add_argument(
        "--min_class_size", type=int, default=10, help="Min samples per class (default: 10).",
    )
    arg_parser.add_argument(
        "--mmap_dir", type=Path, default=None,
        help="Directory for the HDF5 mmap cache (default: <logdir>/mmap_cache). "
             "On HPC set to $SLURM_TMPDIR for fast local-disk writes.",
    )
    arg_parser.add_argument(
        "--folds", type=Path, default=None,
        help='JSON file of explicit fold membership: '
             '{"fold1": {"md5sum": [ids...]}, ...}. The inner key names a '
             'metadata category matched 1:1 to samples (each value must resolve '
             'to exactly one signal). Overrides --n_fold, --min_class_size and '
             'oversampling.',
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    cli = parse_arguments()
    category = cli.category

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)
    hdf5_resolution = my_datasource.hdf5_resolution()

    with open(cli.hyperparameters, "r", encoding="utf-8") as file:
        hparams: Dict[str, Any] = json.load(file)

    restore_model = cli.restore

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = cli.min_class_size

    # --- Pre-specified folds (override n_fold / min_class_size / oversampling) ---
    fold_definitions = None
    if cli.folds is not None:
        with open(cli.folds, "r", encoding="utf-8") as file:
            fold_definitions = json.load(file)
        print(
            f"Using pre-specified folds from {cli.folds}: "
            f"{len(fold_definitions)} folds. "
            "Overriding --n_fold, --min_class_size and oversampling."
        )

    # --- Load signals (stratified k-fold, no UUID grouping) ---
    loading_begin = time_now()
    mmap_dir = (
        cli.mmap_dir if cli.mmap_dir is not None else Path(cli.logdir) / "mmap_cache"
    )
    fold_factory = GeneralFoldFactory(
        datasource=my_datasource,
        label_category=category,
        min_class_size=min_class_size,
        n_fold=cli.n_fold,
        mmap_dir=mmap_dir,
        fold_definitions=fold_definitions,
    )
    loading_time = time_now() - loading_begin

    to_log = {
        "loading_time": loading_time.total_seconds(),
        "hdf5_resolution": str(hdf5_resolution),
        "category": category,
        "folds_file": str(cli.folds) if cli.folds is not None else None,
    }

    min_split = int(os.getenv("MIN_SPLIT", "0"))
    max_split = int(os.getenv("MAX_SPLIT", "42"))

    time_before_split = time_now()
    oversample = hparams.get("oversample", hparams.get("oversampling", True))
    for i, my_data in enumerate(fold_factory.yield_split(oversample=oversample)):
        # Skip if not in inclusive range
        if not (min_split <= i <= max_split):  # pylint: disable=superfluous-parens
            continue

        split_time = time_now() - time_before_split
        to_log.update({"split_time": split_time.total_seconds()})

        # --- Startup LOGGER ---
        is_online = not cli.offline
        logdir = Path(cli.logdir) / f"split{i}"
        create_dirs(logdir)

        exp_name = "-".join(cli.logdir.parts[-3:]) + f"_split{i}"
        comet_logger = pl_loggers.CometLogger(
            project="EpiClass",
            name=exp_name,
            offline_directory=logdir,  # type: ignore
            online=is_online,
            auto_metric_logging=False,
        )

        comet_logger.experiment.add_tag("AVE")
        log_pre_training(logger=comet_logger, to_log=to_log, step=i)

        # Shared training + reconstruction-error scoring (same as ave_training.py)
        do_one_experiment(
            split_nb=i,
            my_data=my_data,
            hparams=hparams,
            logger=comet_logger,
            restore=restore_model,
        )

        time_before_split = time_now()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
