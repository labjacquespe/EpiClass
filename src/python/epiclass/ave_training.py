"""Train the hybrid AVE anomaly-detection model with cross-validation.

Mirrors epiatlas_training.py (UUID-grouped EpiAtlas folds, Comet-ML logging)
but trains the unsupervised LightningAVE instead of the dense classifier. After
each fold, validation samples are scored by reconstruction error and flagged as
outliers against an adaptive threshold; per-sample scores are written to
``ave_validation_scores.csv``.

The metadata `category` is still required: it drives the fold factory's
stratification / UUID grouping. The AVE itself ignores labels in its loss.
"""
# pylint: disable=duplicate-code, wrong-import-position, ungrouped-imports
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
import torch
from lightning.pytorch import loggers as pl_loggers

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import metadata
from epiclass.core.data.dataset import DataSet
from epiclass.core.data_source import EpiDataSource
from epiclass.core.lazy.lazy_fold_factory import (
    LazyEpiAtlasFoldFactory as EpiAtlasFoldFactory,
)
from epiclass.core.model_ave import LightningAVE
from epiclass.core.trainer import MyTrainer, define_callbacks
from epiclass.utils import modify_metadata
from epiclass.utils.check_dir import create_dirs
from epiclass.utils.my_logging import log_dset_composition, log_pre_training
from epiclass.utils.time import time_now
from epiclass.utils.torch_data import create_torch_datasets


class DatasetError(Exception):
    """Custom error"""

    def __init__(self, *args: object) -> None:
        print(
            "\n--- ERROR : Verify source files, filters, and min_class_size. ---\n",
            file=sys.stderr,
        )
        super().__init__(*args)


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
        "--mmap_dir", type=Path, default=None,
        help="Directory for the HDF5 mmap cache (default: <logdir>/mmap_cache). "
             "On HPC set to $SLURM_TMPDIR for fast local-disk writes.",
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments()

    category = cli.category

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)
    hdf5_resolution = my_datasource.hdf5_resolution()

    with open(cli.hyperparameters, "r", encoding="utf-8") as file:
        hparams: Dict[str, Any] = json.load(file)

    my_metadata = metadata.UUIDMetadata(my_datasource.metadata_file)

    # --- Prefilter metadata ---
    my_metadata.remove_missing_labels(category)

    if category in {"paired", "paired_end_mode"}:
        category = "paired_end_mode"
        modify_metadata.merge_pair_end_info(my_metadata)

    label_list = metadata.env_filtering(my_metadata, category)

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = hparams.get("min_class_size", 10)

    # --- Load signals ---
    loading_begin = time_now()

    restore_model = cli.restore
    n_fold = hparams.get("n_fold", 10)

    mmap_dir = (
        cli.mmap_dir if cli.mmap_dir is not None else Path(cli.logdir) / "mmap_cache"
    )
    ea_handler = EpiAtlasFoldFactory.from_datasource(
        my_datasource,
        category,
        label_list,
        n_fold=n_fold,
        test_ratio=0,
        min_class_size=min_class_size,
        force_filter=True,
        metadata=my_metadata,
        mmap_dir=mmap_dir,
    )
    loading_time = time_now() - loading_begin

    to_log = {
        "loading_time": loading_time.total_seconds(),
        "hdf5_resolution": str(hdf5_resolution),
        "category": category,
    }

    min_split = int(os.getenv("MIN_SPLIT", "0"))
    max_split = int(os.getenv("MAX_SPLIT", "42"))

    time_before_split = time_now()
    oversample = hparams.get("oversample", hparams.get("oversampling", True))
    for i, my_data in enumerate(ea_handler.yield_split(oversample=oversample)):
        # Skip if not in inclusive range
        if not (min_split <= i <= max_split):  # pylint: disable=superfluous-parens
            continue

        split_time = time_now() - time_before_split
        to_log.update({"split_time": split_time.total_seconds()})

        # --- Startup LOGGER ---
        is_online = not cli.offline  # additional logging fails when offline
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

        comet_logger.experiment.add_tag("EpiAtlas")
        comet_logger.experiment.add_tag("AVE")
        log_pre_training(logger=comet_logger, to_log=to_log, step=i)

        # Everything happens in there
        do_one_experiment(
            split_nb=i,
            my_data=my_data,
            hparams=hparams,
            logger=comet_logger,
            restore=restore_model,
        )

        time_before_split = time_now()


def do_one_experiment(
    split_nb: int,
    my_data: DataSet,
    hparams: Dict,
    logger: pl_loggers.CometLogger,
    restore: bool,
) -> None:
    """Train one AVE fold, then score the validation set by reconstruction error."""
    begin_loop = time_now()

    log_dset_composition(my_data, logdir=None, logger=logger, split_nb=split_nb)

    dsets_dict = create_torch_datasets(
        data=my_data,
        batch_size=hparams.get("batch_size", 64),
    )
    _, train_dataloader = dsets_dict["training"]
    valid_dataset, valid_dataloader = dsets_dict["validation"]

    if my_data.train.num_examples == 0 or my_data.validation.num_examples == 0:
        raise DatasetError("Trying to train without any training or validation data.")

    # Warning : output mapping of model created from training dataset
    mapping_file = Path(logger.save_dir) / "training_mapping.tsv"  # type: ignore

    if not restore:
        # --- CREATE a brand new MODEL ---
        my_data.save_mapping(mapping_file)
        mapping = my_data.load_mapping(mapping_file)
        logger.experiment.log_asset(mapping_file)

        input_size = my_data.train.signal_length

        my_model = LightningAVE(
            input_size=input_size,
            hparams=hparams,
            mapping=mapping,
        )

        if split_nb == 0:
            print("--MODEL STRUCTURE--\n", my_model)
            my_model.print_model_summary()  # torchinfo summary

        gpu_available = torch.cuda.device_count() > 0
        print(f"GPU available: {gpu_available}")

        # --- TRAIN the model ---
        callbacks = define_callbacks(
            early_stop_limit=hparams.get("early_stop_limit", 20),
            show_summary=(split_nb == 0),
            show_progress_bar=not gpu_available,  # Show progress bar only on CPU
            monitor="valid_loss",
            mode="min",
        )

        # Always tee metrics to a local CSV under the logdir. Offline Comet
        # (online=False) disables logging entirely, so this CSV is the only
        # persisted record of train_loss / train_recon / train_kl / valid_*.
        csv_logger = pl_loggers.CSVLogger(
            save_dir=str(logger.save_dir), name="metrics"  # type: ignore
        )

        before_train = time_now()
        trainer_kwargs = {
            "general_log_dir": logger.save_dir,  # type: ignore
            "model": my_model,
            "max_epochs": hparams.get("training_epochs", 50),
            "check_val_every_n_epoch": hparams.get("measure_frequency", 1),
            "logger": [logger, csv_logger],
            "callbacks": callbacks,
            "accelerator": "gpu" if gpu_available else "cpu",
            "devices": 1,
        }
        if gpu_available:
            trainer_kwargs["precision"] = 16
            trainer_kwargs["enable_progress_bar"] = False

        trainer = MyTrainer(**trainer_kwargs)

        if split_nb == 0:
            trainer.print_hyperparameters()

        trainer.fit(
            my_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            verbose=(split_nb == 0),
        )
        trainer.save_model_path()
        training_time = time_now() - before_train
        print(f"training time: {training_time}")

        logger.experiment.log_metric(
            "Training time", training_time.total_seconds(), step=split_nb
        )
        logger.experiment.log_metric("Last epoch", my_model.current_epoch, step=split_nb)

    try:
        my_model = LightningAVE.restore_model(logger.save_dir)
    except FileNotFoundError as e:
        logger.experiment.add_tag("ModelNotFoundError")
        logger.finalize(status="ModelNotFoundError")
        raise e

    # --- SCORING : reconstruction error + adaptive thresholding ---
    score_validation_set(
        model=my_model,
        valid_dataset=valid_dataset,
        valid_data=my_data.validation,
        save_dir=Path(logger.save_dir),  # type: ignore
        logger=logger,
        split_nb=split_nb,
    )

    end_loop = time_now()
    loop_time = end_loop - begin_loop
    logger.experiment.log_metric("Loop time", loop_time.total_seconds(), step=split_nb)
    print(f"Loop time (excludes split time): {loop_time}")

    logger.experiment.add_tag("Finished")
    logger.finalize(status="Finished")
    logger.experiment.end()


def score_validation_set(
    model: LightningAVE,
    valid_dataset,
    valid_data,
    save_dir: Path,
    logger: pl_loggers.CometLogger,
    split_nb: int,
) -> Path:
    """Score validation samples by reconstruction error and write per-sample CSV.

    Returns the path to ``ave_validation_scores.csv``.
    """
    errors = model.reconstruction_errors(valid_dataset)
    threshold = model.threshold_from_contamination(errors, model.contamination_rate)
    flags = model.predict_outliers(errors, threshold)

    # valid_dataset is unshuffled, so dataset index order matches valid_data.ids.
    sample_ids = valid_data.ids

    scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "reconstruction_error": errors,
            "outlier_flag": flags,
        }
    )
    out_path = save_dir / "ave_validation_scores.csv"
    scores.to_csv(out_path, index=False)
    print(f"Wrote {len(scores)} validation scores to {out_path}")

    logger.experiment.log_asset(out_path)
    logger.experiment.log_metric("outlier_threshold", threshold, step=split_nb)
    logger.experiment.log_metric("n_outliers_flagged", int(flags.sum()), step=split_nb)
    return out_path


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
