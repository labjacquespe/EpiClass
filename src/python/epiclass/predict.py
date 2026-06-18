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

warnings.simplefilter("ignore", category=FutureWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.model_checkpoint import last_checkpoint_path
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.prediction_files import (
    build_prediction_tag,
    experiment_id_from_checkpoint,
)
from epiclass.predict_common import (
    add_data_arguments,
    build_test_dataset,
    setup_comet_logger,
    write_test_predictions,
)
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    arg_parser = ArgumentParser()
    add_data_arguments(arg_parser)
    # fmt: off
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for output logs.",
    )
    arg_parser.add_argument(
        "--model", type=DirectoryChecker(),
        help="Directory from which to load the model. Defaults to logdir.",
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """Main called from command line."""
    begin = time_now()
    print(f"begin {begin}")

    cli = parse_arguments()
    logdir = Path(cli.logdir)
    model_dir = Path(cli.model) if cli.model else logdir

    # --- Logger ---
    comet_logger = setup_comet_logger(cli, logdir.parts[-2:], logdir)

    # --- Load data ---
    now = time_now()
    fmt = "chunked" if cli.chunked else "single"
    print(f"Loading data ({fmt} format): {now}")
    _, test_dataset, datasets = build_test_dataset(cli, logdir / "mmap_cache")
    print(f"Registered {datasets.test.num_examples} samples  ({time_now() - now})")

    # --- Restore model ---
    my_model = LightningDenseClassifier.restore_model(model_dir)
    print("Model successfully restored.")

    # --- Predictions (full softmax vector per sample) ---
    # Tag the output with the model's *training* provenance: the original training comet
    # experiment id (recovered from the checkpoint path) + checkpoint stem. The comet id of
    # this prediction run is deliberately not used.
    ckpt = last_checkpoint_path(model_dir)
    tag = build_prediction_tag(experiment_id_from_checkpoint(ckpt), ckpt)
    model_id = f"{model_dir.stem}_{tag}" if tag else model_dir.stem
    predict_path = logdir / f"{model_id}_test_prediction_{cli.hdf5.stem}.csv"
    write_test_predictions(my_model, datasets, test_dataset, comet_logger, predict_path)

    end = time_now()
    main_time = end - begin
    print(f"end {end}")
    print(f"Main() duration: {main_time}")
    comet_logger.experiment.log_other("Main duration", main_time)
    comet_logger.experiment.add_tag("Finished")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
