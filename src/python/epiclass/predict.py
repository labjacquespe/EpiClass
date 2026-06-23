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

import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.core.model_checkpoint import (
    resolve_checkpoint_spec,
    restore_from_checkpoint_file,
)
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.prediction_files import (
    build_prediction_tag,
    experiment_id_from_checkpoint,
)
from epiclass.predict_common import (
    add_data_arguments,
    build_test_dataset,
    prepare_inference_runtime,
    write_test_predictions,
)
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    arg_parser = ArgumentParser()
    add_data_arguments(arg_parser, hdf5_flag=True)
    # fmt: off
    arg_parser.add_argument(
        "--model", type=Path, required=True,
        help="Trained model to use. Either a .ckpt checkpoint file (loaded directly), "
             "or a directory containing a 'best_checkpoint.list' whose last line points "
             "to such a file.",
    )
    arg_parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Directory for the prediction CSV and mmap cache. Defaults to a "
             "'predictions' directory next to the resolved checkpoint file.",
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """Main called from command line."""
    begin = time_now()
    print(f"begin {begin}")

    prepare_inference_runtime()
    cli = parse_arguments()

    # Resolve the checkpoint once: drives both model loading and the default outdir.
    # A directory model is resolved via its best_checkpoint.list; a .ckpt file is used as-is.
    ckpt = resolve_checkpoint_spec(cli.model)
    outdir = Path(cli.outdir) if cli.outdir else ckpt.parent / "predictions"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    now = time_now()
    fmt = "chunked" if cli.chunked else "single"
    print(f"Loading data ({fmt} format): {now}")
    _, test_dataset, datasets = build_test_dataset(cli, outdir / "mmap_cache")
    print(f"Registered {datasets.test.num_examples} samples  ({time_now() - now})")

    # --- Restore model ---
    my_model = restore_from_checkpoint_file(LightningDenseClassifier, ckpt)
    print("Model successfully restored.")

    # --- Predictions (full softmax vector per sample) ---
    # Tag the output with the model's training provenance: the original training comet
    # experiment id (recovered from the checkpoint path) + checkpoint stem.
    tag = build_prediction_tag(experiment_id_from_checkpoint(ckpt), ckpt)
    model_id = tag or ckpt.stem
    predict_path = outdir / f"{model_id}_test_prediction_{cli.hdf5.stem}.csv"
    write_test_predictions(
        my_model, datasets, test_dataset, predict_path, batch_size=cli.batch_size
    )

    end = time_now()
    print(f"end {end}")
    print(f"Main() duration: {end - begin}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
