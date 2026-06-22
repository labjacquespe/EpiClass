"""Predict unlabeled samples with the full ensemble of a cross-validation run.

Where ``predict.py`` loads one model, this loads **every** fold model of a CV run (one per
``splitX`` / ``fold_X`` directory, as produced by ``epiatlas_training.py`` /
``general_training.py``) and scores the same input data with each. The data is materialized
once and reused across folds.

Outputs, written to ``<cv_root>/predictionsCV/`` by default:
  - one prediction CSV per fold, named ``<fold>_<checkpoint-stem>_test_prediction_<hdf5>.csv``;
  - a concatenated CSV (``concatenated_test_prediction_<hdf5>.csv``) stacking every fold's
    rows with an added ``origin`` column naming the per-fold source file.

These per-fold predictions feed downstream conformal prediction during deployment.
"""
# pylint: disable=wrong-import-position, ungrouped-imports
import argparse
import os
import warnings
from pathlib import Path
from typing import List

import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)

import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.core.model_checkpoint import last_checkpoint_path
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.prediction_files import (
    SPLIT_DIR_GLOBS,
    build_prediction_tag,
    experiment_id_from_checkpoint,
)
from epiclass.predict_common import (
    DirectoryChecker,
    add_data_arguments,
    build_test_dataset,
    prepare_inference_runtime,
    write_test_predictions,
)
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    arg_parser = ArgumentParser()
    add_data_arguments(arg_parser)
    # fmt: off
    arg_parser.add_argument(
        "cv_root", type=DirectoryChecker(),
        help="CV run directory holding split*/ (or fold_*/) fold sub-directories.",
    )
    arg_parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write the per-fold and concatenated CSVs. "
             "Defaults to <cv_root>/predictionsCV.",
    )
    # fmt: on
    return arg_parser.parse_args()


def discover_fold_dirs(cv_root: Path) -> List[Path]:
    """Return the fold directories of a CV run, in order.

    A fold directory is a ``split*`` / ``fold_*`` sub-directory that holds a
    ``best_checkpoint.list`` (so output folders like ``predictionsCV`` / ``mmap_cache`` are
    skipped). Prefers ``split*`` and falls back to ``fold_*`` so the two are never mixed.
    """
    for glob in SPLIT_DIR_GLOBS:
        fold_dirs = [
            d
            for d in sorted(cv_root.glob(glob))
            if d.is_dir() and last_checkpoint_path(d) is not None
        ]
        if fold_dirs:
            return fold_dirs
    return []


def main():
    """Main called from command line."""
    begin = time_now()
    print(f"begin {begin}")

    prepare_inference_runtime()
    cli = parse_arguments()
    cv_root = Path(cli.cv_root)
    output_dir = (
        cli.output_dir if cli.output_dir is not None else cv_root / "predictionsCV"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = discover_fold_dirs(cv_root)
    if not fold_dirs:
        raise ValueError(
            f"No fold directories with a 'best_checkpoint.list' found under '{cv_root}' "
            f"(looked for {' / '.join(SPLIT_DIR_GLOBS)})."
        )
    print(f"Found {len(fold_dirs)} fold models: {[d.name for d in fold_dirs]}")

    # --- Load data once, reused by every fold model ---
    now = time_now()
    fmt = "chunked" if cli.chunked else "single"
    print(f"Loading data ({fmt} format): {now}")
    _, test_dataset, datasets = build_test_dataset(cli, output_dir / "mmap_cache")
    print(f"Registered {datasets.test.num_examples} samples  ({time_now() - now})")

    # --- Predict with each fold model ---
    per_fold_csvs: List[Path] = []
    for fold_dir in fold_dirs:
        # Tag each per-fold file with that fold model's *training* provenance: the original
        # training comet experiment id (from the checkpoint path) + checkpoint stem.
        ckpt = last_checkpoint_path(fold_dir)
        tag = build_prediction_tag(experiment_id_from_checkpoint(ckpt), ckpt)
        tag = tag or fold_dir.name
        print(f"\n--- Fold '{fold_dir.name}' (tag '{tag}') ---")

        my_model = LightningDenseClassifier.restore_model(fold_dir)
        out_path = (
            output_dir / f"{fold_dir.name}_{tag}_test_prediction_{cli.hdf5.stem}.csv"
        )
        write_test_predictions(
            my_model, datasets, test_dataset, out_path, batch_size=cli.batch_size
        )
        per_fold_csvs.append(out_path)

    # --- Concatenated long-format CSV with provenance ---
    write_concatenated(per_fold_csvs, output_dir, cli.hdf5.stem)

    end = time_now()
    print(f"\nend {end}")
    print(f"Main() duration: {end - begin}")


def write_concatenated(
    per_fold_csvs: List[Path], output_dir: Path, hdf5_stem: str
) -> Path:
    """Stack the per-fold prediction CSVs, tagging each row with its source filename.

    Columns align by name across folds (every fold model shares the same class set). The
    added ``origin`` column carries the per-fold CSV filename so each prediction is traceable
    to the model that produced it.
    """
    frames = []
    for csv in per_fold_csvs:
        df = pd.read_csv(csv)
        df.insert(0, "origin", csv.name)
        frames.append(df)
    concat = pd.concat(frames, ignore_index=True)
    concat_path = output_dir / f"concatenated_test_prediction_{hdf5_stem}.csv"
    concat.to_csv(concat_path, index=False)
    print(f"Wrote concatenated predictions ({len(concat)} rows) to '{concat_path}'")
    return concat_path


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
