"""Score new samples with the full ensemble of AVE fold models from a CV run.

The AVE counterpart of ``predict_CV.py``: loads **every** fold model of an
``ave_training`` / ``ave_general_training`` run (one per ``splitX`` / ``fold_X``
directory) and scores the same input data with each. The data is materialized
once and reused across folds.

Outputs, written to ``<cv_root>/ave_predictionsCV/`` by default:
  - one CSV per fold, ``<fold>_<checkpoint-tag>_ave_scores_<hdf5>.csv``;
  - a concatenated CSV (``concatenated_ave_scores_<hdf5>.csv``) stacking every fold's
    rows with an added ``origin`` column naming the per-fold source file;
  - a per-sample summary (``ave_scores_summary_<hdf5>.csv``) with one error column per
    fold plus ``mean`` and ``std``.

The summary is the useful artifact downstream: outlier status is decided later, and
that decision needs the spread across folds — a high error under one fold model is
weaker evidence than a high error under all of them. Nothing here thresholds anything.
"""
# pylint: disable=wrong-import-position, ungrouped-imports
# Deliberately parallel to predict_CV.py, its classifier counterpart.
# pylint: disable=duplicate-code
import argparse
import os
import warnings
from pathlib import Path
from typing import List

import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)

import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.core.model_ave import LightningAVE
from epiclass.core.model_checkpoint import last_checkpoint_path
from epiclass.core.prediction_files import (
    SPLIT_DIR_GLOBS,
    build_prediction_tag,
    experiment_id_from_checkpoint,
)
from epiclass.mains.ave_predict import write_reconstruction_scores
from epiclass.mains.predict_common import (
    DirectoryChecker,
    add_data_arguments,
    build_test_dataset,
    prepare_inference_runtime,
)
from epiclass.mains.predict_CV import discover_fold_dirs, write_concatenated
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
        "--output_dir", type=Path, default=None,
        help="Where to write the per-fold, concatenated and summary CSVs. "
             "Defaults to <cv_root>/ave_predictionsCV.",
    )
    # fmt: on
    return arg_parser.parse_args()


def write_summary(per_fold_csvs: List[Path], output_dir: Path, hdf5_stem: str) -> Path:
    """Write one row per sample: an error column per fold, plus ``mean`` and ``std``.

    Fold columns are named after their source CSV's fold directory prefix, so the
    provenance stays readable next to the aggregates.
    """
    merged = pd.DataFrame()
    for csv in per_fold_csvs:
        df = pd.read_csv(csv, index_col="ID")
        # "<fold>_<tag>_ave_scores_<hdf5>.csv" -> "<fold>"
        fold_name = csv.name.split("_", maxsplit=1)[0]
        merged[fold_name] = df["reconstruction_error"]

    fold_cols = list(merged.columns)
    merged["mean"] = merged[fold_cols].mean(axis=1)
    merged["std"] = merged[fold_cols].std(axis=1)

    summary_path = output_dir / f"ave_scores_summary_{hdf5_stem}.csv"
    merged.to_csv(summary_path, index_label="ID")
    print(f"Wrote per-sample summary ({len(merged)} rows) to '{summary_path}'")
    return summary_path


def main():
    """Main called from command line."""
    begin = time_now()
    print(f"begin {begin}")

    prepare_inference_runtime()
    cli = parse_arguments()
    cv_root = Path(cli.cv_root)
    output_dir = (
        cli.output_dir if cli.output_dir is not None else cv_root / "ave_predictionsCV"
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
    test_data, test_dataset, _ = build_test_dataset(cli, output_dir / "mmap_cache")
    print(f"Registered {test_data.num_examples} samples  ({time_now() - now})")

    # --- Score with each fold model ---
    per_fold_csvs: List[Path] = []
    for fold_dir in fold_dirs:
        # Tag each per-fold file with that fold model's *training* provenance: the original
        # training comet experiment id (from the checkpoint path) + checkpoint stem.
        ckpt = last_checkpoint_path(fold_dir)
        tag = build_prediction_tag(experiment_id_from_checkpoint(ckpt), ckpt)
        tag = tag or fold_dir.name
        print(f"\n--- Fold '{fold_dir.name}' (tag '{tag}') ---")

        my_model = LightningAVE.restore_model(fold_dir)
        out_path = output_dir / f"{fold_dir.name}_{tag}_ave_scores_{cli.hdf5.stem}.csv"
        write_reconstruction_scores(
            my_model,
            test_data.ids,
            test_dataset,
            out_path,
            batch_size=cli.batch_size,
        )
        per_fold_csvs.append(out_path)

    # --- Aggregates ---
    write_concatenated(
        per_fold_csvs,
        output_dir,
        cli.hdf5.stem,
        name_prefix="concatenated_ave_scores",
    )
    write_summary(per_fold_csvs, output_dir, cli.hdf5.stem)

    end = time_now()
    print(f"\nend {end}")
    print(f"Main() duration: {end - begin}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
