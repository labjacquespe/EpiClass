"""Score new samples with a trained AVE anomaly-detection model.

The AVE counterpart of ``predict.py``. Where the classifier writes a predicted
class and per-class softmax probabilities, this writes one number per sample:
the mean squared reconstruction error under the model.

Deliberately no thresholding. Outlier *status* is a separate, later decision:
the data is too heterogeneous for one global cutoff, and calling outliers well
needs reference distributions (per assay, per cell type) and agreement across
fold models. This script only produces the raw scores those decisions are made
from.

Supports the same two input formats as ``predict.py``:
  - Single-sample HDF5 (default): one HDF5 file per sample. Requires --chromsize.
  - Chunked HDF5 (--chunked): pre-concatenated multi-sample HDF5 files.

Output: a CSV with one row per sample, columns ``ID`` and
``reconstruction_error``. ``ID`` matches the ``predict.py`` output so classifier
predictions and AVE scores join on sample id.
"""
# pylint: disable=wrong-import-position, ungrouped-imports
# Deliberately parallel to predict.py, its classifier counterpart.
# pylint: disable=duplicate-code
import argparse
import os
import warnings
from pathlib import Path
from typing import Sequence

import pandas as pd

warnings.simplefilter("ignore", category=FutureWarning)

import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
from torch.utils.data import Dataset

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.core.model_ave import LightningAVE
from epiclass.core.model_checkpoint import (
    resolve_checkpoint_spec,
    restore_from_checkpoint_file,
)
from epiclass.core.prediction_files import (
    build_prediction_tag,
    experiment_id_from_checkpoint,
)
from epiclass.mains.predict_common import (
    add_data_arguments,
    build_test_dataset,
    prepare_inference_runtime,
)
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    arg_parser = ArgumentParser()
    add_data_arguments(arg_parser, hdf5_flag=True)
    # fmt: off
    arg_parser.add_argument(
        "--model", type=Path, required=True,
        help="Trained AVE model. Either a .ckpt checkpoint file (loaded directly), "
             "or a directory containing a 'best_checkpoint.list' whose last line points "
             "to such a file.",
    )
    arg_parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Directory for the scores CSV and mmap cache. Defaults to a "
             "'predictions' directory next to the resolved checkpoint file.",
    )
    # fmt: on
    return arg_parser.parse_args()


def write_reconstruction_scores(
    model: LightningAVE,
    sample_ids: Sequence[str],
    dataset: Dataset,
    path: Path,
    batch_size: int = 256,
) -> Path:
    """Score ``dataset`` by reconstruction error and write the per-sample CSV.

    ``dataset`` is iterated unshuffled, so its index order matches ``sample_ids``.
    """
    errors = model.reconstruction_errors(dataset, batch_size=batch_size)
    if len(errors) != len(sample_ids):
        raise ValueError(
            f"Scored {len(errors)} samples but got {len(sample_ids)} sample ids."
        )
    scores = pd.DataFrame({"ID": list(sample_ids), "reconstruction_error": errors})
    scores.to_csv(path, index=False)
    print(f"Wrote {len(scores)} reconstruction errors to '{path}'")
    return path


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
    test_data, test_dataset, _ = build_test_dataset(cli, outdir / "mmap_cache")
    print(f"Registered {test_data.num_examples} samples  ({time_now() - now})")

    # --- Restore model ---
    my_model = restore_from_checkpoint_file(LightningAVE, ckpt)
    print("Model successfully restored.")

    # --- Scoring ---
    # Tag the output with the model's training provenance: the original training comet
    # experiment id (recovered from the checkpoint path) + checkpoint stem.
    tag = build_prediction_tag(experiment_id_from_checkpoint(ckpt), ckpt)
    model_id = tag or ckpt.stem
    write_reconstruction_scores(
        my_model,
        test_data.ids,
        test_dataset,
        outdir / f"{model_id}_ave_scores_{cli.hdf5.stem}.csv",
        batch_size=cli.batch_size,
    )

    end = time_now()
    print(f"end {end}")
    print(f"Main() duration: {end - begin}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
