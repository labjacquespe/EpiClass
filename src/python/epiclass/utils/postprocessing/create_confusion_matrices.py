"""
This Python script is intended to merge or create confusion matrices.
It uses the `epiclass` library's `ConfusionMatrixWriter` for tasks related to confusion matrices.

Command line arguments.
    It expects '--from_existing' to combine CSV matrices or '--from_prediction'
    to convert a classification prediction file into a confusion matrix.

Note: The prediction file used by the "from_prediction" option must
at least have 'True class' and 'Predicted class' headers to generate the confusion matrix.
Other columns, if present, will be ignored.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sklearn.metrics

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.confusion_matrix import ConfusionMatrixWriter


def parse_arguments() -> argparse.Namespace:
    """Return argument line parser."""
    parser = ArgumentParser()
    # fmt: off
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from_existing", metavar="logdir", type=DirectoryChecker(), help="Location of csv matrix logdir to combine.",
    )
    group.add_argument(
        "--from_prediction", metavar="pred-file", type=Path, help="Prediction file to convert to a confusion matrix.",
    )
    parser.add_argument("--confidence_threshold", type=float, default=0, help="Confidence threshold for predictions. Only applies to --from_prediction. Must be within [0, 1].")
    # fmt: on
    return parser.parse_args()


def add_matrices(logdir: str):
    """Add several matrices together from folds."""
    gen = logdir + "/split{i}/validation_confusion_matrix.csv"

    mat = ConfusionMatrixWriter.from_csv(csv_path=gen.format(i=0), relative=False)
    for i in range(1, 10):
        csv_path = Path(gen.format(i=i))
        if csv_path.exists():
            mat2 = ConfusionMatrixWriter.from_csv(csv_path=csv_path, relative=False)
            mat = mat + mat2
        else:
            print(f"File does not exist: {csv_path}")

    mat.to_all_formats(logdir=logdir, name="full-10fold-validation")


def main():
    """Augment a label prediction file with new metadata categories.

    File header format important. Expects [md5sum, true class, predicted class, labels] lines.
    """
    args = parse_arguments()

    if args.from_existing:
        add_matrices(logdir=args.from_logdir.parent)
        return

    pred_file = args.from_prediction
    logdir = pred_file.parent

    threshold = args.confidence_threshold
    if 0 > threshold > 1:
        raise ValueError("Confidence threshold must be within [0, 1].")

    if threshold > 0:
        confidence_label = "Max pred"
        try:
            df = pd.read_csv(
                pred_file,
                sep=",",
                usecols=["True class", "Predicted class", confidence_label],
            )
        except KeyError as exc:
            raise KeyError(
                f"Prediction file must have 'True class', 'Predicted class', and '{confidence_label}' columns."
            ) from exc

        df = df[df[confidence_label] >= threshold]
    else:
        df = pd.read_csv(
            pred_file,
            sep=",",
            usecols=["True class", "Predicted class"],
        )

    true, pred = df.iloc[:, 0], df.iloc[:, 1]
    labels = sorted(set(true.unique().tolist() + pred.unique().tolist()))
    confusion_mat = sklearn.metrics.confusion_matrix(true, pred, labels=labels)

    writer = ConfusionMatrixWriter(labels=labels, confusion_matrix=confusion_mat)
    writer.to_all_formats(
        logdir=logdir,
        name=str(pred_file.stem) + f"-confusion-matrix-threshold-{threshold:.2f}",
    )


if __name__ == "__main__":
    main()
