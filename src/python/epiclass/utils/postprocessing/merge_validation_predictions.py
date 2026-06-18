"""Merge validation prediction files from splitX folders."""
import argparse
from pathlib import Path

import pandas as pd

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.prediction_files import resolve_split_prediction_csvs
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory where the split results directories are."
    )
    arg_parser.add_argument(
        "-n", "--nfold", required=True, type=int, help="Number of validations folds to merge."
    )
    arg_parser.add_argument(
        "-o", "--output", type=Path, help="Output path", default=None
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params ---
    cli = parse_arguments()

    logdir = cli.logdir
    nfold = cli.nfold

    output_path = logdir / f"full-{nfold}fold-validation_prediction.csv"
    if cli.output:
        output_path = cli.output
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Folder does not exist: {output_path.parent}")

    # Resolve one prediction CSV per fold (newest tagged file wins) so re-runs that share a
    # split folder don't produce duplicate or stale matches.
    pred_files = resolve_split_prediction_csvs(logdir, "validation")
    if len(pred_files) != nfold:
        raise ValueError(f"{len(pred_files)} predictions files found. {nfold} expected.")

    dfs = []
    for split_name, file in pred_files.items():
        split_nb = int(str(split_name)[-1])
        if split_nb not in range(0, nfold):
            raise ValueError(f"Unexpected split number: {split_nb}")
        df = pd.read_csv(file, index_col=0, header=0)

        # Insert the split number after true/predicted class
        df.insert(loc=2, column="split_nb", value=split_nb)
        dfs.append(df)

    full_df = pd.concat(dfs)

    print(f"Writing merged results to {output_path}.")
    full_df.to_csv(output_path, sep=",")


if __name__ == "__main__":
    main()
