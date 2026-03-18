"""Merge predictions files produced from epilap program.
Tightly linked with the output of augment_predict_file.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser(
        description="Merge predictions files produced from epilap program. Probabilities only kept for first given file."
    )
    arg_parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="The predictions files to merge. Should have same columns.",
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        nargs="?",
        help="Give a path for the output file.",
        default="all-predictions-merged.csv",
    )
    return arg_parser.parse_args()


def main():
    """Main"""
    cli = parse_arguments()

    dfs = {file.stem: pd.read_csv(file, sep=",") for file in cli.files}

    first_df = list(dfs.values())[0]
    df_shape = first_df.shape
    column_names = first_df.columns
    for name, df in dfs.items():
        if df.shape != df_shape:
            raise AssertionError(
                f"Expected shape {df_shape}, got {df.shape}. File named {name} does not contain same csv shape as first file."
            )
        if not df.columns.equals(column_names):
            for col1, col2 in zip(df.columns.to_list(), column_names.to_list()):
                print(col1, col2, file=sys.stderr)
                if col1 != col2:
                    print("-- PROBLEMATIC COLUMNS DIRECTLY ABOVE --", file=sys.stderr)
            raise AssertionError(
                f"Headers are not identical. {name} differs from first file. See stderr for all columns."
            )

    column_names = pd.Index(column_names)
    join_pos = column_names.get_loc("True class")
    col1 = "1rst/2nd prob ratio"
    col2 = "files/epiRR"
    cut_pos_1 = column_names.get_loc(col1)
    cut_pos_2 = column_names.get_loc(col2)
    print(
        f"Will remove content between '{col1}' and '{col2}' columns except for first file."
    )

    new_df = first_df
    for name, df in dfs.items():
        df = df.drop(df.columns[cut_pos_1 + 1 : cut_pos_2], axis=1)
        new_df = new_df.merge(
            df,
            on=list(column_names)[0 : join_pos + 1],
            suffixes=(None, f"-{name}"),
            validate="1:1",
        )

    print("New file written to {cli.output.resolve()}.")
    new_df.to_csv(cli.output.resolve(), sep=",", index=False)


if __name__ == "__main__":
    main()
