"""Collect values from hdf5 files and save them as csv and/or hdf5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.loaders.hdf5_loader import Hdf5Loader


def module_exists(module_name):
    """Check if a module exists."""
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser(
        prog="collect_hdf5_vals",
        description="Collect values from hdf5 files and save them as csv and/or hdf5.",
    )
    # fmt: off
    arg_parser.add_argument(
        "hdf5_list",
        type=Path,
        help="A file with hdf5 filenames. Filenames in the list must use absolute path!",
    )
    arg_parser.add_argument(
        "chromsize",
        type=Path,
        help="A file with chrom sizes.",
        )
    arg_parser.add_argument(
        "--feature_list",
        type=Path,
        help="A file with feature bin indexes in json format.",
        default=None,
    )
    arg_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the values in the hdf5 files as z-scores.",
    )
    arg_parser.add_argument(
        "--hdf",
        action="store_true",
        help="Save the values as hdf5 file. pytables must be installed.",
    )
    arg_parser.add_argument(
        "--csv",
        action="store_true",
        help="Save the values as csv file.",
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to save the hdf5 values.",
    )
    # fmt: on
    return arg_parser.parse_args()


def main() -> None:
    """Main function to collect values from hdf5 files and save them as csv."""
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    chromsize_path = cli.chromsize
    feature_list_path = cli.feature_list

    normalize_hdf5 = cli.normalize
    save_hdf = cli.hdf
    save_csv = cli.csv

    if not save_hdf and not save_csv:
        raise ValueError("Must provide at least one of --hdf or --csv")

    if save_hdf:
        if not module_exists("tables"):
            raise ImportError(
                "pytables must be installed to save hdf5 files. Try: `pip install tables`."
            )

    output_dir = cli.output_dir

    if feature_list_path:
        feature_list_name = feature_list_path.stem
        with open(feature_list_path, "r", encoding="utf8") as f:
            feature_list = json.load(f)
    else:
        feature_list_name = "all"
        feature_list = []

    hdf5_loader = Hdf5Loader(chrom_file=chromsize_path, normalization=normalize_hdf5)
    hdf5_loader.load_hdf5s(data_file=hdf5_list_path, verbose=False, strict=True)

    selected_values = {}
    for md5sum, signal in hdf5_loader.signals.items():
        if feature_list:
            feature_values = signal[feature_list]
        else:
            feature_values = signal  # Get all data

        selected_values[md5sum] = feature_values

    df_columns = feature_list if feature_list else None

    df = pd.DataFrame.from_dict(selected_values, orient="index", columns=df_columns)

    output_name = f"hdf5_values_{hdf5_list_path.stem}_{feature_list_name}"
    if save_hdf:
        df.to_hdf(
            output_dir / f"{output_name}.h5",
            key="df",
            mode="w",
            complevel=9,
            format="fixed",
            index=True,
        )

    if save_csv:
        df.to_csv(
            output_dir / f"{output_name}.csv",
            index=True,
        )


if __name__ == "__main__":
    main()
