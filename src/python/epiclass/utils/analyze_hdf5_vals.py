"""Make violing plots of the statistics of a set of hdf5 samples."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import plotly.express as px

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.data_source import EpiDataSource
from epiclass.core.epiatlas_treatment import ACCEPTED_TRACKS
from epiclass.core.loaders.hdf5_loader import Hdf5Loader
from epiclass.core.metadata import Metadata

ASSAY = "assay_epiclass"
TRACK_TYPE = "track_type"


def make_plots(
    stats_df: pd.DataFrame, metrics: Set, logdir: Path, name: str = ""
) -> None:
    """Create violin plots, one plot for each metric, and a violin for each assay (per plot)

    stats_df: A dataframe with descriptive statistics for each sample, and metadata. Indexed by md5sum.
    metrics: columns to plot from stats_df
    """
    category_orders = {ASSAY: sorted(stats_df[ASSAY].unique())}

    height = 1000
    for column in stats_df:
        if column not in metrics:
            continue
        fig = px.violin(
            data_frame=stats_df,
            x=column,
            y=ASSAY,
            box=True,
            points="all",
            title=f"Violin plot for {column}",
            color=ASSAY,
            category_orders=category_orders,
            height=height,
            width=height * 4.0 / 3.0,
            hover_data={"md5sum": (stats_df.index)},
        )
        if name:
            output_name = f"100kb_all_none_hdf5_{column}_{name}"
        else:
            output_name = f"100kb_all_none_hdf5_{column}"
        fig.write_image(logdir / (output_name + ".png"))
        fig.write_html(logdir / (output_name + ".html"))


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes."
        )
    arg_parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file."
        )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to create to output the analysis results.",
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """Main function."""
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    chromsize_path = cli.chromsize
    metadata_path = cli.metadata
    logdir = cli.output_dir

    datasource = EpiDataSource(hdf5_list_path, chromsize_path, metadata_path)
    my_meta = Metadata(datasource.metadata_file)

    # exclude samples that are not in ACCEPTED_TRACKS
    md5_per_track_type = my_meta.md5_per_class("track_type")
    md5_to_exclude = []
    for track_type, md5_list in md5_per_track_type.items():
        if track_type not in ACCEPTED_TRACKS:
            print(f"Excluding samples for track type: {track_type}")
            md5_to_exclude.extend(md5_list)

    md5s_to_analyze = set(Hdf5Loader.read_list(hdf5_list_path).keys())
    for md5 in md5_to_exclude:
        try:
            md5s_to_analyze.remove(md5)
            del my_meta[md5]
        except KeyError:
            continue

    # Check that we have metadata on all left samples.
    for md5 in md5s_to_analyze:
        if md5 not in my_meta:
            raise IndexError(f"Missing metadata for {md5}")

    # Create metadata dataframe
    df_md5_metadata = pd.DataFrame([my_meta[md5] for md5 in md5s_to_analyze])
    df_md5_metadata.set_index("md5sum", inplace=True)
    print(f"Analysis will be done into following files {df_md5_metadata.shape[0]} files:")
    print(df_md5_metadata[TRACK_TYPE].value_counts())
    print(df_md5_metadata[ASSAY].value_counts())

    # Load the hdf5 files into a datafram
    hdf5_loader = Hdf5Loader(chrom_file=chromsize_path, normalization=True)
    signals = hdf5_loader.load_hdf5s(hdf5_list_path, md5s_to_analyze, strict=True).signals
    df = pd.DataFrame.from_dict(signals, orient="index")

    # Compute descriptive statistics
    percentiles = [0.01] + list(np.arange(0.05, 1, 0.05)) + [0.99] + [0.999]
    stats_df = df.apply(pd.DataFrame.describe, percentiles=percentiles, axis=1)  # type: ignore
    del df
    del signals

    # Choose which metrics to plot
    metrics = set(stats_df.columns.values)
    allowed_metrics = metrics - set(["count", "mean", "std"])

    # Add metadata to the stats dataframe
    stats_df = stats_df.join(df_md5_metadata)

    # Save results plots.
    make_plots(stats_df, allowed_metrics, logdir, name="all")

    # Separate raw and relative tracks, and make plots for each.
    not_raw = stats_df[TRACK_TYPE].isin(["fc", "pval"])
    make_plots(stats_df[not_raw], allowed_metrics, logdir, name="relative")
    make_plots(stats_df[~not_raw], allowed_metrics, logdir, name="raw")


if __name__ == "__main__":
    main()
