"""Transfer bedgraph values to new positions based on a reference bed file."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the script.

    Returns:
        argparse.Namespace: Namespace object containing the arguments.
    """
    # fmt: off
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "bg_list", type=Path, help="A file with bedgraph paths."
        )
    arg_parser.add_argument(
        "ref_bed", type=Path, help="Reference bed file where bedgraph values are mapped.",
    )
    arg_parser.add_argument(
        "output_dir", type=Path, help="Directory where the output files will be saved."
    )
    # fmt: on
    return arg_parser.parse_args()


def process_bedgraph(bg_path: Path, ref_bed: pd.DataFrame, output_dir: Path) -> None:
    """Process a single bedgraph file to transfer its values to new positions.

    Args:
        bg_path (Path): Path to the bedgraph file.
        ref_bed (pd.DataFrame): Reference bed dataframe to which values are mapped.
        output_dir (Path): Directory where the output file will be saved.
    """
    ref_bed = ref_bed.copy()
    bedgraph_reg = pd.read_csv(bg_path, sep="\t", header=None)
    max_bins = ref_bed.shape[0]

    # take last column as value, transfer to new positions
    ref_bed["value"] = bedgraph_reg.iloc[0:max_bins, -1]

    output_file = output_dir / bg_path.name
    ref_bed.to_csv(
        output_file,
        sep="\t",
        header=False,
        index=False,
        columns=["chrom", "start", "end", "value"],
    )


def main() -> None:
    """Main function to parse arguments and execute parallel processing of bedgraph files."""
    cli = parse_arguments()

    # Check input file and output directories
    if not cli.bg_list.exists():
        raise FileNotFoundError(f"Reference bed file {cli.bg_list} does not exist.")
    if not cli.ref_bed.exists():
        raise FileNotFoundError(f"Reference bed file {cli.ref_bed} does not exist.")
    if not cli.output_dir.exists():
        raise FileNotFoundError(f"Output directory {cli.output_dir} does not exist.")

    ref_bed = pd.read_csv(
        cli.ref_bed, sep="\t", header=None, names=["chrom", "start", "end"]
    )
    bg_list = pd.read_csv(cli.bg_list, sep="\t", header=None, names=["path"])

    nb_core = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    print(f"Using {nb_core} cores for parallel processing.")
    with ProcessPoolExecutor(nb_core) as executor:
        for bg_path in bg_list["path"]:
            executor.submit(process_bedgraph, Path(bg_path), ref_bed, cli.output_dir)


if __name__ == "__main__":
    main()
