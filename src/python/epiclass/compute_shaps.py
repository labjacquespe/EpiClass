"""Compute SHAP values of a model."""
# pylint: disable=import-error, line-too-long
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*nopython.*")

try:
    import shap  # pylint: disable=unused-import
except ImportError as e:
    raise ImportError(
        "SHAP computation requires the `shap` package. "
        "Install with: pip install .[shap]"
    ) from e

import numpy as np

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core.lazy.lazy_data_classes import LazyUnknownData
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.shap_values import NN_SHAP_Handler


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Namespace object with parsed arguments.
    """
    arg_parser = ArgumentParser(
        description="Compute SHAP values for a trained neural network. "
        "Requires the `shap` package: pip install .[shap]"
    )

    # fmt: off
    arg_parser.add_argument(
        "--background_hdf5", required=True, metavar="background-hdf5", type=Path, help="A file with hdf5 filenames for the explainer background. Use absolute path!"
    )
    arg_parser.add_argument(
        "--explain_hdf5", required=True, metavar="explain-hdf5", type=Path, help="A file with hdf5 filenames on which to compute SHAP values. Use absolute path!",
    )
    arg_parser.add_argument(
        "--chromsize", required=True, type=Path, help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "--model_dir", required=True, type=DirectoryChecker(), help="Model directory containing 'best_checkpoint.list'.",
    )
    arg_parser.add_argument(
        "-l", "--logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    arg_parser.add_argument(
        "-o", "--output_name", metavar="--output-name", default="", help="Name (not path) of outputted pickle file containing computed SHAP values",
    )
    # fmt: on
    return arg_parser.parse_args()


def _load_lazy(hdf5_list: Path, chromsize: Path, mmap_dir: Path) -> LazyUnknownData:
    """Register and preload HDF5 files into a LazyUnknownData for SHAP."""
    loader = LazyHdf5Loader(chrom_file=chromsize, normalization=True, mmap_dir=mmap_dir)
    loader.register_hdf5s(hdf5_list, strict=True)
    loader.preload_all()
    sample_ids = list(loader.file_paths.keys())
    n = len(sample_ids)
    return LazyUnknownData(
        ids=sample_ids,
        loader=loader,
        y=np.zeros(n, dtype=np.int64),
        y_str=[""] * n,
    )


def compute_shap(
    cli: argparse.Namespace,
    shap_computer,
    output_name: str,
):
    """Compute SHAP values for the given NN handler."""
    base_mmap = Path("./mmap_cache")
    background_set = _load_lazy(
        cli.background_hdf5, cli.chromsize, base_mmap / "background"
    )
    explain_set = _load_lazy(cli.explain_hdf5, cli.chromsize, base_mmap / "explain")

    shap_computer.compute_shaps(
        background_dset=background_set,
        evaluation_dset=explain_set,
        save=True,
        name=output_name,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", "1")),
    )


def main():
    """main"""
    cli = parse_arguments()

    logdir = cli.logdir if cli.logdir is not None else Path.cwd()

    my_model = LightningDenseClassifier.restore_model(cli.model_dir)
    shap_handler = NN_SHAP_Handler(model=my_model, logdir=logdir)
    compute_shap(cli, shap_handler, cli.output_name)


if __name__ == "__main__":
    main()
