"""Like "main --predict mode", but for when no true label is available (cannot do certain analyses)."""
# pylint: disable=wrong-import-position, ungrouped-imports
import argparse
import os
import warnings
from pathlib import Path

warnings.simplefilter("ignore", category=FutureWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import pytorch_lightning as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import TensorDataset

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import analysis
from epiclass.core.data import DataSet, UnknownData
from epiclass.core.hdf5_loader import Hdf5Loader
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes."
        )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs."
    )
    arg_parser.add_argument(
        "--offline",
        action="store_true",
        help="Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.",
    )
    arg_parser.add_argument(
        "--model",
        type=DirectoryChecker(),
        help="Directory from which to load the desired model. Default is logdir.",
    )
    arg_parser.add_argument(
        "--hdf5_dir", type=Path, help="Directory in which to look for hdf5s, which will override hdf5 complete paths."
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params ---
    cli = parse_arguments()
    logdir = Path(cli.logdir)
    exp_name = "-".join(cli.logdir.parts[-2:])
    is_online = not cli.offline  # additional logging fails when offline

    # --- Startup LOGGER ---
    # api key in config file
    comet_logger = pl_loggers.CometLogger(
        project="EpiClass",
        name=exp_name,
        offline_directory=logdir,  # type: ignore
        online=is_online,
        auto_metric_logging=False,
    )
    exp_key = comet_logger.experiment.get_key()
    print(f"The current experiment key is {exp_key}")
    comet_logger.experiment.log_other("Experience key", f"{exp_key}")

    if "SLURM_JOB_ID" in os.environ:
        comet_logger.experiment.log_other("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        comet_logger.experiment.add_tag("Cluster")

    # --- LOAD DATA ---
    hdf5_loader = Hdf5Loader(chrom_file=cli.chromsize, normalization=True)

    if cli.hdf5_dir is not None:
        hdf5_loader.load_hdf5s(data_file=cli.hdf5, hdf5_dir=cli.hdf5_dir)
    else:
        hdf5_loader.load_hdf5s(data_file=cli.hdf5)
    files = hdf5_loader.signals

    md5s = []
    signals = []
    for md5, signal in files.items():
        md5s.append(md5)
        signals.append(signal)

    y = [0 for _ in md5s]
    y_str = ["No label available" for _ in y]

    test_set = UnknownData(ids=md5s, x=signals, y=y, y_str=y_str)

    if test_set.num_examples == 0:
        raise ValueError("Trying to test without any test data.")

    datasets = DataSet.empty_collection()
    datasets.set_test(test_set)

    test_dataset = TensorDataset(
        torch.from_numpy(test_set.signals).float(), torch.tensor(y, dtype=torch.int)
    )

    # --- RESTORE model ---
    model_dir = cli.logdir
    if cli.model is not None:
        model_dir = cli.model
    my_model = LightningDenseClassifier.restore_model(model_dir)
    print("Model successfully restored.")

    # --- OUTPUTS ---
    my_analyzer = analysis.Analysis(
        my_model,
        datasets,
        comet_logger,
        train_dataset=None,
        val_dataset=None,
        test_dataset=test_dataset,
    )

    # --- Create prediction file ---
    predict_path = (
        cli.logdir / f"{Path(model_dir).stem}_test_prediction_{cli.hdf5.stem}.csv"
    )
    my_analyzer.write_test_prediction(path=predict_path)

    end = time_now()
    main_time = end - begin
    print(f"end {end}")
    print(f"Main() duration: {main_time}")
    comet_logger.experiment.log_other("Main duration", main_time)
    comet_logger.experiment.add_tag("Finished")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
