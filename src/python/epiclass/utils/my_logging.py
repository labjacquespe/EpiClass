"""Implement logging of specific data through comet logger."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict

from pytorch_lightning import loggers as pl_loggers

from epiclass.core.data.dataset import DataSet
from epiclass.utils.general_utility import write_signal_ids_to_file
from epiclass.utils.time import time_now_str


def log_pre_training(
    logger: pl_loggers.CometLogger, to_log: Dict[str, str], step: int | None
):
    """Log a bunch of stuff in comet logger. Return experience key (str).

    to_log expects keys:
    - category
    - hdf5_resolution
    - loading_time (initial, for hdf5)
    When step is an int:
    - split_time (generator yield time)
    """
    # General stuff
    category = to_log.get("category")
    logger.experiment.add_tag(category)
    logger.experiment.log_other("category", category)

    logger.experiment.log_other(
        "HDF5 Resolution", f"{int(to_log['hdf5_resolution'])/1000}kb"
    )

    # job id when done on HPC
    if os.getenv("SLURM_JOB_ID") is not None:
        logger.experiment.log_other("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        logger.experiment.add_tag("Cluster")

    # Code time stuff
    loading_time = to_log.get("loading_time")
    print(f"Initial hdf5 loading time: {loading_time}")
    logger.experiment.log_other("Initial hdf5 loading time", loading_time)

    if step is not None:
        split_time = to_log.get("split_time")
        print(f"Set loading/splitting time: {split_time}")
        logger.experiment.log_metric("Split_time", split_time, step=step)

    # exp id
    exp_key = logger.experiment.get_key()
    print(f"The current experiment key is {exp_key}")
    logger.experiment.log_other("Experience key", str(exp_key))

    # Save git commit. Pass cwd= to subprocess instead of os.chdir so we don't
    # leak a cwd change into downstream code (LazyHdf5Loader's default mmap_dir
    # is a relative path and would silently move after a chdir).
    commit_label = subprocess.check_output(
        ["git", "describe"],
        cwd=Path(__file__).resolve().parent,
        encoding="UTF-8",
    ).strip()
    print(f"The current commit is {commit_label}")
    logger.experiment.log_other("Code version / commit", f"{commit_label}")


def log_dset_composition(
    my_data: DataSet,
    logdir: str | Path | None,
    logger: pl_loggers.CometLogger | None,
    split_nb: int,
):
    """Logs dataset composition to logger and file.

    Either the 'logdir' or 'logger' parameter must be provided, otherwise ValueError is raised.

    This function logs
    - training/validation set size
    - total number of unique files (training + validation),
    - training/valid unique signal ID composition (to a file)

    Args:
        my_data (DataSet): The dataset to log.
        logdir (str | Path | None): The directory where logs will be stored. None if logger is given.
        logger (pl_loggers.CometLogger | None): The logger object to use for logging. None if logdir is given.
        split_nb (int): The split number to log.

    Raises:
        ValueError if:
        - both 'logdir' and 'logger' parameters are None
        - train and validation sets overlap.
    """
    if logger:
        save_dir = logger.save_dir
    elif logdir:
        save_dir = logdir
    else:
        raise ValueError("No logdir or logger given.")

    for dset_name in ["train", "validation", "test"]:
        size = getattr(my_data, dset_name).num_examples
        if logger:
            logger.experiment.log_other(f"{dset_name} size", size)
        print(f"Split {split_nb} {dset_name} size: {size}")

    ids_train = set(my_data.train.ids.tolist())
    ids_valid = set(my_data.validation.ids.tolist())
    if ids_train & ids_valid:
        raise ValueError("Train and validation sets overlap")

    nb_files = len(ids_train) + len(ids_valid)
    if logger:
        logger.experiment.log_other("Total nb of files", nb_files)
    if split_nb == 0:
        print(f"Total nb of files: {nb_files}")

    # Save exactly which files are used for training and validation
    for ids, name in zip([ids_train, ids_valid], ["training", "validation"]):
        if len(ids) > 0:
            name = f"split{split_nb}_{name}_{time_now_str()}"
            write_signal_ids_to_file(sorted(ids), save_dir, name)  # type: ignore
