"""General-purpose training script with stratified k-fold cross-validation.

Does not require UUID-based metadata (unlike epiatlas_training.py).
Designed for datasets like saccer3 where samples are independent.
"""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from lightning.pytorch import loggers as pl_loggers

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import analysis
from epiclass.core.blas_guard import check_blas_backend
from epiclass.core.data.dataset import DataSet
from epiclass.core.data_source import EpiDataSource
from epiclass.core.lazy.general_fold_factory import GeneralFoldFactory
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.trainer import MyTrainer, define_callbacks
from epiclass.utils.check_dir import create_dirs
from epiclass.utils.my_logging import log_dset_composition
from epiclass.utils.time import time_now
from epiclass.utils.torch_data import create_torch_datasets

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Argument parser for general training."""
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "category", type=str, help="The metadata category to classify (e.g. assay).",
    )
    arg_parser.add_argument(
        "hyperparameters", type=Path, help="JSON file with model hyperparameters.",
    )
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="Text file containing absolute paths to HDF5 files.",
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="Chromosome sizes file.",
    )
    arg_parser.add_argument(
        "metadata", type=Path, help="Metadata JSON file.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Output directory.",
    )
    arg_parser.add_argument(
        "--n_fold", type=int, default=4, help="Number of CV folds (default: 4).",
    )
    arg_parser.add_argument(
        "--hl_units", type=int, default=1000, help="Hidden layer units (default: 1000).",
    )
    arg_parser.add_argument(
        "--nb_layer", type=int, default=1, help="Number of hidden layers (default: 1).",
    )
    arg_parser.add_argument(
        "--min_class_size", type=int, default=10, help="Min samples per class (default: 10).",
    )
    arg_parser.add_argument(
        "--mmap_dir", type=Path, default=None,
        help="Directory for the HDF5 mmap cache (default: <logdir>/mmap_cache). "
             "On HPC set to $SLURM_TMPDIR for fast local-disk writes.",
    )
    arg_parser.add_argument(
        "--folds", type=Path, default=None,
        help='JSON file of explicit fold membership: '
             '{"fold1": {"md5sum": [ids...]}, ...}. The inner key names a '
             'metadata category matched 1:1 to samples (each value must resolve '
             'to exactly one signal). Overrides --n_fold, --min_class_size and '
             'oversampling.',
    )
    # fmt: on
    return arg_parser.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def do_one_experiment(
    split_nb: int,
    my_data: DataSet,
    hparams: Dict[str, Any],
    logdir: Path,
    hl_units: int,
    nb_layers: int,
) -> Path:
    """Train one fold. Returns the model directory path."""
    begin_loop = time_now()

    fold_dir = logdir / f"fold_{split_nb}"
    create_dirs(fold_dir)

    logger = pl_loggers.CSVLogger(save_dir=str(fold_dir), name="logs")

    log_dset_composition(my_data, logdir=fold_dir, logger=None, split_nb=split_nb)

    dsets_dict = create_torch_datasets(
        data=my_data,
        batch_size=hparams.get("batch_size", 64),
    )
    train_dataset, train_dataloader = dsets_dict["training"]
    valid_dataset, valid_dataloader = dsets_dict["validation"]

    if my_data.train.num_examples == 0 or my_data.validation.num_examples == 0:
        raise ValueError("Empty training or validation set.")

    # Save mapping
    mapping_file = fold_dir / "training_mapping.tsv"
    my_data.save_mapping(mapping_file)
    mapping = my_data.load_mapping(mapping_file)

    # Model
    input_size = my_data.train.signal_length
    output_size = len(my_data.classes)

    my_model = LightningDenseClassifier(
        input_size=input_size,
        output_size=output_size,
        mapping=mapping,
        hparams=hparams,
        hl_units=hl_units,
        nb_layer=nb_layers,
    )

    if split_nb == 0:
        print("--MODEL STRUCTURE--\n", my_model)
        my_model.print_model_summary()

    # Trainer
    gpu_available = torch.cuda.device_count() > 0
    show_bar = not gpu_available

    callbacks = define_callbacks(
        early_stop_limit=hparams.get("early_stop_limit", 20),
        show_summary=(split_nb == 0),
        show_progress_bar=show_bar,
    )

    trainer_kwargs = {
        "general_log_dir": str(fold_dir),
        "model": my_model,
        "max_epochs": hparams.get("training_epochs", 50),
        "check_val_every_n_epoch": hparams.get("measure_frequency", 1),
        "logger": logger,
        "callbacks": callbacks,
        "accelerator": "gpu" if gpu_available else "cpu",
        "devices": 1,
    }
    if gpu_available:
        trainer_kwargs["precision"] = 16

    trainer = MyTrainer(**trainer_kwargs)

    if split_nb == 0:
        trainer.print_hyperparameters()

    trainer.fit(
        my_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        verbose=(split_nb == 0),
    )
    trainer.save_model_path()

    # Restore best model for analysis
    try:
        my_model = LightningDenseClassifier.restore_model(fold_dir)
    except (FileNotFoundError, OSError) as e:
        print(f"Could not restore model for fold {split_nb}: {e}")
        return fold_dir

    # Analysis (logger=None: metrics printed, assets written to disk only)
    my_analyzer = analysis.Analysis(
        my_model,
        my_data,
        logger=None,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        test_dataset=None,
        save_dir=fold_dir,
    )
    my_analyzer.get_training_metrics(verbose=True)
    my_analyzer.get_validation_metrics(verbose=True)
    my_analyzer.write_validation_prediction()
    my_analyzer.validation_confusion_matrix()

    loop_time = time_now() - begin_loop
    print(f"Fold {split_nb} time: {loop_time}")

    return fold_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """General training with stratified k-fold cross-validation."""
    begin = time_now()
    print(f"begin {begin}")

    # Fail fast on a BLIS <1.1 CPU backend before wasting GPU hours: the forced-CPU
    # end-of-training validation prediction would otherwise segfault. See blas_guard.
    check_blas_backend()

    cli = parse_arguments()
    logdir = Path(cli.logdir)
    create_dirs(logdir)

    # Load hyperparameters
    with open(cli.hyperparameters, "r", encoding="utf-8") as f:
        hparams: Dict[str, Any] = json.load(f)

    hdf5_list_path = cli.hdf5_list
    if not hdf5_list_path.is_file():
        raise FileNotFoundError(f"HDF5 list file not found: {hdf5_list_path}")

    datasource = EpiDataSource(hdf5_list_path, cli.chromsize, cli.metadata)

    # Pre-specified folds override n_fold / min_class_size / oversampling.
    fold_definitions = None
    if cli.folds is not None:
        with open(cli.folds, "r", encoding="utf-8") as f:
            fold_definitions = json.load(f)
        print(
            f"Using pre-specified folds from {cli.folds}: "
            f"{len(fold_definitions)} folds. "
            "Overriding --n_fold, --min_class_size and oversampling."
        )

    # Load data and create fold factory
    loading_begin = time_now()
    fold_factory = GeneralFoldFactory(
        datasource=datasource,
        label_category=cli.category,
        min_class_size=cli.min_class_size,
        n_fold=cli.n_fold,
        mmap_dir=cli.mmap_dir if cli.mmap_dir is not None else logdir / "mmap_cache",
        fold_definitions=fold_definitions,
    )
    print(f"Loading time: {time_now() - loading_begin}")

    # Cross-validation training
    oversample = hparams.get("oversample", hparams.get("oversampling", True))

    for i, my_data in enumerate(fold_factory.yield_split(oversample=oversample)):
        print(f"\n{'='*60}")
        print(f"FOLD {i+1}/{fold_factory.k}")
        print(f"  Training:   {my_data.train.num_examples} samples")
        print(f"  Validation: {my_data.validation.num_examples} samples")
        print(f"{'='*60}\n")

        do_one_experiment(
            split_nb=i,
            my_data=my_data,
            hparams=hparams,
            logdir=logdir,
            hl_units=cli.hl_units,
            nb_layers=cli.nb_layer,
        )

    total_time = time_now() - begin
    print(f"\nTotal time: {total_time}")


if __name__ == "__main__":
    main()
