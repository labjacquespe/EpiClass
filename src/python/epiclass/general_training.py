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
from typing import Any, Dict, Generator, List

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from lightning.pytorch import loggers as pl_loggers
from sklearn.model_selection import StratifiedKFold

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import analysis
from epiclass.core.data.dataset import DataSet
from epiclass.core.data_source import EpiDataSource
from epiclass.core.lazy.lazy_data_classes import LazyKnownData
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.core.metadata import Metadata
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.trainer import MyTrainer, define_callbacks
from epiclass.utils.check_dir import create_dirs
from epiclass.utils.time import time_now
from epiclass.utils.torch_data import create_torch_datasets

# ---------------------------------------------------------------------------
# Data loading and stratified k-fold splitting (no UUID requirement)
# ---------------------------------------------------------------------------


class GeneralFoldFactory:
    """Stratified k-fold cross-validation without UUID grouping.

    Loads signals once and yields DataSet objects per fold.
    """

    def __init__(
        self,
        datasource: EpiDataSource,
        label_category: str,
        min_class_size: int = 3,
        n_fold: int = 4,
        mmap_dir: Path | None = None,
    ):
        self.k = n_fold
        if n_fold < 2:
            raise ValueError(f"Need at least 2 folds. Got {n_fold}.")

        self._label_category = label_category

        # Load and filter metadata
        meta = Metadata(datasource.metadata_file)
        files = LazyHdf5Loader.read_list(datasource.hdf5_file)
        meta.apply_filter(lambda item: item[0] in files)
        meta.remove_missing_labels(label_category)
        meta.remove_small_classes(min_class_size, label_category)
        self._metadata = meta

        self._classes = meta.unique_classes(label_category)
        self._classes_mapping = {label: i for i, label in enumerate(self._classes)}

        # Register HDF5 paths lazily (no signals loaded yet)
        loader = LazyHdf5Loader(
            chrom_file=datasource.chromsize_file,
            normalization=True,
            mmap_dir=mmap_dir,
        )
        loader.register_hdf5s(
            data_file=datasource.hdf5_file,
            md5s=list(meta.md5s),
            strict=True,
            verbose=True,
        )
        loader.preload_all()

        md5s = list(loader.file_paths.keys())
        labels = [meta[md5][label_category] for md5 in md5s]

        self._dataset = LazyKnownData(
            ids=md5s,
            loader=loader,
            y_str=labels,
            y=np.array(
                [self._classes_mapping[label] for label in labels], dtype=np.int64
            ),
            metadata=meta,
        )

        print(f"\nLoaded {len(md5s)} samples across {len(self._classes)} classes.")
        meta.display_labels(label_category)

    @property
    def classes(self) -> List[str]:
        """Get list of class labels."""
        return self._classes

    def yield_split(self, oversample: bool = True) -> Generator[DataSet, None, None]:
        """Yield DataSet for each fold of stratified k-fold CV."""
        dset = self._dataset
        y = dset.encoded_labels

        # StratifiedKFold only inspects sample count; passing a placeholder
        # avoids materializing all signals just to split on indices.
        x_placeholder = np.zeros((len(y), 1), dtype=np.float32)

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        for train_idxs, valid_idxs in skf.split(x_placeholder, y):
            train_idxs = list(train_idxs)
            valid_idxs = list(valid_idxs)

            if oversample:
                ros = RandomOverSampler(random_state=42)
                resampled, _ = ros.fit_resample(
                    np.arange(len(train_idxs)).reshape(-1, 1),
                    y[train_idxs],
                )
                train_idxs = [train_idxs[i] for i in resampled.flatten()]

            train_set = dset.subsample(train_idxs)
            valid_set = dset.subsample(valid_idxs)

            yield DataSet(
                training=train_set,
                validation=valid_set,
                test=LazyKnownData.empty_collection(),
                sorted_classes=self._classes,
            )


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

    # Load data and create fold factory
    loading_begin = time_now()
    fold_factory = GeneralFoldFactory(
        datasource=datasource,
        label_category=cli.category,
        min_class_size=cli.min_class_size,
        n_fold=cli.n_fold,
        mmap_dir=cli.mmap_dir if cli.mmap_dir is not None else logdir / "mmap_cache",
    )
    print(f"Loading time: {time_now() - loading_begin}")

    # Cross-validation training
    oversample = hparams.get("oversample", hparams.get("oversampling", True))

    for i, my_data in enumerate(fold_factory.yield_split(oversample=oversample)):
        print(f"\n{'='*60}")
        print(f"FOLD {i+1}/{cli.n_fold}")
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
