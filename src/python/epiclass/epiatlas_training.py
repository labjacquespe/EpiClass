"""Main"""
# pylint: disable=duplicate-code, wrong-import-position, ungrouped-imports

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import lightning.pytorch as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
import torch
from lightning.pytorch import loggers as pl_loggers

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import analysis, metadata
from epiclass.core.data import DataSet, create_torch_datasets
from epiclass.core.data_source import EpiDataSource
from epiclass.core.epiatlas_treatment import EpiAtlasFoldFactory
from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.trainer import MyTrainer, define_callbacks
from epiclass.utils import modify_metadata
from epiclass.utils.check_dir import create_dirs
from epiclass.utils.my_logging import log_dset_composition, log_pre_training
from epiclass.utils.time import time_now


class DatasetError(Exception):
    """Custom error"""

    def __init__(self, *args: object) -> None:
        print(
            "\n--- ERROR : Verify source files, filters, and min_class_size. ---\n",
            file=sys.stderr,
        )
        super().__init__(*args)


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "category", type=str, help="The metadata category to analyse.",
    )
    arg_parser.add_argument(
        "hyperparameters", type=Path, help="A json file containing model hyperparameters.",
    )
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!",
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    arg_parser.add_argument(
        "--offline",
        action="store_true",
        help="Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.",
    )
    arg_parser.add_argument(
        "--restore",
        action="store_true",
        help="Skips training, tries to restore existing models in logdir for further analysis. ",
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments()

    category = cli.category

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)
    hdf5_resolution = my_datasource.hdf5_resolution()

    with open(cli.hyperparameters, "r", encoding="utf-8") as file:
        hparams: Dict[str, Any] = json.load(file)

    my_metadata = metadata.UUIDMetadata(my_datasource.metadata_file)

    # --- Prefilter metadata ---
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )
    my_metadata.remove_missing_labels(category)

    if category in {"paired", "paired_end_mode"}:
        category = "paired_end_mode"
        modify_metadata.merge_pair_end_info(my_metadata)
    elif category == "data_generating_centre":
        modify_metadata.fix_roadmap(my_metadata)
    elif category == "upload_date_2":
        modify_metadata.add_formated_date(my_metadata)
    elif category == "random":
        n_splits = os.getenv("RANDOM_SPLITS", "10")
        category = modify_metadata.add_random_group(my_metadata, n_split=int(n_splits))

    label_list = metadata.env_filtering(my_metadata, category)

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = hparams.get("min_class_size", 10)

    if any(
        label in category
        for label in set(
            [
                "harmonized_sample_ontology_intermediate",
                "harm_sample_ontology_intermediate",
                "cell_type",
            ]
        )
    ):
        categories = set(my_metadata.get_categories())
        if "assay_epiclass" in categories:
            assay_cat = "assay_epiclass"
        elif "assay" in categories:
            assay_cat = "assay"
        else:
            raise ValueError("Cannot find assay category for class pairs.")
        my_metadata = modify_metadata.filter_by_pairs(
            my_metadata=my_metadata,
            assay_cat=assay_cat,
            cat2=category,
            nb_pairs=int(os.getenv("NB_PAIRS", "9")),
            min_per_pair=int(os.getenv("MIN_PER_PAIR", "10")),
        )

    # --- Load signals and train ---
    loading_begin = time_now()

    restore_model = cli.restore
    n_fold = hparams.get("n_fold", 10)

    ea_handler = EpiAtlasFoldFactory.from_datasource(
        my_datasource,
        category,
        label_list,
        n_fold=n_fold,
        test_ratio=0,
        min_class_size=min_class_size,
        force_filter=True,
        metadata=my_metadata,
    )
    loading_time = time_now() - loading_begin

    to_log = {
        "loading_time": loading_time.total_seconds(),
        "hdf5_resolution": str(hdf5_resolution),
        "category": category,
    }

    min_split = int(os.getenv("MIN_SPLIT", "0"))
    max_split = int(os.getenv("MAX_SPLIT", "42"))

    time_before_split = time_now()
    oversample = hparams.get("oversample", hparams.get("oversampling", True))
    for i, my_data in enumerate(ea_handler.yield_split(oversample=oversample)):
        # Skip if not in inclusive range
        if not (min_split <= i <= max_split):  # pylint: disable=superfluous-parens
            continue

        split_time = time_now() - time_before_split
        to_log.update({"split_time": split_time.total_seconds()})

        # --- Startup LOGGER ---
        # api key in config file
        is_online = not cli.offline  # additional logging fails when offline
        logdir = Path(cli.logdir)
        create_dirs(logdir)

        exp_name = "-".join(cli.logdir.parts[-3:])
        comet_logger = pl_loggers.CometLogger(
            project="EpiClass",
            name=exp_name,
            offline_directory=logdir,  # type: ignore
            online=is_online,
            auto_metric_logging=False,
        )

        comet_logger.experiment.add_tag("EpiAtlas")
        log_pre_training(logger=comet_logger, to_log=to_log, step=i)

        # Everything happens in there
        do_one_experiment(
            split_nb=i,
            my_data=my_data,
            hparams=hparams,
            logger=comet_logger,
            restore=restore_model,
        )

        time_before_split = time_now()


def do_one_experiment(
    split_nb: int,
    my_data: DataSet,
    hparams: Dict,
    logger: pl_loggers.CometLogger,
    restore: bool,
) -> None:
    """Wrapper for convenience. Skip training if restore is True"""
    begin_loop = time_now()

    log_dset_composition(my_data, logdir=None, logger=logger, split_nb=split_nb)

    dsets_dict = create_torch_datasets(
        data=my_data,
        bs=hparams.get("batch_size", 64),
    )
    train_dataset, train_dataloader = dsets_dict["training"]
    valid_dataset, valid_dataloader = dsets_dict["validation"]

    if my_data.train.num_examples == 0 or my_data.validation.num_examples == 0:
        raise DatasetError("Trying to train without any training or validation data.")

    # Warning : output mapping of model created from training dataset
    mapping_file = Path(logger.save_dir) / "training_mapping.tsv"  # type: ignore

    if not restore:
        # --- CREATE a brand new MODEL ---
        # Create mapping (i --> class string) file
        my_data.save_mapping(mapping_file)
        mapping = my_data.load_mapping(mapping_file)
        logger.experiment.log_asset(mapping_file)

        #  DEFINE sizes for input and output LAYERS of the network
        input_size = my_data.train.signals[0].size  # type: ignore
        output_size = len(my_data.classes)
        hl_units = int(os.getenv("LAYER_SIZE", default="3000"))
        nb_layers = int(os.getenv("NB_LAYER", default="1"))

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
            my_model.print_model_summary()  # torchinfo summary

        gpu_available = torch.cuda.device_count() > 0
        print(f"GPU available: {gpu_available}")

        # --- TRAIN the model ---
        if split_nb == 0:
            callbacks = define_callbacks(
                early_stop_limit=hparams.get("early_stop_limit", 20),
                show_summary=True,  # pytorch_lightning model summary
                show_progress_bar=not gpu_available,  # Show progress bar only on CPU
            )
        else:
            callbacks = define_callbacks(
                early_stop_limit=hparams.get("early_stop_limit", 20),
                show_summary=False,
                show_progress_bar=not gpu_available,
            )

        before_train = time_now()
        if gpu_available:
            trainer = MyTrainer(
                general_log_dir=logger.save_dir,  # type: ignore
                model=my_model,
                max_epochs=hparams.get("training_epochs", 50),
                check_val_every_n_epoch=hparams.get("measure_frequency", 1),
                logger=logger,
                callbacks=callbacks,
                accelerator="gpu",
                devices=1,
                precision=16,
            )
        else:
            trainer = MyTrainer(
                general_log_dir=logger.save_dir,  # type: ignore
                model=my_model,
                max_epochs=hparams.get("training_epochs", 50),
                check_val_every_n_epoch=hparams.get("measure_frequency", 1),
                logger=logger,
                callbacks=callbacks,
                accelerator="cpu",
                devices=1,
            )

        if split_nb == 0:
            trainer.print_hyperparameters()
            trainer.fit(
                my_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                verbose=True,
            )
        else:
            trainer.fit(
                my_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
                verbose=False,
            )
        trainer.save_model_path()
        training_time = time_now() - before_train
        print(f"training time: {training_time}")

        # Make sure to wait for logging to finish
        logger.experiment.end()

        # reload comet logger for further logging, will create new experience in offline mode
        is_online = not bool(type(logger.experiment).__name__ == "OfflineExperiment")

        logger = pl_loggers.CometLogger(
            project="EpiClass",
            save_dir=logger.save_dir,  # type: ignore
            online=is_online,
            auto_metric_logging=False,
            experiment_key=logger.experiment.get_key(),
        )
        logger.experiment.log_metric(
            "Training time", training_time.total_seconds(), step=split_nb
        )
        logger.experiment.log_metric("Last epoch", my_model.current_epoch, step=split_nb)
    try:
        my_model = LightningDenseClassifier.restore_model(logger.save_dir)
    except (FileNotFoundError, OSError) as e:
        print(e)
        print("Closing logger and skipping this split.")
        logger.experiment.add_tag("ModelNotFoundError")
        logger.finalize(status="ModelNotFoundError")
        return

    # --- OUTPUTS ---
    my_analyzer = analysis.Analysis(
        my_model,
        my_data,
        logger,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        test_dataset=None,
    )

    my_analyzer.get_training_metrics(verbose=True)
    my_analyzer.get_validation_metrics(verbose=True)

    my_analyzer.write_validation_prediction()
    my_analyzer.validation_confusion_matrix()

    end_loop = time_now()
    loop_time = end_loop - begin_loop
    logger.experiment.log_metric("Loop time", loop_time.total_seconds(), step=split_nb)
    print(f"Loop time (excludes split time): {loop_time}")

    logger.experiment.add_tag("Finished")
    logger.finalize(status="Finished")

    # Make sure to wait for logging to finish
    logger.experiment.end()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
