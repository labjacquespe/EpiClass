"""Main"""
# pylint: disable=too-many-branches
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

try:
    import lightgbm  # pylint: disable=unused-import
    import optuna  # pylint: disable=unused-import
    import skopt  # pylint: disable=unused-import
except ImportError as e:
    raise ImportError(
        "Missing dependencies for other_estimators.py. "
        "Please install with 'pip install .[other_models]'"
    ) from e

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.argparseutils.directorychecker import DirectoryChecker
from epiclass.core import data, estimators, metadata
from epiclass.core.data_source import EpiDataSource
from epiclass.core.epiatlas_treatment import EpiAtlasFoldFactory
from epiclass.core.lgbm import tune_lgbm
from epiclass.utils import modify_metadata
from epiclass.utils.time import time_now

if os.getenv("CONCURRENT_CV") is not None:
    CONCURRENT_CV = int(os.environ["CONCURRENT_CV"])
else:
    CONCURRENT_CV = 1

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    # fmt: off
    parser = ArgumentParser()
    group1 = parser.add_argument_group("General")
    group1.add_argument(
        "category", type=str, help="The metadata category to analyse."
        )
    group1.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    group1.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes."
        )
    group1.add_argument(
        "metadata", type=Path, help="A metadata JSON file."
        )
    group1.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs."
    )
    group1.add_argument(
        "--models", nargs="+", type=str, help="Specify models to tune and/or predict.",
        choices=["all", "LinearSVC", "RF", "LR", "LGBM"], default=["all"]
        )

    mode = parser.add_argument_group("Mode")
    mode = mode.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tune", action="store_true", help="Search best hyperparameters."
        )
    mode.add_argument(
        "--predict", action="store_true", help="FIT and PREDICT using hyperparameters."
        )
    mode.add_argument(
        "--predict-new", action="store_true", help="Use saved models to predict labels of new samples."
        )
    mode.add_argument(
        "--full-run", action="store_true", help="Tune then predict"
        )

    tune = parser.add_argument_group("Tune")
    tune.add_argument(
        "-n",
        type=int,
        default=30,
        help="Number of BayesSearchCV hyperparameters iterations.",
    )

    predict = parser.add_argument_group("Predictions and Final training")
    predict.add_argument(
        "--hyperparams", type=Path, help="A json file containing model(s) hyperparameters.",
    )
    # fmt: on
    return parser.parse_args()


def main():
    """Takes command line arguments."""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments()

    category = cli.category

    if cli.tune:
        mode_tune = True
        mode_predict = False
    elif cli.predict:
        mode_tune = False
        mode_predict = True
    elif cli.full_run:
        mode_tune = True
        mode_predict = True
    elif cli.predict_new:
        mode_tune = False
        mode_predict = False
    else:
        raise ValueError("Houston we have a problem.")

    acceptable_models = list(estimators.model_mapping.keys())
    if "all" in cli.models:
        models = acceptable_models
    else:
        models = cli.models

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)

    # --- Prefilter metadata, must put in EpiAtlasDataset to actually use it ---
    my_metadata = metadata.UUIDMetadata(my_datasource.metadata_file)
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )

    label_list = metadata.env_filtering(my_metadata, category)

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = 10

    if category in set(
        [
            "harmonized_sample_ontology_intermediate",
            "harm_sample_ontology_intermediate",
            "cell_type",
        ]
    ):
        categories = set(my_metadata.get_categories())
        if "assay_epiclass" in categories:
            assay_cat = "assay_epiclass"
        elif "assay" in categories:
            assay_cat = "assay"
        else:
            raise ValueError("Cannot find assay category for class pairs.")
        my_metadata = modify_metadata.filter_by_pairs(
            my_metadata, assay_cat=assay_cat, cat2=category, nb_pairs=9, min_per_pair=10
        )

    # Tuning mode
    loading_begin = time_now()
    if mode_tune is True:  # type: ignore
        print("Entering tuning mode")
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            my_datasource,
            category,
            label_list,
            n_fold=estimators.NFOLD_TUNE,
            test_ratio=0.1,
            min_class_size=min_class_size,
            md5_list=list(my_metadata.md5s),
            force_filter=True,
        )
        loading_time = time_now() - loading_begin
        print(f"Initial hdf5 loading time: {loading_time}")

        n_iter = cli.n

        for name in models:
            try:
                if name == "LGBM":
                    # optuna.logging.set_verbosity(optuna.logging.DEBUG)  # type: ignore
                    tune_lgbm(ea_handler, cli.logdir)
                else:
                    estimators.optimize_estimator(ea_handler, cli.logdir, n_iter, name)
            except OverflowError as error:
                print("{name} model failed with OverflowError. Logging error to stderr")
                logger.exception(error)
                continue

    # Predict mode
    if mode_predict is True:  # type: ignore
        print("Entering fit/prediction mode")

        if cli.hyperparams is None:
            raise ValueError(
                "No hyperparameter file given for final fit(s) and prediction."
            )

        with open(cli.hyperparams, "r", encoding="utf-8") as file:
            loaded_hparams = json.load(file)

        # Intersect available model hparams with chosen models.
        model_w_hparams = set(loaded_hparams.keys())
        selected_models = model_w_hparams & set(models)

        if not selected_models:
            print("No parameters found for selected models {models}, finishing now.")
            sys.exit()

        # Load hdf5s
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            my_datasource,
            category,
            label_list,
            n_fold=estimators.NFOLD_PREDICT,
            test_ratio=0,
            min_class_size=min_class_size,
            md5_list=list(my_metadata.md5s),
            force_filter=True,
        )
        loading_time = time_now() - loading_begin
        print(f"Initial hdf5 loading time: {loading_time}")

        # Create model with loaded hyperparameters and run predictions.
        for model_name in selected_models:
            model_hparams = loaded_hparams[model_name]

            if model_name == "LGBM":
                model_hparams = {
                    k: v
                    for k, v in model_hparams.items()
                    if k in estimators.lgbm_allowed_params
                }

            print(f"Using {model_name}.")
            estimator = estimators.model_mapping[model_name]
            estimator.set_params(**model_hparams)
            print("Parameters set:")
            for param, value in estimator.get_params(deep=True).items():
                print(f"{param}: {value}")

            estimators.run_predictions(ea_handler, estimator, model_name, cli.logdir)

    # Giving predictions with chosen models, for all files in hdf5 list.
    if cli.predict_new:
        pattern = "{log}/**{name}*.pickle"
        to_load = []
        for model in models:
            save_name = estimators.save_mapping[model]
            to_load += glob.glob(pattern.format(log=cli.logdir, name=save_name))

        if to_load:
            my_data = data.DataSetFactory.from_epidata(
                my_datasource,
                my_metadata,
                category,
                min_class_size=min_class_size,
                validation_ratio=0,
                test_ratio=1,
                onehot=False,
                oversample=False,
            )

            for model_path in to_load:
                my_model = estimators.EstimatorAnalyzer.restore_model_from_path(
                    full_path=model_path
                )
                my_analyzer = estimators.EstimatorAnalyzer(
                    classes=my_metadata.unique_classes(category), estimator=my_model
                )

                predict_path = (
                    cli.logdir / f"{Path(model_path).stem}_prediction_{cli.hdf5.stem}.csv"
                )

                print(f"Saving predictions to: {predict_path}")
                my_analyzer.predict_file(
                    ids=my_data.test.ids,
                    X=my_data.test.signals,
                    y=my_data.test.encoded_labels,
                    log=predict_path,
                )

        else:
            print("No saved model file found, finishing now.")
            sys.exit()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
