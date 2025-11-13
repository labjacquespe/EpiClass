"""Module to define how LightGBM is handled."""
import json
import pickle
from pathlib import Path

try:
    from lightgbm import log_evaluation
except ImportError as e:
    raise ImportError(
        "LightGBM model required for tuning, regardless of whether LightGBM is used as estimator. "
        "Install all models requirements with: pip install .[other_models]"
    ) from e

import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd

from epiclass.core.epiatlas_treatment import EpiAtlasFoldFactory

# TODO: Permit native saving/loading. # https://stackoverflow.com/questions/55208734/save-lgbmregressor-model-from-python-lightgbm-package-to-disc


def print_last_trial(study, trial):  # pylint: disable=unused-argument
    """Optuna callback to always print more information on last finished trial."""
    print("\nLast trial used the following hyper-parameters:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    print(f"to achieve objective function score of {trial.value}")
    print("Full parameters are:")
    print(trial.system_attrs["lightgbm_tuner:lgbm_params"])


def tune_lgbm(ea_handler: EpiAtlasFoldFactory, logdir: Path):
    """
    It takes an EpiAtlasFoldFactory object and a log directory, and it tunes the
    hyperparameters of a LightGBM classifier using Optuna.

    Args:
      ea_handler (EpiAtlasFoldFactory): Dataset splits creator.
      logdir (Path): The directory where the results will be saved.

    Returns:
      The Optuna study object.
    """
    params = {
        "objective": "multiclass",
        "num_class": len(ea_handler.classes),
        "metric": ["multi_logloss", "multi_error"],
        "boosting_type": "dart",
        "seed": 42,
        "force_col_wise": True,
        "device_type": "cpu",
        "num_threads": 9,
    }

    # Using first fold to tune hyperparams
    dsets = next(ea_handler.yield_split())

    dtrain = lgb.Dataset(  # type: ignore
        dsets.train.signals, label=dsets.train.encoded_labels, free_raw_data=True
    )
    dvalid = lgb.Dataset(  # type: ignore
        dsets.validation.signals,
        label=dsets.validation.encoded_labels,
        free_raw_data=True,
    )
    dsets = None
    del dsets

    # Create tuner
    max_iter = 100  # boosting rounds per trial
    study = optuna.create_study(
        study_name="LGBM Classifier",
    )

    tuner = lgb.LightGBMTuner(  # type: ignore
        params,
        dtrain,
        num_boost_round=max_iter,
        study=study,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            log_evaluation(period=5),  # type: ignore
        ],
        show_progress_bar=False,
        optuna_callbacks=[print_last_trial],
    )

    # Optimize and log results.
    print("Tuning feature fraction")
    tuner.tune_feature_fraction(3)
    print("Tuning number of leaves")
    tuner.tune_num_leaves(5)
    print("Tuning bagging")
    tuner.tune_bagging(4)

    print("Best score:", tuner.best_score)

    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in list(best_params.items()):
        print(f"    {key}: {value}")

        # Put it in the pipeline format
        del best_params[key]
        best_params[f"model__{key}"] = value

    name = "LGBM"
    with open(logdir / f"{name}_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, sort_keys=True, indent=4)

    with open(logdir / f"{name}_study.pickle", "wb") as f:
        pickle.dump(study, f)

    df = pd.DataFrame(study.trials_dataframe())
    df.to_csv(logdir / f"{name}_trials.csv")

    return study
