"""Script for debugging LightGBM."""
# pylint: disable=import-outside-toplevel
# pyright: reportPrivateImportUsage=false
import warnings

warnings.filterwarnings("ignore", message=".*IPython display.*")

import glob
import json
from pathlib import Path

import pytest
from lightgbm import LGBMClassifier

from tests.epilap_test_data import EpiAtlasTreatmentTestData


@pytest.fixture(name="logdir")
def tmp_logdir(mk_logdir) -> Path:
    """Test logdir"""
    return mk_logdir("lgbm")


@pytest.mark.skip(
    reason="Sometimes gets stuck. Needs investigation. Doesn't happen when run alone."
)
@pytest.mark.filterwarnings("ignore:IPython")
def test_lgbm_save_load(logdir):
    """Test LGBM tuning + subsequent fit pipeline."""

    from epiclass.core import estimators, lgbm as lgbm_funcs

    ea_handler = EpiAtlasTreatmentTestData.default_test_data()

    # tune (saves hparams)
    lgbm_funcs.tune_lgbm(ea_handler=ea_handler, logdir=logdir)

    # load haprams
    pattern = f"{logdir / estimators.best_params_file_format.format(name='*')}"
    hparam_files = glob.glob(pattern)
    with open(hparam_files[0], "r", encoding="utf-8") as file:
        hparams = json.load(file)

    # filter hparams
    hparams = {k: v for k, v in hparams.items() if k in estimators.lgbm_allowed_params}

    assert "model__boosting_type" in hparams
    assert "model__metric" not in hparams
    for param_name in hparams.keys():
        assert "model__" in param_name


def minimal_bug_example():
    """Script for debugging LightGBM."""
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold

    from epiclass.core import estimators

    print(estimators.lgbm_allowed_params)

    params = {
        "objective": "multiclass",
        "num_class": 2,
        "metric": ["multi_logloss", "multi_error"],
        "n_estimators": 1,
        "verbose": 0,
    }

    X, y = load_iris(return_X_y=True)
    X, y = zip(
        *[[feat, targ] for feat, targ in zip(X, y) if targ != 2]
    )  # remove third class
    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=3, shuffle=False)
    splits = skf.split(np.zeros(X.shape), y)
    train_idx_1, valid_idx_1 = next(splits)

    X_train = X[train_idx_1]
    y_train = y[train_idx_1]

    X_valid = X[valid_idx_1]
    y_valid = y[valid_idx_1]

    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

    estimator = LGBMClassifier()
    estimator.set_params(**params)
    estimator.fit(X_train, y_train)
    preds = estimator.predict_proba(X_valid)
    print(preds.shape)  # type: ignore

    new_estimator = LGBMClassifier()
    new_estimator.set_params(**{})
    new_estimator.fit(X_train, y_train)
    preds = new_estimator.predict_proba(X_valid)
    print(preds.shape)  # type: ignore


def main():
    """main"""
    minimal_bug_example()


if __name__ == "__main__":
    main()
