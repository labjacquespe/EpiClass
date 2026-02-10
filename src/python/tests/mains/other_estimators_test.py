"""Test module for other_estimators.py content"""
import os
import sys
from pathlib import Path

import pytest

import epiclass.core.estimators
from epiclass.other_estimators import main as main_module
from tests.epilap_test_data import FIXTURES_DIR, EpiAtlasTreatmentTestData


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("other_estimators")


# @pytest.mark.filterwarnings("ignore:Cannot read file directly.*")
@pytest.mark.slow
def test_hyperparams(test_dir: Path):
    """Test if hyperparameter file is handled properly."""
    os.environ["MIN_CLASS_SIZE"] = "3"
    epiclass.core.estimators.NFOLD_PREDICT = 2

    datasource = EpiAtlasTreatmentTestData.default_test_data().epiatlas_dataset.datasource

    hparams_file = FIXTURES_DIR / "other_estimators_hparams.json"

    sys.argv = [
        "other_estimators.py",
        "--models",
        "LinearSVC",
        "--predict",
        "biomaterial_type",
        f"{datasource.hdf5_file}",
        f"{datasource.chromsize_file}",
        f"{datasource.metadata_file}",
        str(test_dir),
        "--hyperparams",
        f"{hparams_file}",
    ]

    main_module()


@pytest.mark.slow
def test_binary_classifier(test_dir: Path):
    """Test if binary models are properly supported."""
    # Setting up environment variables
    import multiprocessing as mp  # pylint: disable=import-outside-toplevel

    mp.set_start_method("spawn", force=True)

    os.environ["MIN_CLASS_SIZE"] = "1"
    os.environ["LABEL_LIST"] = '["female", "male"]'
    epiclass.core.estimators.NFOLD_PREDICT = 2

    # Loading paths
    datasource = EpiAtlasTreatmentTestData.default_test_data(
        label_category="sex"
    ).epiatlas_dataset.datasource

    hparams_file = FIXTURES_DIR / "other_estimators_hparams.json"

    sys.argv = [
        "other_estimators.py",
        "--predict",
        "sex",
        f"{datasource.hdf5_file}",
        f"{datasource.chromsize_file}",
        f"{datasource.metadata_file}",
        str(test_dir),
        "--hyperparams",
        f"{hparams_file}",
    ]

    main_module()
