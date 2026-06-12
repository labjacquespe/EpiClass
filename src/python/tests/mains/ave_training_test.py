"""Integration test for ave_training.py using the EpiAtlas mock data fixture."""
# pylint: disable=duplicate-code
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from epiclass.ave_training import main as main_module
from tests.epilap_test_data import EpiAtlasTreatmentTestData


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("ave_training")


def _write_ave_hparams(path: Path) -> Path:
    """Write a tiny AVE hyperparameters file for a fast CPU run."""
    hparams = {
        "batch_size": 4,
        "training_epochs": 2,
        "n_fold": 2,
        "early_stop_limit": 1,
        "measure_frequency": 1,
        "learning_rate": 1e-3,
        "l2_scale": 0.0,
        "dropout": 0.0,
        "kl_weight": 0.1,
        "fusion_weight": 0.5,
        "latent_dim": 8,
        "contamination_rate": 0.1,
        "oversample": True,
        "min_class_size": 1,
    }
    path.write_text(json.dumps(hparams), encoding="utf-8")
    return path


@pytest.mark.filterwarnings("ignore:Resolution not found in HDF5:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:The 'val_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.filterwarnings("ignore:The number of training batches")
@pytest.mark.slow
def test_ave_training(test_dir: Path):
    """Basic AVE training + reconstruction-error scoring succeeds on the mock data."""
    os.environ["MIN_CLASS_SIZE"] = "1"
    os.environ["MAX_SPLIT"] = "0"  # only run the first fold to keep the test fast

    datasource = EpiAtlasTreatmentTestData.test_data(
        min_class_size=1,
    ).epiatlas_dataset.datasource

    hparams_file = _write_ave_hparams(test_dir / "ave_hparams.json")

    sys.argv = [
        "ave_training.py",
        "biomaterial_type",
        str(hparams_file),
        str(datasource.hdf5_file),
        str(datasource.chromsize_file),
        str(datasource.metadata_file),
        str(test_dir),
        "--offline",
    ]

    main_module()

    split_dir = test_dir / "split0"
    assert split_dir.is_dir()
    assert (split_dir / "best_checkpoint.list").is_file()
    assert (split_dir / "training_mapping.tsv").is_file()

    scores_csv = split_dir / "ave_validation_scores.csv"
    assert scores_csv.is_file(), "ave_validation_scores.csv was not created"

    scores = pd.read_csv(scores_csv)
    assert list(scores.columns) == [
        "sample_id",
        "reconstruction_error",
        "outlier_flag",
    ]
    assert len(scores) > 0
    assert scores["outlier_flag"].isin([0, 1]).all()
