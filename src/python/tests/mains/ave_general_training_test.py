"""Integration test for ave_general_training.py using saccer3 fixtures."""
# pylint: disable=duplicate-code
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from epiclass.ave_general_training import main as main_module
from tests.epilap_test_data import FIXTURES_DIR

SACCER3_DIR = FIXTURES_DIR / "saccer3"


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("ave_general_training")


def _write_ave_hparams(path: Path) -> Path:
    """Write a tiny AVE hyperparameters file for a fast CPU run."""
    hparams = {
        "batch_size": 4,
        "training_epochs": 2,
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
def test_ave_general_training(
    test_dir: Path, saccer3_small_training_data: tuple[Path, Path]
):
    """AVE training + scoring succeeds on saccer3 (no-UUID) data with assay folds."""
    os.environ["MAX_SPLIT"] = "0"  # only run the first fold to keep the test fast

    hdf5_list, metadata = saccer3_small_training_data
    hparams_file = _write_ave_hparams(test_dir / "ave_hparams.json")

    # fmt: off
    sys.argv = [
        "ave_general_training.py",
        "assay",
        str(hparams_file),
        str(hdf5_list),
        str(SACCER3_DIR / "saccer3.can.chrom.sizes"),
        str(metadata),
        str(test_dir),
        "--n_fold", "2",
        "--min_class_size", "10",
        "--offline",
    ]
    # fmt: on
    main_module()

    split_dir = test_dir / "split0"
    assert split_dir.is_dir()
    assert (split_dir / "best_checkpoint.list").is_file()

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
