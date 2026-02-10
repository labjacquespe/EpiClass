"""Integration test for general_training.py using saccer3 fixtures."""
import sys
from pathlib import Path

import pytest

from epiclass.general_training import main as main_module
from tests.epilap_test_data import FIXTURES_DIR

SACCER3_DIR = FIXTURES_DIR / "saccer3"


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("general_training")


@pytest.mark.filterwarnings(
    "ignore:The 'val_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*:UserWarning"
)
@pytest.mark.slow
def test_cross_validation_training(test_dir: Path, saccer3_hdf5_file_list: Path):
    """Test if basic training succeeds."""
    # fmt: off
    sys.argv = [
        "general_training.py",
        "assay",
        str(SACCER3_DIR / "saccer3_hparams.json"),
        str(saccer3_hdf5_file_list),
        str(SACCER3_DIR / "saccer3.can.chrom.sizes"),
        str(SACCER3_DIR / "saccer3_2016-07_metadata.json"),
        str(test_dir),
        "--n_fold", "2",
        "--hl_units", "500",
        "--min_class_size", "10",
    ]
    # fmt: on
    print("Running general_training.py with args:", sys.argv)
    main_module()

    for fold_i in range(2):
        fold_dir = test_dir / f"fold_{fold_i}"
        assert fold_dir.is_dir()
        assert (fold_dir / "training_mapping.tsv").is_file()
        assert (fold_dir / "best_checkpoint.list").is_file()
        assert list(fold_dir.glob("*validation_prediction*"))

    assert (test_dir / "hdf5_files.list").is_file()
