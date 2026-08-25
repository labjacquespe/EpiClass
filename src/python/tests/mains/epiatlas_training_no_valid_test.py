"""Test module for epiatlas_training_no_valid.py."""
import os
import sys
from pathlib import Path

import pytest

from epiclass.mains.epiatlas_training_no_valid import main as main_module
from tests.epilap_test_data import FIXTURES_DIR, EpiAtlasTreatmentTestData


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("epiatlas_training_no_valid")


@pytest.mark.filterwarnings("ignore:Resolution not found in HDF5:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:You defined a `validation_step` but have no `val_dataloader`. Skipping val loop."
)
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers which may be a bottleneck."
)
@pytest.mark.filterwarnings(
    "ignore:The number of training batches \\(2\\) is smaller than the logging interval Trainer\\(log_every_n_steps=50\\)."
)
def test_training(test_dir: Path):
    """Test if basic training succeeds.

    Default test data is splitting into 2 folds because the fold factory
    is made for cross-validation. This means you need at least 2 samples per class.
    """
    os.environ["MIN_CLASS_SIZE"] = "1"  # for main script

    datasource = EpiAtlasTreatmentTestData.test_data(
        test_set="test-epilap-empty-biotype-n8",
        min_class_size=1,  # to avoid creating empty mock dataset
    ).epiatlas_dataset.datasource

    hparams_file = FIXTURES_DIR / "test_human_hparams.json"

    sys.argv = [
        "epiatlas_training_no_valid.py",
        "biomaterial_type",
        f"{hparams_file}",
        f"{datasource.hdf5_file}",
        f"{datasource.chromsize_file}",
        f"{datasource.metadata_file}",
        str(test_dir),
        "--offline",
    ]

    main_module()

    ckpt_list = test_dir / "best_checkpoint.list"
    assert ckpt_list.is_file(), "best_checkpoint.list was not created"
    ckpt_path = Path(ckpt_list.read_text(encoding="utf-8").splitlines()[-1].split(" ")[0])
    assert ckpt_path.is_file(), f"checkpoint file not found: {ckpt_path}"
    assert (test_dir / "training_mapping.tsv").is_file()
