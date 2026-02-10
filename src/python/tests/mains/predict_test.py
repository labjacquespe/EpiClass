"""Test module for predict.py."""
import sys
from pathlib import Path

import pytest

from epiclass.predict import main as main_module
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("predict")


@pytest.mark.slow
def test_training(test_dir: Path, saccer3_hdf5_file_list: Path):
    """Test if basic prediction succeeds."""

    saccer3_fixtures_dir = FIXTURES_DIR / "saccer3"
    chroms = saccer3_fixtures_dir / "saccer3.can.chrom.sizes"

    sys.argv = [
        "predict.py",
        str(saccer3_hdf5_file_list),
        str(chroms),
        str(test_dir),  # logdir
        "--offline",
        "--model",
        str(saccer3_fixtures_dir),
    ]
    print("Running predict.py with args:", sys.argv)
    main_module()
