"""Test module for collect_hdf5_vals.py"""

import json
import sys
from pathlib import Path
from typing import List

import pytest

from epiclass.utils.preprocessing.collect_hdf5_vals import main as main_module
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("collect_vals_test")


@pytest.fixture(name="feature_list")
def fixture_feature_list() -> List[int]:
    """Read feature list content."""
    with open(FIXTURES_DIR / "test_feature_list.json", "r", encoding="utf8") as f:
        return json.load(f)


def test_collect_hdf5_vals(test_dir: Path, feature_list: List[int]):
    """
    Test for collecting values from hdf5 files and saving them as csv.

    Args:
        tmp_path (Path): Temporary path provided by pytest fixture.
    """
    hdf5_list = test_dir / "collect_vals_test_file.list"
    chroms = FIXTURES_DIR / "hg38.noy.chrom.sizes"
    feature_list_path = FIXTURES_DIR / "test_feature_list.json"
    normalize = "--normalize"
    output_dir = test_dir

    with open(hdf5_list, "w", encoding="utf8") as f:
        f.write(
            f"{FIXTURES_DIR}/89a0dcb635f0e9740f587931437b69f1_100kb_all_none_value.hdf5\n"
        )

    # usage: collect_hdf5_vals [-h] [--feature_list FEATURE_LIST] [--normalize] [--hdf] [--csv]
    #                          hdf5_list chromsize output_dir
    sys.argv = [
        "collect_hdf5_vals.py",
        str(hdf5_list),
        str(chroms),
        str(output_dir),
        "--feature_list",
        str(feature_list_path),
        normalize,
        "--csv",
        "--hdf",
    ]

    main_module()
    expected_output_name = f"hdf5_values_{hdf5_list.stem}_{feature_list_path.stem}.csv"
    assert (test_dir / expected_output_name).exists()

    with open(test_dir / expected_output_name, "r", encoding="utf8") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert lines[0].strip() == "," + ",".join([str(x) for x in feature_list])
