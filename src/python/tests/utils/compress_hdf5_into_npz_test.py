"""Sanity-check integration test for compress_hdf5_into_npz.py after lazy migration."""
import sys
from pathlib import Path

import numpy as np
import pytest

from epiclass.utils.preprocessing.compress_hdf5_into_npz import main as main_module
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("compress_hdf5_into_npz")


@pytest.mark.slow
def test_compress_hdf5_into_npz_runs(test_dir: Path, saccer3_hdf5_file_list: Path):
    """End-to-end: register HDF5s, preload mmap, write a single .npz."""
    chroms = FIXTURES_DIR / "saccer3" / "saccer3.can.chrom.sizes"
    output_npz = test_dir / "compressed.npz"

    sys.argv = [
        "compress_hdf5_into_npz.py",
        "--hdf5_list",
        str(saccer3_hdf5_file_list),
        "--chromsizes",
        str(chroms),
        "--output",
        str(output_npz),
    ]
    main_module()

    assert output_npz.is_file(), "Expected output NPZ file."
    with np.load(output_npz, allow_pickle=False) as data:
        assert "signals" in data.files
        assert "ids" in data.files
        signals = data["signals"]
        ids = data["ids"]
        assert signals.ndim == 2
        assert signals.shape[0] == ids.shape[0]
        assert signals.shape[0] > 0
