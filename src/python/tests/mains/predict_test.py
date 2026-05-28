"""Test module for predict.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epiclass.predict import main as main_module
from tests.epilap_test_data import FIXTURES_DIR

SACCER3_FIXTURES_DIR = FIXTURES_DIR / "saccer3"
SACCER3_CHROMS = SACCER3_FIXTURES_DIR / "saccer3.can.chrom.sizes"


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("predict")


@pytest.mark.slow
def test_predict_single_sample(test_dir: Path, saccer3_small_hdf5_file_list: Path):
    """Test if basic prediction succeeds (single-sample HDF5 path).

    Uses the 100-sample subset — prediction is per-sample so 100 is enough
    to exercise the load/predict/csv-write path end-to-end.
    """
    sys.argv = [
        "predict.py",
        str(saccer3_small_hdf5_file_list),
        str(test_dir),  # logdir
        "--chromsize",
        str(SACCER3_CHROMS),
        "--mmap_dir",
        str(test_dir / "mmap_cache"),
        "--offline",
        "--model",
        str(SACCER3_FIXTURES_DIR),
    ]
    print("Running predict.py with args:", sys.argv)
    main_module()


@pytest.mark.slow
def test_predict_chunked(test_dir: Path, saccer3_chunked_dir: Path):
    """Test end-to-end --chunked prediction path.

    Exercises _build_loader's ChunkedHdf5Loader branch and verifies that
    analysis.Analysis.write_test_prediction produces the expected CSV
    (Predicted class + one column per class softmax probability).
    """
    sys.argv = [
        "predict.py",
        str(saccer3_chunked_dir),  # directory of chunk_*.h5
        str(test_dir),  # logdir
        "--chunked",
        "--offline",
        "--model",
        str(SACCER3_FIXTURES_DIR),
    ]
    print("Running predict.py with args:", sys.argv)
    main_module()

    csv_outputs = list(test_dir.glob("*_test_prediction_*.csv"))
    assert csv_outputs, "Expected a test_prediction CSV in logdir."

    df = pd.read_csv(csv_outputs[0], index_col="ID")
    assert df.index.name == "ID"
    assert "Predicted class" in df.columns
    assert "True class" not in df.columns  # test predictions drop true labels
    assert len(df) > 0

    # Remaining columns are the per-class softmax probabilities.
    prob_cols = [c for c in df.columns if c != "Predicted class"]
    assert len(prob_cols) >= 2, "Expected at least one probability column per class."
    row_sums = df[prob_cols].sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)
