"""Test module for ave_predict.py (reconstruction-error scoring of new data)."""
# Deliberately parallel to its classifier counterpart.
# pylint: disable=duplicate-code
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epiclass.core.model_checkpoint import last_checkpoint_path
from epiclass.mains.ave_predict import main as main_module
from tests.epilap_test_data import SACCER3_AVE_DIR, SACCER3_DIR

SACCER3_CHROMS = SACCER3_DIR / "saccer3.can.chrom.sizes"


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("ave_predict")


def _assert_valid_scores(
    csv_path: Path, expected_rows: int | None = None
) -> pd.DataFrame:
    """Check the scores CSV shape: ID + reconstruction_error, finite and non-negative."""
    df = pd.read_csv(csv_path, index_col="ID")
    assert df.index.name == "ID"
    # Raw scores only: no threshold, no outlier flag (that decision lives downstream).
    assert list(df.columns) == ["reconstruction_error"]
    assert len(df) > 0
    if expected_rows is not None:
        assert len(df) == expected_rows
    errors = df["reconstruction_error"].to_numpy()
    assert np.all(np.isfinite(errors))
    assert np.all(errors >= 0)
    return df


@pytest.mark.slow
def test_ave_predict_single_sample(test_dir: Path, saccer3_small_hdf5_file_list: Path):
    """Basic AVE scoring succeeds on the single-sample HDF5 path."""
    sys.argv = [
        "ave_predict.py",
        "--hdf5",
        str(saccer3_small_hdf5_file_list),
        "--outdir",
        str(test_dir),
        "--chromsize",
        str(SACCER3_CHROMS),
        "--mmap_dir",
        str(test_dir / "mmap_cache"),
        "--model",
        str(SACCER3_AVE_DIR),
    ]
    print("Running ave_predict.py with args:", sys.argv)
    main_module()

    csv_outputs = list(test_dir.glob("*_ave_scores_*.csv"))
    assert csv_outputs, "Expected an ave_scores CSV in outdir."
    # The output name carries the model's training provenance: the run id recovered from
    # the checkpoint path and the checkpoint stem.
    ckpt = last_checkpoint_path(SACCER3_AVE_DIR)
    assert ckpt is not None
    assert ckpt.parent.parent.name in csv_outputs[0].name
    assert ckpt.stem in csv_outputs[0].name

    n_samples = len(saccer3_small_hdf5_file_list.read_text().split())
    _assert_valid_scores(csv_outputs[0], expected_rows=n_samples)


@pytest.mark.slow
def test_ave_predict_ckpt_file(test_dir: Path, saccer3_small_hdf5_file_list: Path):
    """Scoring when --model points directly at a .ckpt file, bypassing the list."""
    ckpt = last_checkpoint_path(SACCER3_AVE_DIR)
    assert ckpt is not None and ckpt.is_file(), "Fixture checkpoint not found."
    sys.argv = [
        "ave_predict.py",
        "--hdf5",
        str(saccer3_small_hdf5_file_list),
        "--outdir",
        str(test_dir),
        "--chromsize",
        str(SACCER3_CHROMS),
        "--mmap_dir",
        str(test_dir / "mmap_cache"),
        "--model",
        str(ckpt),
    ]
    print("Running ave_predict.py with args:", sys.argv)
    main_module()

    csv_outputs = list(test_dir.glob("*_ave_scores_*.csv"))
    assert csv_outputs, "Expected an ave_scores CSV in outdir."
    _assert_valid_scores(csv_outputs[0])


@pytest.mark.slow
def test_ave_predict_chunked(test_dir: Path, saccer3_chunked_dir: Path):
    """End-to-end --chunked scoring path (ChunkedHdf5Loader branch)."""
    sys.argv = [
        "ave_predict.py",
        "--hdf5",
        str(saccer3_chunked_dir),  # directory of chunk_*.h5
        "--outdir",
        str(test_dir),
        "--chunked",
        "--model",
        str(SACCER3_AVE_DIR),
    ]
    print("Running ave_predict.py with args:", sys.argv)
    main_module()

    csv_outputs = list(test_dir.glob("*_ave_scores_*.csv"))
    assert csv_outputs, "Expected an ave_scores CSV in outdir."
    _assert_valid_scores(csv_outputs[0])
