"""Sanity-check integration test for compute_pca.py after lazy migration."""
import sys
from pathlib import Path

import pytest

from epiclass.utils.embedding.compute_pca import main as main_module
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("compute_pca")


@pytest.mark.slow
@pytest.mark.embedding
def test_compute_pca_single_sample(test_dir: Path, saccer3_small_hdf5_file_list: Path):
    """Single-sample path: register HDF5s, preload mmap, run IPCA, save skops.

    Uses the 100-sample subset fixture — IPCA scales linearly with samples
    but the full 1055-file list adds 10s of overhead without coverage.
    """
    chroms = FIXTURES_DIR / "saccer3" / "saccer3.can.chrom.sizes"

    sys.argv = [
        "compute_pca.py",
        str(saccer3_small_hdf5_file_list),
        str(test_dir),
        "--chromsize",
        str(chroms),
        "--batch_size",
        "256",
    ]
    main_module()

    fit_files = list(test_dir.glob("IPCA_fit_n*.skops"))
    x_files = list(test_dir.glob("X_IPCA_n*.skops"))
    assert fit_files, "Expected an IPCA fit file in output dir."
    assert x_files, "Expected an X_IPCA file in output dir."
    assert (test_dir / "IPCA_saved_files_requirements.txt").is_file()


@pytest.mark.slow
@pytest.mark.embedding
def test_compute_pca_chunked(test_dir: Path, saccer3_chunked_dir: Path):
    """Chunked path: partial_fit/transform per chunk file; no chromsize needed."""
    sys.argv = [
        "compute_pca.py",
        str(saccer3_chunked_dir),
        str(test_dir),
        "--chunked",
        "--batch_size",
        "256",
    ]
    main_module()

    fit_files = list(test_dir.glob("IPCA_fit_n*.skops"))
    x_files = list(test_dir.glob("X_IPCA_n*.skops"))
    assert fit_files, "Expected an IPCA fit file in output dir."
    assert x_files, "Expected an X_IPCA file in output dir."
