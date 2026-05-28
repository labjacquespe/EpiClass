"""Sanity-check integration test for compute_umap.py after lazy migration."""
import sys
from pathlib import Path

import pytest

from epiclass.utils.embedding.compute_umap import main as main_module
from tests.epilap_test_data import FIXTURES_DIR

pytestmark = pytest.mark.filterwarnings(
    r"ignore:n_jobs value 1 overridden to 1 by setting random_state.*:UserWarning"
)


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("compute_umap")


@pytest.mark.slow
@pytest.mark.embedding
def test_compute_umap_single_sample(test_dir: Path, saccer3_small_hdf5_file_list: Path):
    """Single-sample path: mmap (copy-on-write), build knn, fit one UMAP.

    ``--max_embeddings 1`` keeps the sweep to a single fit; the full
    12-embedding sweep would take minutes. We use the 100-sample subset
    fixture because KNN(correlation) scales superlinearly with sample count
    and dominates the test time on the full 1055-file list.
    """
    chroms = FIXTURES_DIR / "saccer3" / "saccer3.can.chrom.sizes"

    sys.argv = [
        "compute_umap.py",
        str(saccer3_small_hdf5_file_list),
        "--output",
        str(test_dir),
        "--chromsize",
        str(chroms),
        "--max_embeddings",
        "1",
    ]
    main_module()

    assert (test_dir / "precomputed_knn_100.pkl").is_file()
    assert (test_dir / "pickle_requirements.txt").is_file()
    embeddings = list(test_dir.glob("embedding_*.pkl"))
    assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"


@pytest.mark.slow
@pytest.mark.embedding
def test_compute_umap_chunked(test_dir: Path, saccer3_chunked_dir: Path):
    """Chunked path: materialize via load_batch, build knn, fit one UMAP."""
    sys.argv = [
        "compute_umap.py",
        str(saccer3_chunked_dir),
        "--output",
        str(test_dir),
        "--chunked",
        "--max_embeddings",
        "1",
    ]
    main_module()

    assert (test_dir / "precomputed_knn_100.pkl").is_file()
    embeddings = list(test_dir.glob("embedding_*.pkl"))
    assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"
