"""pytest setup/configuration"""
# pylint: disable=unused-argument, import-outside-toplevel
from __future__ import annotations

import os
import re
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Iterator

import pytest

from epiclass.core.data.dataset import DataSet
from epiclass.core.lazy.lazy_fold_factory import (
    LazyEpiAtlasFoldFactory as EpiAtlasFoldFactory,
)
from epiclass.core.model_pytorch import LightningDenseClassifier
from tests.epilap_test_data import (
    DEFAULT_TEST_LOGDIR,
    FIXTURES_DIR,
    EpiAtlasTreatmentTestData,
)

RUN_LOGDIR = DEFAULT_TEST_LOGDIR / uuid.uuid4().hex

SACCER3_DIR = FIXTURES_DIR / "saccer3"

# Some scripts (epiatlas_training_no_valid.py) call LazyEpiAtlasFoldFactory
# without passing mmap_dir, so the loader writes signals_*.npy under the cwd's
# ./mmap_cache. A leftover from a previous run with different mock data shapes
# causes preload_all() to skip generation and downstream loads to fail.
LOCAL_MMAP_DIR = Path("./mmap_cache")


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="do not perform cleanup after tests",
    )


def pytest_exception_interact(node, call, report):
    """Intercept FileNotFoundError only when it points at the (un)extracted
    fixtures directory, so we don't swallow tracebacks for unrelated missing
    files (e.g. stale mmap cache paths) that would otherwise be hard to debug.
    """
    if isinstance(call.excinfo.value, FileNotFoundError):
        missing = str(call.excinfo.value.filename or call.excinfo.value)
        fixtures_marker = str(FIXTURES_DIR)
        if fixtures_marker in missing or "fixtures.tar" in missing:
            report.longrepr = (
                f"\nFileNotFoundError intercepted:\n"
                f"  {call.excinfo.value}\n"
                f"Hint: Did you forget to extract fixtures.tar.zstd? "
                f"Use zstd -d fixtures.tar.zstd\n"
            )


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before performing
    collection and entering the run test loop.
    """
    # Wipe the cwd-relative mmap cache so stale signals_*.npy from a previous
    # run (different dataset shape) don't poison preload_all().
    if LOCAL_MMAP_DIR.exists():
        shutil.rmtree(LOCAL_MMAP_DIR, ignore_errors=True)

    if not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.iterdir()):
        # Stop tests immediately
        message = (
            f"Required fixtures directory '{FIXTURES_DIR}' is missing or empty.\n"
            "Please ensure the fixtures are uncompressed and available before running tests.\n"
            "Search for: fixtures.tar.zstd"
        )
        pytest.exit(reason=message, returncode=1)
    checkpoint_file = FIXTURES_DIR / "saccer3" / "best_checkpoint.list"
    if not checkpoint_file.exists():
        checkpoint_template = checkpoint_file.parent / "best_checkpoint_template.list"
        lines = checkpoint_template.read_text().splitlines()
        lines = [
            re.sub(r"THIS_FOLDER", str(checkpoint_file.parent), line) for line in lines
        ]
        checkpoint_file.write_text("\n".join(lines))


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before returning the exit status
    to the system.
    """
    if session.config.getoption("--no-cleanup"):
        print("Skipping cleanup as per --no-cleanup option.")
        return

    # Remove test logdir
    if RUN_LOGDIR.exists():
        shutil.rmtree(RUN_LOGDIR)

    # Drop the cwd-relative mmap cache so the next session starts clean too.
    if LOCAL_MMAP_DIR.exists():
        shutil.rmtree(LOCAL_MMAP_DIR, ignore_errors=True)


def nottest(obj):
    """Decorator to mark a function or method as not a test"""
    obj.__test__ = False
    return obj


@pytest.fixture(scope="session", autouse=True, name="mk_logdir")
def make_specific_logdir(tmp_path_factory):
    """Return fct to create test subdirectory."""

    def _make_specific_logdir(name: str) -> Path:
        logdir = tmp_path_factory.mktemp(name)
        return logdir

    return _make_specific_logdir


@pytest.fixture(scope="session", name="test_epiatlas_data_handler")
def fixture_epiatlas_data_handler() -> EpiAtlasFoldFactory:
    """Return mock data handler. (in /tmp)."""
    return EpiAtlasTreatmentTestData.default_test_data(logdir=RUN_LOGDIR)


@pytest.fixture(scope="session", name="test_epiatlas_dataset")
def fixture_epiatlas_dataset(
    test_epiatlas_data_handler: EpiAtlasFoldFactory,
) -> DataSet:
    """Return mock dataset."""
    return next(test_epiatlas_data_handler.yield_split())


@pytest.fixture(scope="session", name="test_NN_model")
def fixture_NN_model(
    test_epiatlas_dataset: DataSet, mk_logdir
) -> LightningDenseClassifier:
    """Return small test neural network"""
    test_mapping = mk_logdir("model") / "test_mapping.tsv"
    test_epiatlas_dataset.save_mapping(test_mapping)
    test_mapping = test_epiatlas_dataset.load_mapping(test_mapping)

    return LightningDenseClassifier(
        input_size=test_epiatlas_dataset.train.signal_length,
        output_size=len(test_epiatlas_dataset.classes),
        mapping=test_mapping,
        hparams={},
        nb_layer=1,
        hl_units=100,
    )


@pytest.fixture(scope="session", name="extracted_hdf5_dir")
def saccer3_extracted_hdf5_dir(tmp_path_factory) -> Iterator[Path]:
    """
    Extract saccer3 HDF5 files once per pytest worker (xdist-safe).
    Returns the directory containing extracted HDF5 files.
    """
    # Assign one folder per worker
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    extract_root = tmp_path_factory.getbasetemp() / f"saccer3_{worker_id}"
    extracted_dir = extract_root / "saccer3_2016-07"

    if not extracted_dir.exists():
        hdf5_dir = SACCER3_DIR / "hdf5"
        archive = hdf5_dir / "saccer3_2016-07.tar.xz"
        extract_root.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(path=extract_root)

    yield extracted_dir

    # Teardown using makefile to avoid issues when running in parallel


@pytest.fixture(scope="function", name="saccer3_hdf5_file_list")
def hdf5_file_list(
    extracted_hdf5_dir: Path,
    test_dir: Path,
) -> Path:
    """
    Write a list of HDF5 files and return its path.
    """
    hdf5_files = sorted(extracted_hdf5_dir.glob("*.hdf5"))
    file_list = test_dir / "hdf5_files.list"

    with file_list.open("w") as f:
        for hdf5_file in hdf5_files:
            f.write(f"{hdf5_file}\n")

    return file_list


@pytest.fixture(scope="session", name="saccer3_chunked_dir")
def fixture_saccer3_chunked_dir(tmp_path_factory, extracted_hdf5_dir: Path) -> Path:
    """Convert saccer3 single-sample HDF5s into chunked format for tests.

    Normalization mirrors what LazyHdf5Loader applies on the single-sample
    path, so the chunked signals match the expected distribution. Session-
    scoped (per xdist worker) so the ~3-9s conversion is paid once across
    predict_test and the embedding util tests.
    """
    from epiclass.utils.preprocessing.hdf5_chunks_creation import convert

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    chunk_root = tmp_path_factory.getbasetemp() / f"chunked_{worker_id}"
    chunk_dir = chunk_root / "chunked"

    if chunk_dir.exists() and any(chunk_dir.glob("chunk_*.h5")):
        return chunk_dir

    # Build the hdf5 list once, reusing the per-worker extracted fixture.
    chunk_root.mkdir(parents=True, exist_ok=True)
    hdf5_list = chunk_root / "hdf5_files.list"
    with hdf5_list.open("w") as f:
        for hdf5_file in sorted(extracted_hdf5_dir.glob("*.hdf5")):
            f.write(f"{hdf5_file}\n")

    convert(
        hdf5_list=hdf5_list,
        chrom_file=SACCER3_DIR / "saccer3.can.chrom.sizes",
        output_dir=chunk_dir,
        samples_per_chunk=50,
        normalize=True,
        strict=True,
    )
    return chunk_dir
