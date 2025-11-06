"""pytest setup/configuration"""
# pylint: disable=unused-argument
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

import pytest

from epiclass.core.data import DataSet
from epiclass.core.epiatlas_treatment import EpiAtlasFoldFactory
from epiclass.core.model_pytorch import LightningDenseClassifier
from tests.epilap_test_data import (
    DEFAULT_TEST_LOGDIR,
    FIXTURES_DIR,
    EpiAtlasTreatmentTestData,
)

# def pytest_collection_modifyitems(session, config, items):
#     """Ignore certain names from collection."""
#     items[:] = [item for item in items if item.name != "test_logdir"]

RUN_LOGDIR = DEFAULT_TEST_LOGDIR / uuid.uuid4().hex


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="do not perform cleanup after tests",
    )


def pytest_exception_interact(node, call, report):
    """Intercept test exceptions and customize FileNotFoundError messages."""
    if isinstance(call.excinfo.value, FileNotFoundError):
        report.longrepr = (
            f"\nFileNotFoundError intercepted:\n"
            f"  {call.excinfo.value}\n"
            f"Hint: Did you forget to extract fixtures.tar.xz?\n"
        )


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before performing
    collection and entering the run test loop.
    """
    if not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.iterdir()):
        # Stop tests immediately
        message = (
            f"Required fixtures directory '{FIXTURES_DIR}' is missing or empty.\n"
            "Please ensure the fixtures are uncompressed and available before running tests.\n"
            "Search for: fixtures.tar.xz"
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
        input_size=test_epiatlas_dataset.train.signals.shape[1],
        output_size=len(test_epiatlas_dataset.classes),
        mapping=test_mapping,
        hparams={},
        nb_layer=1,
        hl_units=100,
    )


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
