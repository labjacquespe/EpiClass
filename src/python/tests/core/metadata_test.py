"""Test module for metadata module."""
from __future__ import annotations

import os
import tempfile

import pytest

from epiclass.core.lazy.lazy_fold_factory import (
    LazyEpiAtlasFoldFactory as EpiAtlasFoldFactory,
)
from epiclass.core.metadata import UUIDMetadata, env_filtering

# pytestmark = pytest.mark.filterwarnings("ignore:.*Cannot read file directly.*")


DONOR_SEX = "sex"
ASSAY = "assay"
TISSUE_TYPE = "tissue_type"


@pytest.fixture(name="test_meta")
def test_metadata(test_epiatlas_data_handler: EpiAtlasFoldFactory) -> UUIDMetadata:
    """Return mock metadata."""
    return test_epiatlas_data_handler.epiatlas_dataset.metadata


def test_env_filtering_assay_list(test_meta: UUIDMetadata):
    """Test assay list filtering."""
    print("Before filtering")
    nb_before = len(test_meta)
    test_meta.display_labels(ASSAY)
    test_meta.display_labels(DONOR_SEX)
    os.environ["ASSAY_LIST"] = '["h3k27ac", "h3k27me3"]'
    _ = env_filtering(test_meta, DONOR_SEX)
    del os.environ["ASSAY_LIST"]

    print("After filtering")
    test_meta.display_labels(ASSAY)
    test_meta.display_labels(DONOR_SEX)
    assert len(test_meta) < nb_before


def test_env_filtering_exclude_list(test_meta: UUIDMetadata):
    """Test exclude list filtering."""
    nb_before = len(test_meta)
    os.environ["EXCLUDE_LIST"] = '["other", "--", "NA", ""]'
    cat = TISSUE_TYPE
    _ = env_filtering(test_meta, cat)
    del os.environ["EXCLUDE_LIST"]
    assert len(test_meta) < nb_before


def test_env_filtering_label_list(test_meta: UUIDMetadata):
    """Test label list filtering."""
    nb_before = len(test_meta)
    os.environ["LABEL_LIST"] = '["female", "male"]'
    cat = DONOR_SEX
    _ = env_filtering(test_meta, cat)
    del os.environ["LABEL_LIST"]
    test_meta.display_labels(cat)
    assert len(test_meta) < nb_before


def test_env_filtering_remove_tracks(test_meta: UUIDMetadata):
    """Test remove tracks filtering."""
    nb_before = len(test_meta)
    os.environ["REMOVE_TRACKS"] = '["fc", "pval"]'
    cat = "track_type"
    _ = env_filtering(test_meta, cat)
    del os.environ["REMOVE_TRACKS"]
    test_meta.display_labels(cat)
    assert len(test_meta) < nb_before


def test_save_load_marshal(test_meta: UUIDMetadata):
    """Test save and load through Marshal."""
    meta_save_file = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        mode="wb", delete=False
    )
    test_meta.save_marshal(meta_save_file.name)
    meta_save_file.close()

    for _ in range(2):
        meta = UUIDMetadata.from_marshal(meta_save_file.name)
        assert meta == test_meta
        assert len(meta) == len(test_meta)

    os.remove(meta_save_file.name)
