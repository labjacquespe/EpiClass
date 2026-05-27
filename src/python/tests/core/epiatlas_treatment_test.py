"""EpiAtlas data treatment testing module."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pytest

from epiclass.core.data_source import EpiDataSource
from epiclass.core.epiatlas_constants import (
    ACCEPTED_TRACKS,
    EPIRR_LABEL,
    LEADER_TRACKS,
    OTHER_TRACKS,
)
from epiclass.core.lazy.lazy_fold_factory import (
    LazyEpiAtlasFoldFactory as EpiAtlasFoldFactory,
)
from epiclass.core.metadata import Metadata
from epiclass.utils.general_utility import write_signal_ids_to_file
from epiclass.utils.metadata_utils import count_labels_from_dset
from tests.epilap_test_data import FIXTURES_DIR, EpiAtlasTreatmentTestData


class TestEpiAtlasFoldFactory:
    """Test class EpiAtlasFoldFactory.

    Preconditions: Exact same input labels list. (raw_dset.train.encoded_labels)
    """

    @pytest.fixture(scope="class", name="test_data")
    def test_data(self) -> EpiAtlasFoldFactory:
        """Mock test EpiAtlasFoldFactory."""
        return EpiAtlasTreatmentTestData.test_data()

    @pytest.fixture(name="big_test_data")
    def big_test_data(self, tmp_path) -> EpiAtlasFoldFactory:
        """Mock big EpiAtlasFoldFactory."""
        tmp_path = Path(tmp_path) / "big_test"
        tmp_path.mkdir(exist_ok=True, parents=True)

        target_category = "harmonized_donor_sex"

        # Reduce total nb of files
        meta_path = FIXTURES_DIR / "hg38_2023-epiatlas_dfreeze_limited_categories.json"
        meta = Metadata(meta_path)

        meta.select_category_subsets(target_category, ["female", "male"])
        ids_per_class = meta.ids_per_class(target_category)
        signal_ids = [sid for id_list in ids_per_class.values() for sid in id_list[:300]]

        signal_id_file = write_signal_ids_to_file(
            signal_ids=signal_ids, logdir=str(tmp_path), name="big_test"
        )
        return EpiAtlasTreatmentTestData(
            metadata_path=meta_path,
            logdir=tmp_path,
            signal_id_list_path=signal_id_file,
        ).get_ea_handler(label_category=target_category)

    @pytest.fixture(scope="class")
    def test_metadata(self, test_data) -> Metadata:
        """Basic test metadata, using real data."""
        meta_path = test_data.epiatlas_dataset.datasource.metadata_file
        return Metadata(meta_path)

    @pytest.fixture(scope="class")
    def test_datasource(self, test_data) -> EpiDataSource:
        """Basic test datasource, using real data."""
        return test_data.epiatlas_dataset.datasource

    @staticmethod
    def create_test_data_from_datasource(
        datasource: EpiDataSource, metadata: Metadata, target_category: str
    ) -> EpiAtlasFoldFactory:
        """Create EpiAtlasFoldFactory from datasource."""
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            datasource,
            label_category=target_category,
            min_class_size=1,
            n_fold=2,
            signal_id_list=list(metadata.signal_ids),
            force_filter=True,
            test_ratio=0,
        )
        return ea_handler

    @staticmethod
    def modified_metadata(test_metadata, del_tracks: List[str]) -> Metadata:
        """Remove tracks from metadata."""
        meta = copy.deepcopy(test_metadata)
        for del_track in del_tracks:
            for signal_id, dset in list(meta.items):
                if dset["track_type"] == del_track:
                    del meta[signal_id]
        return meta

    @staticmethod
    def assert_splits(ea_handler: EpiAtlasFoldFactory):
        """Assert that splits are correct."""
        total_data = ea_handler.train_val_dset
        assert total_data.num_examples == len(set(total_data.ids))

        for dset in ea_handler.yield_split():
            trains_ids = set(dset.train.ids)
            valid_ids = set(dset.validation.ids)
            assert len(trains_ids & valid_ids) == 0, "No overlap between train and valid"

            assert len(valid_ids) == len(dset.validation.ids), "No dups in validation"

            assert dset.test.num_examples == 0, "No test set"

            assert total_data.num_examples == len(trains_ids) + len(valid_ids)

            # No EpiRR leakage across folds
            train_epirrs = {dset.train.metadata[sid][EPIRR_LABEL] for sid in trains_ids}
            valid_epirrs = {
                dset.validation.metadata[sid][EPIRR_LABEL] for sid in valid_ids
            }
            assert (
                len(train_epirrs & valid_epirrs) == 0
            ), f"EpiRR leakage: {train_epirrs & valid_epirrs}"

            # No UUID leakage (implied by EpiRR grouping, but verify)
            train_uuids = {dset.train.metadata[sid]["uuid"] for sid in trains_ids}
            valid_uuids = {dset.validation.metadata[sid]["uuid"] for sid in valid_ids}
            assert (
                len(train_uuids & valid_uuids) == 0
            ), f"UUID leakage: {train_uuids & valid_uuids}"

    def test_yield_correct_split_1(self, test_data: EpiAtlasFoldFactory):
        """Test that splits contain the correct number of training and validation samples."""
        ea_handler = test_data

        datasource = test_data.epiatlas_dataset.datasource
        metadata = test_data.epiatlas_dataset.metadata
        ea_handler2 = TestEpiAtlasFoldFactory.create_test_data_from_datasource(
            datasource,
            metadata,
            target_category=test_data.epiatlas_dataset.target_category,
        )

        for handler in [ea_handler, ea_handler2]:
            TestEpiAtlasFoldFactory.assert_splits(handler)

    @pytest.mark.slow
    def test_yield_correct_split_2(self, big_test_data: EpiAtlasFoldFactory):
        """Test that splits contain the correct number of training and validation samples, with real metadata."""
        TestEpiAtlasFoldFactory.assert_splits(big_test_data)

    @pytest.mark.parametrize("del_track,", ["raw", "pval", "fc", "Unique_minusRaw"])
    def test_yield_missing_tracks(self, test_datasource, test_metadata, del_track: str):
        """Make sure splitter can handle missing tracks."""
        meta = TestEpiAtlasFoldFactory.modified_metadata(test_metadata, [del_track])
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            test_datasource,
            label_category="biomaterial_type",
            min_class_size=2,
            n_fold=3,
            signal_id_list=list(meta.signal_ids),
            force_filter=True,
        )
        for _ in ea_handler.yield_split():
            pass

    def test_yield_only_lead(self, test_datasource, test_metadata):
        """Make sure splitter can handle multiple missing tracks."""
        meta = TestEpiAtlasFoldFactory.modified_metadata(test_metadata, ["fc", "pval"])
        labels_count = meta.label_counter("track_type")

        total_size = sum(labels_count[track_type] for track_type in ACCEPTED_TRACKS)
        total_raw_count = labels_count["raw"]

        ea_handler = EpiAtlasFoldFactory.from_datasource(
            test_datasource,
            label_category="biomaterial_type",
            min_class_size=2,
            n_fold=3,
            signal_id_list=list(meta.signal_ids),
            force_filter=True,
            test_ratio=0,
        )
        ref_dset = ea_handler.train_val_dset
        lead_tracks = len(
            [
                sample_id
                for sample_id in ref_dset.ids
                if meta[sample_id]["track_type"] in LEADER_TRACKS
            ]
        )
        other_tracks = len(
            [
                sample_id
                for sample_id in ref_dset.ids
                if meta[sample_id]["track_type"] in OTHER_TRACKS
            ]
        )

        assert total_size == lead_tracks + other_tracks

        valid_raw_sum = 0
        for dset in ea_handler.yield_split():
            trains_ids = set(dset.train.ids)
            valid_ids = set(dset.validation.ids)

            train_unique_size = len(trains_ids)
            valid_unique_size = len(valid_ids)

            assert total_size == train_unique_size + valid_unique_size

            track_type_counter = Counter(
                [meta[sid]["track_type"] for sid in dset.validation.ids]
            )
            valid_raw_sum += track_type_counter["raw"]

        assert valid_raw_sum == total_raw_count

    def test_split_by_track_type(self):
        """Test that track types are distributed correctly but same uuid stay together."""
        raise NotImplementedError

    def test_create_total_data_without_oversampling(self, test_data: EpiAtlasFoldFactory):
        """Test create_total_data without oversampling."""
        result = test_data.create_total_data(oversample=False)

        assert np.array_equal(result.ids, test_data.train_val_dset.ids)
        assert np.array_equal(
            result.materialize()[0], test_data.train_val_dset.materialize()[0]
        )
        assert np.array_equal(
            result.encoded_labels, test_data.train_val_dset.encoded_labels
        )
        assert np.array_equal(
            result.original_labels, test_data.train_val_dset.original_labels
        )

    def test_create_total_data_with_oversampling(self, test_data: EpiAtlasFoldFactory):
        """Test create_total_data with oversampling."""
        result = test_data.create_total_data(oversample=True)

        assert len(result) >= len(test_data.train_val_dset)

        biggest_class_count = count_labels_from_dset(
            test_data.train_val_dset, "biomaterial_type", from_uuid=True
        ).most_common(n=1)[0][1]

        oversampled_counts = count_labels_from_dset(
            result, "biomaterial_type", from_uuid=True
        )
        oversampled_ratios = (
            np.array(list(oversampled_counts.values())) / biggest_class_count
        )
        for ratio in oversampled_ratios:
            assert 0.8 <= ratio <= 1.2

    def test_epirr_grouping_with_oversampling(self, test_data: EpiAtlasFoldFactory):
        """Test that EpiRR grouping is maintained and oversampling works at UUID level."""
        for dset in test_data.yield_split(oversample=True):
            # EpiRR constraint still holds after oversampling
            train_epirrs = {
                dset.train.metadata[sid][EPIRR_LABEL] for sid in dset.train.ids
            }
            valid_epirrs = {
                dset.validation.metadata[sid][EPIRR_LABEL] for sid in dset.validation.ids
            }
            assert (
                len(train_epirrs & valid_epirrs) == 0
            ), f"EpiRR leakage after oversampling: {train_epirrs & valid_epirrs}"

            # Oversampled training set should be >= original
            assert dset.train.num_examples >= dset.validation.num_examples

            # Oversampling duplicates whole UUIDs, so every signal ID in training
            # should still have all its UUID-siblings present
            train_ids_list = list(dset.train.ids)
            for sid in dset.train.ids:
                uuid = dset.train.metadata[sid]["uuid"]
                # All signal IDs sharing this UUID that appear in training
                # should appear the same number of times (they're duplicated as a group)
                siblings = [
                    m for m in train_ids_list if dset.train.metadata[m]["uuid"] == uuid
                ]
                # Count occurrences of each sibling — they should all match
                sibling_counts = Counter(siblings)
                counts = list(sibling_counts.values())
                assert (
                    len(set(counts)) == 1
                ), f"UUID {uuid} siblings have inconsistent duplication: {sibling_counts}"
