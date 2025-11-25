"""Module defining data factory classes, for preprocessing of samples and metadata."""
from __future__ import annotations

import collections
import math
from typing import List

import numpy as np
from sklearn import preprocessing

from epiclass.core.data.dataset import DataSet
from epiclass.core.data.eager import KnownData
from epiclass.core.data_source import EpiDataSource
from epiclass.core.loaders.hdf5_loader import Hdf5Loader
from epiclass.core.metadata import Metadata


class DataSetFactory:
    """Creation of DataSet from different sources."""

    @classmethod
    def from_epidata(
        cls,
        datasource: EpiDataSource,
        metadata: Metadata,
        label_category: str,
        onehot=False,
        oversample=False,
        normalization=True,
        min_class_size=3,
        validation_ratio=0.1,
        test_ratio=0.1,
    ) -> DataSet:
        """Return DataSet created from EpiData."""
        return EpiData(
            datasource,
            metadata,
            label_category,
            onehot,
            oversample,
            normalization,
            min_class_size,
            validation_ratio,
            test_ratio,
        ).dataset


class EpiData:
    """Used to load and preprocess epigenomic data. Data factory.

    Test ratio computed from validation ratio and test ratio. Be sure to set both correctly.
    """

    def __init__(
        self,
        datasource: EpiDataSource,
        metadata: Metadata,
        label_category: str,
        onehot=False,
        oversample=False,
        normalization=True,
        min_class_size=3,
        validation_ratio=0.1,
        test_ratio=0.1,
    ):
        self._label_category = label_category
        self._oversample = oversample
        self._assert_ratios(
            val_ratio=validation_ratio, test_ratio=test_ratio, verbose=True
        )

        # load
        self._metadata = self._load_metadata(metadata)
        self._files = Hdf5Loader.read_list(datasource.hdf5_file)

        # preprocess
        self._keep_meta_overlap()
        self._metadata.remove_small_classes(min_class_size, self._label_category)

        self._hdf5s = (
            Hdf5Loader(datasource.chromsize_file, normalization)
            .load_hdf5s(datasource.hdf5_file, md5s=list(self._files.keys()), strict=True)
            .signals
        )

        self._sorted_classes = self._metadata.unique_classes(label_category)

        # TODO : Create encoder class separate from EpiData
        encoder = EpiData._make_encoder(self._sorted_classes, onehot=onehot)

        self._split_data(validation_ratio, test_ratio, encoder)

    @property
    def dataset(self) -> DataSet:
        """Return data/metadata processed into separate sets."""
        return DataSet(self._train, self._validation, self._test, self._sorted_classes)

    def _assert_ratios(self, val_ratio, test_ratio, verbose):
        """Verify that splitting ratios make sense."""
        train_ratio = 1 - val_ratio - test_ratio
        if val_ratio + test_ratio > 1:
            raise ValueError(
                f"Validation and test ratios are bigger than 100%: {val_ratio} and {test_ratio}"
            )
        if verbose:
            print(
                f"training/validation/test split: {train_ratio*100}%/{val_ratio*100}%/{test_ratio*100}%"
            )
        if np.isclose(train_ratio, 0.0):
            self._oversample = False
            print("Forcing oversampling off, training set is empty.")

    def _load_metadata(self, metadata: Metadata) -> Metadata:
        metadata.remove_missing_labels(self._label_category)
        return metadata

    def _keep_meta_overlap(self):
        self._remove_md5_without_hdf5()
        self._remove_hdf5_without_md5()

    def _remove_md5_without_hdf5(self):
        self._metadata.apply_filter(lambda item: item[0] in self._files)  # type: ignore

    def _remove_hdf5_without_md5(self):
        self._files = {md5: self._files[md5] for md5 in self._metadata.md5s}

    @staticmethod
    def _create_onehot_dict(classes: List[str]) -> dict:
        """Returns {label:onehot vector} dict corresponding given classes.
        TODO : put into an encoder class
        Onehot vectors defined with given classes, no sorting done.
        """
        onehot_dict = {}
        for i, label in enumerate(classes):
            onehot = np.zeros(len(classes))
            onehot[i] = 1
            onehot_dict[label] = onehot
        return onehot_dict

    @staticmethod
    def _make_encoder(classes, onehot=False):
        """Return an int (default) or onehot vector encoder that takes label sets as entry.
        TODO : put into an encoder class
        Classes are sorted beforehand.
        """
        labels = sorted(classes)
        if onehot:
            encoding = EpiData._create_onehot_dict(labels)

            def to_onehot(labels):
                return [encoding[label] for label in labels]  # type: ignore

            return to_onehot

        # else int mapping
        encoding = preprocessing.LabelEncoder().fit(labels)

        def to_int(labels):
            if labels:
                return encoding.transform(labels)
            return []

        return to_int

    def _split_md5s(self, validation_ratio, test_ratio):
        """Return md5s for each set, according to given ratios."""
        size_all_dict = self._metadata.label_counter(self._label_category)
        data = self._metadata.md5_per_class(self._label_category)

        # A minimum of 3 examples are needed for each label (1 for each set), when splitting into three sets
        for label, size in size_all_dict.items():
            if size < 3:
                print(f"The label `{label}` countains only {size} datasets.")

        # The point is to try to create indexes for the slices of each different class
        # the indexes would split this way [valid, test, training]
        size_validation_dict = collections.Counter(
            {
                label: math.ceil(size * validation_ratio)
                for label, size in size_all_dict.items()
            }
        )
        size_test_dict = collections.Counter(
            {label: math.ceil(size * test_ratio) for label, size in size_all_dict.items()}
        )

        # sum(size_validation_dict, size_test_dict) ignores zeros, giving counter without labels, which breaks following lambda
        split_index_dict = collections.Counter(size_validation_dict)
        split_index_dict.update(size_test_dict)

        # Will grab the indexes from the dicts and return md5 slices
        # no end means : [i:None]=[i:]=slice from i to end
        slice_data = lambda begin={}, end={}: sum(
            [
                data[label][begin.get(label, 0) : end.get(label, None)]
                for label in size_all_dict.keys()
            ],
            [],
        )

        validation_md5s = slice_data(end=size_validation_dict)
        test_md5s = slice_data(begin=size_validation_dict, end=split_index_dict)
        train_md5s = slice_data(begin=split_index_dict)

        assert len(self._metadata.md5s) == len(
            set(sum([train_md5s, validation_md5s, test_md5s], []))
        )

        return [train_md5s, validation_md5s, test_md5s]

    def _split_data(self, validation_ratio, test_ratio, encoder):
        """Split loaded data into three sets : Training/Validation/Test.

        The encoder/encoding function for a label list needs to be provided.
        """
        train_md5s, validation_md5s, test_md5s = self._split_md5s(
            validation_ratio, test_ratio
        )

        # separate hdf5 files
        train_signals = [self._hdf5s[md5] for md5 in train_md5s]
        validation_signals = [self._hdf5s[md5] for md5 in validation_md5s]
        test_signals = [self._hdf5s[md5] for md5 in test_md5s]

        # separate label values
        train_labels = [self._metadata[md5][self._label_category] for md5 in train_md5s]
        validation_labels = [
            self._metadata[md5][self._label_category] for md5 in validation_md5s
        ]
        test_labels = [self._metadata[md5][self._label_category] for md5 in test_md5s]

        if self._oversample:
            train_signals, train_labels, idxs = EpiData.oversample_data(
                train_signals, train_labels
            )
            train_md5s = np.take(train_md5s, idxs, axis=0)

        encoded_labels = [
            encoder(labels) for labels in [train_labels, validation_labels, test_labels]
        ]

        self._train = KnownData(
            train_md5s, train_signals, encoded_labels[0], train_labels, self._metadata
        )
        self._validation = KnownData(
            validation_md5s,
            validation_signals,
            encoded_labels[1],
            validation_labels,
            self._metadata,
        )
        self._test = KnownData(
            test_md5s, test_signals, encoded_labels[2], test_labels, self._metadata
        )

        print(f"training size {len(train_labels)}")
        print(f"validation size {len(validation_labels)}")
        print(f"test size {len(test_labels)}")

    @staticmethod
    def oversample_data(X, y):
        """Return oversampled data with sampled indexes. X=signals, y=targets."""
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)  # type: ignore
        return X_resampled, y_resampled, ros.sample_indices_
