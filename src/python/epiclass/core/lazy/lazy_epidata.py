"""Modernized EpiData factory for lazy loading.

This version creates LazyKnownData objects instead of loading all signals into memory.
"""
# pylint: disable=too-many-positional-arguments`
from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import List

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing

from epiclass.core.data.dataset import DataSet
from epiclass.core.data_source import EpiDataSource
from epiclass.core.metadata import Metadata

from .lazy_data_classes import LazyKnownData
from .lazy_hdf5_loader import LazyHdf5Loader


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
        mmap_dir: Path | str | None = None,
    ) -> DataSet:
        """Return DataSet created from EpiData with lazy loading.

        Args:
            datasource: EpiDataSource object
            metadata: Metadata object
            label_category: Target label category
            onehot: Use one-hot encoding (not recommended for lazy loading)
            oversample: Oversample minority classes
            normalization: Normalize signals
            min_class_size: Minimum samples per class
            validation_ratio: Validation split ratio
            test_ratio: Test split ratio
            mmap_dir: Directory for the memory-mapped .npy file. Defaults to
                ./mmap_cache. On HPC, set to $SLURM_TMPDIR for fast local storage.
        """
        return LazyEpiData(
            datasource,
            metadata,
            label_category,
            onehot,
            oversample,
            normalization,
            min_class_size,
            validation_ratio,
            test_ratio,
            mmap_dir,
        ).dataset


class LazyEpiData:
    """Lazy-loading data factory for epigenomic data.

    This version doesn't load signals into memory. Instead, it creates
    LazyKnownData objects that load signals on-demand.
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
        mmap_dir: Path | str | None = None,
    ):
        self._label_category = label_category
        self._oversample = oversample
        self._assert_ratios(
            val_ratio=validation_ratio, test_ratio=test_ratio, verbose=True
        )

        # Load metadata
        self._metadata = self._load_metadata(metadata)
        self._files = LazyHdf5Loader.read_list(datasource.hdf5_file)

        # Preprocess metadata
        self._keep_meta_overlap()
        self._metadata.remove_small_classes(min_class_size, self._label_category)

        # Create lazy loader (doesn't load data yet!)
        self._loader = LazyHdf5Loader(
            datasource.chromsize_file,
            normalization,
            mmap_dir=mmap_dir,
        )
        self._loader.register_hdf5s(
            datasource.hdf5_file, md5s=list(self._files.keys()), strict=True
        )
        # Convert registered HDF5s to a single memory-mapped .npy file.
        # Skips if the file already exists (idempotent).
        self._loader.preload_all()

        self._sorted_classes = self._metadata.unique_classes(label_category)

        # Create encoder
        encoder = LazyEpiData._make_encoder(self._sorted_classes, onehot=onehot)

        # Split data (creates lazy data objects)
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
        self._metadata.apply_filter(lambda item: item[0] in self._files)

    def _remove_hdf5_without_md5(self):
        self._files = {md5: self._files[md5] for md5 in self._metadata.md5s}

    @staticmethod
    def _create_onehot_dict(classes: List[str]) -> dict:
        """Returns {label:onehot vector} dict corresponding given classes."""
        onehot_dict = {}
        for i, label in enumerate(classes):
            onehot = np.zeros(len(classes))
            onehot[i] = 1
            onehot_dict[label] = onehot
        return onehot_dict

    @staticmethod
    def _make_encoder(classes, onehot=False):
        """Return an int (default) or onehot vector encoder."""
        labels = sorted(classes)
        if onehot:
            encoding = LazyEpiData._create_onehot_dict(labels)

            def to_onehot(labels):
                return [encoding[label] for label in labels]

            return to_onehot

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

        for label, size in size_all_dict.items():
            if size < 3:
                print(f"The label `{label}` contains only {size} datasets.")

        size_validation_dict = collections.Counter(
            {
                label: math.ceil(size * validation_ratio)
                for label, size in size_all_dict.items()
            }
        )
        size_test_dict = collections.Counter(
            {label: math.ceil(size * test_ratio) for label, size in size_all_dict.items()}
        )

        split_index_dict = collections.Counter(size_validation_dict)
        split_index_dict.update(size_test_dict)

        def slice_data(begin={}, end={}):  # pylint: disable=dangerous-default-value
            """Will grab the indexes from the dicts and return md5 slices.
            No end means: [i:None]=[i:]=slice from i to end.
            """
            return sum(
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
        """Split data into three sets WITHOUT loading signals.

        Key difference from old version: We only store MD5s and metadata,
        not the actual signals.
        """
        train_md5s, validation_md5s, test_md5s = self._split_md5s(
            validation_ratio, test_ratio
        )

        # Get labels for each set
        train_labels = [self._metadata[md5][self._label_category] for md5 in train_md5s]
        validation_labels = [
            self._metadata[md5][self._label_category] for md5 in validation_md5s
        ]
        test_labels = [self._metadata[md5][self._label_category] for md5 in test_md5s]

        # Handle oversampling (only affects MD5 list, not loaded data)
        if self._oversample:
            train_md5s, train_labels = LazyEpiData.oversample_md5s(
                train_md5s, train_labels
            )

        # Encode labels
        encoded_labels = [
            encoder(labels) for labels in [train_labels, validation_labels, test_labels]
        ]

        # Create lazy data objects (no signal loading!)
        self._train = LazyKnownData(
            train_md5s, self._loader, encoded_labels[0], train_labels, self._metadata
        )
        self._validation = LazyKnownData(
            validation_md5s,
            self._loader,
            encoded_labels[1],
            validation_labels,
            self._metadata,
        )
        self._test = LazyKnownData(
            test_md5s, self._loader, encoded_labels[2], test_labels, self._metadata
        )

        print(f"training size {len(train_labels)}")
        print(f"validation size {len(validation_labels)}")
        print(f"test size {len(test_labels)}")

    @staticmethod
    def oversample_md5s(md5s: List[str], labels: List[str]):
        """Oversample MD5s (not signals) to balance classes.

        This is much more memory efficient than oversampling loaded signals.
        """
        # Need to reshape for sklearn
        md5s_array = np.array(md5s).reshape(-1, 1)

        ros = RandomOverSampler(random_state=42)
        md5s_resampled, labels_resampled = ros.fit_resample(md5s_array, labels)

        # Flatten back to list
        md5s_resampled = md5s_resampled.flatten().tolist()
        labels_resampled = list(labels_resampled)

        return md5s_resampled, labels_resampled
