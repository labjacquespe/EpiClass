"""Module to define data/datasets processing and representation classes."""
# pylint: disable=unnecessary-lambda-assignment, too-many-positional-arguments
from __future__ import annotations

import abc
import collections
import copy
import math
import sys
from typing import Generic, List, Type, TypeVar

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing

from ..data_source import EpiDataSource
from ..hdf5_loader import Hdf5Loader
from ..metadata import Metadata

# Define a type variable that can be either KnownData or UnknownData
DataType = TypeVar("DataType", bound="Data")


class Data(abc.ABC):
    """Generalized object to deal with numerical data.

    Does not have metadata.
    """

    # TODO: actually make a data class without any true labels which is supported within analysis.
    def __init__(self, ids, x, y, y_str):
        self._ids = ids
        self._num_examples = len(x)
        self._signals = np.array(x, dtype=np.float32)
        self._labels = np.array(y)
        self._labels_str = y_str
        self._shuffle_order = np.arange(
            self._num_examples
        )  # To be able to find back ids correctly
        self._index = 0

    def __len__(self):
        return self._num_examples

    @property
    def ids(self) -> np.ndarray:
        """Return md5s in current signals order."""
        return np.take(self._ids, list(self._shuffle_order), axis=0)

    def get_id(self, index: int):
        """Return unique identifier associated with signal position."""
        return self._ids[self._shuffle_order[index]]  # type: ignore

    @property
    def signals(self) -> np.ndarray:
        """Return signals in current order."""
        return self._signals

    def get_signal(self, index: int):
        """Return current signal at given position. (signals can be shuffled)"""
        return self._signals[index]  # type: ignore

    @property
    def encoded_labels(self) -> np.ndarray:
        """Return encoded labels of examples in current signal order."""
        return self._labels

    def get_encoded_label(self, index: int):
        """Return encoded label at given signal position."""
        return self._labels[index]

    @property
    def original_labels(self) -> np.ndarray:
        """Return string labels of examples in current signal order."""
        return np.take(self._labels_str, list(self._shuffle_order), axis=0)

    def get_original_label(self, index: int):
        """Return original label at given signal position."""
        return self._labels_str[self._shuffle_order[index]]

    @property
    def num_examples(self) -> int:
        """Return the number of examples contained in the set.

        Repeated/oversampled signals are part of that count.
        """
        return self._num_examples

    def __eq__(self, other):
        if type(other) is type(self):
            bools = []
            bools.append(np.array_equal(self.ids, other.ids))
            bools.append(np.array_equal(self.signals, other.signals))
            bools.append(np.array_equal(self.encoded_labels, other.encoded_labels))
            bools.append(np.array_equal(self.original_labels, other.original_labels))
            bools.append(self.num_examples == other.num_examples)
            return all(bools)
        return False

    def preprocess(self, f):
        """Apply a preprocessing function on signals."""
        self._signals = np.apply_along_axis(f, 1, self._signals)

    def next_batch(self, batch_size, shuffle=True):
        """Return next (signals, targets) batch"""
        # if index exceeded num examples, start over
        if self._index >= self._num_examples:
            self._index = 0
        if self._index == 0:
            if shuffle:
                self._shuffle()
        start = self._index
        self._index += batch_size
        end = self._index
        return self._signals[start:end], self._labels[start:end]

    def _shuffle(self, seed=False):
        """Shuffle signals and labels together"""
        if seed:
            np.random.seed(42)

        rng_state = np.random.get_state()
        for array in [self._shuffle_order, self._signals, self._labels]:
            np.random.shuffle(array)
            np.random.set_state(rng_state)

    def shuffle(self, seed=False):
        """Shuffle signals and labels together"""
        self._shuffle(seed)

    @abc.abstractmethod
    def subsample(self, idxs: List[int]):
        """Abstact method, raises NotImplementedError."""
        raise NotImplementedError("This is an abstract method. Use child class.")

    @classmethod
    @abc.abstractmethod
    def empty_collection(cls):
        """Abstact method, raises NotImplementedError."""
        raise NotImplementedError("This is an abstract class method. Use child class.")


class KnownData(Data):
    """Generalised object to deal with numerical data.

    ids : Signal identifier
    x : features
    y : targets (int)
    y_str : targets (str)
    metadata : Metadata object containing signal metadata.
    """

    def __init__(self, ids, x, y, y_str, metadata: Metadata):
        super().__init__(ids, x, y, y_str)
        self._metadata = metadata

    @property
    def metadata(self) -> Metadata:
        """Return the metadata of the dataset. Careful, modifications to it will affect this object."""
        return self._metadata

    def get_metadata(self, index: int) -> dict:
        """Get the metadata from the signal at the given position in the set."""
        return self._metadata[self.get_id(index)]

    @classmethod
    def empty_collection(cls) -> KnownData:
        """Returns an empty object."""
        obj = cls.__new__(cls)
        obj._ids = []
        obj._num_examples = 0
        obj._signals = np.array([], dtype=np.float32)
        obj._labels = np.array([])
        obj._labels_str = []
        obj._shuffle_order = []  # To be able to find back ids correctly
        obj._index = 0
        obj._metadata = {}
        return obj

    def subsample(self, idxs: List[int]) -> KnownData:
        """Return Data object with subsample of current Data.

        Indexed along current order, not original order.
        """
        try:
            new_ids = np.take(self.ids, idxs, axis=0)
            new_signals = np.take(self.signals, idxs, axis=0)
            new_targets = np.take(self.encoded_labels, idxs, axis=0)
            new_str_targets = np.take(self.original_labels, idxs, axis=0)

            new_meta = copy.deepcopy(self.metadata)
            ok_md5 = set(new_ids)
            for md5 in list(new_meta.md5s):
                if md5 not in ok_md5:
                    del new_meta[md5]
        except IndexError as e:
            if len(self) == 0:
                print("Empty Data object, cannot subsample.")
                return self
            raise e

        return KnownData(new_ids, new_signals, new_targets, new_str_targets, new_meta)


class UnknownData(Data):
    """Generalised object to deal with numerical data without any labels/metadata.

    ids : Signal identifier
    x : features
    y : targets (int)
    y_str : targets (str)
    """

    @classmethod
    def empty_collection(cls) -> UnknownData:
        """Returns an empty object."""
        obj = cls.__new__(cls)
        obj._ids = []
        obj._num_examples = 0
        obj._signals = np.array([], dtype=np.float32)
        obj._labels = np.array([])
        obj._labels_str = []
        obj._shuffle_order = []  # To be able to find back ids correctly
        obj._index = 0
        return obj

    def subsample(self, idxs: List[int]) -> UnknownData:
        """Return Data object with subsample of current Data.

        Indexed along current order, not original order.
        """
        try:
            new_ids = np.take(self.ids, idxs, axis=0)
            new_signals = np.take(self.signals, idxs, axis=0)
            new_targets = np.take(self.encoded_labels, idxs, axis=0)
            new_str_targets = np.take(self.original_labels, idxs, axis=0)
        except IndexError as e:
            if len(self) == 0:
                print("Empty Data object, cannot subsample.")
                return self
            raise e

        return UnknownData(new_ids, new_signals, new_targets, new_str_targets)


class DataSet(Generic[DataType], abc.ABC):
    """Contains training/valid/test Data objects."""

    def __init__(
        self,
        training: DataType,
        validation: DataType,
        test: DataType,
        sorted_classes: List[str],
    ):
        self._train = training
        self._validation = validation
        self._test = test
        self._sorted_classes = sorted_classes

    @property
    def train(self) -> DataType:
        """Training set"""
        return self._train

    @property
    def validation(self) -> DataType:
        """Validation set"""
        return self._validation

    @property
    def test(self) -> DataType:
        """Test set"""
        return self._test

    @property
    def classes(self) -> List[str]:
        """Return sorted classes present through datasets"""
        return self._sorted_classes

    @classmethod
    def empty_collection(
        cls, data_class: Type[DataType] | None = None
    ) -> "DataSet[DataType]":
        """Returns an empty object.

        Args:
            data_class: The data class to use (KnownData, UnknownData, lazy versions.)
        """
        obj = cls.__new__(cls)

        # Narrow the type
        if data_class is None:
            data_class = KnownData  # type: ignore
            print(
                "Warning: data_class not provided, using KnownData.",
                file=sys.stderr,
            )
        else:
            if not issubclass(data_class, Data):
                raise AssertionError("data_class must be a subclass of Data")

        assert data_class is not None  # for pylance

        obj._train = data_class.empty_collection()
        obj._validation = data_class.empty_collection()
        obj._test = data_class.empty_collection()
        obj._sorted_classes = []
        return obj

    def set_train(self, dset: DataType):
        """Set training set."""
        self._train = dset
        self._reset_classes()

    def set_validation(self, dset: DataType):
        """Set validation set."""
        self._validation = dset
        self._reset_classes()

    def set_test(self, dset: DataType):
        """Set testing set."""
        self._test = dset
        self._reset_classes()

    def _reset_classes(self):
        """Reset classes property."""
        new_classes = []
        for dset in [self._train, self._validation, self._test]:
            if dset.num_examples:
                new_classes.extend(dset.original_labels)
        self._sorted_classes = sorted(list(set(new_classes)))

    def preprocess(self, f):
        """Apply preprocessing function to all datasets."""
        for dset in [self._train, self._validation, self._test]:
            if dset.num_examples:
                dset.preprocess(f)

    def save_mapping(self, path):
        """Write the 'output position --> label' mapping to path."""
        with open(path, "w", encoding="utf-8") as map_file:
            for i, label in enumerate(self._sorted_classes):
                map_file.write(f"{i}\t{label}\n")

    def load_mapping(self, path):
        """Return dict object representation 'output position --> label' mapping from path."""
        with open(path, "r", encoding="utf-8") as map_file:
            mapping = {}
            for line in map_file:
                i, label = line.rstrip().split("\t")
                mapping[int(i)] = label
        return mapping

    def get_encoder(self, mapping, using_file=False) -> preprocessing.LabelEncoder:
        """Load and return int label encoder.

        Requires the model mapping file itself, or its path (with using_file=True)
        """
        if using_file:
            mapping = self.load_mapping(mapping)

        labels = sorted(list(mapping.values()))
        return preprocessing.LabelEncoder().fit(labels)


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
