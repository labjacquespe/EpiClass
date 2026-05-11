"""Module defining DataSet class to hold training, validation, and test datasets."""
from __future__ import annotations

import abc
from typing import Generic, List, Type, TypeVar

from sklearn import preprocessing

from epiclass.core.lazy.lazy_data_classes import LazyData, LazyKnownData

DataType = TypeVar("DataType", bound="LazyData")


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
        cls, data_class: Type[DataType] = LazyKnownData  # type: ignore
    ) -> "DataSet[DataType]":
        """Return an empty DataSet whose train/val/test are empty instances of data_class."""
        if not issubclass(data_class, LazyData):
            raise AssertionError("data_class must be a subclass of LazyData")

        obj = cls.__new__(cls)
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
