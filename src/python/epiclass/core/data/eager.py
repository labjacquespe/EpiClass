"""Module containing Data classes for eager loading. (full data in memory)"""
from __future__ import annotations

import abc
import copy
from typing import List

import numpy as np

from epiclass.core.metadata import Metadata


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
