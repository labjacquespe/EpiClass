"""Lazy-loading data classes for ML training.

These classes store sample IDs and a loader reference instead of loaded signals,
enabling on-demand loading for datasets too large to fit in memory.
"""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

import abc
import copy
from typing import List, Protocol, runtime_checkable

import numpy as np

from epiclass.core.metadata import Metadata


@runtime_checkable
class SignalLoader(Protocol):
    """Protocol satisfied by both LazyHdf5Loader and ChunkedHdf5Loader."""

    def load_signal(  # pylint: disable=missing-function-docstring
        self, sample_id: str
    ) -> np.ndarray:
        ...


class _NumpyLoader:
    """Private SignalLoader backed by an in-memory numpy matrix.

    Used by LazyData.from_array() — not part of the public API.
    """

    def __init__(self, ids: list[str], array: np.ndarray):
        self._index: dict[str, int] = {id_: i for i, id_ in enumerate(ids)}
        self._array = np.asarray(array, dtype=np.float32)

    def load_signal(self, sample_id: str) -> np.ndarray:
        """Return the signal row for the given sample ID."""
        return self._array[self._index[sample_id]]


class LazyData(abc.ABC):
    """Base class for lazy-loaded data.

    Stores sample IDs and a loader reference instead of loaded signals.
    Signals are loaded on-demand via __getitem__.
    """

    def __init__(
        self,
        ids: List[str],
        loader: SignalLoader,
        y: np.ndarray,
        y_str: List[str],
    ):
        self._ids = np.array(ids)
        self._num_examples = len(ids)
        self._loader = loader
        self._labels = np.array(y)
        self._labels_str = np.array(y_str)
        self._shuffle_order = np.arange(self._num_examples)
        self._index = 0

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, str]:
        """Load and return (signal, encoded_label, original_label) at index."""
        sample_id = self.get_id(index)
        signal = self._loader.load_signal(sample_id)
        return signal, self.get_encoded_label(index), self.get_original_label(index)

    @property
    def ids(self) -> np.ndarray:
        """Return sample IDs in current order."""
        return np.take(self._ids, list(self._shuffle_order), axis=0)

    def get_id(self, index: int) -> str:
        """Return sample ID at position."""
        return self._ids[self._shuffle_order[index]]

    @property
    def loader(self) -> SignalLoader:
        """Return the signal loader."""
        return self._loader

    @property
    def encoded_labels(self) -> np.ndarray:
        """Return encoded labels in current order."""
        return np.take(self._labels, list(self._shuffle_order), axis=0)

    def get_encoded_label(self, index: int):
        """Return encoded label at position."""
        return self._labels[self._shuffle_order[index]]

    @property
    def original_labels(self) -> np.ndarray:
        """Return string labels in current order."""
        return np.take(self._labels_str, list(self._shuffle_order), axis=0)

    def get_original_label(self, index: int):
        """Return original label at position."""
        return self._labels_str[self._shuffle_order[index]]

    @property
    def num_examples(self) -> int:
        """Return number of examples."""
        return self._num_examples

    def __eq__(self, other):
        if type(other) is type(self):
            bools = [
                np.array_equal(self.ids, other.ids),
                np.array_equal(self.encoded_labels, other.encoded_labels),
                np.array_equal(self.original_labels, other.original_labels),
                self.num_examples == other.num_examples,
            ]
            return all(bools)
        return False

    def shuffle(self, seed=False):
        """Shuffle order of samples."""
        if seed:
            np.random.seed(42)
        np.random.shuffle(self._shuffle_order)
        self._index = 0

    def load_all_signals(self) -> np.ndarray:
        """Load all signals into memory. Defeats lazy loading — use only for debugging."""
        return np.array(
            [self._loader.load_signal(self.get_id(i)) for i in range(len(self))],
            dtype=np.float32,
        )

    @property
    def signal_length(self) -> int:
        """Number of genomic bins in each signal (i.e. the input feature dimension).

        Loads one signal from the loader to determine the length. For HDF5-backed
        loaders this is cheap after preload_all() since it reads a single mmap row.
        """
        signal, _, _ = self[0]
        return int(signal.size)

    def materialize(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all signals into RAM and return (signals, encoded_labels).

        Use when the full dataset must be in memory at once (SHAP, sklearn, LGBM).
        For training, prefer the DataLoader path which streams via the mmap.
        """
        return self.load_all_signals(), self.encoded_labels

    @abc.abstractmethod
    def subsample(self, idxs: List[int]):
        """Return a new instance containing only the samples at the given indices."""
        raise NotImplementedError("Use child class.")

    @classmethod
    @abc.abstractmethod
    def empty_collection(cls):
        """Return an empty instance of this class."""
        raise NotImplementedError("Use child class.")


class LazyKnownData(LazyData):
    """Lazy-loaded data with metadata."""

    @classmethod
    def from_array(
        cls,
        ids: List[str],
        array: np.ndarray,
        y: np.ndarray,
        y_str: List[str],
        metadata: Metadata,
    ) -> LazyKnownData:
        """Create from an in-memory numpy array without a file-based loader."""
        return cls(ids, _NumpyLoader(ids, array), y, y_str, metadata)

    def __init__(
        self,
        ids: List[str],
        loader: SignalLoader,
        y: np.ndarray,
        y_str: List[str],
        metadata: Metadata,
    ):
        super().__init__(ids, loader, y, y_str)
        self._metadata = metadata

    @property
    def metadata(self) -> Metadata:
        """Return metadata."""
        return self._metadata

    def get_metadata(self, index: int) -> dict:
        """Get metadata for sample at position."""
        return self._metadata[self.get_id(index)]

    @classmethod
    def empty_collection(cls) -> LazyKnownData:
        """Return empty object."""
        obj = cls.__new__(cls)
        obj._ids = np.array([])
        obj._num_examples = 0
        obj._loader = None
        obj._labels = np.array([])
        obj._labels_str = np.array([])
        obj._shuffle_order = np.array([])
        obj._index = 0
        obj._metadata = {}
        return obj

    def subsample(self, idxs: List[int]) -> LazyKnownData:
        """Return subsampled object sharing the same loader."""
        try:
            new_ids = np.take(self.ids, idxs, axis=0).tolist()
            new_targets = np.take(self.encoded_labels, idxs, axis=0)
            new_str_targets = np.take(self.original_labels, idxs, axis=0).tolist()

            new_meta = copy.deepcopy(self.metadata)
            ok_ids = set(new_ids)
            for sample_id in list(new_meta.signal_ids):
                if sample_id not in ok_ids:
                    del new_meta[sample_id]
        except IndexError as e:
            if len(self) == 0:
                print("Empty Data object, cannot subsample.")
                return self
            raise e

        return LazyKnownData(
            new_ids, self._loader, new_targets, new_str_targets, new_meta
        )


class LazyUnknownData(LazyData):
    """Lazy-loaded data without metadata."""

    @classmethod
    def from_array(
        cls,
        ids: List[str],
        array: np.ndarray,
        y: np.ndarray,
        y_str: List[str],
    ) -> LazyUnknownData:
        """Create from an in-memory numpy array without a file-based loader."""
        return cls(ids, _NumpyLoader(ids, array), y, y_str)

    @classmethod
    def empty_collection(cls) -> LazyUnknownData:
        """Return empty object."""
        obj = cls.__new__(cls)
        obj._ids = np.array([])
        obj._num_examples = 0
        obj._loader = None
        obj._labels = np.array([])
        obj._labels_str = np.array([])
        obj._shuffle_order = np.array([])
        obj._index = 0
        return obj

    def subsample(self, idxs: List[int]) -> LazyUnknownData:
        """Return subsampled object sharing the same loader."""
        try:
            new_ids = np.take(self.ids, idxs, axis=0).tolist()
            new_targets = np.take(self.encoded_labels, idxs, axis=0)
            new_str_targets = np.take(self.original_labels, idxs, axis=0).tolist()
        except IndexError as e:
            if len(self) == 0:
                print("Empty Data object, cannot subsample.")
                return self
            raise e

        return LazyUnknownData(new_ids, self._loader, new_targets, new_str_targets)
