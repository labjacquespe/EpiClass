"""Tests for lazy_data_classes.py and lazy_torch_dataset.py.

Mirrors data_test.py (which covers eager.KnownData) using in-memory fixtures
built via from_array() — no HDF5 files needed.
"""
# pylint: disable=too-many-positional-arguments, missing-function-docstring, missing-class-docstring, import-outside-toplevel, unbalanced-tuple-unpacking
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from epiclass.core.data.dataset import DataSet
from epiclass.core.lazy.lazy_data_classes import LazyKnownData, LazyUnknownData
from epiclass.core.lazy.lazy_epidata import LazyEpiData
from epiclass.core.lazy.lazy_torch_dataset import LazyHdf5Dataset, create_lazy_dataloaders
from epiclass.core.metadata import Metadata
from epiclass.utils.torch_data import create_torch_datasets

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_signal(i: int) -> list:
    return [42 + i, 42 + 2 * i]


def _make_known(n: int = 10) -> LazyKnownData:
    ids = [f"id{i}" for i in range(n)]
    array = np.array([_mock_signal(i) for i in range(n)], dtype=np.float32)
    y = np.array([i % 2 for i in range(n)])
    y_str = [f"target{i % 2}" for i in range(n)]
    meta = Metadata.from_dict({id_: {} for id_ in ids}, allow_variable_length_id=True)
    return LazyKnownData.from_array(ids, array, y, y_str, meta)


def _make_unknown(n: int = 10) -> LazyUnknownData:
    ids = [f"id{i}" for i in range(n)]
    array = np.array([_mock_signal(i) for i in range(n)], dtype=np.float32)
    y = np.array([i % 2 for i in range(n)])
    y_str = [f"target{i % 2}" for i in range(n)]
    return LazyUnknownData.from_array(ids, array, y, y_str)


# ---------------------------------------------------------------------------
# LazyKnownData
# ---------------------------------------------------------------------------


class TestLazyKnownData:
    @pytest.fixture
    def some_data(self) -> LazyKnownData:
        return _make_known(10)

    @pytest.fixture
    def empty_data(self) -> LazyKnownData:
        return LazyKnownData.empty_collection()

    def test_len(self, some_data: LazyKnownData):
        assert len(some_data) == 10

    def test_subsample_empty(self, empty_data: LazyKnownData):
        result = empty_data.subsample([1])
        assert result is empty_data

    def test_subsample_over(self, some_data: LazyKnownData):
        match = r"index \d+ is out of bounds for axis \d with size \d+"
        with pytest.raises(IndexError, match=match):
            some_data.subsample([666])

    def test_subsample(self, some_data: LazyKnownData):
        nb1, nb2 = 4, 9
        new_data = some_data.subsample([nb1, nb2])

        assert new_data.num_examples == 2
        assert list(new_data.ids) == [f"id{nb1}", f"id{nb2}"]
        assert list(new_data.encoded_labels) == [nb1 % 2, nb2 % 2]

        sig0, _, _ = new_data[0]
        sig1, _, _ = new_data[1]
        assert np.allclose(sig0, _mock_signal(nb1))
        assert np.allclose(sig1, _mock_signal(nb2))

    def test_shuffle_reproducible(self, some_data: LazyKnownData):
        other = _make_known(10)
        some_data.shuffle(seed=True)
        other.shuffle(seed=True)
        assert some_data == other

    def test_shuffle_internal(self, some_data: LazyKnownData):
        """After shuffle, ids, signals, and labels remain in sync."""
        some_data.shuffle(seed=True)
        for i in range(len(some_data)):  # pylint: disable=consider-using-enumerate
            sig_id = some_data.get_id(i)
            nb = int(sig_id[2:])  # "id{nb}" → nb
            signal, label, _ = some_data[i]
            assert sig_id == some_data.ids[i]
            assert np.allclose(signal, _mock_signal(nb))
            assert label == nb % 2

    def test_signal_length(self, some_data: LazyKnownData):
        assert some_data.signal_length == 2

    def test_materialize_shape(self, some_data: LazyKnownData):
        signals, labels = some_data.materialize()
        assert signals.shape == (10, 2)
        assert labels.shape == (10,)

    def test_materialize_values(self, some_data: LazyKnownData):
        signals, labels = some_data.materialize()
        for i in range(10):
            assert np.allclose(signals[i], _mock_signal(i))
            assert labels[i] == i % 2

    def test_eq_same(self, some_data: LazyKnownData):
        other = _make_known(10)
        assert some_data == other

    def test_eq_after_shuffle(self, some_data: LazyKnownData):
        other = _make_known(10)
        other.shuffle(seed=True)
        assert some_data != other

    def test_getitem(self, some_data: LazyKnownData):
        signal, label, label_str = some_data[3]
        assert np.allclose(signal, _mock_signal(3))
        assert label == 3 % 2
        assert label_str == f"target{3 % 2}"


# ---------------------------------------------------------------------------
# LazyUnknownData
# ---------------------------------------------------------------------------


class TestLazyUnknownData:
    @pytest.fixture
    def some_data(self) -> LazyUnknownData:
        return _make_unknown(10)

    @pytest.fixture
    def empty_data(self) -> LazyUnknownData:
        return LazyUnknownData.empty_collection()

    def test_len(self, some_data: LazyUnknownData):
        assert len(some_data) == 10

    def test_subsample_empty(self, empty_data: LazyUnknownData):
        result = empty_data.subsample([1])
        assert result is empty_data

    def test_subsample_over(self, some_data: LazyUnknownData):
        match = r"index \d+ is out of bounds for axis \d with size \d+"
        with pytest.raises(IndexError, match=match):
            some_data.subsample([666])

    def test_subsample(self, some_data: LazyUnknownData):
        nb1, nb2 = 3, 7
        new_data = some_data.subsample([nb1, nb2])

        assert new_data.num_examples == 2
        assert list(new_data.ids) == [f"id{nb1}", f"id{nb2}"]

        sig0, _, _ = new_data[0]
        assert np.allclose(sig0, _mock_signal(nb1))

    def test_shuffle_internal(self, some_data: LazyUnknownData):
        some_data.shuffle(seed=True)
        for i in range(len(some_data)):  # pylint: disable=consider-using-enumerate
            sig_id = some_data.get_id(i)
            nb = int(sig_id[2:])
            signal, label, _ = some_data[i]
            assert np.allclose(signal, _mock_signal(nb))
            assert label == nb % 2

    def test_signal_length(self, some_data: LazyUnknownData):
        assert some_data.signal_length == 2

    def test_materialize(self, some_data: LazyUnknownData):
        signals, labels = some_data.materialize()
        assert signals.shape == (10, 2)
        assert labels.shape == (10,)
        for i in range(10):
            assert np.allclose(signals[i], _mock_signal(i))


# ---------------------------------------------------------------------------
# LazyHdf5Dataset
# ---------------------------------------------------------------------------


class TestLazyHdf5Dataset:
    @pytest.fixture
    def known_data(self) -> LazyKnownData:
        return _make_known(5)

    def test_len(self, known_data: LazyKnownData):
        ds = LazyHdf5Dataset(known_data)
        assert len(ds) == 5

    def test_getitem_types(self, known_data: LazyKnownData):
        ds = LazyHdf5Dataset(known_data)
        signal, label = ds[0]  # type: ignore
        assert isinstance(signal, torch.Tensor)
        assert signal.dtype == torch.float32
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long

    def test_getitem_values(self, known_data: LazyKnownData):
        ds = LazyHdf5Dataset(known_data)
        for i in range(5):
            signal, label = ds[i]  # type: ignore
            assert np.allclose(signal.numpy(), _mock_signal(i))
            assert label.item() == i % 2

    def test_return_id(self, known_data: LazyKnownData):
        ds = LazyHdf5Dataset(known_data, return_id=True)
        signal, _, sample_id = ds[3]  # type: ignore
        assert sample_id == "id3"
        assert isinstance(signal, torch.Tensor)

    def test_unknown_data(self):
        ds = LazyHdf5Dataset(_make_unknown(4))
        assert len(ds) == 4
        signal, _ = ds[0]  # type: ignore
        assert signal.dtype == torch.float32


# ---------------------------------------------------------------------------
# create_lazy_dataloaders
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore:'pin_memory' argument is set as true but no accelerator is found.*:UserWarning"
)
class TestCreateLazyDataloaders:
    def test_all_splits_present(self):
        loaders = create_lazy_dataloaders(
            train_data=_make_known(20),
            val_data=_make_known(5),
            test_data=_make_unknown(5),
            batch_size=4,
            num_workers=0,
        )
        assert set(loaders.keys()) == {"training", "validation", "test"}

    def test_train_only(self):
        loaders = create_lazy_dataloaders(
            train_data=_make_known(20), batch_size=4, num_workers=0
        )
        assert "training" in loaders
        assert "validation" not in loaders
        assert "test" not in loaders

    def test_none_splits_omitted(self):
        loaders = create_lazy_dataloaders(
            val_data=_make_known(5), batch_size=4, num_workers=0
        )
        assert "validation" in loaders
        assert "training" not in loaders

    def test_train_drop_last(self):
        loaders = create_lazy_dataloaders(
            train_data=_make_known(20), batch_size=4, num_workers=0
        )
        assert loaders["training"][1].drop_last is True

    def test_val_no_drop_last(self):
        loaders = create_lazy_dataloaders(
            val_data=_make_known(5), batch_size=4, num_workers=0
        )
        assert loaders["validation"][1].drop_last is False

    def test_batch_shapes(self):
        loaders = create_lazy_dataloaders(
            train_data=_make_known(20), batch_size=4, num_workers=0
        )
        signals, labels = next(iter(loaders["training"][1]))
        assert signals.dtype == torch.float32
        assert labels.dtype == torch.long
        assert signals.shape == (4, 2)
        assert labels.shape == (4,)


# ---------------------------------------------------------------------------
# materialize() alignment after shuffle
# ---------------------------------------------------------------------------


class TestMaterializeAfterShuffle:
    """Verify that signals and labels stay aligned after shuffle.

    If materialize() ignored _shuffle_order, signals and labels would become
    misaligned — a silent data-corruption bug during ML training.
    """

    def test_known_data_aligned(self):
        data = _make_known(20)
        data.shuffle(seed=True)
        signals, labels = data.materialize()
        for i in range(len(data)):
            nb = int(data.ids[i][2:])
            assert np.allclose(signals[i], _mock_signal(nb))
            assert labels[i] == nb % 2

    def test_unknown_data_aligned(self):
        data = _make_unknown(20)
        data.shuffle(seed=True)
        signals, labels = data.materialize()
        for i in range(len(data)):
            nb = int(data.ids[i][2:])
            assert np.allclose(signals[i], _mock_signal(nb))
            assert labels[i] == nb % 2


# ---------------------------------------------------------------------------
# create_torch_datasets lazy dispatch (torch_data.py)
# ---------------------------------------------------------------------------


class TestCreateTorchDatasetsDispatch:
    """Verify create_torch_datasets routes lazy data to the lazy path."""

    def test_lazy_dispatch_returns_dataloaders(self):
        """With lazy input, create_torch_datasets must return working loaders."""
        train = _make_known(20)
        val = _make_known(8)
        test = _make_known(4)
        dset = DataSet(
            training=train, validation=val, test=test, sorted_classes=["0", "1"]
        )

        loaders = create_torch_datasets(dset, batch_size=4)
        assert "training" in loaders
        assert "validation" in loaders

    def test_lazy_dispatch_correct_shapes(self):
        train = _make_known(20)
        val = _make_known(8)
        dset = DataSet(
            training=train,
            validation=val,
            test=LazyKnownData.empty_collection(),
            sorted_classes=["0", "1"],
        )

        loaders = create_torch_datasets(dset, batch_size=4)
        signals, _ = next(iter(loaders["training"][1]))
        assert signals.shape[1] == 2
        assert signals.dtype == torch.float32

    def test_lazy_dispatch_not_in_val_shuffle(self):
        """Validation loader must not shuffle."""
        val = _make_known(8)
        dset = DataSet(
            training=_make_known(20),
            validation=val,
            test=LazyKnownData.empty_collection(),
            sorted_classes=["0", "1"],
        )

        loaders = create_torch_datasets(dset, batch_size=4)
        assert loaders["validation"][1].sampler.__class__.__name__ == "SequentialSampler"


# ---------------------------------------------------------------------------
# LazyEpiData.oversample_signal_ids
# ---------------------------------------------------------------------------


class TestOversampleSignalIds:
    """LazyEpiData.oversample_signal_ids is used for training class balancing."""

    def test_balances_classes(self):
        signal_ids = ["a"] * 10 + ["b"] * 3
        labels = ["cls_a"] * 10 + ["cls_b"] * 3
        _, resampled_labels = LazyEpiData.oversample_signal_ids(signal_ids, labels)

        counts = Counter(resampled_labels)
        assert counts["cls_a"] == counts["cls_b"]

    def test_minority_class_duplicated(self):
        signal_ids = ["a"] * 5 + ["b"] * 2
        labels = ["cls_a"] * 5 + ["cls_b"] * 2
        resampled_ids, resampled_labels = LazyEpiData.oversample_signal_ids(
            signal_ids, labels
        )
        assert len(resampled_ids) == len(resampled_labels)
        assert len(resampled_ids) > len(signal_ids)

    def test_majority_class_unchanged(self):
        signal_ids = ["a"] * 10 + ["b"] * 3
        labels = ["cls_a"] * 10 + ["cls_b"] * 3
        _, resampled_labels = LazyEpiData.oversample_signal_ids(signal_ids, labels)
        counts = Counter(resampled_labels)
        assert counts["cls_a"] == 10
