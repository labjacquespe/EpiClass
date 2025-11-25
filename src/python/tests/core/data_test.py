"""Test data.py module."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from epiclass.core import metadata
from epiclass.core.data import eager


class TestData:
    """Test Data class."""

    @staticmethod
    def mock_signal(i: int):
        """Mock signal"""
        return [42 + i, 42 + 2 * i]

    @pytest.fixture
    def empty_data(self) -> eager.KnownData:
        """Empty Data object."""
        return eager.KnownData.empty_collection()

    @pytest.fixture
    def some_metadata(self) -> metadata.Metadata:
        """Mock Metadata object"""
        return metadata.Metadata.from_dict({"dcc1c4c0514d3465b55900970459dab6": {}})

    @pytest.fixture
    def rng_state(self):
        """Mock Metadata object"""
        return np.random.get_state()

    @pytest.fixture
    def some_data(self, some_metadata) -> eager.KnownData:
        """Mock Data object."""
        some_data = eager.KnownData(
            ids=[f"id{i}" for i in range(50)],
            x=[TestData.mock_signal(i) for i in range(50)],
            y=[0 + i % 2 for i in range(50)],
            y_str=[f"target{0+i%2}" for i in range(50)],
            metadata=some_metadata,
        )
        return some_data

    def test_subsample_empty(self, empty_data: eager.KnownData):
        """Test execution of subsampling under empty dataset."""
        empty_data.subsample([1])

    # pylint: disable=unused-variable
    def test_subsample_over(self, some_data: eager.KnownData):
        """Test execution of subsampling with out of bound idxs."""
        match = r"index \d+ is out of bounds for axis \d with size \d+"
        with pytest.raises(IndexError, match=match):
            some_data.subsample([666])

    def test_subsample(self, some_data: eager.KnownData):
        """Test correctness."""
        nb1, nb2 = 4, 9
        new_data = some_data.subsample([nb1, nb2])

        assert new_data.num_examples == 2
        assert list(new_data.ids) == [f"id{nb1}", f"id{nb2}"]
        assert list(new_data.encoded_labels) == [0, 1]

        assert np.array_equal(new_data.signals[0], TestData.mock_signal(nb1))  # type: ignore
        assert np.array_equal(new_data.signals[1], TestData.mock_signal(nb2))  # type: ignore

    def test_shuffle(self, some_data: eager.KnownData):
        """Test shuffle reproducability."""
        other_data = copy.deepcopy(some_data)

        some_data.shuffle(seed=True)
        other_data.shuffle(seed=True)
        assert some_data == other_data

        some_data.shuffle(seed=False)
        other_data.shuffle(seed=False)
        assert some_data == other_data

    def test_shuffle_internal(self, some_data: eager.KnownData):
        """Test if internal arrays shuffle together."""
        some_data.shuffle(seed=True)
        ids = some_data.ids
        signals = some_data.signals
        targets = some_data.encoded_labels
        shuffle_order = some_data._shuffle_order  # pylint: disable=protected-access

        for sig_id in ids:
            nb = int(sig_id[-1])
            position = np.where(shuffle_order == nb)[0][0]

            assert ids[position] == f"id{nb}"
            assert list(signals[position]) == TestData.mock_signal(nb)
            assert targets[position] == nb % 2
