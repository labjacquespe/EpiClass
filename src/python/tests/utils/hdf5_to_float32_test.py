"""
This module contains tests for HDF5 file operations. These tests include copying HDF5 files,
casting datasets within these files to the float32 data type, repacking files to reduce size,
and a workflow that performs these operations in sequence.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

from epiclass.utils.preprocessing.convert_hdf5_to_float32 import (
    cast_datasets_to_float32,
    copy_hdf5_file,
    repack_hdf5_file,
)
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_hdf5")
def hdf5_test_file() -> Path:
    """
    Provides a Path object to an existing HDF5 test file.

    Returns:
        Path: A Path object pointing to the HDF5 test file.
    """
    return FIXTURES_DIR / "89a0dcb635f0e9740f587931437b69f1_100kb_all_none_value.hdf5"


def test_copy_hdf5_file(tmp_path, test_hdf5):
    """
    Test for copying an HDF5 file to a new location.

    Args:
        tmp_path (Path): Temporary path provided by pytest fixture.
        test_hdf5 (Path): Path to the test HDF5 file.
    """
    copied_file_path = copy_hdf5_file(test_hdf5, tmp_path)
    assert copied_file_path is not None
    assert copied_file_path.is_file()
    assert copied_file_path.stem == test_hdf5.stem + "_float32"


def test_cast_datasets_to_float32(tmp_path, test_hdf5):
    """
    Test for casting datasets in an HDF5 file to float32.

    Args:
        tmp_path (Path): Temporary path provided by pytest fixture.
        test_hdf5 (Path): Path to the test HDF5 file.
    """
    copied_file_path = copy_hdf5_file(test_hdf5, tmp_path)
    assert copied_file_path is not None
    assert cast_datasets_to_float32(copied_file_path) is not None


def test_repack_hdf5_file(tmp_path, test_hdf5):
    """
    Test for repacking an HDF5 file to reduce its size.

    Args:
        tmp_path (Path): Temporary path provided by pytest fixture.
        test_hdf5 (Path): Path to the test HDF5 file.
    """
    copied_file_path = copy_hdf5_file(test_hdf5, tmp_path)
    assert copied_file_path is not None
    cast_datasets_to_float32(copied_file_path)
    repack_hdf5_file(copied_file_path)
    assert copied_file_path.is_file()


def test_workflow(tmp_path, test_hdf5):
    """
    Test for performing the workflow of copying, casting, and repacking an HDF5 file.

    Args:
        tmp_path (Path): Temporary path provided by pytest fixture.
        test_hdf5 (Path): Path to the test HDF5 file.
    """
    copied_file_path = copy_hdf5_file(test_hdf5, tmp_path)
    assert copied_file_path is not None
    assert copied_file_path.is_file()
    cast_datasets_to_float32(copied_file_path)
    repack_hdf5_file(copied_file_path)
    assert copied_file_path.is_file()


def test_casting_changes_data(tmp_path):
    """
    Test for verifying that casting datasets in a fake HDF5 file changes the data.

    Args:
        tmp_path (Path): Temporary path provided by pytest fixture.
    """
    # Create a fake float64 dataset
    # Range is expected typical values in the dataset.
    rng = np.random.default_rng(42)
    original_data = rng.uniform(0, 400, size=(10, 10)).astype(np.float64)

    # Create an HDF5 file with this dataset
    original_file_path = tmp_path / "original.hdf5"
    with h5py.File(original_file_path, "w") as f:
        group = f.create_group("fake_data")
        group.create_dataset("fake_dataset", data=original_data, dtype=np.float64)

    # Copy the HDF5 file to a new location
    copied_file_path = copy_hdf5_file(original_file_path, tmp_path)
    assert copied_file_path is not None

    # Perform the casting
    assert cast_datasets_to_float32(copied_file_path)

    # Load the dataset after casting
    with h5py.File(copied_file_path, "r") as f:
        casted_data = np.array(f["fake_data"]["fake_dataset"])  # type: ignore

    # Check that the data type has changed to float32
    assert casted_data.dtype == np.float32

    # Casting actually introduced rounding.
    # Numpy will upcasts 'casted data' to float64 if it is float32
    # See Numerical promotion rules.
    assert np.any(original_data != casted_data)

    # Check that the data values are correctly cast to float32
    expected = original_data.astype(np.float32)
    assert np.array_equal(
        casted_data, expected
    ), "Casted data should match expected float32 values"


def test_cast_datasets_overflow_raises(tmp_path):
    """
    Test that casting a float64 value that overflows float32 raises a ValueError.
    """
    val_overflow = np.float64(3.5e40)

    test_file = tmp_path / "warning_test.hdf5"
    with h5py.File(test_file, "w") as f:
        g = f.create_group("warn_group")
        data = np.array([1.0, val_overflow], dtype=np.float64)
        g.create_dataset("warn_dataset", data=data)

    with pytest.warns(RuntimeWarning, match="overflow encountered in cast"):
        with pytest.raises(ValueError, match="would overflow float32"):
            cast_datasets_to_float32(test_file)
