"""
This module contains tests for HDF5 file operations. These tests include
casting datasets within these files to the float32 data type, repacking files to reduce size,
and a workflow that performs these operations in sequence.
"""
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from epiclass.utils.preprocessing.convert_hdf5_to_float32 import (
    cast_datasets_to_float32,
    process_file,
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


@pytest.fixture(name="work_copy")
def hdf5_work_copy(tmp_path, test_hdf5) -> Path:
    """
    Provides a working copy of the test HDF5 file in tmp_path.
    """
    dest = tmp_path / test_hdf5.name
    shutil.copy(test_hdf5, dest)
    return dest


@pytest.fixture(name="test_hdf5_float64")
def hdf5_test_file_float64(tmp_path, test_hdf5) -> Path:
    """
    Provides a copy of the test HDF5 file with all float32 datasets promoted to float64.
    """
    dest = tmp_path / "float64_input.hdf5"
    shutil.copy(test_hdf5, dest)

    with h5py.File(dest, "r+") as f:
        for group in f.values():
            if not isinstance(group, h5py.Group):
                continue
            for name, dataset in list(group.items()):
                if not isinstance(dataset, h5py.Dataset):
                    continue
                if dataset.dtype == np.float32:
                    attrs = dict(dataset.attrs.items())
                    data = dataset[...].astype(np.float64)
                    del group[name]
                    group.create_dataset(name, data=data, dtype=np.float64)
                    group[name].attrs.update(attrs)

    return dest


def test_cast_datasets_to_float32(work_copy):
    """
    Test for casting datasets in an HDF5 file to float32.
    """
    assert cast_datasets_to_float32(work_copy) is not None


def test_repack_hdf5_file(work_copy):
    """
    Test for repacking an HDF5 file to reduce its size.
    """
    cast_datasets_to_float32(work_copy)
    repack_hdf5_file(work_copy)
    assert work_copy.is_file()


def test_workflow_to_outdir(tmp_path, test_hdf5_float64):
    """
    Test the normal (non-overwrite) workflow via process_file:
    copies to outdir with a "_float32.hdf5" suffix, casts, and repacks.
    """
    outdir = tmp_path / "output"
    outdir.mkdir()

    process_file(test_hdf5_float64, outdir=outdir, overwrite=False)

    expected_output = outdir / (test_hdf5_float64.stem + "_float32.hdf5")
    assert expected_output.is_file()


def test_workflow_to_outdir_skips_existing(tmp_path, test_hdf5_float64):
    """
    Test that the normal workflow skips files that already exist in outdir.
    """
    outdir = tmp_path / "output"
    outdir.mkdir()

    expected_output = outdir / (test_hdf5_float64.stem + "_float32.hdf5")
    expected_output.write_bytes(b"sentinel")

    process_file(test_hdf5_float64, outdir=outdir, overwrite=False)

    # File should be untouched (still our sentinel content)
    assert expected_output.read_bytes() == b"sentinel"


def test_workflow_overwrite(tmp_path, test_hdf5):
    """
    Test the --overwrite workflow via process_file:
    modifies the file in-place (via a temp copy), replacing the original.
    """
    # Work on a copy so we don't modify the fixture file
    hdf5_copy = tmp_path / test_hdf5.name
    shutil.copy(test_hdf5, hdf5_copy)
    original_size = hdf5_copy.stat().st_size

    process_file(hdf5_copy, outdir=None, overwrite=True)

    assert hdf5_copy.is_file()

    # Verify float32 casting actually happened
    with h5py.File(hdf5_copy, "r") as f:
        for group in f.values():
            if not isinstance(group, h5py.Group):
                continue
            for dataset in group.values():
                if isinstance(dataset, h5py.Dataset) and np.issubdtype(
                    dataset.dtype, np.floating
                ):
                    assert dataset.dtype == np.float32

    # Repacking + float32 should reduce file size
    assert hdf5_copy.stat().st_size <= original_size


def test_casting_changes_data(tmp_path):
    """
    Test for verifying that casting datasets in a fake HDF5 file changes the data.
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

    # Copy the HDF5 file to a working copy
    work_copy = tmp_path / "work_copy.hdf5"
    shutil.copy(original_file_path, work_copy)

    # Perform the casting
    assert cast_datasets_to_float32(work_copy)

    # Load the dataset after casting
    with h5py.File(work_copy, "r") as f:
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
