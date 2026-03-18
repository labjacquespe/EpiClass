"""Tests for the winsorize_hdf5 module.""" ""
import sys

import h5py
import numpy as np
from scipy.stats import mstats

from epiclass.utils.preprocessing.winsorize_hdf5 import main


def test_winsorization(tmp_path):
    """Integrative test: test winsorize_hdf5 main."""
    # Prepare a small test hdf5 file
    test_file_name = tmp_path / "test.h5"
    test_paths_list_name = tmp_path / "test_list.txt"

    expected_output_name = tmp_path / "test_winsorized-0-0.01.hdf5"

    upper_limit = 0.01

    # Prepare a small test hdf5 file
    # Input data is a fixed numpy array
    input_data = np.arange(0, 100)

    # Expected data is the winsorized version of the input_data
    expected_data = mstats.winsorize(input_data, limits=(0, upper_limit))

    with h5py.File(test_file_name, "w") as f:
        group = f.create_group("header")
        for i in range(1, 23):
            data = group.create_dataset(f"chr{i}", (100,), dtype="i")
            data[:] = expected_data
        # Add chrX
        data = group.create_dataset("chrX", (100,), dtype="i")
        data[:] = expected_data

    with open(test_paths_list_name, "w", encoding="utf-8") as f:
        f.write(str(test_file_name))

    # Run the winsorization function
    sys.argv = ["script.py", str(test_paths_list_name), str(tmp_path), f"{upper_limit}"]
    main()

    # Check that the data in the new file is as expected
    with h5py.File(expected_output_name, "r") as f:
        header = list(f.keys())[0]
        hdf5_data = f[header]

        for dataset in hdf5_data.values():  # type: ignore
            assert np.allclose(
                dataset[:],
                expected_data,
            )
