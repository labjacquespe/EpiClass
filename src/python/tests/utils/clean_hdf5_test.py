"""Tests for the `clean_hdf5` module."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from epiclass.utils.preprocessing.clean_hdf5 import (
    get_positions_to_treat,
    load_bed,
    main,
    preprocess_bed,
    process_file,
)
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_hdf5")
def test_hdf5_file_path() -> Path:
    """Returns the path to a test HDF5 file."""
    return Path(
        FIXTURES_DIR / "89a0dcb635f0e9740f587931437b69f1_100kb_all_none_value.hdf5"
    )


@pytest.fixture(name="real_test_bed")
def test_bed_file_path() -> Path:
    """Returns the path to a test bed file."""
    return Path(FIXTURES_DIR / "hg38_unified_blacklist.bed")


@pytest.fixture(name="mock_test_bed")
def mock_bed_data_path(tmpdir) -> Path:
    """Writes mock bed data to a file and returns the file path."""
    # fmt: off
    bed_data = [
        ("chr1", 0, 1000),
        ("chr1", 2000, 3000),
        ("chr1", 4000, 5500),
        ("chr1", 6000, 8000),
        ("chr1", 8500, 90000),  # Spans multiple 100kb bins
        ("chr5", 0, 1500),
        ("chr5", 3000, 4800),
        ("chr5", 5000, 99000),  # Spans multiple 100kb bins
        ("chr5", 100000, 101500),  # Starts at the beginning of a new 100kb bin
        ("chr5", 101501, 200000),  # Spans multiple 100kb bins, starts in one and ends in another
        ("chr5", 200001, 300000),  # Spans multiple 100kb bins, starts in one and ends in another
        ("chr5", 300000, 400000),  # Ends right at the boundary of a 100kb bin
        ("chr5", 400001, 500000),  # Starts right at the beginning of a new 100kb bin
    ]
    # fmt: on
    bed_file = tmpdir / "test.bed"
    with open(bed_file, "w", encoding="utf8") as file:
        for line in bed_data:
            file.write("\t".join(map(str, line)) + "\n")
    return bed_file


def test_load_bed(real_test_bed):
    """Tests the `load_bed` function."""
    bed_data = load_bed(real_test_bed)
    assert bed_data is not None
    assert len(bed_data) > 0


def test_preprocess_bed(real_test_bed):
    """Tests the `preprocess_bed` function."""
    bed_data = load_bed(real_test_bed)
    processed_data = preprocess_bed(bed_data)
    assert processed_data is not None
    assert len(processed_data) > 0


def test_get_positions_to_treat(mock_test_bed):
    """Tests the `get_positions_to_treat` function."""
    bed_data = preprocess_bed(load_bed(mock_test_bed))
    positions_to_treat = get_positions_to_treat(bed_data, bin_resolution=100000)
    assert positions_to_treat is not None
    assert len(positions_to_treat) > 0


@pytest.mark.parametrize("test_bed", ["mock_test_bed", "real_test_bed"])
def test_process_file(tmpdir, test_bed, test_hdf5, request):
    """Tests the `process_file` function."""
    test_bed = request.getfixturevalue(test_bed)  # get the bed file path

    # Compute the positions of zeroes in the input HDF5 before processing
    initial_zero_positions = {}
    with h5py.File(test_hdf5, "r") as f:
        header = list(f.keys())[0]
        hdf5_data: h5py.Group = f[header]  # type: ignore
        for chrom, dataset in hdf5_data.items():
            data = dataset[:]
            zero_positions = np.where(data == 0)[0]
            initial_zero_positions[chrom] = set(zero_positions)
            # print(f"{chrom}, initial zeros: {len(set(zero_positions))}")

    # Perform the processing
    positions_to_treat = get_positions_to_treat(
        preprocess_bed(load_bed(test_bed)), bin_resolution=100000
    )

    # Compute the number of modifications that should be made (new zeroes)
    real_modifications = sum(
        len(set(to_treat) - set(initial_zeros))
        for to_treat, initial_zeros in zip(
            positions_to_treat.values(), initial_zero_positions.values()
        )
    )

    output_dir = Path(tmpdir.mkdir("sub"))
    process_file(test_hdf5, positions_to_treat, output_dir)

    # Ensure the output file was created
    output_file_path = output_dir / (test_hdf5.stem + "_0blklst.hdf5")
    assert output_file_path.is_file()

    # Check that the data in the new file is as expected
    with h5py.File(output_file_path, "r") as f:
        header = list(f.keys())[0]
        hdf5_data: h5py.Group = f[header]  # type: ignore

        final_zero_positions = {}
        for chrom, dataset in hdf5_data.items():
            data = dataset[:]

            # Get positions of new zeroes in the output HDF5
            zero_positions = np.where(data == 0)[0]
            final_zero_positions[chrom] = set(zero_positions)
            # print(f"{chrom}, final zeros: {len(zero_positions)}")

            # Get indices for this chromosome
            indices_to_treat = positions_to_treat.get(chrom, [])

            # Check that indices_to_treat are set to zero in the dataset
            assert np.all(data[indices_to_treat] == 0)

    new_zeros_count = sum(
        len(new - initial)
        for new, initial in zip(
            final_zero_positions.values(), initial_zero_positions.values()
        )
    )
    assert new_zeros_count == real_modifications


def test_main(tmpdir, mock_test_bed, test_hdf5, mocker):
    """Tests the `main` function."""
    tmpdir = Path(tmpdir)
    hdf5_list_file = tmpdir / "hdf5_list.txt"
    hdf5_list_file.write_text(str(test_hdf5))

    # Create a simple function to replace argument parsing
    def mock_parse_arguments():
        class Args:
            """Mock arguments."""

            hdf5_list = hdf5_list_file
            bed_filter = mock_test_bed
            output_dir = tmpdir
            n_jobs = 1
            check_only = False

        return Args()

    # Replace the original parse_arguments function with our mock
    mocker.patch(
        "epiclass.utils.preprocessing.clean_hdf5.parse_arguments",
        side_effect=mock_parse_arguments,
    )

    # Run the main function
    main()

    # Make assertions about the output directory and files
    output_file_path = tmpdir / (test_hdf5.stem + "_0blklst.hdf5")
    assert output_file_path.is_file()
