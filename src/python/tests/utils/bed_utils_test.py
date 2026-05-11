"""Test bed utils functions"""

import pytest

from epiclass.core.lazy.lazy_fold_factory import (
    LazyEpiAtlasFoldFactory as EpiAtlasFoldFactory,
)
from epiclass.utils.bed_utils import (
    bed_ranges_to_bins,
    bins_to_bed_ranges,
    read_bed_to_ranges,
    write_to_bed,
)


def test_bins_to_bed_ranges() -> None:
    """Tests the function bins_to_bed_ranges().

    This function tests bins_to_bed_ranges() with various input arguments, including single bin indexes,
    multiple bin indexes, and out of range bin indexes. The test checks if the function correctly converts
    bin indexes to chromosome ranges and raises an IndexError when a bin index is out of range.
    """
    chroms = [("chr1", 100), ("chr2", 150), ("chr3", 300), ("chrX", 120)]
    resolution = 100

    # Test single bin indexes
    assert bins_to_bed_ranges([0], chroms, resolution) == [("chr1", 0, 100)]
    assert bins_to_bed_ranges([1], chroms, resolution) == [("chr2", 0, 100)]
    assert bins_to_bed_ranges([2], chroms, resolution) == [("chr2", 100, 150)]
    assert bins_to_bed_ranges([3], chroms, resolution) == [("chr3", 0, 100)]
    assert bins_to_bed_ranges([4], chroms, resolution) == [("chr3", 100, 200)]
    assert bins_to_bed_ranges([5], chroms, resolution) == [("chr3", 200, 300)]
    assert bins_to_bed_ranges([6], chroms, resolution) == [("chrX", 0, 100)]
    assert bins_to_bed_ranges([7], chroms, resolution) == [("chrX", 100, 120)]

    # Test multiple bin indexes
    assert bins_to_bed_ranges([0, 2, 4], chroms, resolution) == [
        ("chr1", 0, 100),
        ("chr2", 100, 150),
        ("chr3", 100, 200),
    ]
    # Test out of order bin indexes
    assert bins_to_bed_ranges([5, 1, 3], chroms, resolution) == [
        ("chr3", 200, 300),
        ("chr2", 0, 100),
        ("chr3", 0, 100),
    ]

    # Test out of range bin indexes
    with pytest.raises(IndexError):
        bins_to_bed_ranges([8], chroms, resolution)
    with pytest.raises(IndexError):
        bins_to_bed_ranges([-1], chroms, resolution)
    with pytest.raises(IndexError):
        bins_to_bed_ranges([1, 2, 8], chroms, resolution)


def test_bed_ranges_to_bins() -> None:
    """Tests the function bed_ranges_to_bins().

    This function tests bed_ranges_to_bins() with various bed ranges, including single and multiple ranges,
    all of which are multiples of the resolution. The test checks if the function correctly converts bed ranges
    to bin indexes.
    """
    chroms = [("chr1", 5000), ("chr2", 5000), ("chr3", 5000)]
    resolution = 1000

    # Test single bed ranges
    assert bed_ranges_to_bins([("chr1", 0, 1000)], chroms, resolution) == [0]
    assert bed_ranges_to_bins([("chr1", 1000, 2000)], chroms, resolution) == [1]
    assert bed_ranges_to_bins([("chr2", 0, 1000)], chroms, resolution) == [5]
    assert bed_ranges_to_bins([("chr3", 4000, 5000)], chroms, resolution) == [14]

    # Test multiple bed ranges
    assert bed_ranges_to_bins(
        [
            ("chr1", 0, 1000),
            ("chr1", 2000, 3000),
            ("chr2", 0, 1000),
        ],
        chroms,
        resolution,
    ) == [0, 2, 5]

    assert bed_ranges_to_bins(
        [
            ("chr1", 1000, 2000),
            ("chr3", 3000, 4000),
            ("chr2", 1000, 2000),
        ],
        chroms,
        resolution,
    ) == [1, 13, 6]

    # Test ranges that span across multiple bins
    assert bed_ranges_to_bins([("chr1", 0, 3000)], chroms, resolution) == [0, 1, 2]
    assert bed_ranges_to_bins([("chr2", 1000, 3000)], chroms, resolution) == [6, 7]


def test_bed_bin_conversion():
    """
    Test conversion of bin indexes to bed ranges and back to bin indexes.

    This function tests the `bins_to_bed_ranges` and `bed_ranges_to_bins` functions
    by converting a set of bin indexes to bed ranges, and then converting these
    ranges back to bin indexes. It tests a variety of bin index cases including
    the first and last bins, bins that span across two chromosomes, bins that are
    at the boundary of a chromosome, and no bins. The test asserts that the
    original and returned bin indexes are the same after these conversions,
    disregarding the order of the elements.

    Raises:
        AssertionError: If the returned bin indexes do not match the original bin indexes
        after conversion to ranges and back to bins for any of the tested cases.
    """
    # Given
    chroms = [("chr1", 5000), ("chr2", 5000), ("chr3", 5000)]
    resolution = 1000

    # Test with various types of bin indexes
    test_cases = [
        [0],  # The first bin
        [4],  # The last bin of the first chromosome
        [5],  # The first bin of the second chromosome
        [14],  # The last bin of the genome
        [0, 4, 5, 14],  # Various bins
        [2, 3, 4, 5, 6],  # Bins that span across two chromosomes
        [],  # No bins
    ]

    for bin_indexes in test_cases:
        ranges = bins_to_bed_ranges(bin_indexes, chroms, resolution)
        returned_bin_indexes = bed_ranges_to_bins(ranges, chroms, resolution)

        assert set(returned_bin_indexes) == set(
            bin_indexes
        ), f"Returned bin indexes should match the original bin indexes after conversion to ranges and back to bins. Failed for bin_indexes={bin_indexes}"


def test_bed_bin_conversion_2(test_epiatlas_data_handler: EpiAtlasFoldFactory):
    """Test conversion of bin indexes to bed ranges and back to bin indexes."""
    chroms = test_epiatlas_data_handler.epiatlas_dataset.datasource.load_chrom_sizes()
    resolution = 1000 * 100

    # fmt: off
    bin_indexes = [29956, 28774, 28775, 16809, 26345, 29551, 15888, 5651, 15219, 28889, 11325, 8574]
    # fmt: on

    ranges = bins_to_bed_ranges(bin_indexes, chroms, resolution)
    returned_bin_indexes = bed_ranges_to_bins(ranges, chroms, resolution)

    assert set(returned_bin_indexes) == set(
        bin_indexes
    ), f"Returned bin indexes should match the original bin indexes after conversion to ranges and back to bins. Failed for bin_indexes={bin_indexes}"


def test_bed_bin_conversion_3(test_epiatlas_data_handler: EpiAtlasFoldFactory, tmp_path):
    """Test conversion of bin indexes to bed files and back to bin indexes."""
    chroms = test_epiatlas_data_handler.epiatlas_dataset.datasource.load_chrom_sizes()
    resolution = 1000 * 100

    # fmt: off
    bin_indexes = [29956, 28774, 28775, 16809, 26345, 29551, 15888, 5651, 15219, 28889, 11325, 8574]
    # fmt: on

    ranges = bins_to_bed_ranges(bin_indexes, chroms, resolution)

    bed_path = tmp_path / "test.bed"
    write_to_bed(bed_ranges=ranges, bed_path=bed_path)

    returned_ranges = read_bed_to_ranges(bed_source=bed_path)
    returned_bin_indexes = bed_ranges_to_bins(returned_ranges, chroms, resolution)

    assert set(returned_ranges) == set(
        ranges
    ), f"Returned bin ranges should match the original bin ranges after reading back from written bed. Failed for ranges={ranges}"

    assert set(returned_bin_indexes) == set(
        bin_indexes
    ), f"Returned bin indexes should match the original bin indexes after conversion to ranges and back to bins. Failed for bin_indexes={bin_indexes}"


def test_bins_to_bed_ranges_preserves_order() -> None:
    """Ensure that bins_to_bed_ranges returns output in the same order as input bin indexes."""
    chroms = [("chr1", 100), ("chr2", 150), ("chr3", 300)]
    resolution = 100

    bin_indexes = [2, 0, 4, 1]  # deliberately out of order
    expected = [
        ("chr2", 100, 150),  # bin 2
        ("chr1", 0, 100),  # bin 0
        ("chr3", 100, 200),  # bin 4
        ("chr2", 0, 100),  # bin 1
    ]

    actual = bins_to_bed_ranges(bin_indexes, chroms, resolution)
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_bed_ranges_to_bins_preserves_order() -> None:
    """Ensure that bed_ranges_to_bins returns output in the same order as input ranges."""
    chroms = [("chr1", 100), ("chr2", 150), ("chr3", 300)]
    resolution = 100

    bed_ranges = [
        ("chr3", 100, 200),  # bin 4
        ("chr2", 0, 100),  # bin 1
        ("chr2", 100, 150),  # bin 2
        ("chr1", 0, 100),  # bin 0
    ]
    expected = [4, 1, 2, 0]

    actual = bed_ranges_to_bins(bed_ranges, chroms, resolution)
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_bed_bin_roundtrip_preserves_order() -> None:
    """Test that bin → bed → bin conversion preserves order."""
    chroms = [("chr1", 5000), ("chr2", 5000)]
    resolution = 1000
    bin_indexes = [5, 1, 9, 0]

    ranges = bins_to_bed_ranges(bin_indexes, chroms, resolution)
    returned_bin_indexes = bed_ranges_to_bins(ranges, chroms, resolution)

    assert (
        returned_bin_indexes == bin_indexes
    ), f"Order was not preserved. Expected {bin_indexes}, got {returned_bin_indexes}"
