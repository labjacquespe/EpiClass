"""Tests for compute_bin_metrics.py after lazy migration.

Covers the pure metrics function with synthetic input and a sanity-check
end-to-end run against the small saccer3 fixture.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from epiclass.utils.metrics.compute_bin_metrics import (
    compute_metrics,
    main as main_module,
)
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("compute_bin_metrics")


class TestComputeMetrics:
    """Unit tests for the pure compute_metrics function."""

    def test_known_values(self):
        """Values match hand-computed mean/std/median/IQR."""
        signals = {
            "a": np.array([0.0, 2.0, 10.0], dtype=np.float32),
            "b": np.array([2.0, 4.0, 20.0], dtype=np.float32),
            "c": np.array([4.0, 6.0, 30.0], dtype=np.float32),
            "d": np.array([6.0, 8.0, 40.0], dtype=np.float32),
        }
        metrics = compute_metrics(signals)

        assert set(metrics) == {"mean", "std", "median", "iqr"}
        np.testing.assert_allclose(metrics["mean"], [3.0, 5.0, 25.0])
        np.testing.assert_allclose(metrics["median"], [3.0, 5.0, 25.0])
        # std with ddof=0 (np.std default)
        np.testing.assert_allclose(
            metrics["std"], np.std(np.array(list(signals.values())), axis=0)
        )
        # IQR = p75 - p25
        np.testing.assert_allclose(metrics["iqr"], [3.0, 3.0, 15.0])

    def test_output_shape_matches_signal_length(self):
        """Each metric array has the same length as the input signals."""
        signal_length = 17
        signals = {
            f"sig{i}": np.arange(signal_length, dtype=np.float32) + i for i in range(5)
        }
        metrics = compute_metrics(signals)
        for name, arr in metrics.items():
            assert arr.shape == (signal_length,), f"{name} has wrong shape"

    def test_single_signal(self):
        """One signal: mean/median equal the signal, std == 0, iqr == 0."""
        signal = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        metrics = compute_metrics({"only": signal})
        np.testing.assert_allclose(metrics["mean"], signal)
        np.testing.assert_allclose(metrics["median"], signal)
        np.testing.assert_allclose(metrics["std"], np.zeros_like(signal))
        np.testing.assert_allclose(metrics["iqr"], np.zeros_like(signal))


def test_main_writes_expected_outputs(test_dir: Path, saccer3_small_hdf5_file_list: Path):
    """End-to-end: register HDF5s, preload mmap, write npz + signal list.

    Asserts that the npz contains all four metrics with consistent shape
    and that the companion .list file enumerates one signal per HDF5 input.
    """
    chroms = FIXTURES_DIR / "saccer3" / "saccer3.can.chrom.sizes"
    sys.argv = [
        "compute_bin_metrics.py",
        str(saccer3_small_hdf5_file_list),
        str(chroms),
        str(test_dir),
    ]
    main_module()

    stem = saccer3_small_hdf5_file_list.stem
    npz_path = test_dir / f"{stem}_metrics.npz"
    list_path = test_dir / f"{stem}_metrics_files.list"
    assert npz_path.exists(), f"Expected metrics npz at {npz_path}"
    assert list_path.exists(), f"Expected file list at {list_path}"

    with np.load(npz_path) as data:
        assert set(data.files) == {"mean", "std", "median", "iqr"}
        shapes = {
            arr.shape for arr in (data["mean"], data["std"], data["median"], data["iqr"])
        }
        assert len(shapes) == 1, f"Metric shapes diverge: {shapes}"
        signal_length = data["mean"].shape[0]
        assert signal_length > 0

    expected_n = len(saccer3_small_hdf5_file_list.read_text().splitlines())
    written_ids = [line for line in list_path.read_text().splitlines() if line]
    assert len(written_ids) == expected_n
