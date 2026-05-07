"""Test module for chunked_hdf5_loader."""
# pylint: disable=protected-access, too-many-public-methods, redefined-outer-name, use-implicit-booleaness-not-comparison
from __future__ import annotations

from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest

from epiclass.core.lazy.chunked_hdf5_loader import ChunkedHdf5Loader

# --- Fixtures ---


@pytest.fixture
def signal_length() -> int:
    """Signal length used across test fixtures."""
    return 100


@pytest.fixture
def sample_ids() -> List[str]:
    """Sample IDs for the primary test chunk file."""
    return [f"sample_{i:04d}" for i in range(50)]


@pytest.fixture
def chunk_file(tmp_path: Path, sample_ids: List[str], signal_length: int) -> Path:
    """Create a single chunked HDF5 file with random signals."""
    path = tmp_path / "chunk_0.h5"
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((len(sample_ids), signal_length)).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("signals", data=signals)
        f.create_dataset(
            "sample_ids",
            data=np.array(sample_ids, dtype=h5py.string_dtype()),
        )

    return path


@pytest.fixture
def chunk_file_second(tmp_path: Path, signal_length: int) -> Path:
    """Create a second chunk file with different sample IDs."""
    path = tmp_path / "chunk_1.h5"
    ids = [f"other_{i:04d}" for i in range(30)]
    rng = np.random.default_rng(99)
    signals = rng.standard_normal((len(ids), signal_length)).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("signals", data=signals)
        f.create_dataset(
            "sample_ids",
            data=np.array(ids, dtype=h5py.string_dtype()),
        )

    return path


@pytest.fixture
def chunk_dir(chunk_file: Path) -> Path:
    """Return the directory containing chunk files."""
    return chunk_file.parent


@pytest.fixture
def loader(chunk_file: Path) -> ChunkedHdf5Loader:
    """Return a loader registered with the primary chunk file."""
    ldr = ChunkedHdf5Loader()
    ldr.register_chunked_hdf5s(chunk_file, strict=True)
    return ldr


# --- Registration ---


class TestRegistration:
    """Tests for file registration and validation."""

    def test_register_single_file(
        self, chunk_file: Path, sample_ids: List[str], signal_length: int
    ):
        """Verify registration from a single chunk file."""
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s(chunk_file, strict=True)

        assert ldr.num_registered == len(sample_ids)
        assert ldr.signal_length == signal_length
        assert set(ldr.sample_ids) == set(sample_ids)

    def test_register_file_list(self, chunk_file: Path, chunk_file_second: Path):
        """Verify registration from a list of files."""
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s([chunk_file, chunk_file_second], strict=True)

        assert ldr.num_registered == 80  # 50 + 30

    def test_register_directory(self, chunk_dir: Path):
        """Verify registration from a directory of chunk files."""
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s(chunk_dir, strict=True)

        assert ldr.num_registered == 80
        assert len(ldr.chunk_files) == 2

    def test_register_filters_by_sample_id(self, chunk_file: Path):
        """Verify that only requested sample IDs are registered."""
        subset = ["sample_0000", "sample_0010", "sample_0020"]
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s(chunk_file, sample_ids=subset, strict=True)

        assert ldr.num_registered == len(subset)
        assert set(ldr.sample_ids) == set(subset)

    def test_register_absent_sample_ids(self, chunk_file: Path, capsys):
        """Verify that absent sample IDs are reported."""
        fake_ids = ["nonexistent_0", "nonexistent_1"]
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s(chunk_file, sample_ids=fake_ids, verbose=True)

        captured = capsys.readouterr()
        assert "2" in captured.out
        assert "not found" in captured.out
        assert ldr.num_registered == 0

    def test_register_missing_signals_dataset(self, tmp_path: Path):
        """Verify error when 'signals' dataset is missing."""
        bad_file = tmp_path / "bad.h5"
        with h5py.File(bad_file, "w") as f:
            f.create_dataset(
                "sample_ids",
                data=np.array(["a"], dtype=h5py.string_dtype()),
            )

        ldr = ChunkedHdf5Loader()
        with pytest.raises(KeyError, match="Missing 'signals'"):
            ldr.register_chunked_hdf5s(bad_file, strict=True)

    def test_register_missing_sample_ids_dataset(self, tmp_path: Path):
        """Verify error when 'sample_ids' dataset is missing."""
        bad_file = tmp_path / "bad.h5"
        with h5py.File(bad_file, "w") as f:
            f.create_dataset("signals", data=np.zeros((1, 10)))

        ldr = ChunkedHdf5Loader()
        with pytest.raises(KeyError, match="Missing.*'sample_ids'"):
            ldr.register_chunked_hdf5s(bad_file, strict=True)

    def test_register_mismatched_lengths(self, tmp_path: Path):
        """Verify error when signals and sample_ids have different lengths."""
        bad_file = tmp_path / "bad.h5"
        with h5py.File(bad_file, "w") as f:
            f.create_dataset("signals", data=np.zeros((5, 10), dtype=np.float32))
            f.create_dataset(
                "sample_ids",
                data=np.array(["a", "b", "c"], dtype=h5py.string_dtype()),
            )

        ldr = ChunkedHdf5Loader()
        with pytest.raises(ValueError, match="Mismatched lengths"):
            ldr.register_chunked_hdf5s(bad_file, strict=True)

    def test_register_signal_length_mismatch_across_files(
        self, chunk_file: Path, tmp_path: Path
    ):
        """Verify error when chunk files have different signal lengths."""
        bad_file = tmp_path / "wrong_length.h5"
        with h5py.File(bad_file, "w") as f:
            f.create_dataset("signals", data=np.zeros((3, 999), dtype=np.float32))
            f.create_dataset(
                "sample_ids",
                data=np.array(["x_0", "x_1", "x_2"], dtype=h5py.string_dtype()),
            )

        ldr = ChunkedHdf5Loader()
        with pytest.raises(ValueError, match="Signal length mismatch"):
            ldr.register_chunked_hdf5s([chunk_file, bad_file], strict=True)

    def test_register_duplicate_sample_id_warns(
        self, chunk_file: Path, tmp_path: Path, capsys
    ):
        """Verify warning when a sample ID appears in multiple files."""
        dup_file = tmp_path / "dup.h5"
        with h5py.File(dup_file, "w") as f:
            f.create_dataset("signals", data=np.zeros((1, 100), dtype=np.float32))
            f.create_dataset(
                "sample_ids",
                data=np.array(["sample_0000"], dtype=h5py.string_dtype()),
            )

        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s([chunk_file, dup_file], strict=True)

        captured = capsys.readouterr()
        assert "duplicate" in captured.err.lower()

    def test_register_corrupted_file_strict(self, tmp_path: Path):
        """Verify that a corrupted file raises in strict mode."""
        bad_file = tmp_path / "corrupt.h5"
        bad_file.write_bytes(b"not an hdf5 file")

        ldr = ChunkedHdf5Loader()
        with pytest.raises(OSError):
            ldr.register_chunked_hdf5s(bad_file, strict=True)

    def test_register_corrupted_file_non_strict(self, tmp_path: Path, capsys):
        """Verify that a corrupted file is skipped in non-strict mode."""
        bad_file = tmp_path / "corrupt.h5"
        bad_file.write_bytes(b"not an hdf5 file")

        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s(bad_file, strict=False)

        assert ldr.num_registered == 0
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_register_empty_directory(self, tmp_path: Path):
        """Verify registration from empty directory registers nothing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s(empty_dir, verbose=False)

        assert ldr.num_registered == 0

    def test_register_bytes_sample_ids(self, tmp_path: Path):
        """Verify that byte-encoded sample IDs are decoded properly."""
        path = tmp_path / "bytes_ids.h5"
        ids = [b"byte_0", b"byte_1", b"byte_2"]
        with h5py.File(path, "w") as f:
            f.create_dataset("signals", data=np.zeros((3, 10), dtype=np.float32))
            f.create_dataset("sample_ids", data=np.array(ids))

        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s(path, strict=True)

        assert ldr.num_registered == 3
        assert "byte_0" in ldr.sample_ids


# --- Signal loading ---


class TestLoadSignal:
    """Tests for single signal loading."""

    def test_load_returns_array(self, loader: ChunkedHdf5Loader):
        """Verify load_signal returns a float32 numpy array."""
        signal = loader.load_signal("sample_0000")
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float32

    def test_load_correct_shape(self, loader: ChunkedHdf5Loader, signal_length: int):
        """Verify loaded signal has the expected length."""
        signal = loader.load_signal("sample_0000")
        assert signal.shape == (signal_length,)

    def test_load_consistent(self, loader: ChunkedHdf5Loader):
        """Verify that loading the same signal twice gives identical results."""
        sig1 = loader.load_signal("sample_0000")
        sig2 = loader.load_signal("sample_0000")
        np.testing.assert_array_equal(sig1, sig2)

    def test_load_different_samples_differ(self, loader: ChunkedHdf5Loader):
        """Verify that different samples have different signals."""
        sig0 = loader.load_signal("sample_0000")
        sig1 = loader.load_signal("sample_0001")
        assert not np.array_equal(sig0, sig1)

    def test_load_matches_source_data(self, chunk_file: Path, loader: ChunkedHdf5Loader):
        """Verify loaded signal matches the original HDF5 data."""
        with h5py.File(chunk_file, "r") as f:
            expected = np.array(f["signals"][0], dtype=np.float32)

        actual = loader.load_signal("sample_0000")
        np.testing.assert_array_equal(actual, expected)

    def test_load_unregistered_raises(self, loader: ChunkedHdf5Loader):
        """Verify KeyError for unregistered sample ID."""
        with pytest.raises(KeyError, match="not registered"):
            loader.load_signal("nonexistent")

    def test_load_from_multiple_files(self, chunk_file: Path, chunk_file_second: Path):
        """Verify loading signals from different chunk files."""
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s([chunk_file, chunk_file_second], strict=True)

        sig_first = ldr.load_signal("sample_0000")
        sig_second = ldr.load_signal("other_0000")

        assert sig_first.shape == sig_second.shape
        assert not np.array_equal(sig_first, sig_second)
        ldr.close()


# --- Batch loading ---


class TestLoadBatch:
    """Tests for batch signal loading."""

    def test_batch_shape(self, loader: ChunkedHdf5Loader, signal_length: int):
        """Verify batch returns correct shape."""
        ids = ["sample_0000", "sample_0001", "sample_0002"]
        batch = loader.load_batch(ids)
        assert batch.shape == (3, signal_length)
        assert batch.dtype == np.float32

    def test_batch_matches_individual(self, loader: ChunkedHdf5Loader):
        """Verify batch results match individual load_signal calls."""
        ids = ["sample_0005", "sample_0010", "sample_0015"]

        batch = loader.load_batch(ids)
        for i, sample_id in enumerate(ids):
            individual = loader.load_signal(sample_id)
            np.testing.assert_array_equal(batch[i], individual)

    def test_batch_preserves_order(self, loader: ChunkedHdf5Loader):
        """Verify batch output matches the requested order."""
        ids_forward = ["sample_0000", "sample_0001", "sample_0002"]
        ids_reverse = list(reversed(ids_forward))

        batch_fwd = loader.load_batch(ids_forward)
        batch_rev = loader.load_batch(ids_reverse)

        np.testing.assert_array_equal(batch_fwd[0], batch_rev[2])
        np.testing.assert_array_equal(batch_fwd[1], batch_rev[1])
        np.testing.assert_array_equal(batch_fwd[2], batch_rev[0])

    def test_batch_across_files(self, chunk_file: Path, chunk_file_second: Path):
        """Verify batch loading across multiple chunk files."""
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s([chunk_file, chunk_file_second], strict=True)

        ids = ["sample_0000", "other_0000", "sample_0001", "other_0001"]
        batch = ldr.load_batch(ids)
        assert batch.shape[0] == 4

        for i, sample_id in enumerate(ids):
            individual = ldr.load_signal(sample_id)
            np.testing.assert_array_equal(batch[i], individual)

        ldr.close()

    def test_batch_single_element(self, loader: ChunkedHdf5Loader):
        """Verify batch works with a single element."""
        batch = loader.load_batch(["sample_0000"])
        individual = loader.load_signal("sample_0000")
        np.testing.assert_array_equal(batch[0], individual)

    def test_batch_unregistered_raises(self, loader: ChunkedHdf5Loader):
        """Verify KeyError when batch contains unregistered ID."""
        with pytest.raises(KeyError, match="not registered"):
            loader.load_batch(["sample_0000", "nonexistent"])


# --- File handle management ---


class TestFileHandles:
    """Tests for file handle lifecycle."""

    def test_close_clears_handles(self, loader: ChunkedHdf5Loader):
        """Verify close() clears all file handles."""
        # Trigger a file handle to be opened
        loader.load_signal("sample_0000")
        assert len(loader._file_handles) > 0

        loader.close()
        assert len(loader._file_handles) == 0

    def test_context_manager(self, chunk_file: Path):
        """Verify context manager opens and closes properly."""
        with ChunkedHdf5Loader() as ldr:
            ldr.register_chunked_hdf5s(chunk_file, strict=True)
            ldr.load_signal("sample_0000")
            assert len(ldr._file_handles) > 0

        assert len(ldr._file_handles) == 0

    def test_load_after_close_reopens(self, loader: ChunkedHdf5Loader):
        """Verify that loading after close re-opens file handles."""
        loader.load_signal("sample_0000")
        loader.close()

        # Should re-open the file handle transparently
        signal = loader.load_signal("sample_0000")
        assert isinstance(signal, np.ndarray)

    def test_handles_reused_across_calls(self, loader: ChunkedHdf5Loader):
        """Verify the same file handle is reused for multiple loads."""
        loader.load_signal("sample_0000")
        handles_after_first = dict(loader._file_handles)

        loader.load_signal("sample_0001")
        handles_after_second = dict(loader._file_handles)

        assert handles_after_first.keys() == handles_after_second.keys()


# --- Edge cases ---


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_loader_properties(self):
        """Verify properties on a fresh loader."""
        ldr = ChunkedHdf5Loader()
        assert ldr.num_registered == 0
        assert ldr.signal_length is None
        assert ldr.sample_ids == []
        assert ldr.chunk_files == []

    def test_single_sample_file(self, tmp_path: Path):
        """Verify handling of a chunk file with one sample."""
        path = tmp_path / "single.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("signals", data=np.ones((1, 50), dtype=np.float32))
            f.create_dataset(
                "sample_ids",
                data=np.array(["only_one"], dtype=h5py.string_dtype()),
            )

        with ChunkedHdf5Loader() as ldr:
            ldr.register_chunked_hdf5s(path, strict=True)
            assert ldr.num_registered == 1

            signal = ldr.load_signal("only_one")
            np.testing.assert_array_equal(signal, np.ones(50, dtype=np.float32))

    def test_large_batch_all_samples(
        self, loader: ChunkedHdf5Loader, sample_ids: List[str]
    ):
        """Verify batch loading all registered samples at once."""
        batch = loader.load_batch(sample_ids)
        assert batch.shape == (len(sample_ids), loader.signal_length)

    def test_id_to_location_maps_correctly(
        self, chunk_file: Path, chunk_file_second: Path
    ):
        """Verify id_to_location points to the correct file and index."""
        ldr = ChunkedHdf5Loader()
        ldr.register_chunked_hdf5s([chunk_file, chunk_file_second], strict=True)

        # sample_0000 should be in chunk_file at index 0
        loc_file, loc_idx = ldr.id_to_location["sample_0000"]
        assert loc_file == chunk_file
        assert loc_idx == 0

        # other_0000 should be in chunk_file_second at index 0
        loc_file, loc_idx = ldr.id_to_location["other_0000"]
        assert loc_file == chunk_file_second
        assert loc_idx == 0

        ldr.close()

    def test_chunk_files_property_is_copy(self, loader: ChunkedHdf5Loader):
        """Verify chunk_files returns a copy, not internal state."""
        files = loader.chunk_files
        files.clear()
        assert len(loader.chunk_files) > 0
