"""Test module for lazy_hdf5_loader file."""
# pylint: disable=protected-access, too-many-public-methods, redefined-outer-name
from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import h5py  # pylint: disable=unused-import # import to avoid weirdness
import numpy as np
import pytest

from epiclass.core.lazy.lazy_fold_factory import LazyEpiAtlasDataset as EpiAtlasDataset
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader as Hdf5Loader
from tests.epilap_test_data import EpiAtlasTreatmentTestData


class Test_Hdf5Loader:
    """Test class Test_Hdf5Loader"""

    @pytest.fixture(scope="class", autouse=True)
    def test_folder(self, mk_logdir) -> Path:
        """Return temp hdf5 storage folder."""
        return mk_logdir("temp_hdf5s")

    @pytest.fixture(scope="class")
    def test_data(self) -> EpiAtlasDataset:
        """Mock test EpiAtlasFoldFactory."""
        return EpiAtlasTreatmentTestData.test_data().epiatlas_dataset

    @pytest.fixture(scope="function")
    def loader(self, test_data: EpiAtlasDataset, tmp_path: Path) -> Hdf5Loader:
        """Return a registered loader with a temporary mmap directory."""
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        hdf5_loader.register_hdf5s(test_data.datasource.hdf5_file, strict=True)
        return hdf5_loader

    # --- Registration and loading ---

    def test_load_hdf5s(self, test_data: EpiAtlasDataset):
        """Verify that files are loading correctly."""
        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
        hdf5_loader.register_hdf5s(test_data.datasource.hdf5_file, strict=True)
        hdf5_loader.preload_all()
        shutil.rmtree(hdf5_loader._mmap_dir)

    def test_load_hdf5_corrupted(self, test_data: EpiAtlasDataset, tmp_path: Path):
        """Verify that file corruption errors are caught/raised."""
        # Copy the source HDF5 (and write a fresh list pointing at the copy) so
        # corruption does not bleed into the class-shared fixture. The source is
        # a symlink (see EpiAtlasTreatmentTestData.create_temp_hdf5s); resolve
        # it so we don't mangle the real file via the link target.
        hdf5_list = Hdf5Loader.read_list(test_data.datasource.hdf5_file)
        chosen_src = next(iter(hdf5_list.values()))
        chosen_copy = tmp_path / chosen_src.name
        shutil.copy(chosen_src.resolve(), chosen_copy)

        list_copy = tmp_path / "hdf5s.list"
        list_copy.write_text(f"{chosen_copy}\n")

        with open(chosen_copy, "r+b") as f:
            f.seek(0)
            f.write(os.urandom(1024))

        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)

        with pytest.raises(OSError, match="file signature not found"):
            hdf5_loader.register_hdf5s(list_copy, strict=True)

    def test_register_filters_by_signal_id(
        self, test_data: EpiAtlasDataset, tmp_path: Path
    ):
        """Verify that only requested signal IDs are registered."""
        all_files = Hdf5Loader.read_list(test_data.datasource.hdf5_file)
        subset_ids = list(all_files.keys())[:2]

        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        hdf5_loader.register_hdf5s(
            test_data.datasource.hdf5_file, signal_ids=subset_ids, strict=True
        )

        assert set(hdf5_loader.file_paths.keys()) == set(subset_ids)

    def test_register_absent_signal_ids(
        self, test_data: EpiAtlasDataset, tmp_path: Path, capsys
    ):
        """Verify that absent signal IDs are reported."""
        fake_id = "a" * 32
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        hdf5_loader.register_hdf5s(
            test_data.datasource.hdf5_file, signal_ids=[fake_id], verbose=True
        )

        captured = capsys.readouterr()
        assert fake_id in captured.out
        assert len(hdf5_loader.file_paths) == 0

    # --- Preload and mmap ---

    def test_preload_creates_mmap(self, loader: Hdf5Loader):
        """Verify that preload_all creates the mmap file."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        assert mmap_path.exists()

    def test_preload_idempotent(self, loader: Hdf5Loader):
        """Verify that preload_all can be called twice without error."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        mtime_first = mmap_path.stat().st_mtime

        loader.preload_all()
        mtime_second = mmap_path.stat().st_mtime

        assert mtime_first == mtime_second

    def test_preload_no_files_registered(
        self, test_data: EpiAtlasDataset, tmp_path: Path
    ):
        """Verify that preload_all raises when no files are registered."""
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        with pytest.raises(ValueError, match="No files registered"):
            hdf5_loader.preload_all()

    def test_preload_disk_space_check(self, loader: Hdf5Loader):
        """Verify that preload_all raises on insufficient disk space."""
        fake_usage = shutil.disk_usage("/")._replace(free=1)
        with patch("shutil.disk_usage", return_value=fake_usage):
            with pytest.raises(OSError, match="Insufficient disk space"):
                loader.preload_all()

    # --- Signal loading ---

    def test_load_signal_returns_array(self, loader: Hdf5Loader):
        """Verify that load_signal returns a numpy array."""
        loader.preload_all()
        signal_id = next(iter(loader.file_paths))
        signal = loader.load_signal(signal_id)

        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float32

    def test_load_signal_consistent(self, loader: Hdf5Loader):
        """Verify that loading the same signal twice gives identical results."""
        loader.preload_all()
        signal_id = next(iter(loader.file_paths))

        signal1 = loader.load_signal(signal_id)
        signal2 = loader.load_signal(signal_id)

        np.testing.assert_array_equal(signal1, signal2)

    def test_load_signal_all_samples_same_length(self, loader: Hdf5Loader):
        """Verify that all signals have the same length."""
        loader.preload_all()
        lengths = set()
        for signal_id in loader.file_paths:
            signal = loader.load_signal(signal_id)
            lengths.add(len(signal))

        assert len(lengths) == 1

    def test_load_signal_normalized(self, test_data: EpiAtlasDataset, tmp_path: Path):
        """Verify that normalized signals have zero mean and unit std."""
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        hdf5_loader.register_hdf5s(test_data.datasource.hdf5_file, strict=True)
        hdf5_loader.preload_all()

        signal_id = next(iter(hdf5_loader.file_paths))
        signal = hdf5_loader.load_signal(signal_id)

        np.testing.assert_almost_equal(signal.mean(), 0.0, decimal=5)
        np.testing.assert_almost_equal(signal.std(), 1.0, decimal=5)

    def test_load_signal_unnormalized(self, test_data: EpiAtlasDataset, tmp_path: Path):
        """Verify that unnormalized signals differ from normalized ones."""
        loader_norm = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "norm"
        )
        loader_norm.register_hdf5s(test_data.datasource.hdf5_file, strict=True)
        loader_norm.preload_all()

        loader_raw = Hdf5Loader(
            test_data.datasource.chromsize_file, False, mmap_dir=tmp_path / "raw"
        )
        loader_raw.register_hdf5s(test_data.datasource.hdf5_file, strict=True)
        loader_raw.preload_all()

        signal_id = next(iter(loader_norm.file_paths))
        sig_norm = loader_norm.load_signal(signal_id)
        sig_raw = loader_raw.load_signal(signal_id)

        assert not np.array_equal(sig_norm, sig_raw)

    def test_load_signal_before_preload(self, loader: Hdf5Loader):
        """Verify that load_signal raises if preload_all was not called."""
        signal_id = next(iter(loader.file_paths))
        with pytest.raises(FileNotFoundError, match="Run preload_all"):
            loader.load_signal(signal_id)

    def test_load_signal_unregistered_signal_id(self, loader: Hdf5Loader):
        """Verify that load_signal raises for unregistered signal ID."""
        loader.preload_all()
        with pytest.raises(KeyError, match="not registered"):
            loader.load_signal("f" * 32)

    # --- Environment adaptation ---

    def test_adapt_to_environment(self, test_folder: Path, test_data: EpiAtlasDataset):
        """Test that the existence of $SLURM_TMPDIR/hdf5s affects hdf5 loading."""
        os.environ["SLURM_TMPDIR"] = str(test_folder)
        shutil.rmtree(test_folder, ignore_errors=True)
        os.makedirs(test_folder / "hdf5s", exist_ok=True)

        try:
            hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
            files = hdf5_loader.read_list(test_data.datasource.hdf5_file)
            files = hdf5_loader.adapt_to_environment(files)

            a_path = list(files.values())[0]
            assert str(test_folder) in str(a_path)
        finally:
            del os.environ["SLURM_TMPDIR"]

    def test_adapt_to_environment_custom_parent(
        self, test_folder: Path, test_data: EpiAtlasDataset
    ):
        """Test that $SLURM_TMPDIR/$HDF5_PARENT affects hdf5 loading."""
        new_parent = "test"
        hdf5_dir = test_folder / new_parent

        os.environ["SLURM_TMPDIR"] = str(test_folder)
        os.environ["HDF5_PARENT"] = new_parent
        shutil.rmtree(test_folder, ignore_errors=True)
        os.makedirs(hdf5_dir, exist_ok=True)

        try:
            hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
            files = hdf5_loader.read_list(test_data.datasource.hdf5_file)
            files = hdf5_loader.adapt_to_environment(files)

            a_path = list(files.values())[0]
            assert str(hdf5_dir) in str(a_path)
        finally:
            del os.environ["SLURM_TMPDIR"]
            del os.environ["HDF5_PARENT"]

    # --- Crash-safe creation / corruption handling ---

    @staticmethod
    def _chop_one_row(mmap_path: Path, n_cols: int) -> None:
        """Shrink the mmap body by one row, leaving the full-shape header intact.

        Simulates a process killed (e.g. SIGSEGV) mid-preload: the .npy header still
        declares every row but the body is short, which would otherwise hang/SIGBUS
        the next reader.
        """
        row_bytes = n_cols * np.dtype(np.float32).itemsize
        os.truncate(mmap_path, mmap_path.stat().st_size - row_bytes)

    def _fresh_loader(self, test_data: EpiAtlasDataset, mmap_dir: Path) -> Hdf5Loader:
        """Loader registered on the test files, pointing at an existing mmap_dir."""
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=mmap_dir
        )
        hdf5_loader.register_hdf5s(test_data.datasource.hdf5_file, strict=True)
        return hdf5_loader

    def test_preload_leaves_no_tmp(self, loader: Hdf5Loader):
        """Verify the atomic-rename write leaves no leftover .tmp file on success."""
        loader.preload_all()
        assert not list(loader._mmap_dir.glob("*.tmp"))

    def test_mmap_exists(self, loader: Hdf5Loader):
        """Verify mmap_exists() reflects whether the cache has been built."""
        assert loader.mmap_exists() is False
        loader.preload_all()
        assert loader.mmap_exists() is True

    def test_integrity_flags_truncated_mmap(self, loader: Hdf5Loader):
        """Verify a truncated mmap is reported as corrupt rather than trusted."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        self._chop_one_row(mmap_path, loader.signal_length)

        err = loader._mmap_integrity_error(mmap_path, len(loader.file_paths))
        assert err is not None and "truncated" in err

    def test_load_signal_truncated_mmap_raises(
        self, test_data: EpiAtlasDataset, loader: Hdf5Loader
    ):
        """Verify reading a truncated mmap raises a clear error instead of hanging."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        self._chop_one_row(mmap_path, loader.signal_length)

        fresh = self._fresh_loader(test_data, loader._mmap_dir)
        signal_id = next(iter(fresh.file_paths))
        with pytest.raises(RuntimeError, match="Corrupt mmap"):
            fresh.load_signal(signal_id)

    def test_preload_rebuilds_truncated_mmap(
        self, test_data: EpiAtlasDataset, loader: Hdf5Loader
    ):
        """Verify preload_all rebuilds a crash-truncated mmap rather than reusing it."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        n_cols = loader.signal_length
        self._chop_one_row(mmap_path, n_cols)

        fresh = self._fresh_loader(test_data, loader._mmap_dir)
        fresh.preload_all()

        assert fresh._mmap_integrity_error(mmap_path, len(fresh.file_paths)) is None
        signal_id = next(iter(fresh.file_paths))
        assert fresh.load_signal(signal_id).shape[0] == n_cols

    def test_preload_rebuilds_stale_row_count(
        self, test_data: EpiAtlasDataset, loader: Hdf5Loader
    ):
        """Verify a mmap built for a different sample set is rebuilt, not reused."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        if len(loader.file_paths) < 2:
            pytest.skip("Need >1 sample to exercise the row-count mismatch path.")

        subset_ids = list(loader.file_paths)[:1]
        fresh = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=loader._mmap_dir
        )
        fresh.register_hdf5s(
            test_data.datasource.hdf5_file, signal_ids=subset_ids, strict=True
        )

        err = fresh._mmap_integrity_error(mmap_path, len(subset_ids))
        assert err is not None and "row-count mismatch" in err

        fresh.preload_all()
        assert np.load(mmap_path, mmap_mode="r").shape[0] == len(subset_ids)

    # --- Ordered id manifest (row-order validation) ---

    def _reordered_loader(
        self, test_data: EpiAtlasDataset, mmap_dir: Path, list_dir: Path
    ) -> Hdf5Loader:
        """Loader registered on the same files in reversed order, sharing mmap_dir.

        Same file *set* and row count as ``_fresh_loader`` -- only the order
        differs, which is exactly the silent-corruption case the manifest guards.
        """
        files = Hdf5Loader.read_list(test_data.datasource.hdf5_file)
        reversed_paths = list(files.values())[::-1]
        reordered_list = list_dir / "reversed.list"
        reordered_list.write_text(
            "\n".join(str(p) for p in reversed_paths) + "\n", encoding="utf-8"
        )
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=mmap_dir
        )
        hdf5_loader.register_hdf5s(reordered_list, strict=True)
        return hdf5_loader

    def test_preload_writes_id_manifest(self, loader: Hdf5Loader):
        """Verify preload_all writes an ordered id manifest matching the row order."""
        loader.preload_all()
        manifest = loader._get_manifest_path()
        assert manifest.is_file()
        assert manifest.read_text(encoding="utf-8").splitlines() == list(
            loader.file_paths.keys()
        )

    def test_integrity_flags_reordered_mmap(
        self, test_data: EpiAtlasDataset, loader: Hdf5Loader, tmp_path: Path
    ):
        """Verify a same-size cache built in a different order is rejected, then rebuilt.

        This is the regression test for the predict_CV scramble: identical files,
        identical row count, different order -- the row-count check alone passed and
        every sample was fed another sample's signal.
        """
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        if len(loader.file_paths) < 2:
            pytest.skip("Need >1 sample to exercise a reordering.")

        reordered = self._reordered_loader(test_data, loader._mmap_dir, tmp_path)
        # Same set + count, different order -> manifest mismatch (not row-count).
        err = reordered._mmap_integrity_error(mmap_path, len(reordered.file_paths))
        assert err is not None and "manifest" in err

        # Rebuild, then every id must map to its *own* signal in the new order.
        reordered.preload_all()
        assert (
            reordered._mmap_integrity_error(mmap_path, len(reordered.file_paths)) is None
        )
        ref = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "ref_mmap"
        )
        ref.register_hdf5s(tmp_path / "reversed.list", strict=True)
        ref.preload_all()
        for sid in reordered.file_paths:
            assert np.array_equal(reordered.load_signal(sid), ref.load_signal(sid))

    def test_integrity_flags_missing_manifest(self, loader: Hdf5Loader):
        """Verify a cache without a manifest (legacy/crash) is treated as stale."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        loader._get_manifest_path().unlink()

        # Manifest is gone -> cannot verify order -> treat as stale.
        err = loader._mmap_integrity_error(mmap_path, len(loader.file_paths))
        assert err is not None and "manifest" in err

        loader.preload_all()  # rebuilds and re-writes the manifest
        assert loader._get_manifest_path().is_file()
        assert loader._mmap_integrity_error(mmap_path, len(loader.file_paths)) is None

    def test_load_signal_reordered_mmap_raises(
        self, test_data: EpiAtlasDataset, loader: Hdf5Loader, tmp_path: Path
    ):
        """Verify reading a reordered cache raises rather than returning wrong rows."""
        loader.preload_all()
        if len(loader.file_paths) < 2:
            pytest.skip("Need >1 sample to exercise a reordering.")

        reordered = self._reordered_loader(test_data, loader._mmap_dir, tmp_path)
        signal_id = next(iter(reordered.file_paths))
        with pytest.raises(RuntimeError, match="Corrupt mmap"):
            reordered.load_signal(signal_id)

    def test_reuse_same_order_ok(self, test_data: EpiAtlasDataset, loader: Hdf5Loader):
        """Verify an identical (same files, same order) cache is reused, not rebuilt."""
        loader.preload_all()
        mmap_path = loader._get_mmap_path()
        built_mtime = mmap_path.stat().st_mtime_ns

        fresh = self._fresh_loader(test_data, loader._mmap_dir)
        assert fresh._mmap_integrity_error(mmap_path, len(fresh.file_paths)) is None
        fresh.preload_all()  # no-op: must not rewrite the cache
        assert mmap_path.stat().st_mtime_ns == built_mtime

    # --- Mmap path naming ---

    def test_mmap_path_differs_by_normalization(
        self, test_data: EpiAtlasDataset, tmp_path: Path
    ):
        """Verify that norm and raw mmap files have different paths."""
        loader_norm = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path
        )
        loader_raw = Hdf5Loader(
            test_data.datasource.chromsize_file, False, mmap_dir=tmp_path
        )

        assert loader_norm._get_mmap_path() != loader_raw._get_mmap_path()
