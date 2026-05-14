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

    def test_register_filters_by_md5(self, test_data: EpiAtlasDataset, tmp_path: Path):
        """Verify that only requested md5s are registered."""
        all_files = Hdf5Loader.read_list(test_data.datasource.hdf5_file)
        subset_md5s = list(all_files.keys())[:2]

        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        hdf5_loader.register_hdf5s(
            test_data.datasource.hdf5_file, md5s=subset_md5s, strict=True
        )

        assert set(hdf5_loader.file_paths.keys()) == set(subset_md5s)

    def test_register_absent_md5s(
        self, test_data: EpiAtlasDataset, tmp_path: Path, capsys
    ):
        """Verify that absent md5s are reported."""
        fake_md5 = "a" * 32
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        hdf5_loader.register_hdf5s(
            test_data.datasource.hdf5_file, md5s=[fake_md5], verbose=True
        )

        captured = capsys.readouterr()
        assert fake_md5 in captured.out
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
        md5 = next(iter(loader.file_paths))
        signal = loader.load_signal(md5)

        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float32

    def test_load_signal_consistent(self, loader: Hdf5Loader):
        """Verify that loading the same signal twice gives identical results."""
        loader.preload_all()
        md5 = next(iter(loader.file_paths))

        signal1 = loader.load_signal(md5)
        signal2 = loader.load_signal(md5)

        np.testing.assert_array_equal(signal1, signal2)

    def test_load_signal_all_samples_same_length(self, loader: Hdf5Loader):
        """Verify that all signals have the same length."""
        loader.preload_all()
        lengths = set()
        for md5 in loader.file_paths:
            signal = loader.load_signal(md5)
            lengths.add(len(signal))

        assert len(lengths) == 1

    def test_load_signal_normalized(self, test_data: EpiAtlasDataset, tmp_path: Path):
        """Verify that normalized signals have zero mean and unit std."""
        hdf5_loader = Hdf5Loader(
            test_data.datasource.chromsize_file, True, mmap_dir=tmp_path / "mmap"
        )
        hdf5_loader.register_hdf5s(test_data.datasource.hdf5_file, strict=True)
        hdf5_loader.preload_all()

        md5 = next(iter(hdf5_loader.file_paths))
        signal = hdf5_loader.load_signal(md5)

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

        md5 = next(iter(loader_norm.file_paths))
        sig_norm = loader_norm.load_signal(md5)
        sig_raw = loader_raw.load_signal(md5)

        assert not np.array_equal(sig_norm, sig_raw)

    def test_load_signal_before_preload(self, loader: Hdf5Loader):
        """Verify that load_signal raises if preload_all was not called."""
        md5 = next(iter(loader.file_paths))
        with pytest.raises(FileNotFoundError, match="Run preload_all"):
            loader.load_signal(md5)

    def test_load_signal_unregistered_md5(self, loader: Hdf5Loader):
        """Verify that load_signal raises for unregistered md5."""
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
