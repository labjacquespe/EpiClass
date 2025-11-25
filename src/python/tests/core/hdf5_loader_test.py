"""Test module for hdf5_loader file."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py  # pylint: disable=unused-import # import to avoid weirdness
import pytest

from epiclass.core.epiatlas_treatment import EpiAtlasDataset
from epiclass.core.loaders.hdf5_loader import Hdf5Loader
from tests.epilap_test_data import EpiAtlasTreatmentTestData


class Test_Hdf5Loader:
    """Test class Test_Hdf5Loader"""

    @pytest.fixture(scope="class", autouse=True)
    def test_folder(self, mk_logdir) -> Path:
        """Return temp hdf5 storage folder."""
        return mk_logdir("temp_hdf5s")

    @pytest.fixture(scope="function")
    def test_data(self) -> EpiAtlasDataset:
        """Mock test EpiAtlasFoldFactory."""
        return EpiAtlasTreatmentTestData.default_test_data().epiatlas_dataset

    def test_load_hdf5s(self, test_data: EpiAtlasDataset):
        """Verify that files are loading correctly."""
        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
        hdf5_loader.load_hdf5s(test_data.datasource.hdf5_file, strict=True)

    def test_load_hdf5_corrupted(self, test_data: EpiAtlasDataset):
        """Verify that file corruption errors are caught/raised."""
        hdf5_list = Hdf5Loader.read_list(test_data.datasource.hdf5_file)
        chosen_file = list(hdf5_list.values())[0]

        with open(chosen_file, "r+b") as f:
            f.seek(0)
            f.write(os.urandom(1024))

        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)

        with pytest.raises(OSError, match="file signature not found"):
            hdf5_loader.load_hdf5s(test_data.datasource.hdf5_file, strict=True)

    def test_adapt_to_environment(self, test_folder: Path, test_data: EpiAtlasDataset):
        """Test that the existence of $SLURM_TMPDIR/hdf5s affects hdf5 loading."""
        # setup
        os.environ["SLURM_TMPDIR"] = str(test_folder)
        shutil.rmtree(test_folder)
        os.makedirs(test_folder / "hdf5s", exist_ok=True)

        # test
        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
        files = hdf5_loader.read_list(test_data.datasource.hdf5_file)
        files = hdf5_loader.adapt_to_environment(files)

        a_path = list(files.values())[0]
        assert str(test_folder) in str(a_path)

        # tearup
        del os.environ["SLURM_TMPDIR"]

    def test_adapt_to_environment_2(self, test_folder: Path, test_data: EpiAtlasDataset):
        """Test that the existence of $SLURM_TMPDIR/$HDF5_PARENT affects hdf5 loading."""
        # setup
        new_parent = "test"
        hdf5_dir = test_folder / new_parent

        os.environ["SLURM_TMPDIR"] = str(test_folder)
        os.environ["HDF5_PARENT"] = str(new_parent)
        shutil.rmtree(test_folder)

        os.makedirs(hdf5_dir, exist_ok=True)

        # test
        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
        files = hdf5_loader.read_list(test_data.datasource.hdf5_file)
        files = hdf5_loader.adapt_to_environment(files)

        a_path = list(files.values())[0]
        assert str(hdf5_dir) in str(a_path)

        # tearup
        del os.environ["SLURM_TMPDIR"]
        del os.environ["HDF5_PARENT"]
