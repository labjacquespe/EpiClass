"""EpiAtlas data treatment testing module."""
from __future__ import annotations

import multiprocessing
import os
import uuid
from pathlib import Path
from typing import List

import h5py

from epiclass.core.data_source import EpiDataSource
from epiclass.core.epiatlas_treatment import EpiAtlasFoldFactory
from epiclass.core.hdf5_loader import Hdf5Loader
from epiclass.core.metadata import Metadata

DEFAULT_TEST_LOGDIR = Path("/tmp/pytest")
DEFAULT_TEST_LOGDIR.mkdir(exist_ok=True, parents=True)

FIXTURES_DIR = (Path(__file__).parent / "fixtures").resolve()
if not FIXTURES_DIR.exists():
    raise FileNotFoundError(
        (
            f"Could not find fixtures directory: {FIXTURES_DIR}\n"
            "Hint: Did you extract fixtures.tar.xz?\n"
        )
    )


class EpiAtlasTreatmentTestData:
    """Create and handle mock/test EpiAtlasFoldFactory"""

    def __init__(self, metadata_path: Path, md5_list_path: Path, logdir: Path):
        self.hdf5_logdir = Path(logdir) / "hdf5"
        print(f"Using hdf5 logdir: {self.hdf5_logdir}")
        self.hdf5_logdir.mkdir(exist_ok=True, parents=True)

        self.dir = FIXTURES_DIR.resolve()
        self.chroms_file = self.dir.parents[3] / "input-format/hg38.noy.chrom.sizes"
        self.chroms = Hdf5Loader.load_chroms(self.chroms_file)

        tmp_hdf5 = self.create_temp_hdf5s(md5_list_path.resolve())

        self.datasource = self.create_mock_datasource(
            metadata_path, tmp_hdf5, self.chroms_file
        )

    def get_ea_handler(self, label_category: str, min_class_size=3, n_fold=2):
        """Return a EpiAtlasFoldFactory object from mock datasource."""
        return EpiAtlasFoldFactory.from_datasource(
            datasource=self.datasource,
            label_category=label_category,
            min_class_size=min_class_size,
            n_fold=n_fold,
        )

    @staticmethod
    def _create_symlink(source: Path, link_name: Path):
        """Create a symbolic link pointing to source named link_name"""
        try:
            os.symlink(source, link_name)
        except FileExistsError:
            pass

    def create_temp_hdf5s(
        self, md5_list_path: Path, name="_100kb_all_none_value.hdf5"
    ) -> List[Path]:
        """Create temporary files and returns paths"""
        tmp_files = []
        with open(md5_list_path, "r", encoding="utf8") as md5_list:
            md5s = [md5.strip() for md5 in md5_list.readlines()]

        if len(md5s) < 100:
            for md5 in md5s:
                md5 = md5.strip()
                tmp_file = self.hdf5_logdir / f"{md5 + name}"
                tmp_files.append(tmp_file)
                self.write_mock_hdf5(tmp_file, md5)
        else:
            md5 = md5s[0]
            real_tmp_file = self.hdf5_logdir / f"{md5 + name}"
            tmp_files.append(real_tmp_file)
            self.write_mock_hdf5(real_tmp_file, md5)

            for md5 in md5s[1:]:
                tmp_file = self.hdf5_logdir / f"{md5 + name}"
                tmp_files.append(tmp_file)

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.starmap(
                    self._create_symlink,
                    [(real_tmp_file, fake_file) for fake_file in tmp_files[1:]],
                )

        return tmp_files

    def write_mock_hdf5(self, path: Path, md5: str):
        """Write a hdf5 file to the given path with the expected general structure."""
        if path.exists():
            raise FileExistsError(f"Mock hdf5 already exists at '{path}'")
        with h5py.File(name=path, mode="w") as f:
            grp = f.create_group(md5)
            for chrom in self.chroms:
                grp.create_dataset(name=chrom, data=[1, 2], dtype=int)
        f.close()

    def create_temp_file_list(self, temp_files: List[Path]) -> Path:
        """Create a file containing a list of given paths.

        Returns path of created file.
        """
        tmp_file = self.hdf5_logdir / "hdf5s.list"
        with open(tmp_file, "w", encoding="utf-8") as f:
            for path in temp_files:
                f.write(f"{path}\n")

        return tmp_file

    def create_mock_datasource(
        self, metadata: Path, tmp_hdf5s: List[Path], chroms_file: Path
    ) -> EpiDataSource:
        """Return a datasource object for testing purposes."""
        return EpiDataSource(
            hdf5=self.create_temp_file_list(tmp_hdf5s),
            chromsize=chroms_file,
            metadata=metadata,
        )

    @classmethod
    def default_test_data(
        cls,
        logdir=DEFAULT_TEST_LOGDIR,  # type: ignore
        test_set="test-epilap-empty-biotype-n40",
        label_category="biomaterial_type",
    ) -> EpiAtlasFoldFactory:
        """Create mock EpiAtlasFoldFactory"""
        md5_list = FIXTURES_DIR / f"{test_set}.md5"
        metadata_path = FIXTURES_DIR / f"{test_set}-metadata.json"

        logdir = Path(logdir) / uuid.uuid4().hex
        print(f"Creating test data in logdir: {logdir}")
        return cls(metadata_path, md5_list, logdir).get_ea_handler(label_category)


# standalone
def create_test_metadata(metadata_source_path: Path, md5_list: Path):
    """Create a metadata json file with a subset of information, for testing purposes."""
    my_metadata = Metadata(metadata_source_path)

    with open(md5_list, "r", encoding="utf8") as f:
        md5_set = set(md5.strip() for md5 in f.readlines())

    for md5 in list(my_metadata.md5s):
        if md5 not in md5_set:
            del my_metadata[md5]

    my_metadata.save(md5_list.parent / (md5_list.stem + "-metadata.json"))


def main():
    """Create test data."""
    test_set = "test-epilap-empty-biotype-n40"
    md5_list = FIXTURES_DIR / f"{test_set}.md5"
    metadata_path = FIXTURES_DIR / f"{test_set}-metadata.json"

    label_category = "biomaterial_type"

    tester = EpiAtlasTreatmentTestData(metadata_path, md5_list, logdir=DEFAULT_TEST_LOGDIR)  # type: ignore
    print(tester.get_ea_handler(label_category).classes)


if __name__ == "__main__":
    main()
