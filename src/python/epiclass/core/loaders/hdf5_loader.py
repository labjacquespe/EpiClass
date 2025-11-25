"""Module for hdf5 loading handling."""

# pylint: disable=unexpected-keyword-arg
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


class Hdf5Loader:
    """Handles loading/creating signals from hdf5 files"""

    def __init__(self, chrom_file: Path | str, normalization: bool):
        self._normalization = normalization
        self._chroms = Hdf5Loader.load_chroms(chrom_file)
        self._files = {}
        self._signals = {}

    @property
    def loaded_files(self) -> Dict[str, Path]:
        """Return a {md5:path} dict with last loaded files."""
        return self._files

    @property
    def signals(self) -> Dict[str, np.ndarray]:
        """Return a {md5:signal} dict with the last loaded signals,
        where the signal has concanenated chromosomes, and is normalized if set so.
        """
        return self._signals

    @staticmethod
    def load_chroms(chrom_file: Path | str) -> List[str]:
        """Return sorted chromosome names list."""
        with open(chrom_file, "r", encoding="utf-8") as file:
            chroms = []
            for line in file:
                line = line.rstrip()
                if line:
                    chroms.append(line.split()[0])
        chroms.sort()
        return chroms

    @staticmethod
    def read_list(data_file: Path, adapt: bool = False) -> Dict[str, Path]:
        """Return {md5:file} dict from file of paths list."""
        with open(data_file, "r", encoding="utf-8") as file_of_paths:
            files = {}
            for path in file_of_paths:
                path = Path(path.rstrip())
                files[Hdf5Loader.extract_md5(path)] = path
        if adapt:
            files = Hdf5Loader.adapt_to_environment(files)
        return files

    def load_hdf5s(
        self,
        data_file: Path,
        md5s: List[str] | None = None,
        verbose=True,
        strict=False,
        hdf5_dir: Path | None = None,
    ) -> Hdf5Loader:
        """Load hdf5s from path list file, into self.signals
        If a list of md5s is given, load only the corresponding files.
        Normalize if internal flag set so.

        Check adapt_to_environment for details on dynamic path changes.

        hdf5_dir is a directory in which to look for hdf5s, which will override data_file complete paths.

        If strict, will raise OSError if an hdf5 cannot be opened.

        Loads them as float32.
        """
        files = self.read_list(data_file)

        files = Hdf5Loader.adapt_to_environment(files)

        if hdf5_dir is not None:
            files = {md5: hdf5_dir / path.name for md5, path in files.items()}

        self._files = files

        # Remove undesired files
        if md5s is not None:
            chosen_md5s = set(md5s)
            # fmt: off
            files = {
                md5: path for md5, path in files.items()
                if md5 in chosen_md5s
                }  # fmt: on

            absent_md5s = chosen_md5s - set(files.keys())
            if absent_md5s and verbose:
                print("Following given md5s are absent of hdf5 list")
                for md5 in absent_md5s:
                    print(md5)

        # Load hdf5s and concatenate chroms into signals
        signals = {}
        for md5, file in files.items():
            # Trying to open hdf5 file.
            try:
                with h5py.File(file, "r") as f:
                    signals[md5] = self._normalize(self._read_hdf5(f, md5))
            except (OSError, FloatingPointError) as err:
                print(f"Error occured with {md5}: {file}. {err}", file=sys.stderr)
                if strict:
                    print(
                        "Strict hdf5 loading policy true, raising original error.",
                        file=sys.stderr,
                    )
                    raise err from None
                continue

        self._signals = signals

        return self

    def _read_hdf5(self, file: h5py.File, md5: str) -> np.ndarray:
        """Read and return concatenated genome signal for open hdf5 file."""
        try:
            header = list(file.keys())[0]
        except IndexError as e:
            raise OSError(f"Header not found in {md5}") from e

        hdf5_data = file[header]

        chrom_signals = [hdf5_data[chrom][...] for chrom in self._chroms]  # type: ignore
        return np.concatenate(chrom_signals, dtype=np.float32)  # type: ignore

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        """Normalize array if internal flag set so.

        If normalization is not set, return array as is.

        Raises:
            FloatingPointError: if operation fails
        """
        if self._normalization:
            with np.errstate(all="raise"):
                return (array - array.mean()) / array.std()
        return array

    @staticmethod
    def extract_md5(file_name: Path, verbose: bool = False) -> str:
        """Extract the md5 string from file path with specific naming convention.

        Expecting the md5 to be the first part of the file name, separated by an underscore.

        If there is no md5sum, extract the filename without extension.
        """
        md5 = file_name.name.split("_")[0]
        if len(md5) != 32:
            if verbose:
                print(
                    f"Warning: '{file_name}' does not begin with a md5sum.",
                    file=sys.stderr,
                )
            return file_name.stem
        return file_name.name.split("_")[0]

    @staticmethod
    def adapt_to_environment(files: Dict[str, Path]) -> Dict[str, Path]:
        """Change files paths if they exist on cluster scratch.
        Checks for $SLURM_TMPDIR/$HDF5_PARENT, default is $SLURM_TMPDIR/hdf5s.

        Files : {md5:path} dict.
        """
        local_tmp = Path(os.getenv("SLURM_TMPDIR", "./bleh"))
        local_tmp = local_tmp / os.getenv("HDF5_PARENT", "hdf5s")

        if local_tmp.exists():
            print(f"Using files in {local_tmp}")
            for md5, path in list(files.items()):
                files[md5] = local_tmp / Path(path).name

        return files
