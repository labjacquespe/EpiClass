"""Module for lazy-loading hdf5 handling with memory mapping support."""
# pylint: disable=too-many-positional-arguments, too-many-branches
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np


class LazyHdf5Loader:
    """Handles lazy loading of signals from hdf5 files.

    Instead of loading all files into memory, this class stores file paths
    and loads signals on-demand.
    """

    def __init__(
        self,
        chrom_file: Path | str,
        normalization: bool,
        mmap_dir: Path | str | None = None,
    ):
        """Initialize lazy loader.

        Args:
            chrom_file: Path to chromosome file
            normalization: Whether to normalize signals
            mmap_dir: Directory for memory-mapped files
        """
        self._normalization = normalization
        self._chroms = LazyHdf5Loader.load_chroms(chrom_file)
        self._files: Dict[str, Path] = {}

        # Setup mmap cache directory
        if mmap_dir is None:
            mmap_dir = Path("./mmap_cache")
        self._mmap_dir = Path(mmap_dir)
        self._mmap_dir.mkdir(parents=True, exist_ok=True)

        self._mmap_array = None
        self._md5_to_index: Dict[str, int] = {}

    def _get_mmap_path(self) -> Path:
        """Get path for the combined memory-mapped file."""
        suffix = "_norm" if self._normalization else "_raw"
        return self._mmap_dir / f"signals{suffix}.npy"

    @property
    def loaded_files(self) -> Dict[str, Path]:
        """Return a {md5:path} dict with registered files."""
        return self._files

    @property
    def file_paths(self) -> Dict[str, Path]:
        """Return file paths dictionary."""
        return self._files

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
                files[LazyHdf5Loader.extract_md5(path)] = path
        if adapt:
            files = LazyHdf5Loader.adapt_to_environment(files)
        return files

    def register_hdf5s(
        self,
        data_file: Path,
        md5s: List[str] | None = None,
        verbose: bool = True,
        strict: bool = False,
        hdf5_dir: Path | None = None,
    ) -> LazyHdf5Loader:
        """Register hdf5 file paths without loading data.

        This replaces the old load_hdf5s method. Files are validated but not loaded.

        Args:
            data_file: Path to file containing HDF5 paths
            md5s: Optional list of specific MD5s to register
            verbose: Print validation messages
            strict: Raise error if file cannot be opened
            hdf5_dir: Override directory for HDF5 files
        """
        files = self.read_list(data_file)
        files = LazyHdf5Loader.adapt_to_environment(files)

        if hdf5_dir is not None:
            files = {md5: hdf5_dir / path.name for md5, path in files.items()}

        # Remove undesired files
        if md5s is not None:
            chosen_md5s = set(md5s)
            files = {md5: path for md5, path in files.items() if md5 in chosen_md5s}

            absent_md5s = chosen_md5s - set(files.keys())
            if absent_md5s and verbose:
                print("Following given md5s are absent of hdf5 list:")
                for md5 in absent_md5s:
                    print(md5)

        # Validate files can be opened (optional)
        if strict:
            validated_files = {}
            for md5, file in files.items():
                try:
                    with h5py.File(file, "r") as f:
                        # Just check we can open it
                        _ = list(f.keys())
                    validated_files[md5] = file
                except (OSError, Exception) as err:
                    print(f"Error with {md5}: {file}. {err}", file=sys.stderr)
                    raise err from None
            files = validated_files

        self._files = files

        if verbose:
            print(f"Registered {len(self._files)} HDF5 files for lazy loading")

        return self

    def _read_hdf5(self, file: h5py.File, md5: str) -> np.ndarray:
        """Read and return concatenated genome signal for open hdf5 file."""
        try:
            header = list(file.keys())[0]
        except IndexError as e:
            raise OSError(f"Header not found in {md5}") from e

        hdf5_data = file[header]

        # Load chromosomes and concatenate
        chrom_signals: List[Sequence[float]] = [
            hdf5_data[chrom][...] for chrom in self._chroms  # type: ignore
        ]

        # pylint: disable-next=unexpected-keyword-arg # false positive
        full_signal = np.concatenate(chrom_signals, dtype=np.float32)

        return full_signal

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        """Normalize array if internal flag set so."""
        if self._normalization:
            with np.errstate(all="raise"):
                return (array - array.mean()) / array.std()
        return array

    @staticmethod
    def extract_md5(file_name: Path, verbose: bool = False) -> str:
        """Extract the md5 string from file path with specific naming convention."""
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
        """Change files paths if they exist on cluster scratch."""
        local_tmp = Path(os.getenv("SLURM_TMPDIR", "./bleh"))
        local_tmp = local_tmp / os.getenv("HDF5_PARENT", "hdf5s")

        if local_tmp.exists():
            print(f"Using files in {local_tmp}")
            for md5, path in list(files.items()):
                files[md5] = local_tmp / Path(path).name

        return files

    @property
    def signal_length(self) -> int | None:
        """Return signal length after preload_all(), or None if not yet preloaded."""
        mmap_path = self._get_mmap_path()
        if not mmap_path.exists():
            return None
        if self._mmap_array is None:
            self._mmap_array = np.load(mmap_path, mmap_mode="r")
        return int(self._mmap_array.shape[1])

    def load_signal(self, md5: str) -> np.ndarray:
        """Load a single signal by MD5 from the memory-mapped file."""
        if md5 not in self._files:
            raise KeyError(f"MD5 {md5} not registered")

        mmap_path = self._get_mmap_path()

        if not mmap_path.exists():
            raise FileNotFoundError(
                f"Mmap file not found: {mmap_path}. Run preload_all() first."
            )

        # Lazy-load the mmap array reference
        if self._mmap_array is None:
            self._mmap_array = np.load(mmap_path, mmap_mode="r")

        if not self._md5_to_index:
            # Rebuild index mapping from file order
            self._md5_to_index = {md5: i for i, md5 in enumerate(self._files)}

        return self._mmap_array[self._md5_to_index[md5]]

    def preload_all(self, verbose: bool = True) -> None:
        """Convert all registered HDF5 files to a single mmap .npy file.

        Creates a single memory-mapped array of shape (n_samples, signal_length).
        Useful to run once before training to avoid per-sample conversion overhead.
        """
        mmap_path = self._get_mmap_path()

        if mmap_path.exists():
            if verbose:
                print(f"Mmap file already exists: {mmap_path}")
            return

        n_samples = len(self._files)
        if n_samples == 0:
            raise ValueError("No files registered")

        # Determine signal length from first sample
        first_md5, first_path = next(iter(self._files.items()))
        with h5py.File(first_path, "r") as f:
            first_signal = self._read_hdf5(f, first_md5)
        first_signal = self._normalize(first_signal)
        signal_length = len(first_signal)

        # Check available disk space
        total_bytes = n_samples * signal_length * np.dtype(np.float32).itemsize
        disk_usage = shutil.disk_usage(self._mmap_dir)
        headroom = 0.9  # Don't fill past 90%
        if total_bytes > disk_usage.free * headroom:
            raise OSError(
                f"Insufficient disk space in {self._mmap_dir}: "
                f"need {total_bytes / 1e9:.1f} GB, "
                f"have {disk_usage.free / 1e9:.1f} GB available"
            )

        if verbose:
            print(
                f"Converting {n_samples} files to single mmap "
                f"({total_bytes / 1e9:.1f} GB)..."
            )

        # Build the ordered list of md5s (this defines the index mapping)
        self._md5_to_index = {}

        # Create the memory-mapped file and fill it
        mmap_array = np.lib.format.open_memmap(
            mmap_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, signal_length),
        )

        try:
            for i, (md5, hdf5_path) in enumerate(self._files.items(), 0):
                self._md5_to_index[md5] = i

                with h5py.File(hdf5_path, "r") as f:
                    signal = self._read_hdf5(f, md5)
                signal = self._normalize(signal)

                if len(signal) != signal_length:
                    raise ValueError(
                        f"Signal length mismatch for {md5}: "
                        f"expected {signal_length}, got {len(signal)}. "
                        f"File: {hdf5_path}"
                    )

                mmap_array[i] = signal

                if verbose and (i + 1) % 1000 == 0:
                    print(f"  [{i + 1}/{n_samples}]")

            mmap_array.flush()
        except Exception:
            # Clean up partial file
            del mmap_array
            if mmap_path.exists():
                mmap_path.unlink()
            raise

        if verbose:
            print("Conversion complete!")
