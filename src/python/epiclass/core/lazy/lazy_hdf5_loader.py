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


def read_npy_header(npy_path: Path | str) -> tuple[tuple[int, ...], np.dtype, int]:
    """Return (shape, dtype, header_end_offset) of a .npy file without mapping data."""
    with open(npy_path, "rb") as file:
        version = np.lib.format.read_magic(file)
        (
            shape,
            _fortran,
            dtype,
        ) = np.lib.format._read_array_header(  # pylint: disable=protected-access
            file, version
        )
        return shape, dtype, file.tell()


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

        # Setup mmap cache directory. EPICLASS_MMAP_DIR overrides the default so
        # concurrent processes can be pointed at private caches (pytest-xdist
        # workers, or $SLURM_TMPDIR on HPC) instead of racing on ./mmap_cache.
        if mmap_dir is None:
            mmap_dir = Path(os.environ.get("EPICLASS_MMAP_DIR", "./mmap_cache"))
        self._mmap_dir = Path(mmap_dir)
        self._mmap_dir.mkdir(parents=True, exist_ok=True)

        self._mmap_array = None
        self._signal_id_to_index: Dict[str, int] = {}

    def _get_mmap_path(self) -> Path:
        """Get path for the combined memory-mapped file."""
        suffix = "_norm" if self._normalization else "_raw"
        return self._mmap_dir / f"signals{suffix}.npy"

    def _get_manifest_path(self) -> Path:
        """Path of the ordered signal-id manifest written beside the mmap.

        The mmap is row-ordered, so reuse is only safe when the registered files
        match -- in content *and* order -- those it was built from. This sidecar
        records that order (one signal id per line) so reuse can be validated
        beyond a bare row count.
        """
        suffix = "_norm" if self._normalization else "_raw"
        return self._mmap_dir / f"signals{suffix}.ids"

    @staticmethod
    def _read_manifest(manifest_path: Path) -> List[str] | None:
        """Return the ordered signal ids recorded in ``manifest_path``.

        Returns ``None`` when the manifest is missing or unreadable, which the
        integrity check treats as "cannot verify -- rebuild".
        """
        try:
            text = manifest_path.read_text(encoding="utf-8")
        except OSError:
            return None
        return text.splitlines()

    def _mmap_integrity_error(
        self, mmap_path: Path, expected_rows: int | None
    ) -> str | None:
        """Return why the on-disk mmap cache is unusable, or None if it looks sound.

        Cheap header-only checks (no data read): the header is readable, the file is
        not truncated (at least as large as the header declares), the row count
        matches the registered sample count, and the ordered signal-id manifest
        matches the registered files. The mmap is row-ordered, so a matching row
        count is *not* sufficient — a reuse against the same files in a different
        order would silently feed every sample another sample's signal. The
        manifest check catches that: ``predict_CV`` reuses one mmap_dir across
        classifiers, and the same file set can reach it in different orders (e.g.
        ``tar -tf`` archive order vs ``find | sort``).
        """
        try:
            shape, dtype, header_end = read_npy_header(mmap_path)
        except Exception as err:  # pylint: disable=broad-except
            return f"unreadable .npy header: {err}"

        expected_bytes = header_end + int(np.prod(shape)) * dtype.itemsize
        if mmap_path.stat().st_size < expected_bytes:
            return (
                f"truncated: file is {mmap_path.stat().st_size} bytes but header "
                f"declares {expected_bytes} (likely a crash-interrupted preload)"
            )
        if expected_rows is not None and shape and shape[0] != expected_rows:
            return (
                f"row-count mismatch: mmap has {shape[0]} rows but "
                f"{expected_rows} samples are registered (stale mmap)"
            )

        # Row count alone cannot detect a reordering or a same-size file swap;
        # compare the ordered signal-id manifest against the registered files.
        if self._files:
            expected_ids = list(self._files.keys())
            manifest_ids = self._read_manifest(self._get_manifest_path())
            if manifest_ids is None:
                return (
                    "no id manifest; cannot verify cache order — rebuilding "
                    "(cache predates manifest support, or was crash-interrupted)"
                )
            if manifest_ids != expected_ids:
                diff_at = next(
                    (
                        i
                        for i, (got, want) in enumerate(zip(manifest_ids, expected_ids))
                        if got != want
                    ),
                    min(len(manifest_ids), len(expected_ids)),
                )
                return (
                    "id manifest mismatch: cache was built from a different file "
                    f"set or order (first difference at row {diff_at}); stale mmap"
                )
        return None

    def mmap_exists(self) -> bool:
        """Return True if the combined mmap .npy cache file is already present."""
        return self._get_mmap_path().exists()

    @property
    def loaded_files(self) -> Dict[str, Path]:
        """Return a {signal_id:path} dict with registered files."""
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
        """Return {signal_id:file} dict from file of paths list."""
        with open(data_file, "r", encoding="utf-8") as file_of_paths:
            files = {}
            for path in file_of_paths:
                path = Path(path.rstrip())
                files[LazyHdf5Loader.extract_signal_id(path)] = path
        if adapt:
            files = LazyHdf5Loader.adapt_to_environment(files)
        return files

    def register_hdf5s(
        self,
        data_file: Path,
        signal_ids: List[str] | None = None,
        verbose: bool = True,
        strict: bool = False,
        hdf5_dir: Path | None = None,
    ) -> LazyHdf5Loader:
        """Register hdf5 file paths without loading data.

        This replaces the old load_hdf5s method. Files are validated but not loaded.

        Args:
            data_file: Path to file containing HDF5 paths
            signal_ids: Optional list of specific signal IDs to register
            verbose: Print validation messages
            strict: Raise error if file cannot be opened
            hdf5_dir: Override directory for HDF5 files
        """
        files = self.read_list(data_file)
        files = LazyHdf5Loader.adapt_to_environment(files)

        if hdf5_dir is not None:
            files = {sid: hdf5_dir / path.name for sid, path in files.items()}

        # Remove undesired files
        if signal_ids is not None:
            chosen_ids = set(signal_ids)
            files = {sid: path for sid, path in files.items() if sid in chosen_ids}

            absent_ids = chosen_ids - set(files.keys())
            if absent_ids and verbose:
                print("Following given signal IDs are absent from hdf5 list:")
                for sid in absent_ids:
                    print(sid)

        # Validate files can be opened (optional)
        if strict:
            validated_files = {}
            for sid, file in files.items():
                try:
                    with h5py.File(file, "r") as f:
                        # Just check we can open it
                        _ = list(f.keys())
                    validated_files[sid] = file
                except (OSError, Exception) as err:
                    print(f"Error with {sid}: {file}. {err}", file=sys.stderr)
                    raise err from None
            files = validated_files

        self._files = files

        if verbose:
            print(f"Registered {len(self._files)} HDF5 files for lazy loading")

        return self

    def _read_hdf5(self, file: h5py.File, signal_id: str) -> np.ndarray:
        """Read and return concatenated genome signal for open hdf5 file."""
        try:
            header = list(file.keys())[0]
        except IndexError as e:
            raise OSError(f"Header not found in {signal_id}") from e

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
    def extract_signal_id(file_name: Path, verbose: bool = False) -> str:
        """Extract the signal ID from file path with specific naming convention."""
        signal_id = file_name.name.split("_")[0]
        if len(signal_id) != 32:
            if verbose:
                print(
                    f"Warning: '{file_name}' does not begin with a 32-char signal ID.",
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
            for sid, path in list(files.items()):
                files[sid] = local_tmp / Path(path).name

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

    def as_mmap(self, mmap_mode: str = "r") -> np.ndarray:
        """Return the full mmap-backed (n_samples, signal_length) array.

        The returned ndarray is disk-backed; the OS pages chunks in on demand,
        so this does NOT load the full dataset into RAM. Row order matches
        ``self.file_paths.keys()``. ``preload_all()`` must have been called
        first.

        ``mmap_mode`` (numpy semantics):
          - ``"r"`` (default): read-only. Safest; works for IncrementalPCA,
            np.savez_compressed, and other read-only consumers.
          - ``"c"`` (copy-on-write): readable + writable, but writes are kept
            in RAM and never reach the file. Required by consumers whose
            inner loops are numba-jitted with a writable-array signature —
            pynndescent / UMAP being the canonical case. Pages still page
            in lazily; only pages that get written use extra RAM.

        Use ``"c"`` for UMAP; ``"r"`` for everything else.
        """
        mmap_path = self._get_mmap_path()
        if not mmap_path.exists():
            raise FileNotFoundError(
                f"Mmap file not found: {mmap_path}. Run preload_all() first."
            )
        if (
            self._mmap_array is None
            or getattr(self._mmap_array, "mode", None) != mmap_mode
        ):
            err = self._mmap_integrity_error(mmap_path, len(self._files) or None)
            if err is not None:
                raise RuntimeError(
                    f"Corrupt mmap cache {mmap_path} ({err}). "
                    f"Delete it and re-run preload_all()."
                )
            self._mmap_array = np.load(mmap_path, mmap_mode=mmap_mode)
        return self._mmap_array

    def load_signal(self, signal_id: str) -> np.ndarray:
        """Load a single signal by signal ID from the memory-mapped file."""
        if signal_id not in self._files:
            raise KeyError(f"Signal ID {signal_id} not registered")

        mmap_path = self._get_mmap_path()

        if not mmap_path.exists():
            raise FileNotFoundError(
                f"Mmap file not found: {mmap_path}. Run preload_all() first."
            )

        # Copy-on-write: PyTorch's from_numpy rejects read-only arrays.
        # COW pages are only dirtied if a caller writes to the slice, which the
        # training loop never does, so no real copying occurs in practice.
        if self._mmap_array is None:
            err = self._mmap_integrity_error(mmap_path, len(self._files) or None)
            if err is not None:
                raise RuntimeError(
                    f"Corrupt mmap cache {mmap_path} ({err}). "
                    f"Delete it and re-run preload_all()."
                )
            self._mmap_array = np.load(mmap_path, mmap_mode="c")

        if not self._signal_id_to_index:
            # Rebuild index mapping from file order
            self._signal_id_to_index = {sid: i for i, sid in enumerate(self._files)}

        return self._mmap_array[self._signal_id_to_index[signal_id]]

    def preload_all(self, verbose: bool = True) -> None:
        """Convert all registered HDF5 files to a single mmap .npy file.

        Creates a single memory-mapped array of shape (n_samples, signal_length).
        Useful to run once before training to avoid per-sample conversion overhead.
        """
        mmap_path = self._get_mmap_path()

        n_samples = len(self._files)
        if n_samples == 0:
            raise ValueError("No files registered")

        if mmap_path.exists():
            err = self._mmap_integrity_error(mmap_path, n_samples)
            if err is None:
                if verbose:
                    print(f"Mmap file already exists: {mmap_path}")
                return
            # A truncated or stale cache would mis-feed every future reader. Drop it
            # and rebuild from scratch.
            print(
                f"Existing mmap is unusable and will be rebuilt ({err}): {mmap_path}",
                file=sys.stderr,
            )
            self._mmap_array = None
            mmap_path.unlink()

        # Determine signal length from first sample
        first_sid, first_path = next(iter(self._files.items()))
        with h5py.File(first_path, "r") as f:
            first_signal = self._read_hdf5(f, first_sid)
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

        # Build the ordered list of signal IDs (this defines the index mapping)
        self._signal_id_to_index = {}

        # Fill a private temp file, then atomically rename into place. os.replace is
        # atomic on POSIX, so the final path only ever exists fully written: a crash
        # (SIGSEGV/SIGKILL bypasses the except below) leaves only a stale .tmp, never
        # a partial cache at the final path that the next run would reuse.
        tmp_path = mmap_path.with_name(f"{mmap_path.name}.{os.getpid()}.tmp")
        mmap_array = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_samples, signal_length),
        )

        try:
            for i, (sid, hdf5_path) in enumerate(self._files.items(), 0):
                self._signal_id_to_index[sid] = i

                with h5py.File(hdf5_path, "r") as f:
                    signal = self._read_hdf5(f, sid)
                signal = self._normalize(signal)

                if len(signal) != signal_length:
                    raise ValueError(
                        f"Signal length mismatch for {sid}: "
                        f"expected {signal_length}, got {len(signal)}. "
                        f"File: {hdf5_path}"
                    )

                mmap_array[i] = signal

                if verbose and (i + 1) % 1000 == 0:
                    print(f"  [{i + 1}/{n_samples}]")

            mmap_array.flush()
            mmap_array = None  # close the memmap before renaming it into place
            os.replace(tmp_path, mmap_path)
        except Exception:
            # Clean up partial temp file (final path is never touched on failure)
            mmap_array = None
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        # Record the row order so reuse can be validated beyond a bare row count.
        # Written after the mmap is in place: a crash between the two renames
        # leaves an mmap without a manifest, which the integrity check rejects
        # (rebuild) rather than trusting -- fail safe.
        self._write_manifest(list(self._files.keys()))

        if verbose:
            print("Conversion complete!")

    def _write_manifest(self, signal_ids: List[str]) -> None:
        """Atomically write the ordered signal-id manifest beside the mmap."""
        manifest_path = self._get_manifest_path()
        tmp_path = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text("\n".join(signal_ids) + "\n", encoding="utf-8")
        os.replace(tmp_path, manifest_path)
