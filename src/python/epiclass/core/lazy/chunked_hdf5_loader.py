"""Lazy loader for chunked HDF5 files with pre-concatenated signals.

This module handles HDF5 files where multiple samples are stored together,
with signals already concatenated (and optionally normalized).

Expected HDF5 structure:
    chunk_file.h5
    ├── signals: shape (n_samples, signal_length), dtype float32
    ├── sample_ids: shape (n_samples,), dtype str
    └── attrs (optional metadata)
"""
# pylint: disable=too-many-positional-arguments, too-many-branches
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

SampleId = str


class ChunkedHdf5Loader:
    """Handles lazy loading from chunked HDF5 files.

    Unlike LazyHdf5Loader which loads one file per sample (with
    per-chromosome datasets requiring concatenation and normalization),
    this loader handles HDF5 files containing multiple pre-concatenated
    samples. Signals are read directly from HDF5 without conversion,
    relying on the OS page cache for performance after the first pass.
    """

    def __init__(self):
        """Initialize chunked HDF5 loader."""
        self._id_to_location: Dict[SampleId, Tuple[Path, int]] = {}
        self._chunk_files: List[Path] = []
        self._signal_length: Optional[int] = None
        self._file_handles: Dict[Path, h5py.File] = {}

    @property
    def chunk_files(self) -> List[Path]:
        """Return list of registered chunk files."""
        return list(self._chunk_files)

    @property
    def id_to_location(self) -> Dict[SampleId, Tuple[Path, int]]:
        """Return mapping from sample ID to (file_path, index_in_file)."""
        return self._id_to_location

    @property
    def num_registered(self) -> int:
        """Return number of registered samples."""
        return len(self._id_to_location)

    @property
    def signal_length(self) -> Optional[int]:
        """Return signal length, or None if no files registered."""
        return self._signal_length

    @property
    def sample_ids(self) -> List[SampleId]:
        """Return list of registered sample IDs in registration order."""
        return list(self._id_to_location.keys())

    def _get_dataset(self, f: h5py.File, name: str) -> h5py.Dataset:
        """Get a dataset from an HDF5 file, raising if it's not a dataset."""
        item = f[name]
        if not isinstance(item, h5py.Dataset):
            raise TypeError(f"Expected dataset, got {type(item)} for '{name}'")
        return item

    def register_chunked_hdf5s(
        self,
        chunk_files: List[Path] | Path,
        sample_ids: List[SampleId] | None = None,
        verbose: bool = True,
        strict: bool = False,
    ) -> ChunkedHdf5Loader:
        """Register chunked HDF5 files without loading signals.

        Args:
            chunk_files: Single path, list of paths, or directory
                containing .h5/.hdf5 files.
            sample_ids: Optional list of specific sample IDs to register.
            verbose: Print registration messages.
            strict: Raise error if a file cannot be opened.

        Returns:
            self for chaining.
        """
        resolved_files = self._resolve_file_list(chunk_files)

        if verbose:
            print(f"Registering {len(resolved_files)} chunk file(s)...")

        id_filter = set(sample_ids) if sample_ids is not None else None
        registered_count = 0

        for chunk_file in resolved_files:
            try:
                registered = self._register_file(chunk_file, id_filter)
                registered_count += registered
            except (OSError, KeyError, ValueError) as err:
                print(
                    f"Error registering {chunk_file}: {err}",
                    file=sys.stderr,
                )
                if strict:
                    raise
                continue

        self._chunk_files = resolved_files

        if verbose:
            print(
                f"Registered {registered_count} samples "
                f"from {len(resolved_files)} file(s)"
            )
            if id_filter:
                missing = id_filter - set(self._id_to_location.keys())
                if missing:
                    print(
                        f"Warning: {len(missing)} requested sample ID(s) "
                        f"not found in chunk files"
                    )

        return self

    def _resolve_file_list(self, chunk_files: List[Path] | Path) -> List[Path]:
        """Resolve input to a list of HDF5 file paths."""
        if isinstance(chunk_files, (str, Path)):
            chunk_files = Path(chunk_files)
            if chunk_files.is_dir():
                return sorted(
                    list(chunk_files.glob("*.h5")) + list(chunk_files.glob("*.hdf5"))
                )
            return [chunk_files]
        return [Path(f) for f in chunk_files]

    def _register_file(
        self,
        chunk_file: Path,
        id_filter: set[SampleId] | None,
    ) -> int:
        """Register samples from a single chunked HDF5 file.

        Validates that 'signals' and 'sample_ids' datasets exist and
        that signal length is consistent across all chunk files.
        """
        with h5py.File(chunk_file, "r") as f:
            if "signals" not in f or "sample_ids" not in f:
                raise KeyError(
                    f"Missing 'signals' or 'sample_ids' dataset " f"in {chunk_file}"
                )

            signals_dset = self._get_dataset(f, "signals")
            ids_dset = self._get_dataset(f, "sample_ids")

            if len(signals_dset) != len(ids_dset):
                raise ValueError(
                    f"Mismatched lengths in {chunk_file}: "
                    f"signals has {len(signals_dset)}, "
                    f"sample_ids has {len(ids_dset)}"
                )

            # Validate signal length consistency
            file_signal_length = signals_dset.shape[1]  # pylint: disable=no-member
            if self._signal_length is None:
                self._signal_length = file_signal_length
            elif file_signal_length != self._signal_length:
                raise ValueError(
                    f"Signal length mismatch in {chunk_file}: "
                    f"expected {self._signal_length}, "
                    f"got {file_signal_length}"
                )

            # Read all sample IDs at once (small string dataset)
            all_ids = ids_dset[:]

            registered = 0
            for idx, sample_id in enumerate(all_ids):
                sample_id = self._decode_id(sample_id)

                if id_filter is not None and sample_id not in id_filter:
                    continue

                if sample_id in self._id_to_location:
                    print(
                        f"Warning: duplicate sample ID {sample_id}, "
                        f"overwriting with {chunk_file}",
                        file=sys.stderr,
                    )

                self._id_to_location[sample_id] = (chunk_file, idx)
                registered += 1

        return registered

    @staticmethod
    def _decode_id(sample_id) -> SampleId:
        """Decode sample ID from HDF5 dataset (may be bytes or str)."""
        if isinstance(sample_id, (bytes, np.bytes_)):
            return sample_id.decode("utf-8")
        return str(sample_id)

    def _get_file_handle(self, chunk_file: Path) -> h5py.File:
        """Get or open a file handle for a chunk file.

        Keeps file handles open for reuse across calls. This avoids
        repeated open/close overhead, especially with persistent
        DataLoader workers.
        """
        if chunk_file not in self._file_handles:
            self._file_handles[chunk_file] = h5py.File(chunk_file, "r")
        return self._file_handles[chunk_file]

    def load_signal(self, sample_id: SampleId) -> np.ndarray:
        """Load a single signal by sample ID.

        Args:
            sample_id: Sample identifier.

        Returns:
            Signal array of shape (signal_length,) and dtype float32.

        Raises:
            KeyError: If sample_id is not registered.
        """
        if sample_id not in self._id_to_location:
            raise KeyError(f"Sample ID {sample_id} not registered")

        chunk_file, idx = self._id_to_location[sample_id]
        f = self._get_file_handle(chunk_file)
        return np.array(f["signals"][idx], dtype=np.float32)  # type: ignore

    def load_batch(self, sample_ids: List[SampleId]) -> np.ndarray:
        """Load multiple signals as a single array.

        Groups requests by chunk file and uses sorted index access
        for efficient sequential reads within each file.

        Args:
            sample_ids: List of sample identifiers.

        Returns:
            Array of shape (len(sample_ids), signal_length), dtype float32.

        Raises:
            KeyError: If any sample_id is not registered.
        """
        for sample_id in sample_ids:
            if sample_id not in self._id_to_location:
                raise KeyError(f"Sample ID {sample_id} not registered")

        # Group by chunk file: {path: [(request_pos, file_idx), ...]}
        file_groups: Dict[Path, List[Tuple[int, int]]] = {}
        for request_pos, sample_id in enumerate(sample_ids):
            chunk_file, file_idx = self._id_to_location[sample_id]
            if chunk_file not in file_groups:
                file_groups[chunk_file] = []
            file_groups[chunk_file].append((request_pos, file_idx))

        # Allocate output
        result = np.empty(
            (len(sample_ids), self._signal_length), dtype=np.float32  # type: ignore
        )

        # Read from each file using fancy indexing
        for chunk_file, entries in file_groups.items():
            f = self._get_file_handle(chunk_file)
            signals_dset = self._get_dataset(f, "signals")

            # Sort by file index for sequential access
            entries.sort(key=lambda x: x[1])
            file_indices = [file_idx for _, file_idx in entries]
            request_positions = [pos for pos, _ in entries]

            # Read all needed rows from this file
            batch = signals_dset[file_indices]
            result[request_positions] = batch

        return result

    def close(self) -> None:
        """Close all open file handles."""
        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:  # pylint: disable=broad-except
                pass
        self._file_handles.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    def __enter__(self) -> ChunkedHdf5Loader:
        """Support context manager usage."""
        return self

    def __exit__(self, *exc) -> None:
        """Close on context exit."""
        self.close()
