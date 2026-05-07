"""Integration tests for convert_to_chunked.py.

These tests create temporary single-sample HDF5 files matching the
per-chromosome structure, run the conversion pipeline, and
verify the output chunked files are correct and round-trip faithfully.
"""
# pylint: disable=too-many-lines, redefined-outer-name, too-many-positional-arguments, use-implicit-booleaness-not-comparison
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pytest

from epiclass.utils.preprocessing.hdf5_chunks_creation import (
    _extract_sample_id,
    convert,
    load_chroms,
    main,
    read_hdf5_list,
    read_signal,
    verify_chunks,
)

# --- Constants ---

CHROMS = ["chr1", "chr2", "chr3"]
CHROM_SIZES = {"chr1": 50, "chr2": 30, "chr3": 20}
SIGNAL_LENGTH = sum(CHROM_SIZES.values())  # 100
N_SAMPLES = 25
RNG_SEED = 42


# --- Fixtures ---


@pytest.fixture
def chrom_file(tmp_path: Path) -> Path:
    """Create a chromosome file."""
    path = tmp_path / "chroms.txt"
    path.write_text("\n".join(f"{c}\t{CHROM_SIZES[c]}" for c in CHROMS))
    return path


def _make_single_hdf5(
    path: Path,
    sample_id: str,
    chroms: List[str],
    chrom_sizes: Dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a single-sample HDF5 file with per-chromosome datasets.

    Returns the raw concatenated signal (unnormalized) for verification.
    """
    chrom_signals = {}
    for chrom in chroms:
        chrom_signals[chrom] = rng.standard_normal(chrom_sizes[chrom]).astype(np.float32)

    with h5py.File(path, "w") as f:
        grp = f.create_group(sample_id)
        for chrom in chroms:
            grp.create_dataset(chrom, data=chrom_signals[chrom])

    # Return concatenated signal in sorted chromosome order
    sorted_chroms = sorted(chroms)
    return np.concatenate(  # pylint: disable=unexpected-keyword-arg
        [chrom_signals[c] for c in sorted_chroms],
        dtype=np.float32,
    )


@pytest.fixture
def hdf5_dir(tmp_path: Path) -> Path:
    """Create a directory of single-sample HDF5 files."""
    hdf5_path = tmp_path / "hdf5s"
    hdf5_path.mkdir()

    rng = np.random.default_rng(RNG_SEED)
    for i in range(N_SAMPLES):
        # Use md5-style filenames for some, plain names for others
        if i < 15:
            sample_id = f"{i:032x}"
            filename = f"{sample_id}_10kb_all_none.hdf5"
        else:
            sample_id = f"sample_{i:04d}"
            filename = f"{sample_id}.hdf5"

        _make_single_hdf5(
            hdf5_path / filename,
            sample_id,
            CHROMS,
            CHROM_SIZES,
            rng,
        )

    return hdf5_path


@pytest.fixture
def hdf5_list(tmp_path: Path, hdf5_dir: Path) -> Path:
    """Create a file listing all HDF5 paths."""
    list_path = tmp_path / "file_list.txt"
    paths = sorted(hdf5_dir.glob("*.hdf5"))
    list_path.write_text("\n".join(str(p) for p in paths))
    return list_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create an output directory for chunks."""
    out = tmp_path / "chunks"
    out.mkdir()
    return out


@pytest.fixture
def raw_signals(hdf5_dir: Path, chrom_file: Path) -> Dict[str, np.ndarray]:
    """Load all raw signals via read_signal for ground-truth comparison."""
    chroms = load_chroms(chrom_file)
    signals = {}
    for hdf5_path in sorted(hdf5_dir.glob("*.hdf5")):
        sample_id = _extract_sample_id(hdf5_path)
        signals[sample_id] = read_signal(hdf5_path, chroms, normalize=False)
    return signals


# --- Helper ---


def _read_all_chunks(chunk_dir: Path) -> Dict[str, np.ndarray]:
    """Read all sample signals from chunk files in a directory."""
    result = {}
    for chunk_file in sorted(chunk_dir.glob("*.h5")):
        with h5py.File(chunk_file, "r") as f:
            sigs: h5py.Dataset = f["signals"]  # type: ignore[assignment]
            ids: h5py.Dataset = f["sample_ids"]  # type: ignore[assignment]
            for i in range(len(sigs)):  # pylint: disable=consider-using-enumerate
                sid = ids[i]
                if isinstance(sid, bytes):
                    sid = sid.decode("utf-8")
                result[str(sid)] = np.array(sigs[i], dtype=np.float32)
    return result


# --- Tests: convert() function ---


class TestConvert:
    """Tests for the convert() function."""

    def test_basic_conversion(self, hdf5_list: Path, chrom_file: Path, output_dir: Path):
        """Verify basic conversion produces chunk files."""
        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        assert len(chunks) == 3  # 25 samples / 10 per chunk = 3 chunks
        for chunk_path in chunks:
            assert chunk_path.exists()

    def test_all_samples_present(
        self,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
        raw_signals: Dict[str, np.ndarray],
    ):
        """Verify all input samples appear in the output chunks."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        chunked = _read_all_chunks(output_dir)
        assert set(chunked.keys()) == set(raw_signals.keys())

    def test_signal_roundtrip_unnormalized(
        self,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
        raw_signals: Dict[str, np.ndarray],
    ):
        """Verify unnormalized signals are identical after round-trip."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
            normalize=False,
        )

        chunked = _read_all_chunks(output_dir)
        for sample_id, expected in raw_signals.items():
            np.testing.assert_array_equal(
                chunked[sample_id],
                expected,
                err_msg=f"Mismatch for {sample_id}",
            )

    def test_signal_roundtrip_normalized(
        self,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
    ):
        """Verify normalized signals have zero mean and unit std."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
            normalize=True,
        )

        chunked = _read_all_chunks(output_dir)
        for sample_id, signal in chunked.items():
            np.testing.assert_almost_equal(
                signal.mean(),
                0.0,
                decimal=5,
                err_msg=f"Mean not zero for {sample_id}",
            )
            np.testing.assert_almost_equal(
                signal.std(),
                1.0,
                decimal=5,
                err_msg=f"Std not one for {sample_id}",
            )

    def test_signal_shape(self, hdf5_list: Path, chrom_file: Path, output_dir: Path):
        """Verify chunk file dataset shapes are correct."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        for chunk_file in sorted(output_dir.glob("*.h5")):
            with h5py.File(chunk_file, "r") as f:
                sigs: h5py.Dataset = f["signals"]  # type: ignore[assignment]
                ids: h5py.Dataset = f["sample_ids"]  # type: ignore[assignment]
                assert sigs.shape[1] == SIGNAL_LENGTH  # pylint: disable=no-member
                assert sigs.dtype == np.float32  # pylint: disable=no-member
                assert len(sigs) == len(ids)

    def test_chunk_file_count(self, hdf5_list: Path, chrom_file: Path, output_dir: Path):
        """Verify the number of chunk files matches expected splits."""
        # 25 samples, 7 per chunk → ceil(25/7) = 4 chunks
        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=7,
        )

        assert len(chunks) == 4
        # Last chunk should have 25 - 3*7 = 4 samples
        with h5py.File(chunks[-1], "r") as f:
            assert len(f["signals"]) == 4  # type: ignore[arg-type]

    def test_single_chunk(self, hdf5_list: Path, chrom_file: Path, output_dir: Path):
        """Verify all samples fit in one chunk when chunk size is large."""
        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=100,
        )

        assert len(chunks) == 1
        with h5py.File(chunks[0], "r") as f:
            assert len(f["signals"]) == N_SAMPLES  # type: ignore[arg-type]

    def test_chunk_metadata_attrs(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify chunk file attributes are written correctly."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
            normalize=True,
        )

        chunk_file = next(output_dir.glob("*.h5"))
        with h5py.File(chunk_file, "r") as f:
            assert f.attrs["signal_length"] == SIGNAL_LENGTH
            assert bool(f.attrs["normalized"]) is True
            assert (  # pylint: disable=unsupported-membership-test
                str(chrom_file) in f.attrs["source_chrom_file"]
            )

    def test_idempotent_skips_existing(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify re-running conversion skips already-written chunks."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        mtimes = {p: p.stat().st_mtime for p in output_dir.glob("*.h5")}

        # Run again
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        for path, mtime in mtimes.items():
            assert path.stat().st_mtime == mtime


# --- Tests: filtering ---


class TestFiltering:
    """Tests for sample ID filtering."""

    def test_sample_id_filter(
        self,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
        raw_signals: Dict[str, np.ndarray],
    ):
        """Verify only filtered sample IDs are converted."""
        all_ids = list(raw_signals.keys())
        subset = all_ids[:5]

        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
            sample_id_filter=subset,
        )

        chunked = _read_all_chunks(output_dir)
        assert set(chunked.keys()) == set(subset)
        assert len(chunks) == 1  # 5 samples < 10 per chunk

    def test_sample_id_filter_file(
        self,
        tmp_path: Path,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
        raw_signals: Dict[str, np.ndarray],
    ):
        """Verify filtering via --sample-ids-file through main()."""
        all_ids = list(raw_signals.keys())
        subset = all_ids[:3]

        filter_file = tmp_path / "wanted.txt"
        filter_file.write_text("\n".join(subset))

        ret = main(
            [
                "convert",
                "--hdf5-list",
                str(hdf5_list),
                "--chrom-file",
                str(chrom_file),
                "--output-dir",
                str(output_dir),
                "--samples-per-chunk",
                "10",
                "--sample-ids-file",
                str(filter_file),
            ]
        )

        assert ret == 0
        chunked = _read_all_chunks(output_dir)
        assert set(chunked.keys()) == set(subset)

    def test_empty_after_filter(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify conversion with no matching IDs returns empty list."""
        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
            sample_id_filter=["nonexistent_id"],
        )

        assert chunks == []
        assert list(output_dir.glob("*.h5")) == []


# --- Tests: hdf5_dir override ---


class TestHdf5DirOverride:
    """Tests for the --hdf5-dir option."""

    def test_hdf5_dir_override(
        self,
        tmp_path: Path,
        hdf5_dir: Path,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
    ):
        """Verify --hdf5-dir redirects file reads to a different directory."""
        # Copy files to a new location
        alt_dir = tmp_path / "alt_hdf5s"
        alt_dir.mkdir()
        for f in hdf5_dir.glob("*.hdf5"):
            (alt_dir / f.name).write_bytes(f.read_bytes())

        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
            hdf5_dir=alt_dir,
        )

        chunked = _read_all_chunks(output_dir)
        assert len(chunked) == N_SAMPLES
        assert len(chunks) > 0


# --- Tests: dry run ---


class TestDryRun:
    """Tests for the --dry-run option."""

    def test_dry_run_writes_nothing(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify dry run does not create any files."""
        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
            dry_run=True,
        )

        assert chunks == []
        assert list(output_dir.glob("*.h5")) == []


# --- Tests: error handling ---


class TestErrorHandling:
    """Tests for error handling and strict/non-strict modes."""

    def test_corrupted_file_strict(
        self,
        hdf5_dir: Path,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
    ):
        """Verify strict mode raises on a corrupted file."""
        # Corrupt a non-first file (first is used for signal length probe)
        files = sorted(hdf5_dir.glob("*.hdf5"))
        files[1].write_bytes(os.urandom(1024))

        with pytest.raises(RuntimeError, match="Error reading"):
            convert(
                hdf5_list=hdf5_list,
                chrom_file=chrom_file,
                output_dir=output_dir,
                samples_per_chunk=100,
                strict=True,
            )

    def test_corrupted_file_non_strict(
        self,
        hdf5_dir: Path,
        hdf5_list: Path,
        chrom_file: Path,
        output_dir: Path,
    ):
        """Verify non-strict mode skips corrupted files and continues."""
        # Corrupt a non-first file (first is used for signal length probe)
        files = sorted(hdf5_dir.glob("*.hdf5"))
        files[1].write_bytes(os.urandom(1024))

        chunks = convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=100,
            strict=False,
        )

        chunked = _read_all_chunks(output_dir)
        assert len(chunked) == N_SAMPLES - 1
        assert len(chunks) > 0

    def test_no_samples(self, tmp_path: Path, chrom_file: Path, output_dir: Path):
        """Verify conversion with empty file list returns empty."""
        empty_list = tmp_path / "empty.txt"
        empty_list.write_text("")

        chunks = convert(
            hdf5_list=empty_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        assert chunks == []


# --- Tests: verify_chunks() ---


class TestVerifyChunks:
    """Tests for the verify subcommand."""

    def test_verify_valid_chunks(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify that valid chunks pass verification."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        assert verify_chunks(output_dir) is True

    def test_verify_with_expected_samples(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify expected sample count check passes."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        assert verify_chunks(output_dir, expected_samples=N_SAMPLES) is True

    def test_verify_wrong_expected_samples(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify expected sample count check fails on mismatch."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        assert verify_chunks(output_dir, expected_samples=999) is False

    def test_verify_empty_dir(self, tmp_path: Path):
        """Verify empty directory fails verification."""
        empty = tmp_path / "empty"
        empty.mkdir()
        assert verify_chunks(empty) is False

    def test_verify_corrupted_chunk(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify corrupted chunk file fails verification."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        # Corrupt one chunk
        first_chunk = sorted(output_dir.glob("*.h5"))[0]
        first_chunk.write_bytes(os.urandom(1024))

        assert verify_chunks(output_dir) is False

    def test_verify_missing_dataset(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify chunk with missing dataset fails verification."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        # Overwrite one chunk with a bad structure
        first_chunk = sorted(output_dir.glob("*.h5"))[0]
        with h5py.File(first_chunk, "w") as f:
            f.create_dataset("signals", data=np.zeros((1, 10)))
            # Missing sample_ids

        assert verify_chunks(output_dir) is False


# --- Tests: main() CLI interface ---


class TestMainCLI:
    """Tests for the main() CLI entry point."""

    def test_convert_via_main(self, hdf5_list: Path, chrom_file: Path, output_dir: Path):
        """Verify conversion works through main()."""
        ret = main(
            [
                "convert",
                "--hdf5-list",
                str(hdf5_list),
                "--chrom-file",
                str(chrom_file),
                "--output-dir",
                str(output_dir),
                "--samples-per-chunk",
                "10",
            ]
        )

        assert ret == 0
        assert len(list(output_dir.glob("*.h5"))) == 3

    def test_convert_normalized_via_main(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify --normalize flag works through main()."""
        ret = main(
            [
                "convert",
                "--hdf5-list",
                str(hdf5_list),
                "--chrom-file",
                str(chrom_file),
                "--output-dir",
                str(output_dir),
                "--samples-per-chunk",
                "10",
                "--normalize",
            ]
        )

        assert ret == 0
        chunked = _read_all_chunks(output_dir)
        for signal in chunked.values():
            np.testing.assert_almost_equal(signal.mean(), 0.0, decimal=5)

    def test_dry_run_via_main(self, hdf5_list: Path, chrom_file: Path, output_dir: Path):
        """Verify --dry-run flag works through main()."""
        ret = main(
            [
                "convert",
                "--hdf5-list",
                str(hdf5_list),
                "--chrom-file",
                str(chrom_file),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )

        assert ret == 0
        assert list(output_dir.glob("*.h5")) == []

    def test_verify_via_main(self, hdf5_list: Path, chrom_file: Path, output_dir: Path):
        """Verify verify subcommand works through main()."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        ret = main(["verify", str(output_dir)])
        assert ret == 0

    def test_verify_with_expected_via_main(
        self, hdf5_list: Path, chrom_file: Path, output_dir: Path
    ):
        """Verify --expected-samples flag works through main()."""
        convert(
            hdf5_list=hdf5_list,
            chrom_file=chrom_file,
            output_dir=output_dir,
            samples_per_chunk=10,
        )

        ret = main(
            [
                "verify",
                str(output_dir),
                "--expected-samples",
                str(N_SAMPLES),
            ]
        )
        assert ret == 0

        ret = main(
            [
                "verify",
                str(output_dir),
                "--expected-samples",
                "999",
            ]
        )
        assert ret == 1


# --- Tests: helper functions ---


class TestHelpers:
    """Tests for standalone helper functions."""

    def test_load_chroms(self, chrom_file: Path):
        """Verify chromosome loading and sorting."""
        chroms = load_chroms(chrom_file)
        assert chroms == sorted(CHROMS)

    def test_read_hdf5_list(self, hdf5_list: Path):
        """Verify HDF5 list parsing."""
        files = read_hdf5_list(hdf5_list)
        assert len(files) == N_SAMPLES
        for path in files.values():
            assert path.suffix == ".hdf5"

    def test_extract_sample_id_md5(self):
        """Verify md5-style ID extraction."""
        path = Path("00000000000000000000000000000001_10kb_all_none.hdf5")
        assert _extract_sample_id(path) == "00000000000000000000000000000001"

    def test_extract_sample_id_plain(self):
        """Verify plain filename ID extraction."""
        path = Path("my_sample.hdf5")
        assert _extract_sample_id(path) == "my_sample"

    def test_extract_sample_id_non_hex(self):
        """Verify non-hex 32-char prefix falls back to stem."""
        path = Path("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_foo.hdf5")
        assert _extract_sample_id(path) == "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_foo"

    def test_read_signal_shape(self, hdf5_dir: Path, chrom_file: Path):
        """Verify read_signal produces correct shape."""
        chroms = load_chroms(chrom_file)
        first_file = sorted(hdf5_dir.glob("*.hdf5"))[0]
        signal = read_signal(first_file, chroms, normalize=False)

        assert signal.shape == (SIGNAL_LENGTH,)
        assert signal.dtype == np.float32

    def test_read_signal_normalize(self, hdf5_dir: Path, chrom_file: Path):
        """Verify normalized signal has zero mean and unit std."""
        chroms = load_chroms(chrom_file)
        first_file = sorted(hdf5_dir.glob("*.hdf5"))[0]
        signal = read_signal(first_file, chroms, normalize=True)

        np.testing.assert_almost_equal(signal.mean(), 0.0, decimal=5)
        np.testing.assert_almost_equal(signal.std(), 1.0, decimal=5)
