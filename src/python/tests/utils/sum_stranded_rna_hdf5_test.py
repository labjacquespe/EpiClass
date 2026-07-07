"""Tests for sum_stranded_rna_hdf5.py.

Builds temporary stranded RNA single-sample HDF5 pairs (per-chromosome
datasets under one group), runs the summing pipeline, and checks that both
output modes (per-pair single files, chunked matrix) hold the element-wise
sum, that the deterministic md5-of-filenames ID is order-independent, and that
the mapping TSV bridges each new ID back to its source files/IDs.
"""
# pylint: disable=redefined-outer-name, too-many-positional-arguments
# pylint: disable=missing-function-docstring, missing-class-docstring
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pytest

from epiclass.utils.preprocessing.hdf5_chunks_creation import verify_chunks
from epiclass.utils.preprocessing.sum_stranded_rna_hdf5 import (
    derive_pair_id,
    main,
    parse_pair_list,
    process,
    read_group,
    sum_pair,
)

CHROMS = ["chr1", "chr2", "chr3"]
CHROM_SIZES = {"chr1": 50, "chr2": 30, "chr3": 20}
SIGNAL_LENGTH = sum(CHROM_SIZES.values())  # 100
# Source samples carry only md5-style ids (no EpiRR), like the real inputs.
SAMPLE_MD5S = [
    ("00000000000000000000000000000001", "00000000000000000000000000000002"),
    ("0000000000000000000000000000000a", "0000000000000000000000000000000b"),
    ("0000000000000000000000000000000c", "0000000000000000000000000000000d"),
]
RNG_SEED = 7


def _write_single(path: Path, header: str, data: Dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as f:
        grp = f.create_group(header)
        for chrom, arr in data.items():
            grp.create_dataset(chrom, data=arr.astype(np.float32))


@pytest.fixture
def stranded(tmp_path: Path) -> Tuple[Path, Dict[str, np.ndarray], Dict[str, tuple]]:
    """Create stranded pairs. Returns (pair_list, {new_id: sum}, {new_id: (a_md5, b_md5)})."""
    hdf5_dir = tmp_path / "hdf5s"
    hdf5_dir.mkdir()
    rng = np.random.default_rng(RNG_SEED)

    expected: Dict[str, np.ndarray] = {}
    sources: Dict[str, tuple] = {}
    lines: List[str] = []
    for a_md5, b_md5 in SAMPLE_MD5S:
        a_d, b_d, sum_d = {}, {}, {}
        for chrom in CHROMS:
            pa = rng.standard_normal(CHROM_SIZES[chrom]).astype(np.float32)
            pb = rng.standard_normal(CHROM_SIZES[chrom]).astype(np.float32)
            a_d[chrom], b_d[chrom], sum_d[chrom] = pa, pb, pa + pb
        a_path = hdf5_dir / f"{a_md5}_ihec_rna_Unique.plusRaw_100kb.hdf5"
        b_path = hdf5_dir / f"{b_md5}_ihec_rna_Unique.minusRaw_100kb.hdf5"
        _write_single(a_path, a_md5, a_d)
        _write_single(b_path, b_md5, b_d)
        # b first to prove order-independence of the derived id + sum
        lines.append(f"{b_path}\t{a_path}")
        new_id = derive_pair_id(a_path, b_path)
        expected[new_id] = np.concatenate([sum_d[c] for c in sorted(CHROMS)])
        sources[new_id] = (a_md5, b_md5)

    pair_list = tmp_path / "pairs.txt"
    pair_list.write_text("\n".join(lines) + "\n")
    return pair_list, expected, sources


@pytest.fixture
def chrom_file(tmp_path: Path) -> Path:
    path = tmp_path / "chroms.txt"
    path.write_text("\n".join(f"{c}\t{CHROM_SIZES[c]}" for c in CHROMS))
    return path


def _read_chunks(chunk_dir: Path) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for chunk_file in sorted(chunk_dir.glob("*.h5")):
        with h5py.File(chunk_file, "r") as f:
            sigs: h5py.Dataset = f["signals"]  # type: ignore[assignment]
            ids: h5py.Dataset = f["sample_ids"]  # type: ignore[assignment]
            for i in range(len(sigs)):  # pylint: disable=consider-using-enumerate
                sid = ids[i]
                sid = sid.decode() if isinstance(sid, bytes) else str(sid)
                result[sid] = np.array(sigs[i], dtype=np.float32)
    return result


def _read_mapping(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return {row["new_id"]: row for row in csv.DictReader(f, delimiter="\t")}


# --- Tests: id derivation & pair-list parsing ---


class TestDeriveId:
    def test_order_independent(self, tmp_path: Path):
        a = tmp_path / "aaa_plusRaw.hdf5"
        b = tmp_path / "bbb_minusRaw.hdf5"
        assert derive_pair_id(a, b) == derive_pair_id(b, a)

    def test_id_is_md5_hex(self, tmp_path: Path):
        a = tmp_path / "aaa.hdf5"
        b = tmp_path / "bbb.hdf5"
        new_id = derive_pair_id(a, b)
        assert len(new_id) == 32
        int(new_id, 16)  # hex-parseable

    def test_path_independent(self, tmp_path: Path):
        # Same basenames in different dirs -> same id (basename-only hashing).
        a1 = tmp_path / "d1" / "aaa.hdf5"
        b1 = tmp_path / "d1" / "bbb.hdf5"
        a2 = tmp_path / "d2" / "aaa.hdf5"
        b2 = tmp_path / "d2" / "bbb.hdf5"
        assert derive_pair_id(a1, b1) == derive_pair_id(a2, b2)


class TestParsePairList:
    def test_two_column(self, stranded):
        pair_list, _, _ = stranded
        pairs = parse_pair_list(pair_list)
        assert len(pairs) == len(SAMPLE_MD5S)
        assert all(eid is None for eid, _, _ in pairs)

    def test_three_column_explicit_id(self, tmp_path: Path):
        pl = tmp_path / "p.txt"
        pl.write_text("my_id /a/x.hdf5 /a/y.hdf5\n")
        pairs = parse_pair_list(pl)
        assert pairs[0][0] == "my_id"

    def test_comments_and_blanks_skipped(self, tmp_path: Path):
        pl = tmp_path / "p.txt"
        pl.write_text("# header\n\n/a/x.hdf5,/a/y.hdf5\n")
        assert len(parse_pair_list(pl)) == 1

    def test_bad_field_count_raises(self, tmp_path: Path):
        pl = tmp_path / "p.txt"
        pl.write_text("only_one_field\n")
        with pytest.raises(ValueError, match="2 or 3 fields"):
            parse_pair_list(pl)


# --- Tests: summing ---


class TestSumPair:
    def test_sum_matches_and_commutes(self, stranded):
        pair_list, expected, _ = stranded
        for _, path_a, path_b in parse_pair_list(pair_list):
            _, s1 = sum_pair(path_a, path_b)
            # pylint: disable-next=arguments-out-of-order  # intentional: test commutativity
            _, s2 = sum_pair(path_b, path_a)
            got1 = np.concatenate([s1[c] for c in sorted(CHROMS)])
            got2 = np.concatenate([s2[c] for c in sorted(CHROMS)])
            np.testing.assert_array_almost_equal(got1, got2)
            new_id = derive_pair_id(path_a, path_b)
            np.testing.assert_array_almost_equal(got1, expected[new_id])

    def test_shape_mismatch_raises(self, tmp_path: Path):
        a = tmp_path / "a.hdf5"
        b = tmp_path / "b.hdf5"
        _write_single(a, "h", {"chr1": np.ones(5, dtype=np.float32)})
        _write_single(b, "h", {"chr1": np.ones(6, dtype=np.float32)})
        with pytest.raises(ValueError, match="shape mismatch"):
            sum_pair(a, b)

    def test_chrom_key_mismatch_raises(self, tmp_path: Path):
        a = tmp_path / "a.hdf5"
        b = tmp_path / "b.hdf5"
        _write_single(a, "h", {"chr1": np.ones(5, dtype=np.float32)})
        _write_single(b, "h", {"chr2": np.ones(5, dtype=np.float32)})
        with pytest.raises(ValueError, match="chromosome datasets differ"):
            sum_pair(a, b)


# --- Tests: mapping TSV ---


class TestMapping:
    def test_mapping_bridges_new_id_to_sources(self, stranded, tmp_path: Path):
        pair_list, _, sources = stranded
        out = tmp_path / "out"
        mapping_path = process(pair_list, out, per_pair=True, chunked=False)
        assert mapping_path == out / "pair_mapping.tsv"
        mapping = _read_mapping(mapping_path)
        assert set(mapping) == set(sources)
        for new_id, (a_md5, b_md5) in sources.items():
            row = mapping[new_id]
            # source ids recovered from the filenames (order may differ)
            assert {row["id_a"], row["id_b"]} == {a_md5, b_md5}
            assert row["per_pair_path"].endswith(f"{new_id}.hdf5")

    def test_mapping_file_override(self, stranded, tmp_path: Path):
        pair_list, _, _ = stranded
        out = tmp_path / "out"
        custom = tmp_path / "custom_map.tsv"
        got = process(pair_list, out, per_pair=True, chunked=False, mapping_file=custom)
        assert got == custom
        assert custom.exists()


# --- Tests: per-pair output ---


class TestPerPair:
    def test_per_pair_files_written_and_correct(self, stranded, tmp_path: Path):
        pair_list, expected, _ = stranded
        out = tmp_path / "out"
        process(pair_list, out, per_pair=True, chunked=False)
        for new_id, exp in expected.items():
            f = out / f"{new_id}.hdf5"
            assert f.exists()
            _, data = read_group(f)
            got = np.concatenate([data[c] for c in sorted(CHROMS)])
            np.testing.assert_array_almost_equal(got, exp)

    def test_per_pair_preserves_group_layout(self, stranded, tmp_path: Path):
        pair_list, expected, _ = stranded
        out = tmp_path / "out"
        process(pair_list, out, per_pair=True, chunked=False)
        first = next(iter(expected))
        with h5py.File(out / f"{first}.hdf5", "r") as f:
            header = list(f.keys())[0]
            assert set(f[header].keys()) == set(CHROMS)


# --- Tests: chunked output ---


class TestChunked:
    def test_chunked_roundtrip(self, stranded, chrom_file, tmp_path: Path):
        pair_list, expected, _ = stranded
        out = tmp_path / "out"
        process(pair_list, out, per_pair=False, chunked=True, chrom_file=chrom_file)
        chunked = _read_chunks(out)
        assert set(chunked) == set(expected)
        for new_id, exp in expected.items():
            np.testing.assert_array_almost_equal(chunked[new_id], exp)
        assert verify_chunks(out) is True

    def test_chunked_normalize(self, stranded, chrom_file, tmp_path: Path):
        pair_list, _, _ = stranded
        out = tmp_path / "out"
        process(
            pair_list,
            out,
            per_pair=False,
            chunked=True,
            chrom_file=chrom_file,
            normalize=True,
        )
        for signal in _read_chunks(out).values():
            np.testing.assert_almost_equal(signal.mean(), 0.0, decimal=4)
            np.testing.assert_almost_equal(signal.std(), 1.0, decimal=4)

    def test_both_modes_separate_dirs(self, stranded, chrom_file, tmp_path: Path):
        pair_list, expected, _ = stranded
        out = tmp_path / "out"
        process(pair_list, out, per_pair=True, chunked=True, chrom_file=chrom_file)
        first = next(iter(expected))
        assert (out / "per_pair" / f"{first}.hdf5").exists()
        assert sorted(out.glob("chunk_*.h5"))
        assert set(_read_chunks(out)) == set(expected)

    def test_chunk_chrom_file_order_independent(self, stranded, tmp_path: Path):
        # A chrom file in non-canonical order still yields the sorted concat order
        # (load_external_chrom_file sorts), matching `expected` (built chr-sorted).
        pair_list, expected, _ = stranded
        unsorted = tmp_path / "chroms_unsorted.txt"
        unsorted.write_text("\n".join(f"{c}\t{CHROM_SIZES[c]}" for c in reversed(CHROMS)))
        out = tmp_path / "out"
        process(pair_list, out, per_pair=False, chunked=True, chrom_file=unsorted)
        chunked = _read_chunks(out)
        for new_id, exp in expected.items():
            np.testing.assert_array_almost_equal(chunked[new_id], exp)

    def test_chunk_signal_length(self, stranded, chrom_file, tmp_path: Path):
        pair_list, _, _ = stranded
        out = tmp_path / "out"
        process(pair_list, out, per_pair=False, chunked=True, chrom_file=chrom_file)
        with h5py.File(next(out.glob("chunk_*.h5")), "r") as f:
            sigs: h5py.Dataset = f["signals"]  # type: ignore[assignment]
            assert sigs.shape[1] == SIGNAL_LENGTH  # pylint: disable=no-member


# --- Tests: dry-run / strict ---


def test_dry_run_writes_nothing(stranded, chrom_file, tmp_path: Path):
    pair_list, _, _ = stranded
    out = tmp_path / "out"
    got = process(
        pair_list,
        out,
        per_pair=True,
        chunked=True,
        chrom_file=chrom_file,
        dry_run=True,
    )
    assert got is None
    assert not out.exists() or not list(out.iterdir())


def test_non_strict_skips_bad_pair(stranded, tmp_path: Path):
    pair_list, expected, _ = stranded
    bad_a = tmp_path / "bad_a.hdf5"
    bad_b = tmp_path / "bad_b.hdf5"
    _write_single(bad_a, "b", {"chr1": np.ones(5, dtype=np.float32)})
    _write_single(bad_b, "b", {"chr1": np.ones(6, dtype=np.float32)})
    bad_id = derive_pair_id(bad_a, bad_b)
    with open(pair_list, "a", encoding="utf-8") as f:
        f.write(f"{bad_a} {bad_b}\n")

    out = tmp_path / "out"
    process(pair_list, out, per_pair=True, chunked=False, strict=False)
    for new_id in expected:
        assert (out / f"{new_id}.hdf5").exists()
    assert not (out / f"{bad_id}.hdf5").exists()
    # bad pair absent from mapping too
    assert bad_id not in _read_mapping(out / "pair_mapping.tsv")


# --- Tests: CLI ---


class TestMainCLI:
    def test_main_both_modes(self, stranded, chrom_file, tmp_path: Path):
        pair_list, _, _ = stranded
        out = tmp_path / "out"
        ret = main(
            [
                "--pair-list",
                str(pair_list),
                "--output-dir",
                str(out),
                "--per-pair",
                "--chunked",
                "--chrom-file",
                str(chrom_file),
            ]
        )
        assert ret == 0
        assert sorted(out.glob("chunk_*.h5"))
        assert (out / "pair_mapping.tsv").exists()

    def test_main_requires_a_mode(self, stranded, tmp_path: Path):
        pair_list, _, _ = stranded
        with pytest.raises(SystemExit):
            main(["--pair-list", str(pair_list), "--output-dir", str(tmp_path / "o")])

    def test_main_chunked_needs_chrom_file(self, stranded, tmp_path: Path):
        pair_list, _, _ = stranded
        with pytest.raises(SystemExit):
            main(
                [
                    "--pair-list",
                    str(pair_list),
                    "--output-dir",
                    str(tmp_path / "o"),
                    "--chunked",
                ]
            )

    def test_main_normalize_requires_chunked(self, stranded, tmp_path: Path):
        pair_list, _, _ = stranded
        with pytest.raises(SystemExit):
            main(
                [
                    "--pair-list",
                    str(pair_list),
                    "--output-dir",
                    str(tmp_path / "o"),
                    "--per-pair",
                    "--normalize",
                ]
            )
