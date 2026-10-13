"""Tests for sum_stranded_rna_hdf5.py.

Builds temporary stranded RNA single-sample HDF5 pairs (per-chromosome
datasets under one group), runs the summing pipeline, and checks that both
output modes (per-pair single files, chunked matrix) hold the element-wise
sum, that the deterministic md5-of-filenames ID is order-independent, that the
per-pair files carry epigeec-style attributes (per-chromosome sumX/sumXX of the
*summed* signal, file-level bin/chrom_sizes_*/ID attribute), and that the mapping
TSV bridges each new ID back to its source files/IDs.

Real epigeec files cover both known epigeec formats (see the module docstring),
and their outputs must match the source's HDF5 types exactly: the saccer3
fixtures (older format, created 2018-01: hash / chrom_sizes_hash, float32 sums)
and the hg38 fixture (newer format, created 2020-12: signal_filename /
chrom_sizes_filename, float64 sums).
"""
# pylint: disable=redefined-outer-name, too-many-positional-arguments
# pylint: disable=missing-function-docstring, missing-class-docstring
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pytest

from epiclass.core.data_source import EpiDataSource
from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader
from epiclass.utils.preprocessing.hdf5_chunks_creation import verify_chunks
from epiclass.utils.preprocessing.sum_stranded_rna_hdf5 import (
    derive_pair_id,
    main,
    parse_pair_list,
    process,
    read_group,
    sum_pair,
)
from tests.epilap_test_data import FIXTURES_DIR, SACCER3_DIR

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
BIN = 100_000
CHROM_SIZES_HASH = b"1f3787a06c60e8306f3db49c3768b4e3"
# Newer epigeec format (signal_filename / chrom_sizes_filename, float64 sums).
HG38_HDF5 = FIXTURES_DIR / "89a0dcb635f0e9740f587931437b69f1_100kb_all_none_value.hdf5"


def _write_single(
    path: Path, header: str, data: Dict[str, np.ndarray], bin_size: int = BIN
) -> None:
    """Write a single-sample file shaped like epigeec output (attrs included)."""
    with h5py.File(path, "w") as f:
        f.attrs["bin"] = np.array([bin_size], dtype=np.int32)
        f.attrs["chrom_sizes_hash"] = np.bytes_(CHROM_SIZES_HASH)
        f.attrs["hash"] = np.bytes_(header)
        grp = f.create_group(header)
        for chrom, arr in data.items():
            dset = grp.create_dataset(chrom, data=arr.astype(np.float32))
            dset.attrs["sumX"] = np.array([arr.sum()], dtype=np.float32)
            dset.attrs["sumXX"] = np.array([(arr**2).sum()], dtype=np.float32)


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


def _h5_type(tid: h5py.h5t.TypeID) -> tuple:
    if tid.get_class() == h5py.h5t.STRING:
        return (
            "str",
            tid.get_size(),
            tid.is_variable_str(),
            tid.get_strpad(),
            tid.get_cset(),
        )
    return (tid.get_class(), tid.get_size(), tid.get_order())


def _attr_types(obj) -> Dict[str, tuple]:
    return {
        name: (
            _h5_type(h5py.h5a.open(obj.id, name.encode()).get_type()),
            obj.attrs[name].shape,
        )
        for name in obj.attrs
    }


def _h5_types(path: Path) -> Dict[str, tuple]:
    """HDF5-level types of every attribute/dataset, keyed by path *below* the group."""
    with h5py.File(path, "r") as f:
        (group_name,) = f.keys()
        group: h5py.Group = f[group_name]  # type: ignore[assignment]
        types: Dict[str, tuple] = {"/": _attr_types(f), "<group>": _attr_types(group)}
        for chrom, dset in group.items():
            types[chrom] = (
                _h5_type(dset.id.get_type()),
                dset.shape,
                dset.chunks,
                dset.compression,
                dset.fillvalue,
                _attr_types(dset),
            )
    return types


def _read_mapping(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return {row["new_id"]: row for row in csv.DictReader(f, delimiter="\t")}


# --- Tests: id derivation & pair-list parsing ---


class TestDeriveId:
    def test_order_independent(self, tmp_path: Path):
        a = tmp_path / "aaa_plusRaw.hdf5"
        b = tmp_path / "bbb_minusRaw.hdf5"
        assert derive_pair_id(a, b) == derive_pair_id(b, a)

    def test_id_derivation_is_pinned(self, tmp_path: Path):
        # Existing mapping TSVs depend on this exact recipe; changing it orphans them.
        a = tmp_path / "aaa.hdf5"
        b = tmp_path / "bbb.hdf5"
        assert derive_pair_id(a, b) == hashlib.md5(b"aaa.hdf5\nbbb.hdf5").hexdigest()

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
            s1, _ = sum_pair(path_a, path_b)
            # pylint: disable-next=arguments-out-of-order  # intentional: test commutativity
            s2, _ = sum_pair(path_b, path_a)
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

    def test_bin_mismatch_raises(self, tmp_path: Path):
        a = tmp_path / "a.hdf5"
        b = tmp_path / "b.hdf5"
        _write_single(a, "h", {"chr1": np.ones(5, dtype=np.float32)}, bin_size=BIN)
        _write_single(b, "h", {"chr1": np.ones(5, dtype=np.float32)}, bin_size=10_000)
        with pytest.raises(ValueError, match="'bin' differs"):
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
            _, data, _ = read_group(f)
            got = np.concatenate([data[c] for c in sorted(CHROMS)])
            np.testing.assert_array_almost_equal(got, exp)

    def test_per_pair_chrom_sum_attrs(self, stranded, tmp_path: Path):
        # sumX/sumXX must describe the summed signal, not be copied from a strand.
        pair_list, expected, sources = stranded
        out = tmp_path / "out"
        process(pair_list, out, per_pair=True, chunked=False)
        for new_id in expected:
            a_md5, _ = sources[new_id]
            with h5py.File(out / f"{new_id}.hdf5", "r") as f:
                group = f[list(f.keys())[0]]
                assert set(group.keys()) == set(CHROMS)  # type: ignore[union-attr]
                for chrom in CHROMS:
                    dset = group[chrom]  # type: ignore[index]
                    arr = dset[...].astype(np.float64)
                    for key, value in (("sumX", arr.sum()), ("sumXX", (arr**2).sum())):
                        attr = dset.attrs[key]
                        assert attr.shape == (1,) and attr.dtype == np.float32
                        np.testing.assert_allclose(attr[0], value, rtol=1e-6)
            # Sanity: the source strand's own sumX differs from the summed one.
            src = next(out.parent.glob(f"hdf5s/{a_md5}_*.hdf5"))
            with h5py.File(src, "r") as f_src, h5py.File(
                out / f"{new_id}.hdf5", "r"
            ) as f_out:
                src_sum = f_src[a_md5]["chr1"].attrs["sumX"]  # type: ignore[index]
                out_sum = f_out[list(f_out.keys())[0]]["chr1"].attrs["sumX"]  # type: ignore[index]
                assert not np.allclose(src_sum, out_sum)

    def test_per_pair_root_attrs_and_group(self, stranded, tmp_path: Path):
        # `bin` is what EpiDataSource reads to get the resolution; like epigeec,
        # the group and `hash` are both named after the file's ID.
        pair_list, expected, _ = stranded
        out = tmp_path / "out"
        process(pair_list, out, per_pair=True, chunked=False)
        for new_id in expected:
            with h5py.File(out / f"{new_id}.hdf5", "r") as f:
                assert int(f.attrs["bin"][0]) == BIN  # type: ignore # pylint: disable=unsubscriptable-object
                assert f.attrs["chrom_sizes_hash"] == CHROM_SIZES_HASH
                assert f.attrs["hash"] == new_id.encode()
                assert list(f.keys()) == [new_id]

    def test_explicit_id_names_output(self, stranded, tmp_path: Path):
        pair_list, _, _ = stranded
        (_, path_a, path_b), *_ = parse_pair_list(pair_list)
        explicit = tmp_path / "explicit.txt"
        explicit.write_text(f"my_sample {path_a} {path_b}\n")
        out = tmp_path / "out"
        mapping = _read_mapping(process(explicit, out, per_pair=True, chunked=False))
        assert set(mapping) == {"my_sample"}
        with h5py.File(out / "my_sample.hdf5", "r") as f:
            assert f.attrs["hash"] == b"my_sample"
            assert list(f.keys()) == ["my_sample"]

    def test_existing_file_not_overwritten(self, stranded, tmp_path: Path):
        pair_list, expected, _ = stranded
        out = tmp_path / "out"
        out.mkdir()
        first = next(iter(expected))
        sentinel = out / f"{first}.hdf5"
        sentinel.write_bytes(b"sentinel")
        process(pair_list, out, per_pair=True, chunked=False)
        assert sentinel.read_bytes() == b"sentinel"
        # still listed in the mapping, pointing at the pre-existing file
        assert _read_mapping(out / "pair_mapping.tsv")[first]["per_pair_path"] == str(
            sentinel
        )

    def test_hdf5_dir_override(self, stranded, tmp_path: Path):
        # Files moved elsewhere (e.g. $SLURM_TMPDIR): same IDs, read from new dir.
        pair_list, expected, _ = stranded
        moved = tmp_path / "moved"
        (tmp_path / "hdf5s").rename(moved)
        out = tmp_path / "out"
        mapping = _read_mapping(
            process(pair_list, out, per_pair=True, chunked=False, hdf5_dir=moved)
        )
        assert set(mapping) == set(expected)
        assert all(row["file_a"].startswith(str(moved)) for row in mapping.values())


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

    def test_split_across_chunks(self, stranded, chrom_file, tmp_path: Path):
        pair_list, expected, _ = stranded
        out = tmp_path / "out"
        process(
            pair_list,
            out,
            per_pair=False,
            chunked=True,
            chrom_file=chrom_file,
            samples_per_chunk=2,
        )
        assert len(sorted(out.glob("chunk_*.h5"))) == 2  # 3 samples -> 2 + 1
        chunked = _read_chunks(out)
        assert set(chunked) == set(expected)
        assert all(sig.shape == (SIGNAL_LENGTH,) for sig in chunked.values())
        assert verify_chunks(out) is True


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


# --- Integration: real saccer3 epigeec files ---


def test_saccer3_end_to_end(extracted_hdf5_dir: Path, tmp_path: Path):
    """Sum real epigeec saccer3 files and read the outputs back like predict.py does.

    The tracks are not real strand pairs, but they have the genuine epigeec layout
    and attributes, which is what this test is about.
    """
    chrom_file = SACCER3_DIR / "saccer3.can.chrom.sizes"
    sources = sorted(extracted_hdf5_dir.glob("*.hdf5"))[:6]
    pairs = [(sources[i], sources[i + 1]) for i in range(0, len(sources), 2)]
    pair_list = tmp_path / "pairs.txt"
    pair_list.write_text("".join(f"{a}\t{b}\n" for a, b in pairs))
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
            "--normalize",
        ]
    )
    assert ret == 0
    mapping = _read_mapping(out / "pair_mapping.tsv")
    assert len(mapping) == len(pairs)

    for path_a, path_b in pairs:
        new_id = derive_pair_id(path_a, path_b)
        out_path = Path(mapping[new_id]["per_pair_path"])
        assert out_path.parent == out / "per_pair"
        assert EpiDataSource.get_file_hdf5_resolution(out_path) == 10_000
        # md5 IDs are as long as epigeec's, so every HDF5 type matches exactly.
        assert _h5_types(out_path) == _h5_types(path_a)

        with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb, h5py.File(
            out_path, "r"
        ) as fo:
            assert fo.attrs["chrom_sizes_hash"] == fa.attrs["chrom_sizes_hash"]
            assert fo.attrs["hash"] == new_id.encode()
            # epigeec convention, which the output mirrors: group name == hash.
            assert [k.encode() for k in fa.keys()] == [fa.attrs["hash"]]
            assert list(fo.keys()) == [new_id]
            ga, gb = fa[list(fa.keys())[0]], fb[list(fb.keys())[0]]
            go = fo[new_id]  # type: ignore[assignment]
            assert set(go.keys()) == set(ga.keys())  # type: ignore[union-attr]
            for chrom, do in go.items():  # type: ignore[union-attr]
                da, db = ga[chrom], gb[chrom]  # type: ignore[index]
                np.testing.assert_array_equal(do[...], da[...] + db[...])
                # Cross-check against epigeec's own sumX: sums add across strands.
                np.testing.assert_allclose(
                    do.attrs["sumX"], da.attrs["sumX"] + db.attrs["sumX"], rtol=1e-5
                )
                arr = do[...].astype(np.float64)
                np.testing.assert_allclose(
                    do.attrs["sumXX"], [np.dot(arr, arr)], rtol=1e-6
                )

    # Both output modes must hand the model the same vector: per-pair files read
    # by the predict-time loader (normalizing) == the normalized chunked matrix.
    per_pair_list = tmp_path / "per_pair.list"
    per_pair_list.write_text("".join(f"{r['per_pair_path']}\n" for r in mapping.values()))
    loader = LazyHdf5Loader(chrom_file, normalization=True, mmap_dir=tmp_path / "mmap")
    loader.register_hdf5s(per_pair_list, verbose=False)
    loader.preload_all(verbose=False)
    chunked = _read_chunks(out)
    assert set(chunked) == set(mapping)
    for new_id, signal in chunked.items():
        np.testing.assert_allclose(
            loader.load_signal(new_id), signal, rtol=1e-5, atol=1e-5
        )
    assert verify_chunks(out) is True


# --- Integration: newer epigeec format (hg38 fixture) ---


class TestNewerEpigeecFormat:
    """The fixture is summed with itself: a valid pair for format purposes, and
    sumX / sumXX then have exact references in the source's own attributes."""

    @pytest.fixture
    def summed(self, tmp_path: Path) -> Tuple[Path, str]:
        pair_list = tmp_path / "pairs.txt"
        pair_list.write_text(f"{HG38_HDF5}\t{HG38_HDF5}\n")
        process(pair_list, tmp_path / "out", per_pair=True, chunked=False)
        new_id = derive_pair_id(HG38_HDF5, HG38_HDF5)
        return tmp_path / "out" / f"{new_id}.hdf5", new_id

    def test_hdf5_types_unchanged(self, summed):
        out_path, _ = summed
        assert _h5_types(out_path) == _h5_types(HG38_HDF5)

    def test_id_attr_and_group(self, summed):
        out_path, new_id = summed
        with h5py.File(out_path, "r") as f, h5py.File(HG38_HDF5, "r") as src:
            assert f.attrs["signal_filename"] == new_id.encode()
            assert "hash" not in f.attrs  # older-format attribute must not appear
            assert f.attrs["chrom_sizes_filename"] == src.attrs["chrom_sizes_filename"]
            assert list(f.keys()) == [new_id]

    def test_sums_match_source_attrs(self, summed):
        # x + x: sumX doubles and sumXX quadruples epigeec's own (float64) values.
        out_path, new_id = summed
        with h5py.File(out_path, "r") as f, h5py.File(HG38_HDF5, "r") as src:
            g_src = src[list(src.keys())[0]]
            for chrom, dset in f[new_id].items():  # type: ignore[union-attr]
                src_attrs = g_src[chrom].attrs  # type: ignore[index]
                np.testing.assert_allclose(
                    dset.attrs["sumX"], 2 * src_attrs["sumX"], rtol=1e-6
                )
                np.testing.assert_allclose(
                    dset.attrs["sumXX"], 4 * src_attrs["sumXX"], rtol=1e-6
                )

    def test_long_explicit_id_string_type(self, tmp_path: Path):
        # Only the string size grows (len + 1, NUL-terminated); class/padding/charset
        # are unchanged. epigeec correlate 1.6.3 was checked to accept such files.
        long_id = "IHECRE00000001_rna_unique_sum_unstranded"
        pair_list = tmp_path / "pairs.txt"
        pair_list.write_text(f"{long_id} {HG38_HDF5} {HG38_HDF5}\n")
        process(pair_list, tmp_path / "out", per_pair=True, chunked=False)
        out_path = tmp_path / "out" / f"{long_id}.hdf5"
        with h5py.File(out_path, "r") as f, h5py.File(HG38_HDF5, "r") as src:
            out_type = _attr_types(f)["signal_filename"]
            src_type = _attr_types(src)["signal_filename"]
            assert f.attrs["signal_filename"] == long_id.encode()
        assert out_type[0][1] == len(long_id) + 1
        assert out_type[0][2:] == src_type[0][2:]  # vlen, strpad, cset
        assert out_type[0][3] == h5py.h5t.STR_NULLTERM
        others_out = {k: v for k, v in _h5_types(out_path).items() if k != "/"}
        others_src = {k: v for k, v in _h5_types(HG38_HDF5).items() if k != "/"}
        assert others_out == others_src

    def test_mixed_formats_raise(self, tmp_path: Path):
        old = tmp_path / "old.hdf5"
        _write_single(old, "h", {"chr1": np.ones(5, dtype=np.float32)})
        with pytest.raises(ValueError, match="different epigeec formats"):
            sum_pair(old, HG38_HDF5)
