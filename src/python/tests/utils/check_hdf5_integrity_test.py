"""
Tests for check_hdf5_integrity.py.

These tests exercise the dtype classification and dataset verification logic
in isolation (unit tests), plus the full `run_h5dump_check` / `check_file`
path with `subprocess.run` monkeypatched to return canned `h5dump -H` /
`h5check` output (integration tests). The value scan (`run_value_scan`) is
exercised against real, tiny HDF5 files written to tmp_path.

Both human (chr1-22, chrX, chrY = 24 chromosomes) and mouse
(chr1-19, chrX, chrY = 21 chromosomes) contexts are exercised.

Run with:
    pytest test_check_hdf5_integrity.py -v

No external HDF5 tools are invoked.
"""
# One test module per source module, so this one is allowed to run long.
# pylint: disable=missing-function-docstring, use-implicit-booleaness-not-comparison, too-many-lines
import logging
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from epiclass.utils.preprocessing import check_hdf5_integrity as mod
from epiclass.utils.preprocessing.check_hdf5_integrity import (
    EXPECTED_CHROMOSOMES_HUMAN,
    EXPECTED_CHROMOSOMES_MOUSE,
    check_file,
    classify_dtype,
    run_h5dump_check,
    run_value_scan,
    verify_dtypes,
)

# Ignoring 'use-implicit-booleaness-not-comparison' warnings.
# Tests deliberately use `== []` rather than `not x` to assert exact list
# shape. The result dicts mix list-valued fields (missing, wrong_dtype,
# oversized, mixed, found, missing_allowed) with `family` which is
# str | None, and rewriting to implicit booleaness would silently accept
# None / "" / 0 / False in place of an empty list — defeating the point
# of the assertion.

# ── Helpers ─────────────────────────────────────────────────────────────────


# Convenience: cast to set the way the production code does, so test inputs
# match what `run_h5dump_check`/`check_file` would actually receive at runtime.
HUMAN_SET = set(EXPECTED_CHROMOSOMES_HUMAN)
MOUSE_SET = set(EXPECTED_CHROMOSOMES_MOUSE)

HUMAN_COUNT = len(EXPECTED_CHROMOSOMES_HUMAN)  # 24
MOUSE_COUNT = len(EXPECTED_CHROMOSOMES_MOUSE)  # 21


def _h5dump_stdout(chrom_dtypes):
    """Build a minimal `h5dump -H` output string for a {chrom: dtype} map."""
    blocks = []
    for chrom, dtype in chrom_dtypes.items():
        blocks.append(
            f'   DATASET "{chrom}" {{\n'
            f"      DATATYPE  {dtype}\n"
            f"      DATASPACE  SIMPLE {{ ( 1000 ) / ( 1000 ) }}\n"
            f"   }}\n"
        )
    return 'HDF5 "foo.h5" {\nGROUP "/" {\n' + "".join(blocks) + "}\n}\n"


def _uniform(dtype, chroms=None):
    """All chromosomes mapped to the same dtype. Defaults to the human set."""
    chroms = chroms if chroms is not None else EXPECTED_CHROMOSOMES_HUMAN
    return {c: dtype for c in chroms}


def _write_hdf5(path, chrom_arrays, group_name="sample", extra_groups=0):
    """Write a single-sample HDF5: one top-level group holding chrom datasets.

    `extra_groups` adds sibling top-level groups, to exercise the group-count
    guard in the value scan.
    """
    with h5py.File(path, "w") as f:
        grp = f.create_group(group_name)
        for chrom, values in chrom_arrays.items():
            grp.create_dataset(chrom, data=np.asarray(values, dtype=np.float32))
        for i in range(extra_groups):
            f.create_group(f"extra{i}")
    return path


def _clean_file(tmp_path, name="clean.h5", chroms=("chr1", "chr2")):
    """A small, all-finite, in-range file."""
    return _write_hdf5(tmp_path / name, {c: [0.0, 1.5, -2.5] for c in chroms})


def _fake_run_factory(
    h5dump_stdout, h5check_returncode=0, h5check_stdout="", h5check_stderr=""
):
    """
    Build a replacement for `subprocess.run` that dispatches by command name.

    - h5check  → returns a CompletedProcess-like object with the given returncode.
    - h5dump   → returns a CompletedProcess-like object with the given stdout.
    """

    # Accept and ignore subprocess.run's extra args/kwargs (capture_output, text, timeout, check).
    def fake_run(cmd, *_args, **_kwargs):
        if cmd[0] == "h5check":
            return SimpleNamespace(
                returncode=h5check_returncode,
                stdout=h5check_stdout,
                stderr=h5check_stderr,
            )
        if cmd[0] == "h5dump":
            return SimpleNamespace(returncode=0, stdout=h5dump_stdout, stderr="")
        raise AssertionError(f"Unexpected command: {cmd!r}")

    return fake_run


# ── Module-level constants ──────────────────────────────────────────────────


class TestExpectedChromosomeSets:
    """Make sure the two species constants are well-formed and disjoint where expected."""

    def test_human_has_22_autosomes_plus_xy(self):
        assert len(EXPECTED_CHROMOSOMES_HUMAN) == 24
        assert "chr22" in EXPECTED_CHROMOSOMES_HUMAN
        assert "chr23" not in EXPECTED_CHROMOSOMES_HUMAN
        assert "chrX" in EXPECTED_CHROMOSOMES_HUMAN
        assert "chrY" in EXPECTED_CHROMOSOMES_HUMAN

    def test_mouse_has_19_autosomes_plus_xy(self):
        assert len(EXPECTED_CHROMOSOMES_MOUSE) == 21
        assert "chr19" in EXPECTED_CHROMOSOMES_MOUSE
        assert "chr20" not in EXPECTED_CHROMOSOMES_MOUSE
        assert "chr22" not in EXPECTED_CHROMOSOMES_MOUSE
        assert "chrX" in EXPECTED_CHROMOSOMES_MOUSE
        assert "chrY" in EXPECTED_CHROMOSOMES_MOUSE

    def test_mouse_is_subset_of_human_naming(self):
        # Mouse chr1-19 + chrX/Y should all be names that also appear in the
        # human list (the human list strictly extends to chr22).
        assert MOUSE_SET.issubset(HUMAN_SET)


# ── classify_dtype ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dtype,expected",
    [
        # Floats
        ("H5T_IEEE_F32LE", "float32"),
        ("H5T_IEEE_F32BE", "float32"),
        ("H5T_IEEE_F64LE", "float64"),
        ("H5T_IEEE_F64BE", "float64"),
        # Signed ints
        ("H5T_STD_I32LE", "int32"),
        ("H5T_STD_I32BE", "int32"),
        ("H5T_STD_I64LE", "int64"),
        ("H5T_STD_I64BE", "int64"),
        # Unsigned ints
        ("H5T_STD_U32LE", "int32"),
        ("H5T_STD_U32BE", "int32"),
        ("H5T_STD_U64LE", "int64"),
        ("H5T_STD_U64BE", "int64"),
        # Rejected: too narrowon-numeric, or malformed
        ("H5T_STD_I16LE", "other"),
        ("H5T_STD_I8LE", "other"),
        ("H5T_STD_U16BE", "other"),
        ("H5T_IEEE_F16LE", "other"),
        ("H5T_STRING", "other"),
        ("H5T_COMPOUND", "other"),
        ("", "other"),
    ],
)
def test_classify_dtype(dtype, expected):
    assert classify_dtype(dtype) == expected


# ── verify_dtypes ──────────────────────────────────────────────────────────
#
# verify_dtypes signature is (expected_chromosomes, dataset_dtypes, mode).
# We parametrize each test class over both human and mouse contexts so the same
# behavioral guarantees are exercised under both chromosome sets.


SPECIES_PARAMS = [
    pytest.param(EXPECTED_CHROMOSOMES_HUMAN, id="human"),
    pytest.param(EXPECTED_CHROMOSOMES_MOUSE, id="mouse"),
]


class TestVerifyDtypesFloatMode:
    """Float mode: F32 passes, F64 is oversized, ints are wrong_dtype."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_f32_clean(self, chroms):
        r = verify_dtypes(chroms, _uniform("H5T_IEEE_F32LE", chroms), "float")
        assert r == {"wrong_dtype": [], "oversized": [], "mixed": [], "family": None}

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_f32be_clean(self, chroms):
        # Big-endian should also be accepted.
        r = verify_dtypes(chroms, _uniform("H5T_IEEE_F32BE", chroms), "float")
        assert r["wrong_dtype"] == [] and r["oversized"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_f64_oversized(self, chroms):
        n = len(chroms)
        r = verify_dtypes(chroms, _uniform("H5T_IEEE_F64LE", chroms), "float")
        assert r["wrong_dtype"] == []
        assert len(r["oversized"]) == n
        assert all(e.endswith("=H5T_IEEE_F64LE") for e in r["oversized"])

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_int_rejected_as_wrong_dtype(self, chroms):
        n = len(chroms)
        r = verify_dtypes(chroms, _uniform("H5T_STD_U32LE", chroms), "float")
        assert len(r["wrong_dtype"]) == n
        assert r["oversized"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_narrow_int_wrong_dtype(self, chroms):
        n = len(chroms)
        r = verify_dtypes(chroms, _uniform("H5T_STD_I16LE", chroms), "float")
        assert len(r["wrong_dtype"]) == n


class TestVerifyDtypesIntMode:
    """Int mode: 32-bit I/U passes, 64-bit is oversized, floats are wrong_dtype."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    @pytest.mark.parametrize(
        "dtype",
        ["H5T_STD_I32LE", "H5T_STD_I32BE", "H5T_STD_U32LE", "H5T_STD_U32BE"],
    )
    def test_all_int32_clean(self, chroms, dtype):
        r = verify_dtypes(chroms, _uniform(dtype, chroms), "int")
        assert r == {"wrong_dtype": [], "oversized": [], "mixed": [], "family": None}

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    @pytest.mark.parametrize(
        "dtype",
        ["H5T_STD_I64LE", "H5T_STD_I64BE", "H5T_STD_U64LE", "H5T_STD_U64BE"],
    )
    def test_all_int64_oversized(self, chroms, dtype):
        n = len(chroms)
        r = verify_dtypes(chroms, _uniform(dtype, chroms), "int")
        assert r["wrong_dtype"] == []
        assert len(r["oversized"]) == n

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_float_rejected_as_wrong_dtype(self, chroms):
        n = len(chroms)
        r = verify_dtypes(chroms, _uniform("H5T_IEEE_F32LE", chroms), "int")
        assert len(r["wrong_dtype"]) == n
        assert r["oversized"] == []


class TestVerifyDtypesAutoMode:
    """Auto mode: accepts either family, flags 64-bit, flags within-file mixing."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_f32_clean_family_float(self, chroms):
        r = verify_dtypes(chroms, _uniform("H5T_IEEE_F32LE", chroms), "auto")
        assert r["family"] == "float"
        assert r["wrong_dtype"] == r["oversized"] == r["mixed"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_u32_clean_family_int(self, chroms):
        r = verify_dtypes(chroms, _uniform("H5T_STD_U32LE", chroms), "auto")
        assert r["family"] == "int"
        assert r["wrong_dtype"] == r["oversized"] == r["mixed"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_mixed_float_int_flagged(self, chroms):
        # chr1 is F32 -> family locked to float; chr5 + chrY are the odd ones out.
        # chr5 and chrY are present in both the human and mouse lists, so this
        # case is portable across species.
        m = _uniform("H5T_IEEE_F32LE", chroms)
        m["chr5"] = "H5T_STD_U32LE"
        m["chrY"] = "H5T_STD_I32LE"
        r = verify_dtypes(chroms, m, "auto")
        assert r["family"] == "float"
        assert set(r["mixed"]) == {"chr5=H5T_STD_U32LE", "chrY=H5T_STD_I32LE"}
        assert r["wrong_dtype"] == r["oversized"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_family_locked_by_first_chromosome_in_order(self, chroms):
        n = len(chroms)
        # If chr1 is int, family locks to int even if most others are float.
        m = _uniform("H5T_IEEE_F32LE", chroms)
        m["chr1"] = "H5T_STD_U32LE"
        r = verify_dtypes(chroms, m, "auto")
        assert r["family"] == "int"
        # Every chromosome other than chr1 is float, so all (n - 1) are 'mixed'.
        assert len(r["mixed"]) == n - 1

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_f64_oversized_family_float(self, chroms):
        n = len(chroms)
        r = verify_dtypes(chroms, _uniform("H5T_IEEE_F64LE", chroms), "auto")
        assert r["family"] == "float"
        assert len(r["oversized"]) == n
        assert r["mixed"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_i64_oversized_family_int(self, chroms):
        n = len(chroms)
        r = verify_dtypes(chroms, _uniform("H5T_STD_I64LE", chroms), "auto")
        assert r["family"] == "int"
        assert len(r["oversized"]) == n
        assert r["mixed"] == []


class TestVerifyDtypesCrossSpecies:
    """
    Sanity checks specific to the human/mouse split:

    A mouse-context check should ignore chr20-22 even if they happen to be
    present in the dtype map (they're not 'expected'), and conversely a
    human-context check shouldn't be tricked into passing when chr20-22 are
    absent (that's a 'missing' problem handled by run_h5dump_check, but
    verify_dtypes itself must just skip them silently).
    """

    def test_mouse_context_ignores_extra_chr20_chr21_chr22(self):
        # Build a dtype map that contains chr20-chr22 (extra) plus a bad dtype on chr20.
        # Mouse mode should not flag chr20 because it's not in the expected set.
        dtypes = _uniform("H5T_IEEE_F32LE", EXPECTED_CHROMOSOMES_HUMAN)
        dtypes["chr20"] = "H5T_STRING"  # garbage, but mouse doesn't care
        r = verify_dtypes(set(EXPECTED_CHROMOSOMES_MOUSE), dtypes, "float")
        assert r["wrong_dtype"] == []
        assert r["oversized"] == []

    def test_human_context_skips_missing_extras_silently(self):
        # Mouse-shaped dtype map (only chr1-19, X, Y) fed into a human check.
        # verify_dtypes itself just skips missing keys; the higher-level
        # `run_h5dump_check` is what reports them as 'missing'.
        dtypes = _uniform("H5T_IEEE_F32LE", EXPECTED_CHROMOSOMES_MOUSE)
        r = verify_dtypes(set(EXPECTED_CHROMOSOMES_HUMAN), dtypes, "float")
        assert r["wrong_dtype"] == []
        assert r["oversized"] == []
        assert r["mixed"] == []


# ── run_h5dump_check ────────────────────────────────────────────────────────


class TestRunH5dumpCheckDatasetsOnly:
    """Dataset-presence behavior (dtype_mode='off'), parametrized over species."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_all_present_passes(self, monkeypatch, chroms):
        n = len(chroms)
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", chroms))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="off", allow_missing_chry=False
        )
        assert r["passed"] is True
        assert r["missing"] == [] and r["missing_allowed"] == []
        assert len(r["found"]) == n

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_missing_chry_fails_when_flag_off(self, monkeypatch, chroms):
        stdout = _h5dump_stdout(
            _uniform("H5T_IEEE_F32LE", [c for c in chroms if c != "chrY"])
        )
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="off", allow_missing_chry=False
        )
        assert r["passed"] is False
        assert r["missing"] == ["chrY"]
        assert r["missing_allowed"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_missing_chry_passes_when_flag_on(self, monkeypatch, chroms):
        stdout = _h5dump_stdout(
            _uniform("H5T_IEEE_F32LE", [c for c in chroms if c != "chrY"])
        )
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="off", allow_missing_chry=True
        )
        assert r["passed"] is True
        assert r["missing"] == []
        assert r["missing_allowed"] == ["chrY"]

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_other_missing_still_fails_with_flag(self, monkeypatch, chroms):
        # chr5 and chrY both missing; chrY is allowed, chr5 is not.
        kept = [c for c in chroms if c not in ("chrY", "chr5")]
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", kept))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="off", allow_missing_chry=True
        )
        assert r["passed"] is False
        assert r["missing"] == ["chr5"]
        assert r["missing_allowed"] == ["chrY"]

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_missing_chrx_still_fails_with_chry_flag(self, monkeypatch, chroms):
        # Only chrY is optional — chrX missing must still fail.
        kept = [c for c in chroms if c != "chrX"]
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", kept))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="off", allow_missing_chry=True
        )
        assert r["passed"] is False
        assert r["missing"] == ["chrX"]
        assert r["missing_allowed"] == []


class TestRunH5dumpCheckSpeciesMismatch:
    """Catch the bug class introduced when species detection is wrong."""

    def test_human_file_checked_as_mouse_passes(self, monkeypatch):
        # A human-shaped file (chr1-22, X, Y) checked against the mouse
        # expected set should still pass: extra chromosomes are not flagged.
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", EXPECTED_CHROMOSOMES_HUMAN))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", MOUSE_SET, dtype_mode="off")
        assert r["passed"] is True
        # Only the expected (mouse) chromosomes are reported in 'found'.
        assert len(r["found"]) == MOUSE_COUNT
        assert "chr22" not in r["found"]

    def test_mouse_file_checked_as_human_reports_missing_autosomes(self, monkeypatch):
        # A mouse-shaped file (no chr20-22) checked against the human expected
        # set must fail with chr20, chr21, chr22 reported as missing.
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", EXPECTED_CHROMOSOMES_MOUSE))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", HUMAN_SET, dtype_mode="off")
        assert r["passed"] is False
        assert set(r["missing"]) == {"chr20", "chr21", "chr22"}
        assert r["missing_allowed"] == []

    def test_mouse_file_checked_as_mouse_passes(self, monkeypatch):
        # Sanity: the matching case must pass cleanly.
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", EXPECTED_CHROMOSOMES_MOUSE))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", MOUSE_SET, dtype_mode="off")
        assert r["passed"] is True
        assert len(r["found"]) == MOUSE_COUNT


class TestRunH5dumpCheckDtypeModes:
    """Dtype verification behavior when chromosomes are present."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_float_mode_all_f32_passes(self, monkeypatch, chroms):
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", chroms))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", set(chroms), dtype_mode="float")
        assert r["passed"] is True

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_float_mode_int_file_fails(self, monkeypatch, chroms):
        n = len(chroms)
        stdout = _h5dump_stdout(_uniform("H5T_STD_U32LE", chroms))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", set(chroms), dtype_mode="float")
        assert r["passed"] is False
        assert len(r["wrong_dtype"]) == n

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_int_mode_accepts_signed_and_unsigned(self, monkeypatch, chroms):
        n = len(chroms)
        m = _uniform("H5T_STD_U32LE", chroms)
        # First half signed, second half unsigned — both should be accepted.
        for c in chroms[: n // 2]:
            m[c] = "H5T_STD_I32LE"
        stdout = _h5dump_stdout(m)
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", set(chroms), dtype_mode="int")
        assert r["passed"] is True

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_int_mode_64bit_oversized(self, monkeypatch, chroms):
        n = len(chroms)
        stdout = _h5dump_stdout(_uniform("H5T_STD_U64LE", chroms))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", set(chroms), dtype_mode="int")
        assert r["passed"] is False
        assert len(r["oversized"]) == n
        assert r["wrong_dtype"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_auto_mode_detects_int_file(self, monkeypatch, chroms):
        stdout = _h5dump_stdout(_uniform("H5T_STD_U32LE", chroms))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", set(chroms), dtype_mode="auto")
        assert r["passed"] is True
        assert r["family"] == "int"

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_auto_mode_flags_mixed_family(self, monkeypatch, chroms):
        # chr10 exists in both human (chr1-22) and mouse (chr1-19) sets.
        m = _uniform("H5T_IEEE_F32LE", chroms)
        m["chr10"] = "H5T_STD_U32LE"
        stdout = _h5dump_stdout(m)
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check("dummy.h5", set(chroms), dtype_mode="auto")
        assert r["passed"] is False
        assert r["mixed"] == ["chr10=H5T_STD_U32LE"]
        assert r["family"] == "float"


class TestRunH5dumpCheckWithAllowMissingChry:
    """Interaction between --allow-missing-chry and dtype verification."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_dtype_check_runs_on_all_but_chry_when_chry_allowed_missing(
        self, monkeypatch, chroms
    ):
        # chrY absent, remaining (n-1) are valid F32. Should pass with flag on.
        kept = [c for c in chroms if c != "chrY"]
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", kept))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="float", allow_missing_chry=True
        )
        assert r["passed"] is True
        assert r["missing_allowed"] == ["chrY"]
        assert r["wrong_dtype"] == r["oversized"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_64bit_still_flagged_when_chry_allowed_missing(self, monkeypatch, chroms):
        # chrY missing + chr3 is F64. chrY is OK; chr3 oversizing is still a failure.
        kept = {c: "H5T_IEEE_F32LE" for c in chroms if c != "chrY"}
        kept["chr3"] = "H5T_IEEE_F64LE"
        stdout = _h5dump_stdout(kept)
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="float", allow_missing_chry=True
        )
        assert r["passed"] is False
        assert r["missing"] == []
        assert r["missing_allowed"] == ["chrY"]
        assert r["oversized"] == ["chr3=H5T_IEEE_F64LE"]

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_auto_mode_int_file_with_chry_allowed_missing(self, monkeypatch, chroms):
        kept = [c for c in chroms if c != "chrY"]
        stdout = _h5dump_stdout(_uniform("H5T_STD_U32LE", kept))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="auto", allow_missing_chry=True
        )
        assert r["passed"] is True
        assert r["family"] == "int"
        assert r["missing_allowed"] == ["chrY"]

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_dtype_check_skipped_when_required_chrom_missing(self, monkeypatch, chroms):
        # chr5 missing → dtype check must be skipped entirely (no false positives
        # from the partial dataset map). chrY also missing but allowed.
        kept = {c: "H5T_IEEE_F32LE" for c in chroms if c not in ("chr5", "chrY")}
        stdout = _h5dump_stdout(kept)
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = run_h5dump_check(
            "dummy.h5", set(chroms), dtype_mode="float", allow_missing_chry=True
        )
        assert r["passed"] is False
        assert r["missing"] == ["chr5"]
        assert r["missing_allowed"] == ["chrY"]
        # Dtype lists should be empty because the dtype pass was gated off.
        assert r["wrong_dtype"] == r["oversized"] == r["mixed"] == []


# ── check_file (full two-phase flow) ───────────────────────────────────────


class TestCheckFile:
    """End-to-end check_file integration with both h5check and h5dump mocked."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_happy_path_float(self, monkeypatch, tmp_path, chroms):
        f = tmp_path / "ok.h5"
        f.write_bytes(b"")  # file must exist on disk
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", chroms))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = check_file(str(f), set(chroms), dtype_mode="float")
        assert r["ok"] is True
        assert r["h5check_ok"] is True
        assert r["datasets_ok"] is True
        assert r["error"] is None
        assert r["missing_allowed"] == []

    def test_h5check_failure_short_circuits(self, monkeypatch, tmp_path):
        # When h5check fails, h5dump should not even be consulted for a result,
        # and the result must reflect the h5check error.
        f = tmp_path / "bad.h5"
        f.write_bytes(b"")
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            _fake_run_factory(
                _h5dump_stdout(_uniform("H5T_IEEE_F32LE")),
                h5check_returncode=1,
                h5check_stderr="bad superblock",
            ),
        )
        r = check_file(str(f), HUMAN_SET)
        assert r["ok"] is False
        assert r["h5check_ok"] is False
        assert r["datasets_ok"] is False
        assert "h5check" in r["error"].lower()

    def test_file_not_found(self):
        # Passing the human set, but the file-not-found path runs before any
        # subprocess, so the species choice is irrelevant.
        r = check_file("/no/such/file.h5", HUMAN_SET)
        assert r["ok"] is False
        assert r["error"] == "File not found"
        # No subprocess was called; h5check_ok stays False but no crash.

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_chry_missing_with_flag_passes(self, monkeypatch, tmp_path, chroms):
        f = tmp_path / "no_chry.h5"
        f.write_bytes(b"")
        kept = [c for c in chroms if c != "chrY"]
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", kept))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = check_file(str(f), set(chroms), dtype_mode="float", allow_missing_chry=True)
        assert r["ok"] is True
        assert r["missing_allowed"] == ["chrY"]
        assert r["missing"] == []

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_chry_missing_without_flag_fails(self, monkeypatch, tmp_path, chroms):
        f = tmp_path / "no_chry.h5"
        f.write_bytes(b"")
        kept = [c for c in chroms if c != "chrY"]
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", kept))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = check_file(str(f), set(chroms), dtype_mode="float", allow_missing_chry=False)
        assert r["ok"] is False
        assert r["missing"] == ["chrY"]
        assert r["missing_allowed"] == []

    def test_mouse_file_against_human_set_fails_on_missing_autosomes(
        self, monkeypatch, tmp_path
    ):
        # Cross-species mismatch caught at the check_file level.
        f = tmp_path / "mouse_misclassified.h5"
        f.write_bytes(b"")
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", EXPECTED_CHROMOSOMES_MOUSE))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))
        r = check_file(str(f), HUMAN_SET, dtype_mode="off")
        assert r["ok"] is False
        assert set(r["missing"]) == {"chr20", "chr21", "chr22"}


# ── format_failure ─────────────────────────────────────────────────────────


class TestFormatFailure:
    """Failure message formatting combines all applicable reasons."""

    def _base(self, **overrides):
        res = {
            "file": "x.h5",
            "error": None,
            "missing": [],
            "wrong_dtype": [],
            "oversized": [],
            "mixed": [],
            "n_outliers": 0,
            "n_nonfinite": 0,
        }
        res.update(overrides)
        return res

    def test_error_takes_precedence(self):
        r = self._base(error="boom", missing=["chr1"])
        assert mod.format_failure(r) == "boom"

    def test_missing_only(self):
        r = self._base(missing=["chr1", "chr2"])
        assert mod.format_failure(r) == "missing chr1, chr2"

    def test_wrong_dtype_only(self):
        r = self._base(wrong_dtype=["chr1=H5T_STD_I16LE"])
        assert mod.format_failure(r) == "wrong dtype chr1=H5T_STD_I16LE"

    def test_oversized_only(self):
        r = self._base(oversized=["chr3=H5T_IEEE_F64LE"])
        assert mod.format_failure(r) == "oversized dtype chr3=H5T_IEEE_F64LE"

    def test_mixed_only(self):
        r = self._base(mixed=["chr5=H5T_STD_U32LE"])
        assert mod.format_failure(r) == "mixed dtype families chr5=H5T_STD_U32LE"

    def test_outliers_only(self):
        r = self._base(n_outliers=3)
        assert mod.format_failure(r) == "3 values above threshold"

    def test_nonfinite_only(self):
        r = self._base(n_nonfinite=2)
        assert mod.format_failure(r) == "2 non-finite values"

    def test_zero_counts_produce_no_reason(self):
        # A file failing for another reason must not gain a "0 values" clause.
        r = self._base(missing=["chr1"], n_outliers=0, n_nonfinite=0)
        assert mod.format_failure(r) == "missing chr1"

    def test_missing_count_keys_tolerated(self):
        # Results built before the value scan existed lack the count keys.
        r = self._base(missing=["chr1"])
        del r["n_outliers"]
        del r["n_nonfinite"]
        assert mod.format_failure(r) == "missing chr1"

    def test_combined_reasons(self):
        r = self._base(
            missing=["chrY"],
            oversized=["chr3=H5T_IEEE_F64LE"],
            mixed=["chr5=H5T_STD_U32LE"],
            n_outliers=1,
            n_nonfinite=4,
        )
        msg = mod.format_failure(r)
        assert "missing chrY" in msg
        assert "oversized dtype chr3=H5T_IEEE_F64LE" in msg
        assert "mixed dtype families chr5=H5T_STD_U32LE" in msg
        assert "1 values above threshold" in msg
        assert "4 non-finite values" in msg
        # Reasons are joined by '; '
        assert msg.count("; ") == 4


# ── run_value_scan (phase 3) ────────────────────────────────────────────────


class TestRunValueScan:
    """Amplitude / non-finite scanning over real (tiny) HDF5 files."""

    def test_clean_file_passes(self, tmp_path):
        f = _clean_file(tmp_path)
        r = run_value_scan(str(f), {"chr1", "chr2"}, threshold=1e10)
        assert r["passed"] is True
        assert r["n_outliers"] == 0
        assert r["n_nonfinite"] == 0
        assert r["records"] == []
        assert r["error"] is None

    def test_single_outlier_flagged_with_position_and_value(self, tmp_path):
        f = _write_hdf5(tmp_path / "hot.h5", {"chr1": [0.0, 5e10, 1.0]})
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["passed"] is False
        assert r["n_outliers"] == 1
        assert r["n_nonfinite"] == 0
        (rec,) = r["records"]
        assert rec["dataset"] == "chr1"
        assert rec["index"] == 1
        assert rec["value"] == pytest.approx(5e10)
        assert rec["kind"] == "outlier"

    def test_negative_outlier_flagged(self, tmp_path):
        # The threshold is on absolute amplitude.
        f = _write_hdf5(tmp_path / "cold.h5", {"chr1": [0.0, -5e10]})
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["n_outliers"] == 1
        assert r["records"][0]["value"] == pytest.approx(-5e10)

    def test_value_exactly_at_threshold_not_flagged(self, tmp_path):
        # Comparison is strictly greater-than.
        f = _write_hdf5(tmp_path / "edge.h5", {"chr1": [100.0, -100.0]})
        r = run_value_scan(str(f), {"chr1"}, threshold=100.0)
        assert r["passed"] is True
        assert r["n_outliers"] == 0

    def test_nan_and_inf_counted_as_nonfinite_not_outliers(self, tmp_path):
        # abs(nan) > t is False, which is exactly why non-finite gets its own
        # check; inf must not be double-counted as an outlier either.
        f = _write_hdf5(
            tmp_path / "nan.h5",
            {"chr1": [np.nan, np.inf, -np.inf, 1.0]},
        )
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["passed"] is False
        assert r["n_nonfinite"] == 3
        assert r["n_outliers"] == 0
        assert {rec["kind"] for rec in r["records"]} == {"nonfinite"}

    def test_outliers_and_nonfinite_both_reported(self, tmp_path):
        f = _write_hdf5(tmp_path / "both.h5", {"chr1": [np.nan, 5e10, 1.0]})
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["n_nonfinite"] == 1
        assert r["n_outliers"] == 1
        assert {rec["kind"] for rec in r["records"]} == {"nonfinite", "outlier"}

    def test_only_expected_chromosomes_scanned(self, tmp_path):
        # A stray dataset outside the expected set is ignored.
        f = _write_hdf5(
            tmp_path / "stray.h5",
            {"chr1": [1.0], "chrM": [5e10]},
        )
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["passed"] is True
        assert r["n_outliers"] == 0

    def test_missing_dataset_is_not_an_error(self, tmp_path):
        # Dataset presence is phase 2's job; the scan just skips absentees.
        f = _write_hdf5(tmp_path / "partial.h5", {"chr1": [1.0]})
        r = run_value_scan(str(f), {"chr1", "chr2"}, threshold=1e10)
        assert r["passed"] is True
        assert r["error"] is None

    @pytest.mark.parametrize("extra_groups, expected", [(1, 2), (2, 3)])
    def test_multiple_top_level_groups_errors(self, tmp_path, extra_groups, expected):
        f = _write_hdf5(tmp_path / "multi.h5", {"chr1": [1.0]}, extra_groups=extra_groups)
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["passed"] is False
        assert r["error"] == f"expected 1 group, found {expected}"

    def test_zero_groups_errors(self, tmp_path):
        f = tmp_path / "empty.h5"
        with h5py.File(f, "w") as handle:
            handle.create_dataset("chr1", data=np.zeros(3, dtype=np.float32))
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["passed"] is False
        assert r["error"] == "expected 1 group, found 0"

    def test_unreadable_file_reports_error(self, tmp_path):
        f = tmp_path / "notanhdf5.h5"
        f.write_bytes(b"definitely not HDF5")
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["passed"] is False
        assert r["error"]

    def test_uncapped_by_default(self, tmp_path):
        # Every offending value is recorded unless a cap is asked for.
        f = _write_hdf5(tmp_path / "many.h5", {"chr1": [5e10] * 50})
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10)
        assert r["n_outliers"] == 50
        assert len(r["records"]) == 50

    def test_max_records_caps_records_but_not_counts(self, tmp_path):
        f = _write_hdf5(tmp_path / "many.h5", {"chr1": [5e10] * 50})
        r = run_value_scan(str(f), {"chr1"}, threshold=1e10, max_records=5)
        assert r["n_outliers"] == 50  # count stays exact
        assert len(r["records"]) == 5  # storage is capped

    def test_records_cap_is_per_file_across_datasets(self, tmp_path):
        f = _write_hdf5(
            tmp_path / "many2.h5",
            {"chr1": [5e10] * 10, "chr2": [5e10] * 10},
        )
        r = run_value_scan(str(f), {"chr1", "chr2"}, threshold=1e10, max_records=12)
        assert r["n_outliers"] == 20
        assert len(r["records"]) == 12


# ── check_file with the value scan wired in ─────────────────────────────────


class TestCheckFileValueScan:
    """Phase 3 gating and its effect on the overall verdict."""

    def _mock_tools(self, monkeypatch, chroms):
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", chroms))
        monkeypatch.setattr(mod.subprocess, "run", _fake_run_factory(stdout))

    def test_scan_skipped_when_threshold_is_none(self, monkeypatch, tmp_path):
        # A file full of outliers still passes when the scan is disabled.
        chroms = ["chr1"]
        f = _write_hdf5(tmp_path / "hot.h5", {"chr1": [5e10]})
        self._mock_tools(monkeypatch, chroms)
        r = check_file(str(f), set(chroms), dtype_mode="off")
        assert r["ok"] is True
        assert r["values_ok"] is True
        assert r["n_outliers"] == 0
        assert r["records"] == []

    def test_outlier_fails_otherwise_valid_file(self, monkeypatch, tmp_path):
        chroms = ["chr1"]
        f = _write_hdf5(tmp_path / "hot.h5", {"chr1": [0.0, 5e10]})
        self._mock_tools(monkeypatch, chroms)
        r = check_file(str(f), set(chroms), dtype_mode="off", outlier_threshold=1e10)
        assert r["ok"] is False
        assert r["h5check_ok"] is True
        assert r["datasets_ok"] is True
        assert r["values_ok"] is False
        assert r["n_outliers"] == 1
        assert r["error"] is None
        assert "1 values above threshold" in mod.format_failure(r)

    def test_clean_file_passes_with_scan_enabled(self, monkeypatch, tmp_path):
        chroms = ["chr1", "chr2"]
        f = _clean_file(tmp_path, chroms=chroms)
        self._mock_tools(monkeypatch, chroms)
        r = check_file(str(f), set(chroms), dtype_mode="off", outlier_threshold=1e10)
        assert r["ok"] is True
        assert r["values_ok"] is True

    def test_scan_not_run_when_h5check_fails(self, monkeypatch, tmp_path):
        # Short-circuit: no scan result is produced for a malformed file.
        f = _write_hdf5(tmp_path / "hot.h5", {"chr1": [5e10]})
        monkeypatch.setattr(
            mod.subprocess,
            "run",
            _fake_run_factory(
                _h5dump_stdout(_uniform("H5T_IEEE_F32LE", ["chr1"])),
                h5check_returncode=1,
                h5check_stderr="bad superblock",
            ),
        )
        r = check_file(str(f), {"chr1"}, dtype_mode="off", outlier_threshold=1e10)
        assert r["ok"] is False
        assert r["n_outliers"] == 0
        assert r["records"] == []

    def test_scan_error_surfaces_as_result_error(self, monkeypatch, tmp_path):
        f = tmp_path / "broken.h5"
        f.write_bytes(b"not HDF5")
        self._mock_tools(monkeypatch, ["chr1"])
        r = check_file(str(f), {"chr1"}, dtype_mode="off", outlier_threshold=1e10)
        assert r["ok"] is False
        assert r["values_ok"] is False
        assert r["error"]


# ── Outlier CSV sink ────────────────────────────────────────────────────────


class TestOutlierCsvWriter:
    """The CSV is created lazily, so a clean run leaves no file behind."""

    def _writer(self, tmp_path, name="out.csv"):
        return mod.OutlierCsvWriter(tmp_path / name, logging.getLogger("test"))

    def test_no_file_created_when_nothing_written(self, tmp_path):
        w = self._writer(tmp_path)
        w.close()
        assert not (tmp_path / "out.csv").exists()
        assert w.n_written == 0

    def test_empty_rows_do_not_create_the_file(self, tmp_path):
        w = self._writer(tmp_path)
        w.writerows([])
        w.close()
        assert not (tmp_path / "out.csv").exists()

    def test_first_write_creates_file_with_header(self, tmp_path):
        w = self._writer(tmp_path)
        w.writerows(
            [
                {
                    "file": "a.h5",
                    "dataset": "chr1",
                    "index": 3,
                    "value": 5e10,
                    "kind": "outlier",
                }
            ]
        )
        w.close()
        lines = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "file,dataset,index,value,kind"
        assert lines[1].startswith("a.h5,chr1,3,")
        assert w.n_written == 1

    def test_header_written_once_across_calls(self, tmp_path):
        w = self._writer(tmp_path)
        row = {
            "file": "a.h5",
            "dataset": "chr1",
            "index": 0,
            "value": 1.0,
            "kind": "outlier",
        }
        w.writerows([row])
        w.writerows([row, row])
        w.close()
        lines = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()
        assert lines.count("file,dataset,index,value,kind") == 1
        assert len(lines) == 4  # header + 3 rows
        assert w.n_written == 3

    def test_existing_file_untouched_when_nothing_written(self, tmp_path):
        # A CSV from an earlier run must not be silently truncated.
        previous = tmp_path / "out.csv"
        previous.write_text("previous content\n", encoding="utf-8")
        w = self._writer(tmp_path)
        w.close()
        assert previous.read_text(encoding="utf-8") == "previous content\n"

    def test_existing_file_diverted_to_timestamped_name(self, tmp_path):
        previous = tmp_path / "out.csv"
        previous.write_text("previous content\n", encoding="utf-8")
        w = self._writer(tmp_path)
        w.writerows(
            [
                {
                    "file": "a.h5",
                    "dataset": "chr1",
                    "index": 0,
                    "value": 1.0,
                    "kind": "outlier",
                }
            ]
        )
        w.close()
        # The earlier run's file survives untouched...
        assert previous.read_text(encoding="utf-8") == "previous content\n"
        # ...and csv_path reports where the rows actually went.
        assert w.csv_path != previous
        assert w.csv_path.exists()
        assert w.csv_path.name.startswith("out_")
        assert w.csv_path.suffix == ".csv"
        assert "chr1" in w.csv_path.read_text(encoding="utf-8")

    def test_context_manager_closes(self, tmp_path):
        with self._writer(tmp_path) as w:
            w.writerows(
                [
                    {
                        "file": "a.h5",
                        "dataset": "chr1",
                        "index": 0,
                        "value": 1.0,
                        "kind": "outlier",
                    }
                ]
            )
        # Readable straight after the block, so the handle was flushed/closed.
        assert (tmp_path / "out.csv").read_text(encoding="utf-8").count("\n") == 2


class TestTimestampedPath:
    """Never clobber an existing file."""

    def test_free_path_returned_unchanged(self, tmp_path):
        p = tmp_path / "scan.csv"
        assert mod.timestamped_path(p) == p

    def test_taken_path_gets_timestamp_before_suffix(self, tmp_path):
        p = tmp_path / "scan.csv"
        p.touch()
        out = mod.timestamped_path(p)
        assert out != p
        assert not out.exists()
        assert out.parent == p.parent
        assert out.name.startswith("scan_")
        assert out.suffix == ".csv"

    def test_log_file_diverted_instead_of_truncated(self, tmp_path):
        # setup_logging must not clobber an earlier run's log either.
        previous = tmp_path / "run.log"
        previous.write_text("previous run\n", encoding="utf-8")
        logger = mod.setup_logging(str(previous))
        try:
            logger.info("hello")
            written = [p for p in tmp_path.glob("run_*.log") if p.name != previous.name]
            assert len(written) == 1
            assert "hello" in written[0].read_text(encoding="utf-8")
            assert previous.read_text(encoding="utf-8") == "previous run\n"
        finally:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

    def test_taken_timestamp_gets_numeric_suffix(self, tmp_path, monkeypatch):
        # Freeze the clock so the first timestamped candidate is taken too.
        monkeypatch.setattr(mod.time, "strftime", lambda _fmt: "20260805-143000")
        p = tmp_path / "scan.csv"
        p.touch()
        (tmp_path / "scan_20260805-143000.csv").touch()
        out = mod.timestamped_path(p)
        assert out.name == "scan_20260805-143000-1.csv"


# ── Regex parsing sanity ────────────────────────────────────────────────────


class TestRegexes:
    """Guard against regressions in the dataset / dtype regex parsing."""

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_dataset_re_picks_up_all(self, chroms):
        stdout = _h5dump_stdout(_uniform("H5T_IEEE_F32LE", chroms))
        names = set(mod.DATASET_RE.findall(stdout))
        assert names == set(chroms)

    @pytest.mark.parametrize("chroms", SPECIES_PARAMS)
    def test_dataset_dtype_re_picks_up_pairs(self, chroms):
        m = _uniform("H5T_IEEE_F32LE", chroms)
        m["chr7"] = "H5T_STD_U32LE"
        stdout = _h5dump_stdout(m)
        pairs = dict(mod.DATASET_DTYPE_RE.findall(stdout))
        assert pairs["chr1"] == "H5T_IEEE_F32LE"
        assert pairs["chr7"] == "H5T_STD_U32LE"
        assert pairs["chrY"] == "H5T_IEEE_F32LE"
