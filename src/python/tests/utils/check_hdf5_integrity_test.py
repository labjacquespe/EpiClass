"""
Tests for check_hdf5_integrity.py.

These tests exercise the dtype classification and dataset verification logic
in isolation (unit tests), plus the full `run_h5dump_check` / `check_file`
path with `subprocess.run` monkeypatched to return canned `h5dump -H` /
`h5check` output (integration tests).

Both human (chr1-22, chrX, chrY = 24 chromosomes) and mouse
(chr1-19, chrX, chrY = 21 chromosomes) contexts are exercised.

Run with:
    pytest test_check_hdf5_integrity.py -v

No external HDF5 tools are invoked.
"""
# pylint: disable=missing-function-docstring, use-implicit-booleaness-not-comparison
from types import SimpleNamespace

import pytest

from epiclass.utils.preprocessing import check_hdf5_integrity as mod
from epiclass.utils.preprocessing.check_hdf5_integrity import (
    EXPECTED_CHROMOSOMES_HUMAN,
    EXPECTED_CHROMOSOMES_MOUSE,
    check_file,
    classify_dtype,
    run_h5dump_check,
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

    def test_combined_reasons(self):
        r = self._base(
            missing=["chrY"],
            oversized=["chr3=H5T_IEEE_F64LE"],
            mixed=["chr5=H5T_STD_U32LE"],
        )
        msg = mod.format_failure(r)
        assert "missing chrY" in msg
        assert "oversized dtype chr3=H5T_IEEE_F64LE" in msg
        assert "mixed dtype families chr5=H5T_STD_U32LE" in msg
        # Reasons are joined by '; '
        assert msg.count("; ") == 2


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
