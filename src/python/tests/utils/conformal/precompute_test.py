"""Tests for the conformal precompute driver (within-fold LOO + deploy wrapper)."""
# Synthetic prediction-CSV construction is shared boilerplate with the sibling
# conformal test modules; that overlap is expected in test fixtures. The driver's
# closed-form LOO helpers are intentionally exercised directly.
# pylint: disable=duplicate-code, protected-access
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torchcp")

# pylint: disable=wrong-import-position
from epiclass.utils.conformal import precompute as pre, prediction as cp

CLASSES = ["assay_A", "assay_B", "assay_C", "assay_D"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _write_fold(fold_dir: Path, true_idx: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = len(true_idx)
    logits = rng.normal(size=(n, len(CLASSES)))
    logits[np.arange(n), true_idx] += 3.0
    probs = _softmax(logits)
    df = pd.DataFrame(probs, columns=CLASSES)
    df.insert(0, "Predicted class", [CLASSES[j] for j in probs.argmax(axis=1)])
    df.insert(0, "True class", [CLASSES[j] for j in true_idx])
    df.index = [f"{fold_dir.name}_{i}" for i in range(n)]
    df.index.name = "ID"
    fold_dir.mkdir(parents=True)
    df.to_csv(fold_dir / "validation_prediction.csv", index_label="ID")


def _make_run(
    tmp_path: Path, n_folds: int = 3, n: int = 500, balanced: bool = True
) -> Path:
    rng = np.random.default_rng(0)
    for fold in range(n_folds):
        if balanced:
            true_idx = rng.integers(0, len(CLASSES), size=n)
        else:
            # assay_D is rare (a handful per fold) -> below the Mondrian floor.
            true_idx = rng.integers(0, len(CLASSES) - 1, size=n)
            true_idx[:3] = len(CLASSES) - 1
        _write_fold(tmp_path / f"split{fold}", true_idx, seed=fold + 1)
    return tmp_path


# --------------------------------------------------------------------------- #
# Closed-form LOO membership vs brute force (honesty + formula).
# --------------------------------------------------------------------------- #
class TestLooMembership:
    """The vectorized LOO p-value must match a brute-force, leave-self-out loop."""

    def test_marginal_matches_bruteforce(self):
        """Marginal LOO membership equals an explicit leave-j-out conformal p-value."""
        rng = np.random.default_rng(1)
        n, k, alpha = 50, 4, 0.2
        t = rng.normal(size=n)
        scores = rng.normal(size=(n, k))
        got = pre._loo_membership(t, scores, alpha)

        expected = np.zeros((n, k), dtype=np.int64)
        for j in range(n):
            for c in range(k):
                ge = sum(1 for m in range(n) if m != j and t[m] >= scores[j, c])
                # calibration is fold \ {j}, of size n - 1, so the denominator is n
                expected[j, c] = int((1 + ge) / n > alpha)
        assert np.array_equal(got, expected)

    def test_mondrian_matches_bruteforce(self):
        """Class-conditional LOO membership equals the brute-force per-class p-value."""
        rng = np.random.default_rng(2)
        n, k, alpha = 60, 3, 0.2
        t = rng.normal(size=n)
        scores = rng.normal(size=(n, k))
        true_idx = rng.integers(0, k, size=n)
        feasible = np.ones(k, dtype=bool)
        got = pre._loo_membership_mondrian(t, scores, true_idx, alpha, feasible)

        expected = np.zeros((n, k), dtype=np.int64)
        for j in range(n):
            for c in range(k):
                pool = [m for m in range(n) if true_idx[m] == c and m != j]
                ge = sum(1 for m in pool if t[m] >= scores[j, c])
                expected[j, c] = int((1 + ge) / (len(pool) + 1) > alpha)
        assert np.array_equal(got, expected)

    def test_mondrian_infeasible_class_falls_back_to_marginal(self):
        """An infeasible class's column uses the marginal rule, not class-conditional."""
        rng = np.random.default_rng(3)
        n, k, alpha = 40, 3, 0.2
        t = rng.normal(size=n)
        scores = rng.normal(size=(n, k))
        true_idx = rng.integers(0, k, size=n)
        feasible = np.array([True, True, False])  # class 2 falls back

        mond = pre._loo_membership_mondrian(t, scores, true_idx, alpha, feasible)
        marg = pre._loo_membership(t, scores, alpha)
        assert np.array_equal(mond[:, 2], marg[:, 2])


# --------------------------------------------------------------------------- #
# End-to-end cv-examine.
# --------------------------------------------------------------------------- #
class TestCvExamine:
    """Driver output: coverage, schema, flag partition, determinism, Mondrian gate."""

    def _load(self, run_dir: Path, alpha: float, mondrian: bool = False) -> pd.DataFrame:
        tag = "_mondrian" if mondrian else ""
        path = run_dir / "conformal_sets" / f"cv_examination_SAPS{tag}_alpha{alpha}.csv"
        return pd.read_csv(path, index_col=0)

    def test_marginal_coverage_near_target(self, tmp_path: Path):
        """Within-fold marginal LOO coverage sits near the 1 - alpha target."""
        run_dir = _make_run(tmp_path, n_folds=3, n=600)
        pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1)
        df = self._load(run_dir, 0.1)
        membership = df[CLASSES].to_numpy()
        idx = {c: i for i, c in enumerate(CLASSES)}
        true_pos = df["True class"].map(idx).to_numpy()
        coverage = (membership[np.arange(len(df)), true_pos] == 1).mean()
        assert abs(coverage - 0.9) < 0.05

    def test_saps_not_all_empty(self, tmp_path: Path):
        """The deterministic p-value formulation does not collapse SAPS to empty sets."""
        run_dir = _make_run(tmp_path, n_folds=2, n=400)
        pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1)
        df = self._load(run_dir, 0.1)
        assert (df["Set size"] > 0).mean() > 0.5

    def test_schema_and_set_size_consistency(self, tmp_path: Path):
        """Columns match the documented schema; Set size matches membership + the set."""
        run_dir = _make_run(tmp_path, n_folds=2, n=300)
        pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1)
        df = self._load(run_dir, 0.1)
        meta_cols = [
            "fold",
            "True class",
            "Predicted class",
            "Prediction set",
            "Set size",
            "flag_category",
        ]
        assert list(df.columns) == meta_cols + CLASSES
        membership = df[CLASSES].to_numpy()
        assert (df["Set size"].to_numpy() == membership.sum(axis=1)).all()
        split_len = (
            df["Prediction set"]
            .fillna("")
            .map(lambda s: 0 if s == "" else len(s.split(";")))
        )
        assert (split_len.to_numpy() == df["Set size"].to_numpy()).all()

    def test_flag_partition(self, tmp_path: Path):
        """Every sample gets exactly one of the four flag categories."""
        run_dir = _make_run(tmp_path, n_folds=2, n=300)
        pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1)
        df = self._load(run_dir, 0.1)
        counts = df["flag_category"].value_counts()
        assert set(counts.index) <= set(cp.FLAG_CATEGORIES)
        assert counts.sum() == len(df)

    def test_parallel_matches_serial(self, tmp_path: Path):
        """--jobs > 1 produces byte-identical membership to the serial run."""
        run_serial = _make_run(tmp_path / "serial", n_folds=3, n=300)
        run_parallel = _make_run(tmp_path / "parallel", n_folds=3, n=300)
        pre.run_cv_examine(run_serial, alphas=(0.1,), jobs=1)
        pre.run_cv_examine(run_parallel, alphas=(0.1,), jobs=2)
        a = self._load(run_serial, 0.1).reset_index(drop=True)
        b = self._load(run_parallel, 0.1).reset_index(drop=True)
        assert a[CLASSES].equals(b[CLASSES])

    def test_force_skips_unless_set(self, tmp_path: Path):
        """A second run is skipped (cached) unless force=True."""
        run_dir = _make_run(tmp_path, n_folds=2, n=200)
        pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1)
        assert not pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1)  # cached -> skip
        assert pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1, force=True)  # recompute

    def test_mondrian_gate_marks_rare_class_degenerate(self, tmp_path: Path):
        """A rare class below the floor is marked degenerate in the feasibility sidecar."""
        run_dir = _make_run(tmp_path, n_folds=2, n=400, balanced=False)
        pre.run_cv_examine(run_dir, alphas=(0.1,), jobs=1)
        feas = pd.read_csv(
            run_dir / "conformal_sets" / "mondrian_feasibility_alpha0.1.csv"
        )
        rare = feas[feas["class"] == "assay_D"]
        assert (rare["status"] == "degenerate").all()
        assert not rare["clears_floor"].any()


class TestDeploy:
    """deploy mode resolves the K calibration + K test CSVs and writes CV+ sets."""

    def test_writes_cv_plus_csv(self, tmp_path: Path):
        """run_deploy writes one CV+ set CSV with the apply contract columns."""
        run_dir = _make_run(tmp_path / "run", n_folds=3, n=200)
        # New data: the same 50 samples scored under each of the 3 fold models.
        new_dir = tmp_path / "new"
        rng = np.random.default_rng(7)
        for fold in range(3):
            probs = _softmax(rng.normal(size=(50, len(CLASSES))))
            df = pd.DataFrame(probs, columns=CLASSES)
            df.insert(0, "Predicted class", [CLASSES[j] for j in probs.argmax(axis=1)])
            df.index = [f"new_{i}" for i in range(50)]
            df.index.name = "ID"
            (new_dir / f"split{fold}").mkdir(parents=True)
            df.to_csv(new_dir / f"split{fold}" / "test_prediction.csv", index_label="ID")

        written = pre.run_deploy(run_dir, new_dir, methods=["SAPS"], alpha=0.1)
        assert len(written) == 1
        out = pd.read_csv(written[0], index_col=0)
        assert list(out.columns)[:3] == ["Predicted class", "Prediction set", "Set size"]
        assert len(out) == 50

    def test_fold_count_mismatch_raises(self, tmp_path: Path):
        """A test/calibration fold-count mismatch is rejected."""
        run_dir = _make_run(tmp_path / "run", n_folds=3, n=100)
        new_dir = tmp_path / "new"
        (new_dir / "split0").mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_csv(new_dir / "split0" / "test_prediction.csv")
        with pytest.raises(ValueError, match="mismatch"):
            pre.run_deploy(run_dir, new_dir)
