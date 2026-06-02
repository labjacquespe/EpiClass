"""Tests for post-hoc conformal prediction on EpiClass prediction CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torchcp")

# pylint: disable=wrong-import-position
from epiclass.utils.postprocessing import conformal_prediction as cp

CLASSES = ["assay_A", "assay_B", "assay_C", "assay_D"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _make_prediction_df(
    n: int, seed: int = 0, separation: float = 3.0, with_labels: bool = True
) -> pd.DataFrame:
    """Build a synthetic prediction DataFrame in write_pred_table schema.

    Probabilities are softmaxes of class-separated logits, so the model is
    informative but imperfect -- exactly the regime conformal prediction targets.
    """
    rng = np.random.default_rng(seed)
    k = len(CLASSES)
    true_idx = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), true_idx] += separation
    probs = _softmax(logits)

    df = pd.DataFrame(probs, columns=CLASSES)
    df.insert(0, "Predicted class", [CLASSES[j] for j in probs.argmax(axis=1)])
    if with_labels:
        df.insert(0, "True class", [CLASSES[j] for j in true_idx])
    df.index = [f"sample_{i}" for i in range(n)]
    df.index.name = "ID"
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, encoding="utf8", index_label="ID")
    return path


class TestLoadPredictionCsv:
    """Round-tripping the EpiClass prediction CSV schema."""

    def test_loads_ids_probs_classes_labels(self, tmp_path: Path):
        """IDs, probabilities, class order and true indices round-trip."""
        df = _make_prediction_df(n=20, seed=1)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        ids, probs, classes, true_idx = cp.load_prediction_csv(path)

        assert classes == CLASSES  # file column order preserved
        assert probs.shape == (20, len(CLASSES))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert ids[0] == "sample_0"
        assert true_idx is not None
        # true_idx must map back to the True class strings
        recovered = [CLASSES[i] for i in true_idx]
        assert recovered == df["True class"].tolist()

    def test_missing_true_class_returns_none(self, tmp_path: Path):
        """A test CSV without 'True class' yields true_idx=None."""
        df = _make_prediction_df(n=10, seed=2, with_labels=False)
        path = _write_csv(df, tmp_path / "test_prediction.csv")

        _, _, classes, true_idx = cp.load_prediction_csv(path)

        assert classes == CLASSES
        assert true_idx is None

    def test_unnamed_index_column(self, tmp_path: Path):
        """The first column loads as the ID even with no 'ID' header."""
        df = _make_prediction_df(n=15, seed=9)
        path = tmp_path / "validation_prediction.csv"
        df.to_csv(path, encoding="utf8")  # default index_label -> unnamed first column

        ids, probs, classes, true_idx = cp.load_prediction_csv(path)

        assert classes == CLASSES  # 'True class'/'Predicted class' still excluded
        assert probs.shape == (15, len(CLASSES))
        assert ids[0] == "sample_0"
        assert true_idx is not None


class TestBuildScore:
    """Score-function factory."""

    @pytest.mark.parametrize("method", cp.DEFAULT_METHODS)
    def test_known_methods(self, method: str):
        """Every default method builds a score function."""
        assert cp.build_score(method) is not None

    def test_unknown_method_raises(self):
        """An unknown method name raises ValueError."""
        with pytest.raises(ValueError):
            cp.build_score("NOPE")


class TestCoverageGuarantee:
    """The core conformal property: empirical coverage >= 1 - alpha."""

    def test_all_methods_reach_target_coverage(self, tmp_path: Path):
        """Each method's empirical coverage meets the 1-alpha target."""
        # Large sample -> tight finite-sample slack around the 1-alpha target.
        df = _make_prediction_df(n=4000, seed=3)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        results = cp.run_evaluate(
            path, methods=cp.DEFAULT_METHODS, alphas=(0.1,), calib_frac=0.5, seed=42
        )

        assert set(results["method"]) == set(cp.DEFAULT_METHODS)
        for _, row in results.iterrows():
            # finite-sample coverage can dip slightly below 1-alpha; allow small slack
            assert row["empirical_coverage"] >= 0.9 - 0.03, row.to_dict()
            # sets may be empty (size 0) when no class clears the threshold -- valid
            # in conformal prediction -- but must never exceed the class count.
            assert 0.0 <= row["avg_set_size"] <= len(CLASSES)
            assert row["n_calib"] == 2000 and row["n_eval"] == 2000

    def test_smaller_alpha_gives_larger_sets(self, tmp_path: Path):
        """Lower alpha (higher target coverage) yields larger sets."""
        df = _make_prediction_df(n=3000, seed=4)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        results = cp.run_evaluate(path, methods=("APS",), alphas=(0.2, 0.01))
        size_lo_cov = results.loc[results["alpha"] == 0.2, "avg_set_size"].iloc[0]
        size_hi_cov = results.loc[results["alpha"] == 0.01, "avg_set_size"].iloc[0]
        assert size_hi_cov >= size_lo_cov


class TestPerClassEvaluation:
    """Per-true-class coverage breakdown."""

    def test_counts_are_consistent(self, tmp_path: Path):
        """Supports sum to n_eval and counts derive the reported rates."""
        df = _make_prediction_df(n=3000, seed=10)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        result = cp.run_evaluate_per_class(path, methods=("APS",), alpha=0.1)

        assert set(result["true_class"]) == set(CLASSES)
        # one calib/eval split -> every method sees the same eval support total
        assert result["support"].sum() == 1500
        covered = result["n_covered"] / result["support"]
        np.testing.assert_allclose(covered, result["coverage"], rtol=0, atol=1e-9)
        sizes = result["set_size_sum"] / result["support"]
        np.testing.assert_allclose(sizes, result["avg_set_size"], rtol=0, atol=1e-9)
        empties = result["n_empty"] / result["support"]
        np.testing.assert_allclose(empties, result["empty_rate"], rtol=0, atol=1e-9)
        assert (result["n_empty"] <= result["support"]).all()

    def test_per_class_aggregates_to_marginal(self, tmp_path: Path):
        """Support-weighted per-class coverage equals the marginal coverage."""
        df = _make_prediction_df(n=3000, seed=11)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        per_class = cp.run_evaluate_per_class(path, methods=("LAC",), alpha=0.1)
        marginal = cp.run_evaluate(path, methods=("LAC",), alphas=(0.1,))

        weighted = per_class["n_covered"].sum() / per_class["support"].sum()
        # TorchCP's coverage_rate returns float32, hence the ~1e-7 slack.
        assert abs(weighted - marginal["empirical_coverage"].iloc[0]) < 1e-6


class TestMondrianFeasibility:
    """Feasibility check from a label distribution."""

    def test_rare_class_flagged_degenerate(self):
        """A class below the alpha floor is flagged degenerate; common ones ok."""
        labels = ["male"] * 1000 + ["female"] * 1000 + ["mixed"] * 8
        result = cp.mondrian_feasibility(labels, alpha=0.1)

        assert result["floor"].iloc[0] == 9  # ceil(1/0.1) - 1
        row = result.set_index("class")
        assert row.loc["mixed", "n_calib"] == 8
        assert not row.loc["mixed", "clears_floor"]
        assert row.loc["mixed", "status"] == "degenerate"
        assert bool(row.loc["male", "clears_floor"]) is True

    def test_counts_mapping_and_projection(self):
        """Accepts a count mapping and projects per-fold via n_splits/calib_frac."""
        # 160 mixed total, 10 folds, half for calibration -> 8 per fold -> degenerate.
        result = cp.mondrian_feasibility(
            {"male": 20000, "female": 20000, "mixed": 160},
            alpha=0.1,
            n_splits=10,
            calib_frac=0.5,
        )
        row = result.set_index("class")
        assert row.loc["mixed", "n_calib"] == 8
        assert not row.loc["mixed", "clears_floor"]
        # plenty of calibration for the majority classes
        assert bool(row.loc["male", "reliable"]) is True

    def test_noisy_status_between_floor_and_reliable(self):
        """A class above the floor but below reliability is flagged noisy."""
        # floor=9, reliable_n=ceil(0.1*0.9/0.05**2)=36 at alpha=0.1, delta=0.05
        result = cp.mondrian_feasibility(
            {"a": 1000, "b": 20}, alpha=0.1, target_delta=0.05
        )
        row = result.set_index("class")
        assert row.loc["b", "status"] == "noisy"
        assert bool(row.loc["b", "clears_floor"]) is True
        assert not row.loc["b", "reliable"]


class TestMondrian:
    """Class-conditional (Mondrian) calibration."""

    def test_mondrian_lifts_rare_class_coverage(self, tmp_path: Path):
        """Class-conditional calibration raises the worst class's coverage."""
        # Imbalanced 4-class problem with one rarer, harder class.
        rng = np.random.default_rng(20)
        n, k = 6000, len(CLASSES)
        weights = np.array([0.45, 0.35, 0.15, 0.05])
        true_idx = rng.choice(k, size=n, p=weights)
        seps = np.array([3.5, 3.5, 2.5, 1.5])  # last class is harder
        logits = rng.normal(size=(n, k))
        logits[np.arange(n), true_idx] += seps[true_idx]
        probs = _softmax(logits)
        df = pd.DataFrame(probs, columns=CLASSES)
        df.insert(0, "Predicted class", [CLASSES[j] for j in probs.argmax(axis=1)])
        df.insert(0, "True class", [CLASSES[j] for j in true_idx])
        df.index = [f"s{i}" for i in range(n)]
        df.index.name = "ID"
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        marginal = cp.run_evaluate_per_class(path, methods=("LAC",), alpha=0.1)
        mondrian = cp.run_evaluate_per_class(
            path, methods=("LAC",), alpha=0.1, mondrian=True
        )

        worst = CLASSES[int(np.argmin(weights))]
        marg_cov = marginal.set_index("true_class").loc[worst, "coverage"]
        mond_cov = mondrian.set_index("true_class").loc[worst, "coverage"]
        # Mondrian targets per-class coverage, so the rare class should improve.
        assert mond_cov >= marg_cov
        # and it should land near / above target for the rare class
        assert mond_cov >= 0.9 - 0.05

    def test_mondrian_flag_recorded(self, tmp_path: Path):
        """run_evaluate tags rows with the mondrian flag and a NaN scalar q_hat."""
        df = _make_prediction_df(n=2000, seed=21)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        result = cp.run_evaluate(path, methods=("APS",), alphas=(0.1,), mondrian=True)
        assert bool(result["mondrian"].iloc[0]) is True
        assert np.isnan(result["q_hat"].iloc[0])  # per-class thresholds, no scalar


class TestApplyMode:
    """Calibrate on labelled CSV, emit sets for an unlabelled CSV."""

    def test_writes_one_csv_per_method(self, tmp_path: Path):
        """Apply mode writes one well-formed set CSV per method."""
        calib = _write_csv(
            _make_prediction_df(n=500, seed=5), tmp_path / "validation_prediction.csv"
        )
        test = _write_csv(
            _make_prediction_df(n=50, seed=6, with_labels=False),
            tmp_path / "test_prediction.csv",
        )
        out_dir = tmp_path / "sets"

        written = cp.run_apply(calib, test, out_dir, methods=("LAC", "APS"), alpha=0.1)

        assert len(written) == 2
        for path in written:
            assert path.exists()
            out = pd.read_csv(path, index_col="ID")
            assert list(out.columns[:3]) == [
                "Predicted class",
                "Prediction set",
                "Set size",
            ]
            # membership columns are the classes, 0/1 valued
            assert list(out.columns[3:]) == CLASSES
            assert out[CLASSES].isin([0, 1]).all().all()
            # Set size equals the membership row sum
            assert (out["Set size"] == out[CLASSES].sum(axis=1)).all()

    def test_mismatched_classes_raises(self, tmp_path: Path):
        """Mismatched calib/test class columns raise ValueError."""
        calib = _write_csv(
            _make_prediction_df(n=100, seed=7), tmp_path / "validation_prediction.csv"
        )
        bad = _make_prediction_df(n=20, seed=8, with_labels=False)
        bad = bad.rename(columns={"assay_A": "assay_Z"})
        bad_path = _write_csv(bad, tmp_path / "test_prediction.csv")

        with pytest.raises(ValueError):
            cp.run_apply(calib, bad_path, tmp_path / "sets", methods=("LAC",))
