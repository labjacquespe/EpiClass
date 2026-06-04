"""Tests for post-hoc conformal prediction on EpiClass prediction CSVs."""
# Synthetic prediction-CSV construction is shared boilerplate with the sibling
# conformal_report_test; that overlap is expected in test fixtures.
# pylint: disable=duplicate-code
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torchcp")

# pylint: disable=wrong-import-position
from epiclass.utils.conformal import prediction as cp

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

    def test_true_false_class_names_not_coerced_to_bool(self, tmp_path: Path):
        """A paired-end CSV with TRUE/FALSE class names loads without bool coercion.

        Mirrors the real header ``,True class,Predicted class,FALSE,TRUE``: pandas would
        infer the all-TRUE/FALSE label columns as booleans, so ``True`` would stop matching
        the string class headers ``"TRUE"``/``"FALSE"`` (and the casing differs too). The
        loader must keep the label columns as strings.
        """
        df = pd.DataFrame(
            {
                "True class": ["TRUE", "FALSE", "TRUE"],
                "Predicted class": ["TRUE", "TRUE", "FALSE"],
                "FALSE": [0.1, 0.8, 0.3],
                "TRUE": [0.9, 0.2, 0.7],
            },
            index=["a", "b", "c"],
        )
        df.index.name = "ID"
        path = tmp_path / "validation_prediction.csv"
        df.to_csv(path, encoding="utf8", index_label="ID")

        _, probs, classes, true_idx = cp.load_prediction_csv(path)

        assert classes == ["FALSE", "TRUE"]  # column order = integer encoding
        assert true_idx is not None
        assert true_idx.tolist() == [1, 0, 1]  # TRUE->1, FALSE->0, TRUE->1
        assert probs.shape == (3, 2)

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

    def test_hparams_default_constants_exist(self):
        """The literature defaults are exposed as the documented module constants."""
        assert set(cp.RAPS_DEFAULTS) == {"kreg", "penalty"}
        assert set(cp.SAPS_DEFAULTS) == {"weight"}

    def test_hparams_change_set_size(self, tmp_path: Path):
        """A heavier RAPS penalty reaches the score and shrinks (or holds) set size.

        TorchCP stores the penalty in a name-mangled private attribute, so assert on the
        behaviour (the contract) rather than introspecting internals: more regularization
        cannot enlarge the average set at a fixed alpha.
        """
        df = _make_prediction_df(n=3000, seed=12)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        light = cp.run_evaluate_per_class(
            path, methods=("RAPS",), alpha=0.1, hparams={"penalty": 0.0}
        )
        heavy = cp.run_evaluate_per_class(
            path, methods=("RAPS",), alpha=0.1, hparams={"penalty": 1.0, "kreg": 1}
        )
        light_size = light["set_size_sum"].sum() / light["support"].sum()
        heavy_size = heavy["set_size_sum"].sum() / heavy["support"].sum()
        assert heavy_size <= light_size + 1e-9

    def test_irrelevant_hparams_ignored(self):
        """A hyperparameter for another method is ignored, not an error."""
        # 'weight' is a SAPS knob; RAPS should silently ignore it (single shared grid).
        assert cp.build_score("RAPS", weight=0.9) is not None
        assert cp.build_score("LAC", penalty=0.5, weight=0.1) is not None


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

    def test_shape_counts_partition_support(self, tmp_path: Path):
        """The four set-shape counts partition each class's support exactly."""
        df = _make_prediction_df(n=3000, seed=13)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        result = cp.run_evaluate_per_class(path, methods=("SAPS",), alpha=0.1)

        shape = result[
            ["n_empty", "n_singleton_correct", "n_singleton_wrong", "n_multi"]
        ].sum(axis=1)
        np.testing.assert_array_equal(shape.to_numpy(), result["support"].to_numpy())
        # a correct singleton is, by definition, also a covered set
        assert (result["n_singleton_correct"] <= result["n_covered"]).all()
        # the shape counts survive fold aggregation (still partition the summed support)
        agg = cp.aggregate_per_class(result, group_cols=["method", "true_class"])
        agg_shape = agg[
            ["n_empty", "n_singleton_correct", "n_singleton_wrong", "n_multi"]
        ].sum(axis=1)
        np.testing.assert_array_equal(agg_shape.to_numpy(), agg["support"].to_numpy())

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

    def test_suppress_quantile_warning(self, tmp_path: Path):
        """The context manager silences TorchCP's degenerate-quantile warning.

        A class far below the Mondrian floor makes the per-class quantile exceed 1, so
        TorchCP warns and sets the threshold to inf. The first call confirms the warning
        really fires (documents the degeneracy); the second confirms it is suppressed.
        """
        k = len(CLASSES)
        # Rare class (idx 3) gets ~6 calibration samples at calib_frac=0.5 -> below the
        # alpha=0.1 floor of 9 -> degenerate per-class threshold.
        true_idx = np.array([0] * 240 + [1] * 240 + [2] * 108 + [3] * 12)
        rng = np.random.default_rng(70)
        rng.shuffle(true_idx)
        n = true_idx.size
        logits = rng.normal(size=(n, k))
        logits[np.arange(n), true_idx] += 3.0
        probs = _softmax(logits)
        df = pd.DataFrame(probs, columns=CLASSES)
        df.insert(0, "Predicted class", [CLASSES[j] for j in probs.argmax(axis=1)])
        df.insert(0, "True class", [CLASSES[j] for j in true_idx])
        df.index = [f"s{i}" for i in range(n)]
        df.index.name = "ID"
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cp.run_evaluate_per_class(path, methods=("APS",), alpha=0.1, mondrian=True)
        assert any("quantile exceeds 1" in str(w.message) for w in caught)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with cp.suppress_quantile_warning():
                cp.run_evaluate_per_class(
                    path, methods=("APS",), alpha=0.1, mondrian=True
                )
        assert not any("quantile exceeds 1" in str(w.message) for w in caught)

    def test_mondrian_flag_recorded(self, tmp_path: Path):
        """run_evaluate tags rows with the mondrian flag and a NaN scalar q_hat."""
        df = _make_prediction_df(n=2000, seed=21)
        path = _write_csv(df, tmp_path / "validation_prediction.csv")

        result = cp.run_evaluate(path, methods=("APS",), alphas=(0.1,), mondrian=True)
        assert bool(result["mondrian"].iloc[0]) is True
        assert np.isnan(result["q_hat"].iloc[0])  # per-class thresholds, no scalar


class TestExpandGrid:
    """Cartesian expansion of a hyperparameter grid."""

    def test_expands_cartesian_product(self):
        """A two-axis grid expands to every combination."""
        combos = cp.expand_grid({"kreg": [1, 2], "penalty": [0.0, 0.1]})
        assert len(combos) == 4
        assert {"kreg": 1, "penalty": 0.0} in combos
        assert {"kreg": 2, "penalty": 0.1} in combos

    def test_empty_grid_is_single_default(self):
        """An empty grid yields a single empty (default) combo."""
        assert cp.expand_grid({}) == [{}]


class TestSweepHparams:
    """Hyperparameter sweep over the folds of a CV run."""

    def test_one_row_per_combo_and_class(self, tmp_path: Path):
        """Result carries one row per (combo, true_class) with the swept columns."""
        paths = [
            _write_csv(
                _make_prediction_df(n=1500, seed=30 + i),
                tmp_path / f"split{i}_prediction.csv",
            )
            for i in range(3)
        ]
        grid = {"penalty": [0.0, 0.1], "kreg": [1, 2]}
        result = cp.sweep_hparams(paths, "RAPS", grid, alpha=0.1)

        n_combos = len(cp.expand_grid(grid))
        assert len(result) == n_combos * len(CLASSES)
        assert {"penalty", "kreg", "combo", "method"} <= set(result.columns)
        assert set(result["true_class"]) == set(CLASSES)
        assert (result["method"] == "RAPS").all()
        # rates derive from the aggregated raw counts
        np.testing.assert_allclose(
            result["coverage"], result["n_covered"] / result["support"], atol=1e-9
        )

    def test_default_combo_matches_direct_call(self, tmp_path: Path):
        """A single-point sweep at the defaults equals a direct per-class call."""
        path = _write_csv(
            _make_prediction_df(n=2000, seed=40), tmp_path / "split0_prediction.csv"
        )
        swept = cp.sweep_hparams(
            path, "SAPS", {"weight": [cp.SAPS_DEFAULTS["weight"]]}, alpha=0.1
        )
        direct = cp.run_evaluate_per_class(path, methods=("SAPS",), alpha=0.1)

        swept = swept.set_index("true_class").sort_index()
        direct = direct.set_index("true_class").sort_index()
        np.testing.assert_allclose(
            swept["coverage"].to_numpy(), direct["coverage"].to_numpy(), atol=1e-9
        )
        np.testing.assert_allclose(
            swept["avg_set_size"].to_numpy(),
            direct["avg_set_size"].to_numpy(),
            atol=1e-9,
        )

    def test_accepts_explicit_combo_list(self, tmp_path: Path):
        """A sequence of combo dicts is accepted in place of a grid mapping."""
        path = _write_csv(
            _make_prediction_df(n=1200, seed=41), tmp_path / "split0_prediction.csv"
        )
        result = cp.sweep_hparams(
            path, "RAPS", [{"penalty": 0.0}, {"penalty": 0.5}], alpha=0.1
        )
        assert set(result["combo"]) == {"penalty=0.0", "penalty=0.5"}


class TestFoldReportCache:
    """Per-fold full report and its on-disk cache."""

    def test_compute_covers_methods_alphas_combos(self, tmp_path: Path):
        """compute_fold_report spans every method x alpha x grid combo for the fold."""
        path = _write_csv(
            _make_prediction_df(n=1500, seed=50), tmp_path / "validation_prediction.csv"
        )
        report = cp.compute_fold_report(path)

        assert set(report["alpha"].unique()) == set(cp.REPORT_ALPHAS)
        assert set(report["method"].unique()) == {"LAC", "APS", "RAPS", "SAPS"}
        # one block per (alpha, combo) for each method, one row per class within a block
        raps = report[(report["method"] == "RAPS") & (report["alpha"] == 0.05)]
        assert raps["combo"].nunique() == len(cp.expand_grid(cp.REPORT_GRIDS["RAPS"]))
        lac = report[(report["method"] == "LAC") & (report["alpha"] == 0.05)]
        assert lac["combo"].unique().tolist() == ["default"]

    def test_cache_is_written_and_reused(self, tmp_path: Path):
        """fold_report writes the cache and reuses it unless forced."""
        path = _write_csv(
            _make_prediction_df(n=1500, seed=51), tmp_path / "validation_prediction.csv"
        )
        first = cp.fold_report(path)
        cache = path.parent / cp.FOLD_REPORT_NAME
        assert cache.exists()

        # Overwrite the (still complete) cache with a sentinel; a non-forced call must read
        # it back verbatim -- proving it loaded the cache rather than recomputing.
        sentinel = first.assign(coverage=-1.0)
        sentinel.to_csv(cache, index=False)
        reused = cp.fold_report(path)
        assert (reused["coverage"] == -1.0).all()

        # force=True ignores and overwrites the cache.
        forced = cp.fold_report(path, force=True)
        assert not (forced["coverage"] == -1.0).all()

    def test_incomplete_cache_triggers_recompute(self, tmp_path: Path):
        """A cache missing a canonical alpha is treated as stale and recomputed."""
        path = _write_csv(
            _make_prediction_df(n=1500, seed=52), tmp_path / "validation_prediction.csv"
        )
        full = cp.fold_report(path)
        cache = path.parent / cp.FOLD_REPORT_NAME
        # Drop one canonical alpha -> incomplete -> next call rebuilds the full set.
        full[full["alpha"] != cp.REPORT_ALPHAS[-1]].to_csv(cache, index=False)
        rebuilt = cp.fold_report(path)
        assert set(rebuilt["alpha"].unique()) == set(cp.REPORT_ALPHAS)

    def test_run_report_tags_folds(self, tmp_path: Path):
        """run_report stacks each fold's cached report, tagged with the fold name."""
        for i in range(3):
            fold_dir = tmp_path / f"split{i}"
            fold_dir.mkdir()
            _write_csv(
                _make_prediction_df(n=1200, seed=60 + i),
                fold_dir / "validation_prediction.csv",
            )
        report = cp.run_report(tmp_path)

        assert set(report["fold"].unique()) == {"split0", "split1", "split2"}
        # every split folder got its own cache file
        assert all(
            (tmp_path / f"split{i}" / cp.FOLD_REPORT_NAME).exists() for i in range(3)
        )


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


def _make_cv_plus_csvs(
    tmp_path: Path, k_folds: int = 5, n_cal: int = 400, n_test: int = 800
):
    """Build K fold validation CSVs + K test CSVs of one shared, labelled test set.

    Each fold's calibration samples and the test set scored "under that fold's model"
    are drawn from the same generative process (separation 3.0), so calibration and test
    are exchangeable -- the regime where CV+ coverage should hold. The K test CSVs share
    the same IDs / true labels (the same samples scored by each fold model).
    """
    rng = np.random.default_rng(0)
    k = len(CLASSES)
    test_true = rng.integers(0, k, size=n_test)
    calib_paths, test_paths = [], []
    for fold in range(k_folds):
        calib_paths.append(
            _write_csv(
                _make_prediction_df(n=n_cal, seed=100 + fold),
                tmp_path / f"val_fold{fold}.csv",
            )
        )
        logits = rng.normal(size=(n_test, k))
        logits[np.arange(n_test), test_true] += 3.0
        probs = _softmax(logits)
        tdf = pd.DataFrame(probs, columns=CLASSES)
        tdf.insert(0, "Predicted class", [CLASSES[j] for j in probs.argmax(axis=1)])
        tdf.insert(0, "True class", [CLASSES[j] for j in test_true])
        tdf.index = [f"t{i}" for i in range(n_test)]
        tdf.index.name = "ID"
        test_paths.append(_write_csv(tdf, tmp_path / f"test_fold{fold}.csv"))
    return calib_paths, test_paths, test_true


class TestCVPlus:
    """Cross-conformal (CV+) prediction sets across the folds."""

    def test_membership_matches_pvalue_formula(self):
        """cv_plus_membership equals a direct evaluation of the CV+ p-value rule."""
        rng = np.random.default_rng(1)
        cal = [rng.normal(size=300), rng.normal(size=250)]
        test = [rng.normal(size=(40, 4)), rng.normal(size=(40, 4))]
        alpha = 0.1

        membership = cp.cv_plus_membership(cal, test, alpha)
        n = sum(len(s) for s in cal)
        ge = sum(
            (s_cal[None, None, :] >= s_test[:, :, None]).sum(axis=-1)
            for s_cal, s_test in zip(cal, test)
        )
        expected = ((1 + ge) / (n + 1) > alpha).astype(np.int64)
        np.testing.assert_array_equal(membership, expected)

    def test_coverage_holds(self, tmp_path: Path):
        """Empirical CV+ coverage on exchangeable data meets ~ 1 - alpha."""
        calib, test, _ = _make_cv_plus_csvs(tmp_path)
        _, classes, membership, mean_probs, coverage = cp.cv_plus_sets(
            calib, test, "SAPS", alpha=0.1
        )

        assert classes == CLASSES
        assert membership.shape == (800, len(CLASSES))
        assert mean_probs.shape == (800, len(CLASSES))
        assert coverage is not None
        # CV+ guarantees >= 1 - 2*alpha worst case; empirically ~ 1 - alpha here.
        assert coverage >= 0.86

    def test_pooled_denominator_is_all_folds(self):
        """The CV+ p-value pools every fold: a label kept in *no* fold is excluded.

        With test scores far above all calibration scores, no calibration point is >=
        the test score, so the p-value collapses to 1/(n+1) <= alpha and the label drops
        -- which only holds if the denominator counts all folds' samples.
        """
        cal = [np.zeros(300), np.zeros(250)]  # n = 550
        test = [np.full((1, 4), 10.0), np.full((1, 4), 10.0)]
        membership = cp.cv_plus_membership(cal, test, alpha=0.1)
        assert membership.sum() == 0  # 1/(550+1) < 0.1 -> empty set

    def test_unlabelled_test_gives_no_coverage(self, tmp_path: Path):
        """Apply on unlabelled test CSVs returns membership but coverage None."""
        calib = [
            _write_csv(_make_prediction_df(n=300, seed=200 + f), tmp_path / f"val{f}.csv")
            for f in range(3)
        ]
        test = [
            _write_csv(
                _make_prediction_df(n=60, seed=300 + f, with_labels=False),
                tmp_path / f"test{f}.csv",
            )
            for f in range(3)
        ]
        _, _, membership, _, coverage = cp.cv_plus_sets(calib, test, "SAPS", alpha=0.1)
        assert coverage is None
        assert membership.shape == (60, len(CLASSES))

    def test_apply_writes_one_csv_per_method(self, tmp_path: Path):
        """run_cv_plus_apply writes one well-formed set CSV per method."""
        calib, test, _ = _make_cv_plus_csvs(tmp_path, k_folds=3, n_cal=300, n_test=120)
        written = cp.run_cv_plus_apply(
            calib, test, tmp_path / "sets", methods=("SAPS", "RAPS"), alpha=0.1
        )
        assert len(written) == 2
        for path in written:
            out = pd.read_csv(path, index_col="ID")
            assert list(out.columns[:3]) == [
                "Predicted class",
                "Prediction set",
                "Set size",
            ]
            assert (out["Set size"] == out[CLASSES].sum(axis=1)).all()

    def test_validation_errors(self, tmp_path: Path):
        """Mismatched K, classes, IDs and missing labels all raise."""
        calib, test, _ = _make_cv_plus_csvs(tmp_path, k_folds=3, n_test=100)

        with pytest.raises(ValueError):  # K mismatch
            cp.cv_plus_sets(calib, test[:2], "SAPS")

        with pytest.raises(ValueError):  # test class columns differ
            bad = pd.read_csv(test[0], index_col=0).rename(columns={"assay_A": "Z"})
            bad_path = tmp_path / "bad_test.csv"
            bad.to_csv(bad_path, index_label="ID")
            cp.cv_plus_sets(calib, [bad_path, test[1], test[2]], "SAPS")

        with pytest.raises(ValueError):  # calibration CSV has no labels
            unlabelled = _write_csv(
                _make_prediction_df(n=100, seed=9, with_labels=False),
                tmp_path / "nolabel.csv",
            )
            cp.cv_plus_sets([unlabelled, calib[1], calib[2]], test, "SAPS")
