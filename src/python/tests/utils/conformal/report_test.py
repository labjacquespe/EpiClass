"""Tests for the conformal report summary figures and their on-disk export."""
# Synthetic prediction-CSV construction is shared boilerplate with the sibling
# conformal_prediction_test; that overlap is expected in test fixtures.
# pylint: disable=duplicate-code
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torchcp")
pytest.importorskip("plotly")

# pylint: disable=wrong-import-position
import plotly.graph_objects as go

from epiclass.utils.conformal import prediction as cp, report as cpr

CLASSES = ["assay_A", "assay_B", "assay_C", "assay_D"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _make_run(tmp_path: Path, n_folds: int = 2, n: int = 1200) -> Path:
    """Write a small multi-fold run (split*/validation_prediction.csv); return the run dir."""
    rng = np.random.default_rng(0)
    for fold in range(n_folds):
        true_idx = rng.integers(0, len(CLASSES), size=n)
        logits = rng.normal(size=(n, len(CLASSES)))
        logits[np.arange(n), true_idx] += 3.0
        probs = _softmax(logits)
        df = pd.DataFrame(probs, columns=CLASSES)
        df.insert(0, "Predicted class", [CLASSES[j] for j in probs.argmax(axis=1)])
        df.insert(0, "True class", [CLASSES[j] for j in true_idx])
        df.index = [f"s{fold}_{i}" for i in range(n)]
        df.index.name = "ID"
        fold_dir = tmp_path / f"split{fold}"
        fold_dir.mkdir()
        df.to_csv(fold_dir / "validation_prediction.csv", index_label="ID")
    return tmp_path


class TestBuildSummaryFigures:
    """Figure construction from a cached per-fold report."""

    def test_returns_named_figures(self, tmp_path: Path):
        """Every expected figure is built and is a plotly Figure."""
        report = cp.run_report(_make_run(tmp_path))
        figs = cpr.build_summary_figures(report, alpha=0.1)

        expected = {
            "per_class_coverage",
            "per_class_set_size",
            "per_class_empty_rate",
            "hparam_sensitivity_setsize",
            "hparam_sensitivity_range",
            "flag_composition",
        }
        assert set(figs) == expected
        assert all(isinstance(fig, go.Figure) for fig in figs.values())

    def test_flag_composition_rates_sum_to_flag_rate(self, tmp_path: Path):
        """The three flag segments equal (1 - clean) of each per-class group."""
        report = cp.run_report(_make_run(tmp_path))
        agg = cp.aggregate_per_class(
            report, group_cols=["method", "alpha", "combo", "true_class"]
        )
        row = agg[agg["alpha"] == 0.1].iloc[0]
        flag_rate = (row["n_multi"] + row["n_singleton_wrong"] + row["n_empty"]) / row[
            "support"
        ]
        clean_rate = row["n_singleton_correct"] / row["support"]
        assert abs(flag_rate + clean_rate - 1.0) < 1e-9
        # the builder uses exactly these three count columns
        assert set(cpr.FLAG_CATEGORIES.values()) == {
            "n_multi",
            "n_singleton_wrong",
            "n_empty",
        }

    def test_unknown_alpha_raises(self, tmp_path: Path):
        """An alpha absent from the cached report is rejected."""
        report = cp.run_report(_make_run(tmp_path))
        with pytest.raises(ValueError):
            cpr.build_summary_figures(report, alpha=0.123)


class TestSaveSummaryFigures:
    """On-disk export next to the split folders."""

    def test_writes_into_run_sibling_folder(self, tmp_path: Path):
        """Figures land in run_dir/conformal_report/, a sibling of the split folders."""
        run_dir = _make_run(tmp_path)
        out = cpr.save_summary_figures(run_dir, alpha=0.1, fmt="html")

        assert out == run_dir / cpr.REPORT_DIR_NAME
        assert out.parent == run_dir  # sits next to split0/split1
        written = sorted(p.name for p in out.glob("*.html"))
        assert written == [
            "flag_composition_alpha0.10.html",
            "hparam_sensitivity_range_alpha0.10.html",
            "hparam_sensitivity_setsize_alpha0.10.html",
            "per_class_coverage_alpha0.10.html",
            "per_class_empty_rate_alpha0.10.html",
            "per_class_set_size_alpha0.10.html",
        ]

    def test_png_export(self, tmp_path: Path):
        """PNG export via kaleido writes non-empty image files."""
        pytest.importorskip("kaleido")
        run_dir = _make_run(tmp_path)
        out = cpr.save_summary_figures(run_dir, alpha=0.1, fmt="png")
        pngs = list(out.glob("*.png"))
        assert len(pngs) == 6
        assert all(p.stat().st_size > 0 for p in pngs)

    def test_no_folds_raises(self, tmp_path: Path):
        """A directory with no fold CSVs raises rather than writing an empty folder."""
        with pytest.raises(ValueError):
            cpr.save_summary_figures(tmp_path / "empty", alpha=0.1)
