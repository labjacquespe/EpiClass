"""Tests for the shared conformal app support (readers, joins, summaries, plotting)."""
# Synthetic frame construction overlaps with the sibling conformal test modules.
# pylint: disable=duplicate-code
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torchcp")
pytest.importorskip("plotly")

# pylint: disable=wrong-import-position
import plotly.graph_objects as go

from epiclass.utils.conformal import app_support as aps, prediction as cp

CLASSES = ["a", "b", "c"]


def _sets_frame(ids, true, pred, membership) -> pd.DataFrame:
    df = pd.DataFrame(membership, index=list(ids), columns=CLASSES)
    df.insert(0, "True class", true)
    df.insert(1, "Predicted class", pred)
    df.insert(2, "Set size", np.asarray(membership).sum(axis=1))
    df.insert(
        3,
        "flag_category",
        cp.classify_flags(
            np.asarray(membership), np.array([CLASSES.index(t) for t in true])
        ),
    )
    df.index.name = "ID"
    return df


class TestClassifyFlags:
    """The four flag categories partition every labelled sample."""

    def test_table_driven(self):
        """Each of the four set shapes maps to the expected flag."""
        membership = np.array(
            [
                [1, 0, 0],  # singleton == true -> clean
                [0, 1, 0],  # singleton != true -> disagree
                [1, 1, 0],  # multi -> hedge
                [0, 0, 0],  # empty
            ]
        )
        true_idx = np.array([0, 0, 0, 0])
        assert cp.classify_flags(membership, true_idx) == [
            cp.FLAG_CLEAN,
            cp.FLAG_DISAGREE,
            cp.FLAG_HEDGE,
            cp.FLAG_EMPTY,
        ]


class TestFlagColours:
    """The colour map and order cover exactly the four flag categories."""

    def test_map_covers_categories(self):
        """FLAG_COLOR_MAP keys and FLAG_ORDER match the canonical categories."""
        assert set(aps.FLAG_COLOR_MAP) == set(cp.FLAG_CATEGORIES)
        assert aps.FLAG_ORDER == list(cp.FLAG_CATEGORIES)
        assert aps.FLAG_ORDER[0] == cp.FLAG_CLEAN  # green/good first
        assert aps.FLAG_ORDER[-1] == cp.FLAG_EMPTY


class TestMetadataJoin:
    """attach_metadata left-joins by md5 and tolerates unmatched IDs."""

    def test_unmatched_does_not_raise(self):
        """An ID absent from metadata is kept (NaN metadata) rather than raising."""
        sets = _sets_frame(
            ["m0", "m1", "ghost"],
            ["a", "b", "a"],
            ["a", "b", "a"],
            [[1, 0, 0], [0, 1, 0], [1, 1, 0]],
        )
        meta = pd.DataFrame({"donor": ["d0", "d1"]}, index=["m0", "m1"])
        meta.index.name = "md5sum"
        merged = aps.attach_metadata(sets, meta)
        assert len(merged) == 3  # ghost kept
        assert merged.loc["ghost", "donor"] != merged.loc["ghost", "donor"]  # NaN


class TestEmbeddingJoin:
    """Mixed-id embeddings keep only md5-matching points; the count is surfaced."""

    def test_umap_md5_filtering(self, tmp_path: Path):
        """UMAP loader drops accession (non-md5) ids and records the match coverage."""
        md5_ids = [f"m{i}" for i in range(5)]
        emb_ids = md5_ids + ["SRX1", "SRX2"]  # 2 accessions that will not match
        coords = np.random.default_rng(0).normal(size=(len(emb_ids), 2))
        path = tmp_path / "embedding_standard_2D_nn15.pkl"
        with open(path, "wb") as handle:
            pickle.dump({"ids": emb_ids, "embedding": coords}, handle)

        meta = pd.DataFrame({"donor": list("vwxyz")}, index=md5_ids)
        meta.index.name = "md5sum"
        emb = aps.load_umap_embedding(path, meta)
        assert len(emb) == 5
        assert emb.attrs["n_total"] == 7 and emb.attrs["n_matched"] == 5
        assert "5 of 7" in aps.match_note(emb)
        assert set(emb["ID"]) == set(md5_ids)

    def test_pca_skops_loader(self, tmp_path: Path):
        """PCA .skops loader yields 3 PCA columns joined on md5."""
        sio = pytest.importorskip("skops.io")
        md5_ids = [f"m{i}" for i in range(4)]
        x_ipca = np.random.default_rng(1).normal(size=(4, 3))
        path = tmp_path / "X_IPCA_n4.skops"
        sio.dump({"file_names": md5_ids, "X_ipca": x_ipca}, path)

        meta = pd.DataFrame({"donor": list("wxyz")}, index=md5_ids)
        meta.index.name = "md5sum"
        emb = aps.load_pca_embedding(path, meta)
        pca_cols = [c for c in emb.columns if c.startswith("PCA ")]
        assert pca_cols == ["PCA 1", "PCA 2", "PCA 3"]
        assert set(emb["ID"]) == set(md5_ids)

    def test_join_sets_to_embedding_brings_flags(self, tmp_path: Path):
        """join_sets_to_embedding attaches the flag column by md5."""
        md5_ids = [f"m{i}" for i in range(4)]
        coords = np.random.default_rng(2).normal(size=(4, 2))
        path = tmp_path / "embedding_standard_2D_nn15.pkl"
        with open(path, "wb") as handle:
            pickle.dump({"ids": md5_ids, "embedding": coords}, handle)
        meta = pd.DataFrame({"donor": list("wxyz")}, index=md5_ids)
        meta.index.name = "md5sum"
        sets = _sets_frame(
            md5_ids,
            ["a", "b", "a", "c"],
            ["a", "b", "a", "c"],
            [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]],
        )
        joined = aps.join_sets_to_embedding(aps.load_umap_embedding(path, meta), sets)
        assert "flag_category" in joined.columns
        assert len(joined) == 4


class TestSummaries:
    """Set-derived summary helpers."""

    def test_per_class_coverage(self):
        """Per-class coverage counts the fraction whose set holds the true class."""
        sets = _sets_frame(
            ["m0", "m1", "m2", "m3"],
            ["a", "a", "b", "b"],
            ["a", "a", "b", "b"],
            [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]],  # a: 1/2, b: 1/2
        )
        cov = aps.per_class_coverage(sets).set_index("True class")["coverage"]
        assert cov["a"] == pytest.approx(0.5)
        assert cov["b"] == pytest.approx(0.5)

    def test_flag_rate_proportions_sum_to_one(self):
        """Each class's flag proportions sum to 1."""
        sets = _sets_frame(
            ["m0", "m1", "m2", "m3"],
            ["a", "a", "a", "a"],
            ["a", "b", "a", "a"],
            [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]],
        )
        tbl = aps.flag_rate_table(sets)
        assert tbl.groupby("True class")["proportion"].sum().round(9).eq(1.0).all()

    def test_set_classes_excludes_metadata(self):
        """set_classes returns only the per-class membership columns."""
        sets = _sets_frame(["m0"], ["a"], ["a"], [[1, 0, 0]])
        assert aps.set_classes(sets) == CLASSES


class TestPlotting:
    """Scatter builds a plotly figure carrying the ID custom field."""

    def test_embedding_scatter_figure(self):
        """The scatter is a plotly Figure with one trace per present flag."""
        df = pd.DataFrame(
            {
                "ID": ["m0", "m1", "m2"],
                "UMAP 1": [0.0, 1.0, 2.0],
                "UMAP 2": [0.0, 1.0, 2.0],
                "flag_category": [cp.FLAG_CLEAN, cp.FLAG_HEDGE, cp.FLAG_EMPTY],
            }
        )
        fig = aps.embedding_scatter(
            df,
            "UMAP 1",
            "UMAP 2",
            "flag_category",
            color_map=aps.FLAG_COLOR_MAP,
            category_order=aps.FLAG_ORDER,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # one trace per flag category present

    def test_ids_from_selection_handles_both_shapes(self):
        """Selection ids parse from both the flat ID and customdata forms."""
        selection = [{"ID": "m0"}, {"customdata": ["m1"]}, {"customdata": "m2"}]
        assert aps.ids_from_selection(selection) == ["m0", "m1", "m2"]
