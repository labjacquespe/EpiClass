"""Shared helpers for the conformal interpretation marimo apps.

Both apps (CV examination, deployment) are READ-ONLY: they glob the per-sample set CSVs
written by ``precompute`` and join them to metadata and to precomputed UMAP/PCA embeddings.
This module holds everything they share -- readers, the metadata/embedding joins (the
embedding ``ids`` are mixed accessions + md5; the md5sum index is the join bridge, so
non-md5 points drop out naturally), the flag colour map, a reusable embedding scatter, and
the plain-language plot explainers.

Kept free of a top-level ``marimo`` import (only ``explainer`` needs it, lazily) so it is a
normal importable, unit-testable library that the precompute CLI could share without
pulling notebook machinery.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from epiclass.utils.conformal import prediction as cp
from epiclass.utils.conformal.precompute import SETS_DIR_NAME

# Semantic colour map for the per-sample QC flags (clean = good -> empty/disagree = bad),
# as ``rgb(...)`` strings to match the IHEC palette style. ``FLAG_ORDER`` fixes the legend
# / stack order. Keys are exactly ``prediction.FLAG_CATEGORIES``.
FLAG_COLOR_MAP: Dict[str, str] = {
    cp.FLAG_CLEAN: "rgb(44,160,44)",  # green
    cp.FLAG_HEDGE: "rgb(255,170,0)",  # amber
    cp.FLAG_DISAGREE: "rgb(214,39,40)",  # red
    cp.FLAG_EMPTY: "rgb(120,120,120)",  # grey
}
FLAG_ORDER: List[str] = list(cp.FLAG_CATEGORIES)

# String columns of a set CSV that must not be coerced (paired-end TRUE/FALSE -> bool).
_SET_STR_COLS = (
    "fold",
    "True class",
    "Predicted class",
    "Prediction set",
    "flag_category",
)


# --------------------------------------------------------------------------- #
# Readers (glob the precomputed conformal_sets/ artifacts).
# --------------------------------------------------------------------------- #
def _read_set_csv(path: Path) -> pd.DataFrame:
    """Read a per-sample set CSV (ID-indexed), keeping label columns as strings."""
    dtypes = {c: str for c in _SET_STR_COLS}
    return pd.read_csv(path, index_col=0, dtype=dtypes)


def load_examination_sets(
    run_dir: str | Path, method: str = "SAPS", alpha: float = 0.1, mondrian: bool = False
) -> pd.DataFrame:
    """Load a cv-examine per-sample set CSV from ``<run_dir>/conformal_sets/``."""
    tag = "_mondrian" if mondrian else ""
    path = (
        Path(run_dir) / SETS_DIR_NAME / f"cv_examination_{method}{tag}_alpha{alpha}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"No examination CSV at '{path}'. Run precompute --mode cv-examine first."
        )
    return _read_set_csv(path)


def load_deployment_sets(
    data_dir: str | Path, method: str = "SAPS", alpha: float = 0.05
) -> pd.DataFrame:
    """Load a CV+ deployment set CSV from ``<data_dir>/conformal_sets/``."""
    path = Path(data_dir) / SETS_DIR_NAME / f"cv_plus_sets_{method}_alpha{alpha}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No deployment CSV at '{path}'. Run precompute --mode deploy first."
        )
    return _read_set_csv(path)


def load_feasibility(run_dir: str | Path, alpha: float = 0.1) -> Optional[pd.DataFrame]:
    """Load the per-fold Mondrian feasibility sidecar, or ``None`` if absent."""
    path = Path(run_dir) / SETS_DIR_NAME / f"mondrian_feasibility_alpha{alpha}.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# Metadata + embedding joins (md5sum is the bridge).
# --------------------------------------------------------------------------- #
def attach_metadata(sets_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join metadata (md5sum-indexed) onto an ID-indexed set frame.

    Unlike the strict ``MetadataHandler.join_metadata`` this does not raise on unmatched
    IDs -- a flagged training sample may legitimately be absent if metadata versions drift;
    a warning count is printed instead. Overlapping columns keep the set frame's version.
    """
    merged = sets_df.merge(
        metadata_df,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=(None, "_meta"),
    )
    merged.drop(columns=[c for c in merged.columns if c.endswith("_meta")], inplace=True)
    n_missing = (
        merged[metadata_df.columns.intersection(merged.columns)].isna().all(axis=1).sum()
    )
    if n_missing:
        print(f"attach_metadata: {n_missing}/{len(merged)} IDs had no metadata match.")
    return merged


def _merge_embedding(
    ids: Sequence[str], coords: Dict[str, np.ndarray], metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """Join embedding coords to metadata on the md5sum index; drop non-md5 ids.

    The inner join against the md5sum-indexed metadata silently drops embedding points
    keyed by accession (SRX/ERX). The match coverage is recorded in ``df.attrs``.
    """
    emb = pd.DataFrame({"ID": list(ids), **coords})
    merged = emb.merge(metadata_df, left_on="ID", right_index=True, how="inner")
    merged.attrs["n_total"] = len(emb)
    merged.attrs["n_matched"] = len(merged)
    return merged


def load_umap_embedding(path: str | Path, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Load a pickled UMAP embedding (``{"ids", "embedding"}``) joined to metadata."""
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    emb = np.asarray(data["embedding"])
    coords = {f"UMAP {i + 1}": emb[:, i] for i in range(emb.shape[1])}
    return _merge_embedding(data["ids"], coords, metadata_df)


def load_pca_embedding(path: str | Path, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Load a ``.skops`` PCA projection (``X_IPCA_n*.skops``) joined to metadata."""
    # lazy: skops is only needed for PCA embeddings
    import skops.io as sio  # pylint: disable=import-outside-toplevel

    untrusted = sio.get_untrusted_types(file=path)
    data = sio.load(path, trusted=untrusted)
    coords_arr = np.asarray(data["X_ipca"])
    coords = {f"PCA {i + 1}": coords_arr[:, i] for i in range(coords_arr.shape[1])}
    return _merge_embedding(data["file_names"], coords, metadata_df)


def load_embedding(
    path: str | Path, metadata_df: pd.DataFrame, kind: str = "umap"
) -> pd.DataFrame:
    """Dispatch to the UMAP (pickle) or PCA (.skops) loader by ``kind``."""
    if kind == "umap":
        return load_umap_embedding(path, metadata_df)
    if kind == "pca":
        return load_pca_embedding(path, metadata_df)
    raise ValueError(f"Unknown embedding kind '{kind}'. Use 'umap' or 'pca'.")


def match_note(emb_df: pd.DataFrame) -> str:
    """One-line 'N of M matched on md5sum' coverage note from an embedding frame."""
    total = emb_df.attrs.get("n_total", len(emb_df))
    matched = emb_df.attrs.get("n_matched", len(emb_df))
    return f"{matched} of {total} embedding points matched on md5sum."


_SET_JOIN_COLS = (
    "True class",
    "Predicted class",
    "Prediction set",
    "Set size",
    "flag_category",
)


def join_sets_to_embedding(emb_df: pd.DataFrame, sets_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the set/flag columns to an embedding frame by md5sum (inner join)."""
    cols = [c for c in _SET_JOIN_COLS if c in sets_df.columns]
    joined = emb_df.merge(
        sets_df[cols],
        left_on="ID",
        right_index=True,
        how="inner",
        suffixes=(None, "_set"),
    )
    joined.attrs["n_total"] = emb_df.attrs.get("n_total", len(emb_df))
    joined.attrs["n_matched"] = len(joined)
    return joined


# --------------------------------------------------------------------------- #
# Set-derived summaries (computed from a per-sample examination frame).
# --------------------------------------------------------------------------- #
_NON_CLASS_SET_COLS = (
    "fold",
    "True class",
    "Predicted class",
    "Prediction set",
    "Set size",
    "flag_category",
)


def set_classes(sets_df: pd.DataFrame) -> List[str]:
    """The per-class membership columns of a set frame (everything else removed)."""
    return [c for c in sets_df.columns if c not in _NON_CLASS_SET_COLS]


def flag_rate_table(sets_df: pd.DataFrame, group_col: str = "True class") -> pd.DataFrame:
    """Long-form per-group flag counts + proportions (one row per group x flag)."""
    grouped = (
        sets_df.groupby([group_col, "flag_category"]).size().rename("count").reset_index()
    )
    grouped["proportion"] = grouped["count"] / grouped.groupby(group_col)[
        "count"
    ].transform("sum")
    return grouped


def flag_composition_bar(
    sets_df: pd.DataFrame,
    *,
    group_col: str = "True class",
    title: str = "Flag composition per class",
) -> go.Figure:
    """Stacked bar of flag proportions per class (green clean -> grey empty)."""
    tbl = flag_rate_table(sets_df, group_col)
    fig = px.bar(
        tbl,
        x=group_col,
        y="proportion",
        color="flag_category",
        category_orders={"flag_category": FLAG_ORDER},
        color_discrete_map=FLAG_COLOR_MAP,
        template="plotly_white",
        title=title,
    )
    fig.update_layout(barmode="stack", height=500, yaxis_range=[0, 1.0])
    return fig


def per_class_coverage(sets_df: pd.DataFrame) -> pd.DataFrame:
    """Empirical per-class coverage (fraction whose set contains the true class)."""
    classes = set_classes(sets_df)
    membership = sets_df[classes].to_numpy()
    idx = {c: i for i, c in enumerate(classes)}
    true_pos = sets_df["True class"].map(idx).to_numpy()
    covered = membership[np.arange(len(sets_df)), true_pos] == 1
    out = pd.DataFrame(
        {"True class": sets_df["True class"].to_numpy(), "covered": covered}
    )
    return out.groupby("True class")["covered"].mean().rename("coverage").reset_index()


def per_class_coverage_bar(
    sets_df: pd.DataFrame, *, alpha: float, title: str = "Per-class coverage"
) -> go.Figure:
    """Per-class coverage bars with the 1-alpha target line."""
    cov = per_class_coverage(sets_df)
    fig = px.bar(cov, x="True class", y="coverage", template="plotly_white", title=title)
    fig.add_hline(
        y=1 - alpha, line={"dash": "dash"}, annotation_text=f"target {1 - alpha:.2f}"
    )
    fig.update_layout(height=450, yaxis_range=[0, 1.02])
    return fig


# --------------------------------------------------------------------------- #
# Plotting + selection.
# --------------------------------------------------------------------------- #
def build_color_map(df: pd.DataFrame, column: str) -> Dict[str, str]:
    """Stable ``{category: colour}`` map over a column's full set of values."""
    if not column or column not in df.columns:
        return {}
    cats = sorted(df[column].dropna().unique())
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    return {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}


def embedding_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str,
    *,
    title: str = "",
    color_map: Optional[Dict[str, str]] = None,
    category_order: Optional[Sequence[str]] = None,
    continuous: bool = False,
    custom_id: str = "ID",
    hover_cols: Optional[Sequence[str]] = None,
    height: int = 700,
) -> go.Figure:
    """Embedding scatter coloured by ``color_col`` with ``custom_data=[custom_id]``.

    Pass ``continuous=True`` for a numeric colour (e.g. Set size); otherwise a categorical
    colour, using ``color_map`` / ``category_order`` (e.g. ``FLAG_COLOR_MAP`` / ``FLAG_ORDER``
    for ``flag_category``).
    """
    # Base hover columns
    hover = [c for c in (custom_id, color_col) if c in df.columns]

    # Append custom hover columns without creating duplicates
    if hover_cols:
        for col in hover_cols:
            if col in df.columns and col not in hover:
                hover.append(col)

    if continuous:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color_col,
            hover_data=hover,
            custom_data=[custom_id],
            template="plotly_white",
            title=title,
            color_continuous_scale="Turbo",
            render_mode="webgl",
        )
    else:
        kwargs: Dict = {}
        if color_map:
            kwargs["color_discrete_map"] = color_map
        if category_order:
            kwargs["category_orders"] = {color_col: list(category_order)}
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color_col,
            hover_data=hover,
            custom_data=[custom_id],
            template="plotly_white",
            title=title,
            render_mode="webgl",
            **kwargs,
        )
    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(legend={"itemsizing": "constant"}, height=height)
    return fig


def ids_from_selection(selection) -> List[str]:
    """Pull sample ids out of a marimo plotly selection (flat ``ID`` or ``customdata``)."""
    ids: List[str] = []
    for point in selection or []:
        if not isinstance(point, dict):
            continue
        if "ID" in point:
            ids.append(point["ID"])
        elif "customdata" in point:
            cdata = point["customdata"]
            ids.append(cdata[0] if isinstance(cdata, (list, tuple)) else cdata)
    return ids


# --------------------------------------------------------------------------- #
# Plain-language explainers (one per plot), shared so wording is identical.
# --------------------------------------------------------------------------- #
_EXPLAINERS: Dict[str, str] = {
    "cv_intro": "A **conformal prediction set** is the group of labels the model still considers plausible for a sample at confidence 1−α. Each cross-validation sample gets one *honestly*: it is calibrated only against other held-out samples from its own fold, never against itself. We then sort every sample into one of four flags — **clean** (one label, the right one), **hedge** (several labels, unsure), **disagree** (one label, the *wrong* one — a mislabel suspect), and **empty** (the model isn't 1−α-confident about *any* label — an abstention). For mislabel hunting, **disagree** and **hedge** are the signal; empties are mostly low confidence, not outliers (see below).",
    "flag_composition": "Each bar splits a class's samples into the four flags. Tall amber (**hedge**) means the model is often unsure for that class; tall red (**disagree**) means confident-but-wrong singletons — the strongest mislabel suspects; grey (**empty**) means the model abstained (its softmax was too diffuse to clear the bar for any class). Note empties live in *score/confidence* space, not the UMAP's feature space, and the empty rate is roughly α by construction — so a centrally-clustered point can still be empty, and a high empty count is mostly the coverage budget, not novelty. A clean class is almost all green.",
    "coverage": "Coverage is the fraction of samples whose set actually contains the true label. By construction it sits near the 1−α target *overall*; reading it **per class** reveals classes the guarantee is quietly failing (usually via empty sets — i.e. the model is systematically under-confident on that class).",
    "mondrian": "**Marginal** sets share one threshold across all classes; **Mondrian** (class-conditional) uses a separate threshold per class, which protects rare classes from being absorbed by common ones — but it needs enough calibration samples *per class*. This panel shows the per-class coverage/flag-rate under each, so you can see where Mondrian genuinely helps.",
    "mondrian_unavailable": "Mondrian needs at least ⌈1/α⌉−1 calibration samples for *every* class in *every* fold. Classes below that floor fall back to the marginal threshold (named below); their Mondrian result would be degenerate, so trust the marginal numbers for them.",
    "flagged_table": "Every non-clean sample, with its prediction set and metadata. Sort by flag — **disagree** rows are confident mislabel suspects, **empty** rows are outliers, **hedge** rows are genuine ambiguity. Cross-check suspicious rows against the embedding below.",
    "embedding_flag": "The same samples in UMAP/PCA space, coloured by flag. The flag comes from the model's *confidence* (score space), not from position here (feature space) — so don't read **empty** (grey) points as geometric outliers; a centrally-clustered point is empty when the model was simply under-confident. The mislabel signal is a **disagree** (red) or **hedge** (amber) point sitting *inside another class's cluster* — box- or lasso-select those to inspect the rows.",
    "deploy_intro": "Each new sample gets a **set** of plausible labels from the 10 fold-models pooled together (CV+). A single-label set means confident; a multi-label set means the sample looks like several classes; an **empty** set means it resembles nothing the models were trained on — a likely novel / out-of-distribution sample.",
    "setsize_hist": "How many new samples got a confident single label vs a hedged multi-label set vs an empty (rejected) set. A long tail of large or empty sets is your manual-review queue.",
    "embedding_setsize": "New samples in embedding space, coloured by set size. Large-set or empty points that sit away from any tight cluster are the likeliest novel / out-of-distribution cases.",
}


def explainer(key: str):
    """Return a one-paragraph ``mo.md`` explainer for a plot (marimo imported lazily)."""
    # lazy: keeps this module importable without marimo
    import marimo as mo  # pylint: disable=import-outside-toplevel

    if key not in _EXPLAINERS:
        raise KeyError(f"No explainer for '{key}'. Known: {sorted(_EXPLAINERS)}.")
    return mo.md(_EXPLAINERS[key])
