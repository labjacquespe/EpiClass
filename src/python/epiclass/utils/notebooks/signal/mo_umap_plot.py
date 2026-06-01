# Plot UMAP embeddings for various datasets.
#
# File-wide pylint disables. Kept as a header comment (above `import marimo`) so
# marimo preserves it on save; an in-cell disable only scopes to that one cell.
# pylint: disable=missing-module-docstring, missing-function-docstring, function-redefined
# pylint: disable=import-error, import-outside-toplevel, reimported
# pylint: disable=redefined-outer-name, use-dict-literal, too-many-lines
# pylint: disable=unused-import, unused-argument, unused-variable, too-many-branches
# Structural to marimo's notebook format (cells are functions that return/display):
# pylint: disable=useless-return, pointless-statement, expression-not-assigned
# pylint: disable=too-many-positional-arguments, too-many-arguments
import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    return


@app.cell
def _():
    from collections import Counter
    from pathlib import Path
    from typing import Dict, Optional, Tuple

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from IPython.display import display
    from plotly.subplots import make_subplots

    from epiclass.core.metadata import Metadata
    from epiclass.utils.notebooks.paper.paper_utilities import (
        ASSAY,
        ASSAY_MERGE_DICT,
        ASSAY_ORDER,
        IHECColorMap,
        MetadataHandler,
    )

    return (
        ASSAY_ORDER,
        Counter,
        IHECColorMap,
        MetadataHandler,
        Path,
        mo,
        np,
        pd,
        px,
    )


@app.cell
def _(ASSAY_ORDER):
    CORE7_ASSAYS = ASSAY_ORDER[0:7]
    return


@app.cell
def _():
    UMAP = "plot_label"
    return


@app.cell
def _(Path):
    base_dir = Path.home() / "Projects/epiclass/output/paper"
    base_data_dir = base_dir / "data"
    base_fig_dir = base_dir / "figures"
    paper_dir = base_dir

    if not base_fig_dir.exists():
        raise FileNotFoundError(f"Directory {base_fig_dir} does not exist.")

    umap_dir = base_dir.parent / "umap" / "data"
    if not umap_dir.exists():
        raise FileNotFoundError(f"Directory {umap_dir} does not exist.")
    return base_fig_dir, paper_dir, umap_dir


@app.cell
def _(MetadataHandler, paper_dir):
    metadata_handler = MetadataHandler(paper_dir)
    metadata_v2 = metadata_handler.load_metadata("v2")
    metadata_df = metadata_v2.to_df()
    return (metadata_df,)


@app.cell
def _(metadata_df):
    metadata_df["id"] = metadata_df.index
    return


@app.cell
def _(IHECColorMap, base_fig_dir):
    IHECColorMap_1 = IHECColorMap(base_fig_dir)
    assay_colors = IHECColorMap_1.assay_color_map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Loading current EpiATLAS content
    """
    )
    return


@app.cell
def _(Path):
    download_dir = Path.home() / "Downloads"
    filepath = download_dir / "EpiATLAS_observed_2026-05.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    with open(filepath, "r", encoding="utf-8") as f:
        urls = f.read().splitlines()
    return (urls,)


@app.cell
def _(urls):
    pairs = [url.lower().split("/")[-3:-1] for url in urls if "readme" not in url]
    return (pairs,)


@app.cell
def _(exp_remapper, pairs):
    experiments = [
        f"{epirr}__{exp_remapper[assay.lower()] if assay in exp_remapper else assay}".lower()
        for epirr, assay in pairs
    ]
    return (experiments,)


@app.cell
def _(experiments):
    exp_assays = [experiment.split("__")[1].lower() for experiment in experiments]
    return (exp_assays,)


@app.cell
def _(Counter, exp_assays):
    Counter(exp_assays)
    return


@app.cell
def _(experiments):
    experiments
    return


@app.cell
def _():
    exp_remapper = {
        "total-rna-seq": "rna_seq",
        "mrna-seq": "mrna_seq",
        "wgbs": "wgbs-standard",
        "pbat": "wgbs-pbat",
    }
    return (exp_remapper,)


@app.cell
def _(metadata_df):
    metadata_df["experiment_id"] = metadata_df.apply(
        lambda row: f"{row['epirr_id_without_version']}__{row['assay_epiclass']}".lower(),
        axis=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Compare dfreeze v2 vs current
    """
    )
    return


@app.cell
def _(experiments, metadata_df):
    dfreeze_exp_ids = set(metadata_df["experiment_id"].unique())
    current_exp_ids = set(experiments)

    # in current, missing from dfreeze
    missing_exp_ids = current_exp_ids - dfreeze_exp_ids
    if missing_exp_ids:
        print(
            f"The following experiment IDs are missing from the metadata (N={len(missing_exp_ids)}):"
        )
        for _exp_id in sorted(missing_exp_ids):
            print(_exp_id)
    else:
        print("All experiment IDs are present in the metadata.")
    return current_exp_ids, dfreeze_exp_ids


@app.cell
def _(current_exp_ids, dfreeze_exp_ids):
    # in dfreeze, missing from current
    missing_in_current = dfreeze_exp_ids - current_exp_ids
    if missing_in_current:
        print(
            f"The following experiment IDs are present in the metadata but missing from the current list (N={len(missing_in_current)}):"
        )
        for _exp_id in sorted(missing_in_current):
            print(_exp_id)
    else:
        print("All experiment IDs in the metadata are present in the current list.")
    return


@app.cell
def _(experiments, metadata_df):
    metadata_df_filtered = metadata_df[metadata_df["experiment_id"].isin(experiments)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Interactive UMAP explorer

    Pick an embedding file, choose a metadata column to color by, then
    box/lasso-select points on the plot to inspect them in the table below.
    """
    )
    return


@app.cell
def _(mo, umap_dir):
    embeddings_dir_ui = mo.ui.text(
        value=str(umap_dir / "C-A_epiatlas"),
        label="Embeddings directory",
        full_width=True,
    )
    embeddings_dir_ui
    return (embeddings_dir_ui,)


@app.cell
def _(px):
    def build_color_map(df, column):
        """Stable {category: color} map from the FULL dataframe.

        Built from explorer_df so every category keeps the same color across
        both plots and across selection subsets, regardless of which categories
        happen to be present in a given view.
        """
        if not column or column not in df.columns:
            return {}
        _cats = sorted(df[column].dropna().unique())
        _palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        return {cat: _palette[i % len(_palette)] for i, cat in enumerate(_cats)}

    return (build_color_map,)


@app.cell
def _(Path, embedding_file_ui, embeddings_dir_ui, metadata_df, np, pd, pickle):
    def _load_embedding_df(file_path, metadata_df, id_column="id"):
        """Load a pickled UMAP embedding and merge it with metadata on sample id."""
        with open(file_path, "rb") as f:
            _data = pickle.load(f)
        _emb = np.asarray(_data["embedding"])
        _coords = {f"UMAP {i + 1}": _emb[:, i] for i in range(_emb.shape[1])}
        _df = pd.DataFrame({"ids": _data["ids"], **_coords})
        return _df.merge(metadata_df, left_on="ids", right_on=id_column, how="inner")

    if embedding_file_ui.value:
        explorer_df = _load_embedding_df(
            Path(embeddings_dir_ui.value) / embedding_file_ui.value,
            metadata_df,
        )
    else:
        explorer_df = pd.DataFrame()
    explorer_df
    return (explorer_df,)


@app.cell
def _(Path, embeddings_dir_ui, mo):
    _dir = Path(embeddings_dir_ui.value)
    _options = (
        sorted(p.name for p in _dir.glob("embedding*2D*.pkl")) if _dir.exists() else []
    )
    embedding_file_ui = mo.ui.dropdown(
        options=_options,
        value=_options[0] if _options else None,
        label="Embedding file",
    )
    return (embedding_file_ui,)


@app.cell
def _(embedding_file_ui, explorer_df, mo):
    _umap_cols = [c for c in explorer_df.columns if c.startswith("UMAP ")]
    _meta_cols = [c for c in explorer_df.columns if c not in _umap_cols and c != "ids"]
    _default_color = "assay_type"
    color_by_ui = mo.ui.dropdown(
        options=_meta_cols,
        value=_default_color
        if _default_color in _meta_cols
        else (_meta_cols[0] if _meta_cols else None),
        label="Color by",
    )
    x_axis_ui = mo.ui.dropdown(
        options=_umap_cols,
        value=_umap_cols[0] if _umap_cols else None,
        label="X axis",
    )
    y_axis_ui = mo.ui.dropdown(
        options=_umap_cols,
        value=_umap_cols[1] if len(_umap_cols) > 1 else None,
        label="Y axis",
    )
    mo.hstack(
        [embedding_file_ui, color_by_ui, x_axis_ui, y_axis_ui],
        justify="start",
        gap=2,
    )
    return color_by_ui, x_axis_ui, y_axis_ui


@app.cell
def _(
    build_color_map,
    color_by_ui,
    embedding_file_ui,
    explorer_df,
    mo,
    px,
    x_axis_ui,
    y_axis_ui,
):
    if explorer_df.empty or not (
        color_by_ui.value and x_axis_ui.value and y_axis_ui.value
    ):
        umap_plot = mo.md("⚠️ No data loaded, or color/axis selection is empty.")
    else:
        _color_map = build_color_map(explorer_df, color_by_ui.value)
        _fig = px.scatter(
            explorer_df,
            x=x_axis_ui.value,
            y=y_axis_ui.value,
            color=color_by_ui.value,
            hover_data=["ids", color_by_ui.value],
            custom_data=["ids"],
            template="plotly_white",
            title=embedding_file_ui.value,
            category_orders={
                color_by_ui.value: sorted(
                    explorer_df[color_by_ui.value].dropna().unique()
                )
            },
            color_discrete_map=_color_map,
            render_mode="webgl",
        )
        _fig.update_traces(marker={"size": 4, "opacity": 0.8})
        _fig.update_layout(legend={"itemsizing": "constant"}, height=700)
        umap_plot = mo.ui.plotly(_fig)
    umap_plot
    return (umap_plot,)


@app.cell
def _(explorer_df, mo, pd, umap_plot):
    def _ids_from_selection(selection):
        """Pull sample ids out of a marimo plotly selection, whatever its shape.

        Each selected point may expose the id either as a flat ``ids`` field or
        inside ``customdata`` (``custom_data=["ids"]`` -> ``["<id>"]``).
        """
        ids = []
        for _pt in selection or []:
            if not isinstance(_pt, dict):
                continue
            if "ids" in _pt:
                ids.append(_pt["ids"])
            elif "customdata" in _pt:
                _cd = _pt["customdata"]
                ids.append(_cd[0] if isinstance(_cd, (list, tuple)) else _cd)
        return ids

    _selection = getattr(umap_plot, "value", None)
    _ids = _ids_from_selection(_selection)
    if _ids:
        # Look the full rows up in explorer_df so we get all metadata + UMAP 1/2
        # and none of the plotly internals (curveNumber/pointNumber/x/y).
        selected_points = explorer_df[explorer_df["ids"].isin(_ids)]
        _view = mo.vstack(
            [
                mo.md(f"**{len(selected_points)} selected point(s)**"),
                mo.ui.table(selected_points),
            ]
        )
    else:
        selected_points = pd.DataFrame()
        _view = mo.md(
            "No points selected: Use box- or lasso-select on the plot to inspect points here."
        )
    _view
    return (selected_points,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Selection re-projection

    Plot only the points selected above, on any embedding you pick. Useful to see where a cluster from one embedding lands in another (e.g. standard vs densMAP).
    """
    )
    return


@app.cell
def _(
    Path,
    embedding_file_ui_2,
    embeddings_dir_ui,
    metadata_df,
    np,
    pd,
    pickle,
    selected_points,
):
    def _load_embedding_df_2(file_path, metadata_df, id_column="id"):
        """Load a pickled UMAP embedding and merge it with metadata on sample id."""
        with open(file_path, "rb") as f:
            _data = pickle.load(f)
        _emb = np.asarray(_data["embedding"])
        _coords = {f"UMAP {i + 1}": _emb[:, i] for i in range(_emb.shape[1])}
        _df = pd.DataFrame({"ids": _data["ids"], **_coords})
        return _df.merge(metadata_df, left_on="ids", right_on=id_column, how="inner")

    # Reuse the ids resolved by the first plot's selection cell.
    _selected_ids = selected_points["ids"].tolist() if not selected_points.empty else []
    if _selected_ids and embedding_file_ui_2.value:
        _full = _load_embedding_df_2(
            Path(embeddings_dir_ui.value) / embedding_file_ui_2.value,
            metadata_df,
        )
        selection_df = _full[_full["ids"].isin(_selected_ids)]
    else:
        selection_df = pd.DataFrame()
    # selection_df
    return (selection_df,)


@app.cell
def _(Path, embeddings_dir_ui, mo):
    _dir = Path(embeddings_dir_ui.value)
    _options = (
        sorted(p.name for p in _dir.glob("embedding*2D*.pkl")) if _dir.exists() else []
    )
    embedding_file_ui_2 = mo.ui.dropdown(
        options=_options,
        value=_options[0] if _options else None,
        label="Embedding file (selection re-projection)",
    )
    return (embedding_file_ui_2,)


@app.cell
def _(embedding_file_ui_2, mo, selection_df):
    _umap_cols = [c for c in selection_df.columns if c.startswith("UMAP ")]
    _meta_cols = [c for c in selection_df.columns if c not in _umap_cols and c != "ids"]
    _default_color = "assay_type"
    sync_color_ui = mo.ui.dropdown(
        options=["Sync with first plot", "Independent"],
        value="Sync with first plot",
        label="Color source",
    )
    color_by_ui_2 = mo.ui.dropdown(
        options=_meta_cols,
        value=_default_color
        if _default_color in _meta_cols
        else (_meta_cols[0] if _meta_cols else None),
        label="Color by (when independent)",
    )
    x_axis_ui_2 = mo.ui.dropdown(
        options=_umap_cols,
        value=_umap_cols[0] if _umap_cols else None,
        label="X axis",
    )
    y_axis_ui_2 = mo.ui.dropdown(
        options=_umap_cols,
        value=_umap_cols[1] if len(_umap_cols) > 1 else None,
        label="Y axis",
    )
    mo.hstack(
        [embedding_file_ui_2, sync_color_ui, color_by_ui_2, x_axis_ui_2, y_axis_ui_2],
        justify="start",
        gap=2,
    )
    return color_by_ui_2, sync_color_ui, x_axis_ui_2, y_axis_ui_2


@app.cell
def _(
    build_color_map,
    color_by_ui,
    color_by_ui_2,
    embedding_file_ui_2,
    explorer_df,
    mo,
    px,
    selection_df,
    sync_color_ui,
    x_axis_ui_2,
    y_axis_ui_2,
):
    if sync_color_ui.value == "Sync with first plot":
        _color_col = color_by_ui.value
    else:
        _color_col = color_by_ui_2.value
    # Fall back if the chosen column isn't present in the re-projected embedding.
    if _color_col not in selection_df.columns:
        _color_col = color_by_ui_2.value
    if selection_df.empty or not (_color_col and x_axis_ui_2.value and y_axis_ui_2.value):
        selection_plot = mo.md(
            "⚠️ Nothing selected in the first plot, or no embedding/columns chosen."
        )
    else:
        # Build the map from the FULL explorer_df so colors match the first plot
        # and stay stable regardless of which categories are in this selection.
        _color_map = build_color_map(explorer_df, _color_col)
        _fig = px.scatter(
            selection_df,
            x=x_axis_ui_2.value,
            y=y_axis_ui_2.value,
            color=_color_col,
            hover_data=["ids", _color_col],
            custom_data=["ids"],
            template="plotly_white",
            title=f"Selection on {embedding_file_ui_2.value}",
            category_orders={
                _color_col: sorted(selection_df[_color_col].dropna().unique())
            },
            color_discrete_map=_color_map,
            render_mode="webgl",
        )
        _fig.update_traces(marker={"size": 4, "opacity": 0.8})
        _fig.update_layout(legend={"itemsizing": "constant"}, height=700)
        selection_plot = mo.ui.plotly(_fig)
    selection_plot
    return


if __name__ == "__main__":
    app.run()
