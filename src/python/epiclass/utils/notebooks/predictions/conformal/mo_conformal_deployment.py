# Conformal deployment explorer — prediction sets for NEW data via CV+.
#
# READ-ONLY consumer of the per-sample CV+ sets written by
# `python -m epiclass.utils.conformal.precompute --mode deploy --run-dir <run> --new-data-dir <dir>`.
# New data is unlabelled, so there is no flag/true class: a single-label set means
# confident, a multi-label set means the sample looks like several classes, an empty set
# means it resembles nothing the models saw (likely out-of-distribution). The app shows the
# set-size distribution, a browsable table, a per-predicted-class breakdown, and the samples
# in UMAP/PCA space coloured by set size. Every plot carries a plain-language explainer.
#
# All logic lives in epiclass.utils.conformal (prediction / app_support); this notebook
# only drives it.
#
# File-wide pylint disables. Kept as a header comment (above `import marimo`) so marimo
# preserves it on save; an in-cell disable only scopes to that one cell.
# pylint: disable=missing-module-docstring, missing-function-docstring, function-redefined
# pylint: disable=import-error, import-outside-toplevel, reimported
# pylint: disable=redefined-outer-name, use-dict-literal, too-many-lines, duplicate-code
# pylint: disable=unused-import, unused-argument, unused-variable, too-many-branches
# Structural to marimo's notebook format (cells are functions that return/display):
# pylint: disable=useless-return, pointless-statement, expression-not-assigned
# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    from epiclass.utils.conformal import app_support as aps, prediction as cp
    from epiclass.utils.notebooks.paper.paper_utilities import MetadataHandler

    return Path, MetadataHandler, aps, cp, mo, np, pd, px


@app.cell
def _(Path):
    paper_dir = Path.home() / "Projects/epiclass/output/paper"
    umap_dir = paper_dir.parent / "umap" / "data"
    # Default deployment directory: edit to wherever the new-data CV+ sets were written.
    default_data_dir = paper_dir / "data/deployment"
    return default_data_dir, paper_dir, umap_dir


@app.cell
def _(MetadataHandler, paper_dir):
    metadata_df = MetadataHandler(paper_dir).load_metadata_df("v2")
    return (metadata_df,)


@app.cell
def _(aps, mo):
    mo.vstack(
        [
            mo.md("# Conformal prediction — deployment explorer"),
            aps.explainer("deploy_intro"),
        ]
    )
    return


@app.cell
def _(cp, default_data_dir, mo):
    data_dir_ui = mo.ui.text(
        value=str(default_data_dir), label="Deployment data directory", full_width=True
    )
    method_dd = mo.ui.dropdown(
        options=list(cp.DEFAULT_METHODS), value="SAPS", label="Method"
    )
    alpha_dd = mo.ui.dropdown(
        options=list(cp.REPORT_ALPHAS), value=cp.REPORT_ALPHAS[0], label="alpha"
    )
    mo.vstack([data_dir_ui, mo.hstack([method_dd, alpha_dd], justify="start", gap=2)])
    return alpha_dd, data_dir_ui, method_dd


@app.cell
def _(alpha_dd, aps, data_dir_ui, method_dd, mo):
    sets_df = None
    try:
        sets_df = aps.load_deployment_sets(
            data_dir_ui.value, method=method_dd.value, alpha=float(alpha_dd.value)
        )
        _msg = mo.md(
            f"Loaded **{len(sets_df)}** samples ({method_dd.value}, α={alpha_dd.value})."
        )
    except FileNotFoundError as _err:
        _msg = mo.md(f"⚠️ {_err}")
    _msg
    return (sets_df,)


@app.cell
def _(aps, mo, px, sets_df):
    if sets_df is None:
        _view = mo.md("")
    else:
        _fig = px.histogram(
            sets_df, x="Set size", template="plotly_white", title="Prediction-set sizes"
        )
        _fig.update_layout(height=400, bargap=0.1)
        _view = mo.vstack(
            [
                mo.md("## Set-size distribution"),
                aps.explainer("setsize_hist"),
                mo.ui.plotly(_fig),
            ]
        )
    _view
    return


@app.cell
def _(mo, sets_df):
    _max = int(sets_df["Set size"].max()) if sets_df is not None else 1
    size_filter = mo.ui.range_slider(
        start=0,
        stop=max(_max, 1),
        step=1,
        value=[0, max(_max, 1)],
        label="Set size range",
        show_value=True,
    )
    size_filter
    return (size_filter,)


@app.cell
def _(aps, metadata_df, mo, sets_df, size_filter):
    if sets_df is None:
        _view = mo.md("")
    else:
        _lo, _hi = size_filter.value
        _sub = sets_df[sets_df["Set size"].between(_lo, _hi)]
        _cols = ["Predicted class", "Prediction set", "Set size"]
        _table = aps.attach_metadata(_sub[_cols], metadata_df)
        _view = mo.vstack(
            [mo.md(f"## Prediction sets ({len(_sub)})"), mo.ui.table(_table)]
        )
    _view
    return


@app.cell
def _(mo, px, sets_df):
    if sets_df is None:
        _view = mo.md("")
    else:
        _agg = (
            sets_df.groupby("Predicted class")
            .agg(count=("Set size", "size"), mean_set_size=("Set size", "mean"))
            .reset_index()
        )
        _fig = px.bar(
            _agg,
            x="Predicted class",
            y="count",
            color="mean_set_size",
            template="plotly_white",
            title="Samples per predicted class (colour = mean set size)",
        )
        _fig.update_layout(height=450)
        _view = mo.vstack([mo.md("## Per-predicted-class breakdown"), mo.ui.plotly(_fig)])
    _view
    return


@app.cell
def _(mo, umap_dir):
    embed_dir_ui = mo.ui.text(
        value=str(umap_dir), label="Embeddings directory", full_width=True
    )
    embed_kind_ui = mo.ui.radio(
        options=["umap", "pca"], value="umap", label="Embedding type"
    )
    mo.hstack([embed_dir_ui, embed_kind_ui], justify="start", gap=2)
    return embed_dir_ui, embed_kind_ui


@app.cell
def _(Path, embed_dir_ui, embed_kind_ui, mo):
    _dir = Path(embed_dir_ui.value)
    _pattern = "embedding*2D*.pkl" if embed_kind_ui.value == "umap" else "*.skops"
    _options = sorted(p.name for p in _dir.glob(_pattern)) if _dir.exists() else []
    embed_file_ui = mo.ui.dropdown(
        options=_options, value=_options[0] if _options else None, label="Embedding file"
    )
    color_ui = mo.ui.dropdown(
        options=["Set size", "Predicted class"], value="Set size", label="Colour by"
    )
    mo.hstack([embed_file_ui, color_ui], justify="start", gap=2)
    return color_ui, embed_file_ui


@app.cell
def _(
    Path,
    aps,
    color_ui,
    embed_dir_ui,
    embed_file_ui,
    embed_kind_ui,
    metadata_df,
    mo,
    sets_df,
):
    embed_join = None
    embedding_plot = None
    if sets_df is None or not embed_file_ui.value:
        _view = mo.md("_(Load deployment sets and pick an embedding file to plot.)_")
    else:
        _emb = aps.load_embedding(
            Path(embed_dir_ui.value) / embed_file_ui.value,
            metadata_df,
            embed_kind_ui.value,
        )
        embed_join = aps.join_sets_to_embedding(_emb, sets_df)
        _axes = [c for c in embed_join.columns if c.startswith(("UMAP ", "PCA "))]
        _continuous = color_ui.value == "Set size"
        _fig = aps.embedding_scatter(
            embed_join,
            _axes[0],
            _axes[1],
            color_ui.value,
            title=embed_file_ui.value,
            color_map=None
            if _continuous
            else aps.build_color_map(embed_join, color_ui.value),
            continuous=_continuous,
        )
        embedding_plot = mo.ui.plotly(_fig)
        _view = mo.vstack(
            [
                mo.md("## New samples in embedding space"),
                aps.explainer("embedding_setsize"),
                mo.md(aps.match_note(embed_join)),
                embedding_plot,
            ]
        )
    _view
    return embed_join, embedding_plot


@app.cell
def _(aps, embed_join, embedding_plot, mo):
    _sel = aps.ids_from_selection(getattr(embedding_plot, "value", None))
    if embed_join is not None and _sel:
        _rows = embed_join[embed_join["ID"].isin(_sel)]
        _view = mo.vstack([mo.md(f"**{len(_rows)} selected**"), mo.ui.table(_rows)])
    else:
        _view = mo.md("_Box/lasso-select points above to inspect them here._")
    _view
    return


if __name__ == "__main__":
    app.run()
