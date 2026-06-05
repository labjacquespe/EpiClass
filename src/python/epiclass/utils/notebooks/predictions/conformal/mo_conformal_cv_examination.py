# Conformal CV examination — validate a training dataset and flag imperfect samples.
#
# READ-ONLY consumer of the per-sample sets written by
# `python -m epiclass.utils.conformal.precompute --mode cv-examine --run-dir <run>`.
# Pick a classifier's 10-fold run and an alpha; the app shows the per-class flag
# composition, per-class coverage, the marginal-vs-Mondrian comparison (where Mondrian is
# feasible), a table of every non-clean sample, and the same samples in UMAP/PCA space
# coloured by flag — so mislabel / outlier suspects can be spotted next to their neighbours.
# Default score is marginal SAPS. Every plot carries a plain-language explainer.
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
    import re
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    from epiclass.utils.conformal import app_support as aps, prediction as cp
    from epiclass.utils.notebooks.paper.paper_utilities import MetadataHandler

    return MetadataHandler, Path, aps, cp, mo, pd, px, re


@app.cell
def _(Path):
    paper_dir = Path.home() / "Projects/epiclass/output/paper"
    training_dir = paper_dir / "data/training_results/dfreeze_v2/hg38_100kb_all_none"
    umap_dir = paper_dir.parent / "umap" / "data"

    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    return paper_dir, training_dir, umap_dir


@app.cell
def _(MetadataHandler, paper_dir, re):
    # md5sum-indexed metadata: the bridge that joins set CSVs (md5) to embeddings (mixed).
    metadata_df = MetadataHandler(paper_dir).load_metadata_df("v2")

    to_drop = [
        _col
        for _col in metadata_df.columns
        if re.search("automated|read_len|input|curie|groups|EpiRR", _col)
    ] + [
        "harmonized_donor_id",
        "upload_date",
        "data_file_path",
        "epirr_id_without_version",
    ]
    for _col in to_drop:
        if _col not in metadata_df.columns:
            print(f"Warning: column '{_col}' not found in metadata_df, skipping drop.")
        else:
            metadata_df.drop(columns=_col, inplace=True)
    return (metadata_df,)


@app.cell
def _(metadata_df):
    metadata_df["hover_label"] = (
        metadata_df["epirr_id"]
        + " | "
        + metadata_df["assay_epiclass"]
        + " | "
        + metadata_df["track_type"]
    )
    return


@app.cell
def _(aps, mo):
    mo.vstack(
        [
            mo.md("# Conformal prediction — cross-validation examination"),
            aps.explainer("cv_intro"),
        ]
    )
    return


@app.cell
def _(training_dir):
    # A "run" is any directory holding split*/validation_prediction.csv (rglob keeps the
    # pattern relative; an absolute glob raises NotImplementedError on py3.11).
    run_dirs = {}
    for _split0_csv in sorted(training_dir.rglob("split0/validation_prediction.csv")):
        _run_dir = _split0_csv.parent.parent
        run_dirs[str(_run_dir.relative_to(training_dir))] = _run_dir
    return (run_dirs,)


@app.cell
def _(run_dirs):
    run_dirs[
        "assay_epiclass_1l_3000n/11c/10fold-oversampling"
    ] = "/home/local/USHERBROOKE/rabj2301/Projects/epiclass/output/paper/data/training_results/dfreeze_v2/hg38_100kb_all_none/assay_epiclass_1l_3000n/11c/10fold-oversampling/"
    return


@app.cell
def _(cp, mo, run_dirs):
    _labels = list(run_dirs)
    _default = next((label for label in _labels if "donor_sex" in label), _labels[0])

    classifier_dd = mo.ui.dropdown(
        options=_labels, value=_default, label="Classifier run"
    )
    alpha_dd = mo.ui.dropdown(
        options=list(cp.REPORT_ALPHAS), value=cp.REPORT_ALPHAS[0], label="alpha"
    )
    mondrian_radio = mo.ui.radio(
        options=["marginal", "Mondrian"], value="marginal", label="Calibration"
    )
    mo.hstack([classifier_dd, alpha_dd, mondrian_radio], justify="start", gap=2)
    return alpha_dd, classifier_dd, mondrian_radio


@app.cell
def _(alpha_dd, aps, classifier_dd, mo, mondrian_radio, run_dirs):
    run_dir = run_dirs[classifier_dd.value]
    _alpha = float(alpha_dd.value)
    _want_mondrian = mondrian_radio.value == "Mondrian"

    sets_df = None
    _notes = []
    try:
        sets_df = aps.load_examination_sets(
            run_dir, alpha=_alpha, mondrian=_want_mondrian
        )
        _variant = "Mondrian" if _want_mondrian else "marginal"
        _notes.append(f"Loaded **{len(sets_df)}** samples ({_variant}, α={_alpha}).")
    except FileNotFoundError as _err:
        if _want_mondrian:
            try:
                sets_df = aps.load_examination_sets(run_dir, alpha=_alpha, mondrian=False)
                _notes.append(
                    f"⚠️ No Mondrian file for α={_alpha} (no class cleared the floor); "
                    f"showing **marginal** instead."
                )
            except FileNotFoundError as _err2:
                _notes.append(f"⚠️ {_err2}")
        else:
            _notes.append(f"⚠️ {_err}")
    mo.md("  \n".join(_notes))
    return run_dir, sets_df


@app.cell
def _(aps, mo, sets_df):
    mo.vstack(
        [
            mo.md("## Flag composition per class"),
            aps.explainer("flag_composition"),
            mo.ui.plotly(aps.flag_composition_bar(sets_df))
            if sets_df is not None
            else mo.md(""),
        ]
    )
    return


@app.cell
def _(alpha_dd, aps, mo, sets_df):
    mo.vstack(
        [
            mo.md("## Per-class coverage"),
            aps.explainer("coverage"),
            mo.ui.plotly(aps.per_class_coverage_bar(sets_df, alpha=float(alpha_dd.value)))
            if sets_df is not None
            else mo.md(""),
        ]
    )
    return


@app.cell
def _(alpha_dd, aps, mo, run_dir):
    # Mondrian feasibility sidecar: which classes clear the floor, per fold.
    _feas = aps.load_feasibility(run_dir, alpha=float(alpha_dd.value))
    if _feas is None:
        _view = mo.md("No feasibility sidecar found.")
    else:
        _degenerate = sorted(_feas.loc[~_feas["clears_floor"], "class"].unique())
        _key = "mondrian_unavailable" if _degenerate else "mondrian"
        _msg = (
            f"Classes below the floor in some fold: **{', '.join(_degenerate)}**."
            if _degenerate
            else "Every class clears the floor in every fold — Mondrian is fully usable."
        )
        _view = mo.vstack(
            [
                mo.md("## Mondrian feasibility"),
                aps.explainer(_key),
                mo.md(_msg),
                mo.ui.table(_feas),
            ]
        )
    _view
    return


@app.cell
def _(alpha_dd, aps, run_dir):
    _feas = aps.load_feasibility(run_dir, alpha=float(alpha_dd.value))
    print(_feas["reliable"].value_counts())
    return


@app.cell
def _(alpha_dd, aps, mo, run_dir):
    # Marginal vs Mondrian per-class coverage, side by side (when a Mondrian file exists).
    _alpha = float(alpha_dd.value)
    try:
        _marg = aps.load_examination_sets(run_dir, alpha=_alpha, mondrian=False)
        _mond = aps.load_examination_sets(run_dir, alpha=_alpha, mondrian=True)
        _mc = aps.per_class_coverage(_marg).assign(calibration="marginal")
        _dc = aps.per_class_coverage(_mond).assign(calibration="Mondrian")
        import pandas as _pd
        import plotly.express as _px

        _cmp = _pd.concat([_mc, _dc], ignore_index=True)
        _fig = _px.bar(
            _cmp,
            x="True class",
            y="coverage",
            color="calibration",
            barmode="group",
            template="plotly_white",
            title=f"Marginal vs Mondrian coverage (α={_alpha})",
        )
        _fig.add_hline(y=1 - _alpha, line={"dash": "dash"})
        _fig.update_layout(height=450, yaxis_range=[0, 1.02])
        _view = mo.vstack([mo.md("### Marginal vs Mondrian lift"), mo.ui.plotly(_fig)])
    except FileNotFoundError:
        _view = mo.md("_(No Mondrian file for this α — nothing to compare.)_")
    _view
    return


@app.cell
def _(mo):
    flag_filter_dd = mo.ui.dropdown(
        options=["disagree", "hedge", "empty", "all non-clean"],
        value="disagree",
        label="Show flag",
    )
    flag_filter_dd
    return (flag_filter_dd,)


@app.cell
def _(aps, flag_filter_dd, metadata_df, mo, sets_df):
    if sets_df is None:
        _view = mo.md("")
    else:
        _flagged = sets_df[sets_df["flag_category"] != "clean"]
        if flag_filter_dd.value != "all non-clean":
            _flagged = _flagged[_flagged["flag_category"] == flag_filter_dd.value]
        _cols = [
            "fold",
            "True class",
            "Predicted class",
            "Prediction set",
            "flag_category",
        ]
        _table = aps.attach_metadata(_flagged[_cols], metadata_df)
        _view = mo.vstack(
            [
                mo.md(f"## Flagged samples ({len(_flagged)})"),
                aps.explainer("flagged_table"),
                mo.ui.table(_table),
            ]
        )
    _view
    return


@app.cell
def _(mo, umap_dir):
    embed_dir_ui = mo.ui.text(
        value=str(umap_dir / "C-A_epiatlas"),
        label="Embeddings directory",
        full_width=True,
    )
    embed_kind_ui = mo.ui.radio(
        options=["umap", "pca"], value="umap", label="Embedding type"
    )
    mo.vstack([embed_dir_ui, embed_kind_ui], justify="start", gap=2)
    return embed_dir_ui, embed_kind_ui


@app.cell
def _(px):
    def build_color_map(df, column):
        """Stable {category: color} map from the FULL dataframe so palettes are uniform."""
        if not column or column not in df.columns:
            return {}
        _cats = sorted(df[column].dropna().unique().astype(str))
        _palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        return {cat: _palette[i % len(_palette)] for i, cat in enumerate(_cats)}

    return (build_color_map,)


@app.cell
def _(Path, embed_dir_ui, embed_kind_ui, metadata_df, mo):
    _dir = Path(embed_dir_ui.value)
    _pattern = "embedding*2D*.pkl" if embed_kind_ui.value == "umap" else "*.skops"
    _options = sorted(p.name for p in _dir.glob(_pattern)) if _dir.exists() else []

    embed_file_ui = mo.ui.dropdown(
        options=_options, value=_options[0] if _options else None, label="Embedding file"
    )

    _color_opts = ["flag_category", "Set size"] + list(metadata_df.columns)
    color_ui = mo.ui.dropdown(
        options=_color_opts, value="flag_category", label="Colour by"
    )

    _hover_opts = ["None"] + list(metadata_df.columns)
    hover_ui = mo.ui.dropdown(options=_hover_opts, value="None", label="Hover label")

    mo.hstack([embed_file_ui, color_ui, hover_ui], justify="start", gap=2)
    return color_ui, embed_file_ui, hover_ui


@app.cell
def _(
    Path,
    aps,
    build_color_map,
    color_ui,
    embed_dir_ui,
    embed_file_ui,
    embed_kind_ui,
    hover_ui,
    metadata_df,
    mo,
    pd,
    sets_df,
):
    embed_join = None
    embedding_plot = None
    if sets_df is None or not embed_file_ui.value:
        _view = mo.md("_(Load a run and pick an embedding file to plot.)_")
    else:
        _emb = aps.load_embedding(
            Path(embed_dir_ui.value) / embed_file_ui.value,
            metadata_df,
            embed_kind_ui.value,
        )
        embed_join = aps.join_sets_to_embedding(_emb, sets_df)
        _axes = [c for c in embed_join.columns if c.startswith(("UMAP ", "PCA "))]

        # Determine continuous scaling safely
        _is_numeric = (
            pd.api.types.is_numeric_dtype(embed_join[color_ui.value])
            if color_ui.value in embed_join.columns
            else False
        )
        _continuous = color_ui.value == "Set size" or _is_numeric

        # Define categorical structures required by aps.embedding_scatter
        if color_ui.value == "flag_category":
            _color_map = aps.FLAG_COLOR_MAP
            _cat_order = aps.FLAG_ORDER
        elif not _continuous:
            _color_map = build_color_map(embed_join, color_ui.value)
            _cat_order = sorted(embed_join[color_ui.value].dropna().unique().astype(str))
        else:
            _color_map = None
            _cat_order = None

        _hover_cols = [hover_ui.value] if hover_ui.value != "None" else []

        _fig = aps.embedding_scatter(
            embed_join,
            _axes[0],
            _axes[1],
            color_ui.value,
            title=embed_file_ui.value,
            color_map=_color_map,
            category_order=_cat_order,
            continuous=_continuous,
            hover_cols=_hover_cols,
        )

        embedding_plot = mo.ui.plotly(_fig)
        _view = mo.vstack(
            [
                mo.md("## Uncertain samples in embedding space"),
                aps.explainer("embedding_flag"),
                mo.md(aps.match_note(embed_join)),
                embedding_plot,
            ]
        )
    _view
    return embed_join, embedding_plot


@app.cell
def _(aps, embed_join, embedding_plot, mo):
    # Extracts point indices from Plotly selection events (boxes/lassos)
    sel = aps.ids_from_selection(getattr(embedding_plot, "value", None))

    if embed_join is not None and sel:
        _rows = embed_join[embed_join["ID"].isin(sel)]
        _view = mo.vstack([mo.md(f"**{len(_rows)} selected**"), mo.ui.table(_rows)])
    else:
        _view = mo.md("_Box/lasso-select points above to inspect them here._")
    _view
    return (sel,)


@app.cell
def _(Path, embed_dir_ui, embed_kind_ui, metadata_df, mo):
    _dir = Path(embed_dir_ui.value)
    _pattern = "embedding*2D*.pkl" if embed_kind_ui.value == "umap" else "*.skops"
    _options = sorted(p.name for p in _dir.glob(_pattern)) if _dir.exists() else []

    embed_file_ui_2 = mo.ui.dropdown(
        options=_options,
        value=_options[0] if _options else None,
        label="Embedding file (selection re-projection)",
    )

    _color_opts = ["flag_category", "Set size"] + list(metadata_df.columns)
    sync_color_ui = mo.ui.dropdown(
        options=["Sync with first plot", "Independent"],
        value="Sync with first plot",
        label="Color source",
    )
    color_ui_2 = mo.ui.dropdown(
        options=_color_opts,
        value="flag_category",
        label="Color by (when independent)",
    )

    mo.hstack(
        [embed_file_ui_2, sync_color_ui, color_ui_2],
        justify="start",
        gap=2,
    )
    return color_ui_2, embed_file_ui_2, sync_color_ui


@app.cell
def _(
    Path,
    aps,
    build_color_map,
    color_ui,
    color_ui_2,
    embed_dir_ui,
    embed_file_ui_2,
    embed_kind_ui,
    hover_ui,
    metadata_df,
    mo,
    pd,
    sel,
    sets_df,
    sync_color_ui,
):
    if sets_df is None or not embed_file_ui_2.value or not sel:
        _view2 = mo.md(
            "_(Load a run, select points in the first plot, and pick a second embedding to view re-projection.)_"
        )
    else:
        _emb2 = aps.load_embedding(
            Path(embed_dir_ui.value) / embed_file_ui_2.value,
            metadata_df,
            embed_kind_ui.value,
        )
        _full_merge = aps.join_sets_to_embedding(_emb2, sets_df)

        # Decide which column rules the second plot's coloring
        _color_col = (
            color_ui.value
            if sync_color_ui.value == "Sync with first plot"
            else color_ui_2.value
        )
        if _color_col not in _full_merge.columns:
            _color_col = color_ui_2.value

        _is_numeric2 = (
            pd.api.types.is_numeric_dtype(_full_merge[_color_col])
            if _color_col in _full_merge.columns
            else False
        )
        _continuous2 = _color_col == "Set size" or _is_numeric2

        # Derive color mapping against the ENTIRE space for categorical stability
        if _color_col == "flag_category":
            _color_map2 = aps.FLAG_COLOR_MAP
            _cat_order2 = aps.FLAG_ORDER
        elif not _continuous2:
            _color_map2 = build_color_map(_full_merge, _color_col)
            _cat_order2 = sorted(_full_merge[_color_col].dropna().unique().astype(str))
        else:
            _color_map2 = None
            _cat_order2 = None

        _axes2 = [c for c in _full_merge.columns if c.startswith(("UMAP ", "PCA "))]

        # Propagate identical hover options
        _hover_cols2 = [hover_ui.value] if hover_ui.value != "None" else []

        # Plot the full dataset in the second view
        _fig2 = aps.embedding_scatter(
            _full_merge,
            _axes2[0],
            _axes2[1],
            _color_col,
            title=f"Selection re-projection on {embed_file_ui_2.value}",
            color_map=_color_map2,
            category_order=_cat_order2,
            continuous=_continuous2,
            hover_cols=_hover_cols2,
        )

        # Visually isolate the selected points by making unselected points highly transparent & small
        sel_set = set(sel)
        for trace in _fig2.data:
            if trace.customdata is not None:
                # Custom data holds custom_id ("ID") in its first index
                trace_ids = [cd[0] for cd in trace.customdata]

                # Arrays configuring opacity & size conditionally
                opacities = [0.9 if tid in sel_set else 0.5 for tid in trace_ids]
                sizes = [6 if tid in sel_set else 3 for tid in trace_ids]

                trace.marker.opacity = opacities
                trace.marker.size = sizes

        _view2 = mo.vstack(
            [
                mo.md("## Selection re-projection"),
                mo.ui.plotly(_fig2),
            ]
        )
    _view2
    return


if __name__ == "__main__":
    app.run()
