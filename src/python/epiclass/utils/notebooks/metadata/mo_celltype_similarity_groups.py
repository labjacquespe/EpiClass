# Explore the within-cohort cell-type Lin-similarity matrix group by group.
#
# Loads what `evaluate_biospecimen_similarity.ipynb` computed
# (`celltype_similarity_matrix.csv`, a square term x term Lin-similarity matrix over
# `harmonized_sample_ontology_curie`) and, optionally, the `*_sweep_groups.csv`
# produced by `create_similar_celltype_folds.py --sweep`. Lets you pick a similarity
# threshold + a cell-type group and inspect it as a similarity heatmap and an
# interactive node-link graph. Node labels are built by joining any chosen columns
# from a sample-metadata CSV (the IHEC harmonization extended table), keyed on the
# shared `harmonized_sample_ontology_curie` column — so you can annotate each curie
# with as much human-readable context as you want and see *why* a term sits where it
# does (especially why a singleton is a singleton).
#
# Run with the biospecimen_similarity venv:
#   marimo edit src/python/epiclass/utils/notebooks/metadata/mo_celltype_similarity_groups.py
#
# File-wide pylint disables. Kept as a header comment (above `import marimo`) so marimo
# preserves it on save; an in-cell disable only scopes to that one cell.
# pylint: disable=missing-module-docstring, missing-function-docstring, function-redefined
# pylint: disable=import-error, import-outside-toplevel, reimported
# pylint: disable=redefined-outer-name, use-dict-literal, too-many-lines, duplicate-code
# pylint: disable=unused-import, unused-argument, unused-variable, too-many-branches
# pylint: disable=too-many-locals, too-many-statements, invalid-name
# Structural to marimo's notebook format (cells are functions that return/display):
# pylint: disable=useless-return, pointless-statement, expression-not-assigned
# pylint: disable=too-many-positional-arguments, too-many-arguments

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import networkx as nx
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    return Path, go, mo, np, nx, pd, px


@app.cell
def _(mo):
    mo.md(
        """
    # Cell-type similarity — group explorer

    Inspect the within-cohort **Lin semantic-similarity** matrix one cell-type group at a time, and annotate each ontology term (`harmonized_sample_ontology_curie`) with any metadata labels you like.

    1. Point the paths below at the similarity matrix (from `evaluate_biospecimen_similarity.ipynb`), the node-info metadata CSV, and — optionally — the `*_sweep_groups.csv` from `create_similar_celltype_folds.py --sweep`.
    2. Choose which metadata columns become the node label.
    3. Pick a threshold + group (or view a whole threshold at once) and read the heatmap / node-link graph.
    4. Use **nearest neighbours** at the bottom to see what a singleton's closest term actually is, and how far below the cut it falls.
    """
    )
    return


@app.cell
def _(Path, mo):
    _meta_dir = (
        Path.home() / "Projects/epiclass/output/paper/data/metadata/epiatlas/official"
    )
    node_info_path = mo.ui.text(
        value=str(_meta_dir / "IHEC_sample_metadata_harmonization.v2.0.extended.csv"),
        full_width=True,
        label="node-info metadata CSV (shares the curie column)",
    )

    sim_matrix_dir = (
        Path.home() / "Projects/epiclass/output/paper/tables/biospecimen_comparison"
    )
    sim_matrix_path = mo.ui.text(
        value=str(sim_matrix_dir / "epiatlas_celltype_similarity_matrix.csv"),
        full_width=True,
        label="similarity matrix CSV (square term x term Lin)",
    )

    sweep_groups_path = mo.ui.text(
        value=str(sim_matrix_dir / "epiatlas_sweep_groups.csv"),
        full_width=True,
        label="sweep groups CSV (optional; *_sweep_groups.csv)",
    )
    curie_col = mo.ui.text(
        value="harmonized_sample_ontology_curie",
        full_width=True,
        label="curie column (shared key)",
    )
    mo.vstack([sim_matrix_path, node_info_path, sweep_groups_path, curie_col])
    return curie_col, node_info_path, sim_matrix_path, sweep_groups_path


@app.cell
def _(Path, mo, pd, sim_matrix_path):
    _p = Path(sim_matrix_path.value)
    if not _p.is_file():
        raise FileNotFoundError(f"Similarity matrix not found: {_p}")
    sim = pd.read_csv(_p, index_col=0)
    # Make it truly square / aligned (the notebook fills only computed pairs).
    _terms = list(sim.index)
    sim = sim.reindex(index=_terms, columns=_terms)
    mo.md(f"Loaded similarity matrix: **{sim.shape[0]}** terms.")
    return (sim,)


@app.cell
def _(Path, curie_col, mo, node_info_path, pd):
    _p = Path(node_info_path.value)
    if not _p.is_file():
        raise FileNotFoundError(f"Node-info CSV not found: {_p}")
    node_info = pd.read_csv(_p, low_memory=False)
    if curie_col.value not in node_info.columns:
        raise KeyError(
            f"Curie column {curie_col.value!r} absent from node-info CSV. "
            f"Available example columns: {list(node_info.columns)[:8]} ..."
        )
    # Records per curie — a quick proxy for "how much" of each term the cohort holds.
    records_per_curie = node_info[curie_col.value].value_counts()
    mo.md(
        f"Loaded node-info CSV: **{len(node_info)}** rows, "
        f"**{node_info[curie_col.value].nunique()}** distinct curies, "
        f"**{len(node_info.columns)}** label columns."
    )
    return node_info, records_per_curie


@app.cell
def _(Path, mo, pd, sweep_groups_path):
    if sweep_groups_path.value.strip():
        _p = Path(sweep_groups_path.value)
        if not _p.is_file():
            raise FileNotFoundError(f"Sweep groups CSV not found: {_p}")
        sweep_groups = pd.read_csv(_p)
        _msg = (
            f"Loaded sweep groups: **{sweep_groups['sim_threshold'].nunique()}** "
            f"threshold(s) — `{sorted(sweep_groups['sim_threshold'].unique())}`."
        )
    else:
        sweep_groups = None
        _msg = (
            "No sweep groups CSV given — groups will be computed **ad-hoc** from the "
            "matrix with a threshold slider (average-linkage agglomerative clustering, "
            "the same recipe as `create_similar_celltype_folds.cluster_terms`)."
        )
    mo.md(_msg)
    return (sweep_groups,)


@app.cell
def _(curie_col, mo, node_info):
    # Pick which metadata columns make up each node's label.
    _candidates = [c for c in node_info.columns if c != curie_col.value]
    _preferred = [
        "harmonized_sample_label",
        # "harmonized_sample_ontology_intermediate",
        "harmonized_biomaterial_type",
        # "harmonized_cell_type",
        # "harmonized_tissue_type",
    ]
    _default = [c for c in _preferred if c in _candidates] or _candidates[:2]
    label_cols = mo.ui.multiselect(
        options=_candidates,
        value=_default,
        label="metadata columns to concatenate into each node label",
    )
    label_cols
    return (label_cols,)


@app.cell
def _(curie_col, label_cols, node_info, pd):
    # Per-curie label: for each chosen column, join the distinct non-empty values
    # seen across that curie's samples; then concatenate the columns with " | ".
    def _join(values):
        seen = sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
        return "; ".join(seen)

    _cols = list(label_cols.value)
    if _cols:
        _agg = (
            node_info[[curie_col.value, *_cols]]
            .dropna(subset=[curie_col.value])
            .groupby(curie_col.value)[_cols]
            .agg(_join)
        )
        per_curie_labels = _agg
        curie_label = {
            curie: " | ".join(v for v in row.tolist() if v) or curie
            for curie, row in _agg.iterrows()
        }
    else:
        per_curie_labels = pd.DataFrame(index=node_info[curie_col.value].unique())
        curie_label = {}

    def label_of(curie: str) -> str:
        return curie_label.get(curie, curie)

    return (label_of,)


@app.cell
def _(mo, np, sweep_groups):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    def cluster_adhoc(sim_matrix, sim_threshold):
        """Ad-hoc clustering mirroring create_similar_celltype_folds.cluster_terms."""
        terms = list(sim_matrix.index)
        if len(terms) == 1:
            return {terms[0]: 1}
        arr = sim_matrix.to_numpy(dtype=float)
        arr = np.where(np.isnan(arr), 0.0, arr)
        arr = (arr + arr.T) / 2.0
        np.fill_diagonal(arr, 1.0)
        dist = 1.0 - arr
        np.fill_diagonal(dist, 0.0)
        dist = np.clip(dist, 0.0, None)
        z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(z, t=1.0 - sim_threshold, criterion="distance")
        return dict(zip(terms, labels))

    # Threshold choices: from the sweep file if present, else a slider.
    if sweep_groups is not None:
        _opts = sorted(sweep_groups["sim_threshold"].unique())
        threshold = mo.ui.dropdown(
            options={f"{t:g}": t for t in _opts},
            value=f"{_opts[0]:g}",
            label="similarity threshold (from sweep file)",
        )
    else:
        threshold = mo.ui.slider(
            start=0.1,
            stop=0.95,
            step=0.05,
            value=0.4,
            label="similarity threshold (ad-hoc clustering)",
        )
    threshold
    return cluster_adhoc, threshold


@app.cell
def _(
    cluster_adhoc,
    curie_col,
    records_per_curie,
    sim,
    sweep_groups,
    threshold,
):
    # Resolve {curie -> group name} + per-group sample counts for the chosen threshold,
    # from the sweep file when available, otherwise by clustering the matrix now.
    if sweep_groups is not None:
        _rows = sweep_groups[sweep_groups["sim_threshold"] == threshold.value]
        term_to_group = dict(zip(_rows[curie_col.value], _rows["group"].astype(str)))
        n_samples_of = dict(zip(_rows[curie_col.value], _rows["n_samples"]))
    else:
        _ids = cluster_adhoc(sim, float(threshold.value))
        # Name groups group1.. by descending member-record count (stable, readable).
        _by_group = {}
        for _curie, _gid in _ids.items():
            _by_group.setdefault(_gid, []).append(_curie)
        _ordered = sorted(
            _by_group.values(),
            key=lambda cs: (-sum(int(records_per_curie.get(c, 0)) for c in cs), min(cs)),
        )
        term_to_group = {}
        for _i, _members in enumerate(_ordered, start=1):
            for _curie in _members:
                term_to_group[_curie] = f"group{_i}"
        n_samples_of = {c: int(records_per_curie.get(c, 0)) for c in term_to_group}

    group_to_curies = {}
    for _curie, _g in term_to_group.items():
        group_to_curies.setdefault(_g, []).append(_curie)
    # Order groups by size (desc) then name, singletons last-ish.
    group_order = sorted(
        group_to_curies,
        key=lambda g: (-len(group_to_curies[g]), g),
    )
    return group_order, group_to_curies, n_samples_of, term_to_group


@app.cell
def _(group_order, group_to_curies, mo):
    _n_singletons = sum(1 for g in group_order if len(group_to_curies[g]) == 1)
    view_mode = mo.ui.radio(
        options=["selected group", "all groups at threshold"],
        value="selected group",
        label="view",
    )
    group_sel = mo.ui.dropdown(
        options={
            f"{g}  ({len(group_to_curies[g])} curie"
            f"{'s' if len(group_to_curies[g]) != 1 else ''})": g
            for g in group_order
        },
        value=next(
            (
                f"{g}  ({len(group_to_curies[g])} curies)"
                for g in group_order
                if len(group_to_curies[g]) > 1
            ),
            f"{group_order[0]}  (1 curie)",
        ),
        label="group",
    )
    mo.vstack(
        [
            mo.md(
                f"**{len(group_order)}** groups at this threshold "
                f"(**{_n_singletons}** singleton"
                f"{'s' if _n_singletons != 1 else ''})."
            ),
            view_mode,
            group_sel,
        ]
    )
    return group_sel, view_mode


@app.cell
def _(group_order, group_sel, group_to_curies, sim, view_mode):
    # Curies in the current view, and the group-color flag.
    if view_mode.value == "all groups at threshold":
        view_curies = [
            c for g in group_order for c in group_to_curies[g] if c in sim.index
        ]
        color_by_group = True
    else:
        view_curies = [c for c in group_to_curies[group_sel.value] if c in sim.index]
        color_by_group = False
    return color_by_group, view_curies


@app.cell
def _(label_of, mo, n_samples_of, pd, sim, term_to_group, view_curies):
    # The clarity table: every term in the view with its label + within-view stats.
    def _within_mean(curie):
        others = [c for c in view_curies if c != curie]
        if not others:
            return float("nan")
        vals = sim.loc[curie, others].to_numpy(dtype=float)
        return float(pd.Series(vals).mean(skipna=True))

    members = pd.DataFrame(
        {
            "curie": view_curies,
            "group": [term_to_group.get(c, "?") for c in view_curies],
            "label": [label_of(c) for c in view_curies],
            "n_samples": [n_samples_of.get(c, 0) for c in view_curies],
            "mean_lin_within_view": [_within_mean(c) for c in view_curies],
        }
    ).sort_values(["group", "n_samples"], ascending=[True, False])
    mo.ui.table(members, selection=None, page_size=20)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Group in context

    A singleton (or any small group) is uninformative on its own — one dot, an empty heatmap. So the views below optionally pull in each group member's **nearest external terms** as context. The group itself is highlighted; context terms are greyed. In the graph, links that clear the cut are solid and links **below** the cut (the near-misses that kept a term out of the group) are dotted — so you see *how close* the nearest thing came.
    """
    )
    return


@app.cell
def _(mo):
    context_k = mo.ui.slider(
        start=0,
        stop=15,
        step=1,
        value=5,
        label="context: add N nearest external terms per group member (0 = group only)",
        show_value=True,
    )
    show_subcut = mo.ui.checkbox(
        value=True, label="draw below-cut edges from the group to its context"
    )
    mo.vstack([context_k, show_subcut])
    return context_k, show_subcut


@app.cell
def _(context_k, sim, view_curies, view_mode):
    # Augment the (possibly singleton) group with each member's nearest *external*
    # terms across the whole matrix, so a one-node group still has something to
    # compare against. The all-groups view already shows everything — leave it be.
    in_view = set(view_curies)
    context_curies = list(view_curies)
    if view_mode.value != "all groups at threshold" and context_k.value > 0:
        _seen = set(view_curies)
        for _c in view_curies:
            _row = sim.loc[_c].drop(labels=[_c], errors="ignore").dropna()
            _row = _row[~_row.index.isin(in_view)].sort_values(ascending=False)
            for _n in _row.head(int(context_k.value)).index:
                if _n not in _seen:
                    _seen.add(_n)
                    context_curies.append(_n)
    return context_curies, in_view


@app.cell
def _(context_curies, go, in_view, label_of, sim):
    # Heatmap over the group + its nearest context terms. Context-only terms are
    # prefixed "+ ". The full metadata label only goes on the y-axis (which has room);
    # x-ticks stay short (curie only) so Plotly doesn't drop them when they collide.
    def _mark(curie):
        return "" if curie in in_view else "+ "

    def _ylabel(curie, n=46):
        lab = label_of(curie)
        lab = lab if len(lab) <= n else lab[: n - 1] + "…"
        return f"{_mark(curie)}{curie} — {lab}"

    _xlabels = [f"{_mark(c)}{c}" for c in context_curies]
    _ylabels = [_ylabel(c) for c in context_curies]
    _sub = sim.loc[context_curies, context_curies]
    _custom = [
        [f"{label_of(r)}<br>vs<br>{label_of(c)}" for c in context_curies]
        for r in context_curies
    ]
    _n_ctx = len(context_curies) - len(in_view)
    heatmap = go.Figure(
        go.Heatmap(
            z=_sub.to_numpy(dtype=float),
            x=_xlabels,
            y=_ylabels,
            customdata=_custom,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            hovertemplate="Lin=%{z:.3f}<br>%{customdata}<extra></extra>",
            colorbar=dict(title="Lin"),
        )
    )
    heatmap.update_layout(
        title=(
            f"Lin similarity — {len(in_view)} group term(s)"
            + (f" + {_n_ctx} context" if _n_ctx else "")
        ),
        width=max(560, 34 * len(context_curies) + 320),
        height=max(380, 26 * len(context_curies) + 220),
        # tickmode="array" forces *every* tick to render even when they crowd.
        xaxis=dict(
            tickmode="array",
            tickvals=_xlabels,
            ticktext=_xlabels,
            tickangle=-45,
            tickfont=dict(size=10),
            automargin=True,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=_ylabels,
            ticktext=_ylabels,
            tickfont=dict(size=10),
            automargin=True,
            autorange="reversed",
        ),
    )
    heatmap
    return


@app.cell
def _(mo, threshold):
    edge_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=float(threshold.value),
        label="draw an edge when Lin >= ",
        show_value=True,
    )
    edge_threshold
    return (edge_threshold,)


@app.cell
def _(
    color_by_group,
    context_curies,
    edge_threshold,
    go,
    in_view,
    label_of,
    n_samples_of,
    np,
    nx,
    px,
    show_subcut,
    sim,
    term_to_group,
):
    # Interactive node-link graph over the group + its context. The spring layout is
    # weighted by *every* positive-Lin pair (not just above-cut ones), so a context
    # term sits as close to the group as its similarity warrants — a singleton's
    # nearest neighbour ends up right next to it, joined by a dotted (below-cut) edge.
    _nodes = context_curies
    _cut = float(edge_threshold.value)
    _glay = nx.Graph()
    _glay.add_nodes_from(_nodes)
    _solid, _subcut = [], []
    for _i, _a in enumerate(_nodes):
        for _b in _nodes[_i + 1 :]:
            _w = sim.loc[_a, _b]
            if not np.isfinite(_w) or _w <= 0:
                continue
            _glay.add_edge(_a, _b, weight=float(_w))
            if _w >= _cut:
                _solid.append((_a, _b))
            elif show_subcut.value and (_a in in_view or _b in in_view):
                _subcut.append((_a, _b))
    _pos = nx.spring_layout(_glay, weight="weight", seed=42)

    def _line_trace(edges, color, dash):
        _x, _y = [], []
        for _a, _b in edges:
            _x += [_pos[_a][0], _pos[_b][0], None]
            _y += [_pos[_a][1], _pos[_b][1], None]
        return go.Scatter(
            x=_x,
            y=_y,
            mode="lines",
            line=dict(width=1, color=color, dash=dash),
            hoverinfo="none",
        )

    _deg = {c: 0 for c in _nodes}
    for _a, _b in _solid:
        _deg[_a] += 1
        _deg[_b] += 1
    _palette = px.colors.qualitative.Dark24
    _groups_here = sorted({term_to_group.get(c, "?") for c in _nodes})
    _gcolor = {g: _palette[i % len(_palette)] for i, g in enumerate(_groups_here)}
    _maxn = np.sqrt(max(n_samples_of.values()) or 1)
    _sizes = [10 + 22 * (np.sqrt(n_samples_of.get(c, 0)) / _maxn) for c in _nodes]
    _hover = [
        f"<b>{c}</b>{'' if c in in_view else ' (context)'}<br>{label_of(c)}"
        f"<br>group={term_to_group.get(c, '?')}<br>n_samples={n_samples_of.get(c, 0)}"
        f"<br>neighbours@cut={_deg.get(c, 0)}"
        for c in _nodes
    ]
    if color_by_group:
        _colors = [_gcolor[term_to_group.get(c, "?")] for c in _nodes]
        _line = dict(width=1, color="white")
    else:
        # Selected-group view: highlight the group, grey out the context terms.
        _colors = ["#d62728" if c in in_view else "#cccccc" for c in _nodes]
        _line = dict(
            width=[1.6 if c in in_view else 0.8 for c in _nodes],
            color=["#7f1113" if c in in_view else "#9a9a9a" for c in _nodes],
        )
    _node_trace = go.Scatter(
        x=[_pos[c][0] for c in _nodes],
        y=[_pos[c][1] for c in _nodes],
        mode="markers+text",
        text=[c.split(":")[-1] for c in _nodes],
        textposition="top center",
        textfont=dict(size=9),
        customdata=_hover,
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(size=_sizes, color=_colors, line=_line),
    )
    graph = go.Figure(
        [
            _line_trace(_subcut, "rgba(214,39,40,0.45)", "dot"),
            _line_trace(_solid, "rgba(110,110,110,0.55)", "solid"),
            _node_trace,
        ]
    )
    graph.update_layout(
        title=(
            f"Similarity graph — solid: Lin ≥ {_cut:g}"
            f"; dotted: below-cut link to the group"
        ),
        showlegend=False,
        height=640,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    graph
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Why is a term a singleton?

    Pick any curie to see its **nearest terms across the whole cohort** (not just its group). A singleton's top neighbour still sits below the cut `1 - threshold` — this table shows exactly how far below, and what that nearest term actually is.
    """
    )
    return


@app.cell
def _(mo, sim, view_curies):
    nn_curie = mo.ui.dropdown(
        options=sorted(view_curies) if view_curies else sorted(sim.index),
        value=(sorted(view_curies)[0] if view_curies else sorted(sim.index)[0]),
        label="curie",
    )
    nn_k = mo.ui.slider(start=3, stop=25, step=1, value=8, label="top-K neighbours")
    mo.vstack([nn_curie, nn_k])
    return nn_curie, nn_k


@app.cell
def _(label_of, nn_curie, nn_k, pd, sim, term_to_group):
    _row = sim.loc[nn_curie.value].drop(labels=[nn_curie.value], errors="ignore")
    _row = _row.dropna().sort_values(ascending=False).head(int(nn_k.value))
    neighbours = pd.DataFrame(
        {
            "neighbour": _row.index,
            "lin": _row.to_numpy(dtype=float),
            "group": [term_to_group.get(c, "?") for c in _row.index],
            "label": [label_of(c) for c in _row.index],
        }
    )
    neighbours
    return


@app.cell
def _(node_info):
    node_info["harmonized_sample_ontology_intermediate"].value_counts()
    return


if __name__ == "__main__":
    app.run()
