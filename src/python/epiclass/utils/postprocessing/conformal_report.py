"""Summary figures for the per-classifier conformal report.

Builds the headline figures of the report notebook (per-class coverage / set size /
empty rate at the default hyperparameters, and the RAPS/SAPS hyperparameter-sensitivity
sweep) straight from a cached per-fold report (see ``conformal_prediction.run_report``),
and writes them into a ``conformal_report/`` folder that sits **next to the split folders**
of a run -- one summary per classifier, alongside its data.

Plotting lives here, not in ``conformal_prediction`` (which stays plotting-free): this
module is the single source of the figure construction, reused by the marimo notebook for
on-screen display and by ``save_summary_figures`` for on-disk export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from epiclass.utils.postprocessing import conformal_prediction as cp

# Folder (sibling of split0/.../splitN) that collects a run's summary figures.
REPORT_DIR_NAME = "conformal_report"
FOCUS_METHODS: Sequence[str] = ("RAPS", "SAPS")
REF_METHODS: Sequence[str] = ("LAC", "APS")


def dim_ref_traces(fig: go.Figure, ref_methods: Sequence[str] = REF_METHODS) -> go.Figure:
    """Fade traces whose name starts with a reference method so RAPS/SAPS stand out."""
    for trace in fig.data:
        name = getattr(trace, "name", "") or ""
        if any(name.startswith(ref) for ref in ref_methods):
            trace.update(opacity=0.35)
    return fig


def per_class_bar(
    per_class: pd.DataFrame,
    y: str,
    *,
    title: str,
    hline: float | None = None,
    range_y: Sequence[float] | None = None,
    hover: Sequence[str] | None = None,
    ref_methods: Sequence[str] = REF_METHODS,
) -> go.Figure:
    """Grouped per-true-class bar of column ``y``, one bar group per method.

    Shared by the notebook's section 2 (coverage / set size / empty rate) and the saved
    summary; the only difference between those three figures is ``y``, the title, and an
    optional ``hline`` (the ``1 - alpha`` target on the coverage plot).
    """
    fig = px.bar(
        per_class.sort_values(["true_class", "method"]),
        x="true_class",
        y=y,
        color="method",
        barmode="group",
        range_y=list(range_y) if range_y is not None else None,
        hover_data=list(hover) if hover is not None else None,
        title=title,
        template="plotly_white",
    )
    if hline is not None:
        fig.add_hline(y=hline, line={"dash": "dash"})
    dim_ref_traces(fig, ref_methods)
    return fig


def sensitivity_line(hp_sweep: pd.DataFrame, *, title: str) -> go.Figure:
    """Per-class average set size across the hyperparameter grid, faceted by method."""
    fig = px.line(
        hp_sweep.sort_values(["method", "true_class", "combo"]),
        x="combo",
        y="avg_set_size",
        color="true_class",
        facet_col="method",
        markers=True,
        title=title,
        template="plotly_white",
    )
    fig.update_xaxes(matches=None, showticklabels=True, tickangle=45)
    return fig


def sensitivity_range_bar(hp_sweep: pd.DataFrame, *, title: str) -> go.Figure:
    """Per-class set-size range (max - min) across the grid: the sensitivity ranking."""
    sensitivity = (
        hp_sweep.groupby(["method", "true_class"], as_index=False)["avg_set_size"]
        .agg(set_size_range=lambda s: s.max() - s.min())
        .sort_values("set_size_range", ascending=False)
    )
    return px.bar(
        sensitivity,
        x="true_class",
        y="set_size_range",
        color="method",
        barmode="group",
        title=title,
        template="plotly_white",
    )


# Flag-composition categories: the three prediction-set shapes that route a file to
# manual review (everything except a confident, correct singleton), ordered most- to
# least-informative for QC. Maps each to its raw per-class count column.
FLAG_CATEGORIES = {
    "ambiguous (hedge)": "n_multi",
    "singleton disagrees": "n_singleton_wrong",
    "empty (reject)": "n_empty",
}


def flag_composition_bar(per_class: pd.DataFrame, *, title: str) -> go.Figure:
    """Stacked per-class **flag rate**, split by flag type, faceted by method.

    A file is *flagged* (routed to manual review) whenever its prediction set is not a
    confident correct singleton. The bar height is the flag rate ``1 - clean`` and its
    segments are the three flag shapes (``FLAG_CATEGORIES``): a hedge (size >= 2), a
    singleton disagreeing with the label (mislabel hypothesis), and an empty set
    (reject / OOD, the uninformative miss). ``per_class`` must carry the summed shape
    counts and ``support`` (from ``cp.aggregate_per_class``).
    """
    rates = per_class.assign(
        **{
            name: lambda d, col=col: d[col] / d["support"]
            for name, col in FLAG_CATEGORIES.items()
        }
    )
    long = rates.melt(
        id_vars=["true_class", "method"],
        value_vars=list(FLAG_CATEGORIES),
        var_name="flag_type",
        value_name="rate",
    )
    return px.bar(
        long.sort_values(["method", "true_class", "flag_type"]),
        x="true_class",
        y="rate",
        color="flag_type",
        barmode="stack",
        facet_col="method",
        title=title,
        template="plotly_white",
        category_orders={"flag_type": list(FLAG_CATEGORIES)},
    )


def _default_combo_rows(at_alpha: pd.DataFrame) -> pd.DataFrame:
    """Keep, per method, only the rows at that method's default hyperparameters.

    LAC/APS have the single ``default`` combo; RAPS/SAPS keep the combo matching their
    literature defaults (``cp.REPORT_DEFAULTS``). This is the headline configuration the
    per-class coverage/size/empty figures report.
    """
    expected = at_alpha["method"].map(
        lambda m: cp.combo_label(cp.REPORT_DEFAULTS.get(m, {}))
    )
    return at_alpha[at_alpha["combo"] == expected]


def build_summary_figures(
    report_df: pd.DataFrame,
    *,
    alpha: float,
    focus_methods: Sequence[str] = FOCUS_METHODS,
) -> Dict[str, go.Figure]:
    """Build the named summary figures for one run from its cached per-fold report.

    ``report_df`` is the stacked per-fold report from ``cp.run_report`` (one row per
    fold x method x alpha x combo x class). Aggregates across folds on raw counts, slices
    to ``alpha``, and returns a ``{name: figure}`` dict: per-class coverage / set size /
    empty rate at the default hyperparameters, plus the RAPS/SAPS sensitivity line and
    range bar over the full grid. Raises ``ValueError`` if ``alpha`` is absent from the
    report (the report is cached only at ``cp.REPORT_ALPHAS``).
    """
    agg = cp.aggregate_per_class(
        report_df, group_cols=["method", "alpha", "combo", "true_class"]
    )
    at_alpha = agg[agg["alpha"] == alpha]
    if at_alpha.empty:
        raise ValueError(
            f"No rows at alpha={alpha} in the report; cached alphas are {cp.REPORT_ALPHAS}."
        )

    per_class = _default_combo_rows(at_alpha)
    hp_sweep = at_alpha[at_alpha["method"].isin(list(focus_methods))]
    target = 1 - alpha

    return {
        "per_class_coverage": per_class_bar(
            per_class,
            "coverage",
            title=f"Per-class coverage (alpha = {alpha:.2f}, target = {target:.2f})",
            hline=target,
            range_y=[0, 1.02],
            hover=["support", "avg_set_size", "empty_rate"],
        ),
        "per_class_set_size": per_class_bar(
            per_class,
            "avg_set_size",
            title=f"Per-class average set size (alpha = {alpha:.2f})",
            hover=["support", "coverage", "empty_rate"],
        ),
        "per_class_empty_rate": per_class_bar(
            per_class,
            "empty_rate",
            title=f"Per-class empty-set rate (alpha = {alpha:.2f})",
            hover=["support", "coverage", "avg_set_size"],
        ),
        "hparam_sensitivity_setsize": sensitivity_line(
            hp_sweep,
            title=f"Set size across the hyperparameter grid (alpha = {alpha:.2f})",
        ),
        "hparam_sensitivity_range": sensitivity_range_bar(
            hp_sweep,
            title=f"Set-size sensitivity to hyperparameters (alpha = {alpha:.2f})",
        ),
        "flag_composition": flag_composition_bar(
            per_class,
            title=f"Per-class flag rate by type (alpha = {alpha:.2f})",
        ),
    }


def write_figures(
    figures: Dict[str, go.Figure],
    out_dir: str | Path,
    *,
    alpha: float,
    fmt: str = "png",
) -> Path:
    """Write each named figure to ``out_dir`` as ``<name>_alpha<alpha>.<fmt>``.

    ``fmt="png"`` uses kaleido (static, embeddable); ``fmt="html"`` writes a standalone
    interactive plot (plotly.js from CDN). Returns the output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = out_dir / f"{name}_alpha{alpha:.2f}.{fmt}"
        if fmt == "html":
            fig.write_html(path, include_plotlyjs="cdn")
        else:
            fig.write_image(path)
    return out_dir


def save_summary_figures(
    run_dir: str | Path,
    *,
    alpha: float = 0.05,
    fmt: str = "png",
    out_subdir: str = REPORT_DIR_NAME,
    force: bool = False,
) -> Path:
    """Build and save a run's summary figures into ``run_dir/<out_subdir>/``.

    ``out_subdir`` sits next to the split folders. Reads the cached per-fold report via
    ``cp.run_report`` (computing/caching it if missing, or rebuilding when ``force``),
    builds the figures at ``alpha`` and writes them. Returns the figure directory.
    """
    report = cp.run_report(run_dir, force=force)
    if report.empty:
        raise ValueError(f"No fold prediction CSVs found under '{run_dir}'.")
    figures = build_summary_figures(report, alpha=alpha)
    return write_figures(figures, Path(run_dir) / out_subdir, alpha=alpha, fmt=fmt)
