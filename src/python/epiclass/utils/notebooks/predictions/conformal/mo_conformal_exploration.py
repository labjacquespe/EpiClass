# Explore split conformal prediction on EpiClass prediction CSVs.
#
# Sweeps LAC/APS/RAPS/SAPS x alpha over one or more folds' validation_prediction.csv
# (calibration uses the held-out validation predictions) and plots empirical coverage
# vs the 1 - alpha target and average prediction-set size vs alpha. All logic lives in
# epiclass.utils.conformal.prediction; this notebook only drives it.
#
# File-wide pylint disables. Kept as a header comment (above `import marimo`) so marimo
# preserves it on save; an in-cell disable only scopes to that one cell.
# pylint: disable=missing-module-docstring, missing-function-docstring, function-redefined
# pylint: disable=import-error, import-outside-toplevel, reimported
# pylint: disable=redefined-outer-name, use-dict-literal, too-many-lines, duplicate-code
# pylint: disable=unused-import, unused-argument, unused-variable, too-many-branches
# Structural to marimo's notebook format (cells are functions that return/display):
# pylint: disable=useless-return, pointless-statement, expression-not-assigned
# pylint: disable=too-many-positional-arguments, too-many-arguments

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

    from epiclass.utils.conformal import prediction as cp

    return Path, cp, mo, pd, px


@app.cell
def _(Path):
    paper_dir = Path.home() / "Projects/epiclass/output/paper"
    data_dir = paper_dir / "data"
    training_dir = data_dir / "training_results"

    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    return (training_dir,)


@app.cell
def _(mo):
    mo.md(
        """
    # Conformal prediction explorer

    Calibrates split conformal prediction on a fold's `validation_prediction.csv` and reports the empirical coverage / average prediction-set size for every score method (LAC, APS, RAPS, SAPS) across a sweep of target miscoverage `alpha`.

    Set the glob below to point at one or more prediction CSVs, then run.
    """
    )
    return


@app.cell
def _(mo, training_dir):
    # Point this at your run: a directory holding fold subdirs (split0, split1, ...),
    # each with a validation_prediction.csv.
    pred_dir = (
        training_dir
        / "dfreeze_v2/hg38_100kb_all_none/harmonized_donor_sex_1l_3000n/10fold-oversampling"
    )
    pattern = "split*/validation_prediction.csv"

    # Glob a *relative* pattern against the (absolute) base dir: calling
    # Path().glob() with an absolute pattern raises NotImplementedError on py3.11.
    pred_paths = sorted(pred_dir.glob(pattern))
    if not pred_paths and pred_dir.is_file():
        pred_paths = [pred_dir]

    mo.md(
        f"Globbing `{pred_dir / pattern}`\n\n"
        f"Found **{len(pred_paths)}** prediction CSV(s)."
    )
    return (pred_paths,)


@app.cell
def _():
    # df = pd.read_csv(pred_paths[0])
    # df.head()
    return


@app.cell
def _(cp, pd, pred_paths):
    # Evaluate-mode sweep on each CSV; tag rows with the fold name for aggregation.
    frames = []
    for path in pred_paths:
        fold = path.parent.name or path.stem
        result = cp.run_evaluate(
            path,
            methods=cp.DEFAULT_METHODS,
            alphas=cp.DEFAULT_ALPHAS,
            calib_frac=0.5,
            seed=42,
        )
        result.insert(0, "fold", fold)
        frames.append(result)

    sweep = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sweep
    return (sweep,)


@app.cell
def _(sweep):
    # Average across folds for a compact summary.
    if sweep.empty:
        summary = sweep
    else:
        summary = (
            sweep.groupby(["method", "alpha"], as_index=False)
            .agg(
                target_coverage=("target_coverage", "first"),
                empirical_coverage=("empirical_coverage", "mean"),
                avg_set_size=("avg_set_size", "mean"),
            )
            .sort_values(["method", "alpha"])
        )
    summary
    return (summary,)


@app.cell
def _(px, summary):
    # Coverage calibration: points should sit on / above the diagonal (>= 1 - alpha).
    if summary.empty:
        fig_cov = None
    else:
        fig_cov = px.scatter(
            summary,
            x="target_coverage",
            y="empirical_coverage",
            color="method",
            title="Empirical vs target coverage (on/above diagonal = valid)",
            template="plotly_white",
        )
        fig_cov.add_shape(
            type="line", x0=0.7, y0=0.7, x1=1.0, y1=1.0, line=dict(dash="dash")
        )
    fig_cov
    return


@app.cell
def _(px, summary):
    # Average set size vs alpha: smaller is better at equal coverage; grows as alpha drops.
    if summary.empty:
        fig_size = None
    else:
        fig_size = px.line(
            summary.sort_values("alpha"),
            x="alpha",
            y="avg_set_size",
            color="method",
            markers=True,
            title="Average prediction-set size vs alpha",
            template="plotly_white",
        )
    fig_size
    return


@app.cell
def _(cp, pd, pred_paths):
    # Per-class coverage at a single alpha. Marginal coverage (above) is forced onto
    # the diagonal for every method and so hides whether a rare class (e.g. the
    # "mixed" sex label) is under-covered. This stratifies coverage by true label.
    PER_CLASS_ALPHA = 0.1

    pc_frames = [
        cp.run_evaluate_per_class(path, alpha=PER_CLASS_ALPHA, calib_frac=0.5, seed=42)
        for path in pred_paths
    ]
    if pc_frames:
        # Aggregate across folds on the raw counts, then derive rates.
        per_class = (
            pd.concat(pc_frames, ignore_index=True)
            .groupby(["method", "true_class"], as_index=False)[
                ["support", "n_covered", "set_size_sum", "n_empty"]
            ]
            .sum()
            .assign(
                coverage=lambda d: d["n_covered"] / d["support"],
                avg_set_size=lambda d: d["set_size_sum"] / d["support"],
                empty_rate=lambda d: d["n_empty"] / d["support"],
            )
        )
    else:
        per_class = pd.DataFrame()
    per_class
    return PER_CLASS_ALPHA, per_class


@app.cell
def _(PER_CLASS_ALPHA, per_class, px):
    # Bars below the dashed 1 - alpha line are under-covered classes; hover shows the
    # support, the average set size, and the empty-set rate (which drives misses).
    if per_class.empty:
        fig_per_class = None
    else:
        fig_per_class = px.bar(
            per_class.sort_values(["true_class", "method"]),
            x="true_class",
            y="coverage",
            color="method",
            barmode="group",
            range_y=[0, 1.02],
            hover_data=["support", "avg_set_size", "empty_rate"],
            title=(
                f"Per-class coverage (alpha={PER_CLASS_ALPHA}, "
                f"target={1 - PER_CLASS_ALPHA})"
            ),
            template="plotly_white",
        )
        fig_per_class.add_hline(y=1 - PER_CLASS_ALPHA, line=dict(dash="dash"))
    fig_per_class
    return


@app.cell
def _(PER_CLASS_ALPHA, per_class, px):
    # The cost side of the coverage above: average prediction-set size per true class.
    # A bar approaching the class count means the method buys coverage by abstaining
    # (sets contain ~everything) rather than by discriminating.
    if per_class.empty:
        fig_pc_size = None
    else:
        fig_pc_size = px.bar(
            per_class.sort_values(["true_class", "method"]),
            x="true_class",
            y="avg_set_size",
            color="method",
            barmode="group",
            hover_data=["support", "coverage", "empty_rate"],
            title=f"Per-class average set size (alpha={PER_CLASS_ALPHA})",
            template="plotly_white",
        )
    fig_pc_size
    return


@app.cell
def _(PER_CLASS_ALPHA, per_class, px):
    # Empty-set rate per class: an empty set is an automatic miss, so this is the
    # mechanism behind under-coverage. Disambiguates a low avg_set_size (many small
    # non-empty sets vs. outright empties). Expect LAC highest on the hard class.
    if per_class.empty:
        fig_pc_empty = None
    else:
        fig_pc_empty = px.bar(
            per_class.sort_values(["true_class", "method"]),
            x="true_class",
            y="empty_rate",
            color="method",
            barmode="group",
            hover_data=["support", "coverage", "avg_set_size"],
            title=f"Per-class empty-set rate (alpha={PER_CLASS_ALPHA})",
            template="plotly_white",
        )
    fig_pc_empty
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Class-conditional (Mondrian) calibration

    Mondrian CP uses a **separate threshold per class**, giving a per-class coverage guarantee instead of a marginal one — but every class needs enough *calibration* samples on its own.

    Below the floor `ceil(1/alpha) - 1` a class is forced into every prediction set (trivial coverage). The feasibility table checks this from the label distribution before we trust the Mondrian results.
    """
    )
    return


@app.cell
def _(PER_CLASS_ALPHA, cp, pred_paths):
    # Each CSV is one fold's validation set, and the evaluate split uses calib_frac of
    # it for calibration -- so check the per-fold calibration count per class against
    # the alpha floor. n_splits is left at 1: the CSV is already a single fold.
    if pred_paths:
        _, _, _fcls, _ftrue = cp.load_prediction_csv(pred_paths[0])
        _labels = [_fcls[i] for i in _ftrue]
        feasibility = cp.mondrian_feasibility(
            _labels, alpha=PER_CLASS_ALPHA, calib_frac=0.5
        )
    else:
        feasibility = None
    feasibility
    return


@app.cell
def _(PER_CLASS_ALPHA, cp, pd, pred_paths):
    # Marginal vs Mondrian per-class coverage, aggregated across folds on raw counts.
    def _per_class(is_mondrian):
        frames = [
            cp.run_evaluate_per_class(
                p,
                alpha=PER_CLASS_ALPHA,
                calib_frac=0.5,
                seed=42,
                mondrian=is_mondrian,
            )
            for p in pred_paths
        ]
        agg = (
            pd.concat(frames, ignore_index=True)
            .groupby(["method", "true_class"], as_index=False)[
                ["support", "n_covered", "set_size_sum"]
            ]
            .sum()
            .assign(
                coverage=lambda d: d["n_covered"] / d["support"],
                avg_set_size=lambda d: d["set_size_sum"] / d["support"],
                calibration="mondrian" if is_mondrian else "marginal",
            )
        )
        return agg

    if pred_paths:
        mondrian_cmp = pd.concat([_per_class(False), _per_class(True)], ignore_index=True)
    else:
        mondrian_cmp = pd.DataFrame()
    mondrian_cmp
    return (mondrian_cmp,)


@app.cell
def _(PER_CLASS_ALPHA, mondrian_cmp, px):
    # Per method: marginal vs Mondrian coverage per class. Mondrian should lift the
    # rare class onto target *if* it cleared the feasibility floor; if not, it "covers"
    # by forcing that class into every set (watch its avg_set_size jump in the hover).
    if mondrian_cmp.empty:
        fig_mondrian = None
    else:
        fig_mondrian = px.bar(
            mondrian_cmp.sort_values(["method", "true_class", "calibration"]),
            x="true_class",
            y="coverage",
            color="calibration",
            barmode="group",
            facet_col="method",
            range_y=[0, 1.02],
            hover_data=["support", "avg_set_size"],
            title=f"Marginal vs Mondrian per-class coverage (alpha={PER_CLASS_ALPHA})",
            template="plotly_white",
        )
        fig_mondrian.add_hline(y=1 - PER_CLASS_ALPHA, line=dict(dash="dash"))
    fig_mondrian
    return


@app.cell
def _(PER_CLASS_ALPHA, mondrian_cmp, px):
    # Companion to the coverage plot: did Mondrian's coverage lift come with a set-size
    # jump? A rare class below the calibration floor is "covered" by being forced into
    # every set -- its avg_set_size balloons (toward the class count) under Mondrian,
    # which is coverage-by-abstention, not a genuine threshold. A near-flat set size
    # means the lift is real.
    if mondrian_cmp.empty:
        fig_mondrian_size = None
    else:
        fig_mondrian_size = px.bar(
            mondrian_cmp.sort_values(["method", "true_class", "calibration"]),
            x="true_class",
            y="avg_set_size",
            color="calibration",
            barmode="group",
            facet_col="method",
            hover_data=["support", "coverage"],
            title=f"Marginal vs Mondrian per-class set size (alpha={PER_CLASS_ALPHA})",
            template="plotly_white",
        )
    fig_mondrian_size
    return


if __name__ == "__main__":
    app.run()
