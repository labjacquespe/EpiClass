# Extensive per-classifier conformal-prediction report over a 10-fold CV run.
#
# Pick one metadata classifier's run directory (a folder of split*/validation_prediction.csv)
# and an alpha; the app reports, aggregated over the folds: marginal coverage sanity,
# per-class coverage / set size / empty rate, Mondrian feasibility + marginal-vs-Mondrian
# lift, and a RAPS/SAPS hyperparameter-sensitivity sweep (which classes move under
# regularization). A final section scans every classifier under the training dir for a
# "hands up" summary (worst-covered class, Mondrian feasibility, hyperparam sensitivity).
#
# Focus methods are RAPS and SAPS; LAC and APS are kept as faded reference lines. All logic
# lives in epiclass.utils.postprocessing.conformal_prediction; this notebook only drives it.
#
# File-wide pylint disables. Kept as a header comment (above `import marimo`) so marimo
# preserves it on save; an in-cell disable only scopes to that one cell.
# pylint: disable=missing-module-docstring, missing-function-docstring, function-redefined
# pylint: disable=import-error, import-outside-toplevel, reimported
# pylint: disable=redefined-outer-name, use-dict-literal, too-many-lines
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
    from tqdm.notebook import tqdm

    from epiclass.utils.postprocessing import (
        conformal_prediction as cp,
        conformal_report as cpr,
    )

    return Path, cp, cpr, mo, pd, px, tqdm


@app.cell
def _():
    # Score methods to headline vs keep as faded reference.
    FOCUS_METHODS = ("RAPS", "SAPS")
    REF_METHODS = ("LAC", "APS")

    # Hyperparameter grids swept in the sensitivity section (section 4).
    RAPS_GRID = {"kreg": [1, 2, 5], "penalty": [0.0, 0.01, 0.1, 0.5]}
    SAPS_GRID = {"weight": [0.05, 0.1, 0.2, 0.5]}
    return FOCUS_METHODS, RAPS_GRID, SAPS_GRID


@app.cell
def _(Path):
    paper_dir = Path.home() / "Projects/epiclass/output/paper"
    training_dir = paper_dir / "data/training_results/dfreeze_v2/hg38_100kb_all_none"

    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    return (training_dir,)


@app.cell
def _(mo):
    mo.md(
        """
    # Conformal prediction — per-classifier report

    Work through a 10-fold cross-validation run one classifier at a time. Calibration uses each fold's `validation_prediction.csv` (a 50/50 calib/eval split per fold); results are aggregated over folds on **raw counts** (sum-then-divide), never mean-of-means.

    **RAPS** and **SAPS** are the focus methods; **LAC**/**APS** are shown as faded reference lines. Marginal coverage is a *sanity* check only — it is forced onto target by construction and hides per-class problems, so the per-class sections below are what actually decide things.
    """
    )
    return


@app.cell
def _(training_dir):
    # Discover every CV run under the training dir: a "run" is any directory holding
    # split*/validation_prediction.csv. rglob keeps the pattern relative (an absolute
    # glob pattern raises NotImplementedError on py3.11).
    run_dirs = {}
    for _split0_csv in sorted(training_dir.rglob("split0/validation_prediction.csv")):
        _run_dir = _split0_csv.parent.parent
        run_dirs[str(_run_dir.relative_to(training_dir))] = _run_dir
    return (run_dirs,)


@app.cell
def _(mo, run_dirs):
    # Default to the donor-sex run (the one already characterized in conformal_methods.md)
    # if present, else the first discovered run.
    _labels = list(run_dirs)
    _default = next((label for label in _labels if "donor_sex" in label), _labels[0])

    classifier_dd = mo.ui.dropdown(
        options=_labels, value=_default, label="Classifier run"
    )
    alpha_slider = mo.ui.slider(
        start=0.01, stop=0.30, step=0.01, value=0.05, label="alpha", show_value=True
    )
    methods_ms = mo.ui.multiselect(
        options=["LAC", "APS", "RAPS", "SAPS"],
        value=["RAPS", "SAPS", "LAC", "APS"],
        label="Methods",
    )
    mo.vstack([classifier_dd, alpha_slider, methods_ms])
    return alpha_slider, classifier_dd, methods_ms


@app.cell
def _(classifier_dd, mo, run_dirs):
    run_dir = run_dirs[classifier_dd.value]
    pred_paths = sorted(run_dir.glob("split*/validation_prediction.csv"))
    mo.md(
        f"Selected **{classifier_dd.value}** — "
        f"**{len(pred_paths)}** fold prediction CSV(s) under `{run_dir}`."
    )
    return (pred_paths,)


@app.cell
def _(cp):
    # Shared driver: per-class coverage/size/empty aggregated over folds on raw counts.
    # The sum-then-divide aggregation lives in the module (cp.aggregate_per_class);
    # figure construction lives in cpr (reused on-screen here and on-disk in the export).
    def per_class_over_folds(paths, alpha, methods, mondrian=False):
        frames = [
            cp.run_evaluate_per_class(
                p,
                methods=methods,
                alpha=alpha,
                calib_frac=0.5,
                seed=42,
                mondrian=mondrian,
            )
            for p in paths
        ]
        return cp.aggregate_per_class(frames)

    return (per_class_over_folds,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Marginal coverage (sanity only)

    Every method lands on the diagonal by construction, so this plot can never distinguish methods or reveal a per-class problem — it only confirms the machinery is calibrated. The deciding diagnostics are in section 2.
    """
    )
    return


@app.cell
def _(cp, methods_ms, pd, pred_paths):
    sweep_frames = []
    for _path in pred_paths:
        _res = cp.run_evaluate(
            _path,
            methods=tuple(methods_ms.value),
            alphas=cp.DEFAULT_ALPHAS,
            calib_frac=0.5,
            seed=42,
        )
        sweep_frames.append(_res)

    marginal_summary = (
        pd.concat(sweep_frames, ignore_index=True)
        .groupby(["method", "alpha"], as_index=False)
        .agg(
            target_coverage=("target_coverage", "first"),
            empirical_coverage=("empirical_coverage", "mean"),
            avg_set_size=("avg_set_size", "mean"),
        )
        if sweep_frames
        else pd.DataFrame()
    )
    return (marginal_summary,)


@app.cell
def _(cpr, marginal_summary, px):
    if marginal_summary.empty:
        fig_cov = None
    else:
        fig_cov = px.scatter(
            marginal_summary,
            x="target_coverage",
            y="empirical_coverage",
            color="method",
            title="Empirical vs target coverage (on/above diagonal = valid)",
            template="plotly_white",
        )
        fig_cov.add_shape(
            type="line", x0=0.7, y0=0.7, x1=1.0, y1=1.0, line=dict(dash="dash")
        )
        cpr.dim_ref_traces(fig_cov)
    fig_cov
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Per-class breakdown

    Bars below the dashed `1 - alpha` line are **under-covered** classes — the classifier "raising its hand". Set size shows whether coverage is bought by discrimination (size ≈ 1) or abstention (size → class count); empty rate isolates outright misses (an empty set is an automatic miss, the mechanism behind LAC under-covering hard classes).
    """
    )
    return


@app.cell
def _(alpha_slider, methods_ms, per_class_over_folds, pred_paths):
    per_class = per_class_over_folds(
        pred_paths, alpha_slider.value, tuple(methods_ms.value)
    )
    per_class
    return (per_class,)


@app.cell
def _(alpha_slider, cpr, per_class):
    fig_pc_cov = (
        None
        if per_class.empty
        else cpr.per_class_bar(
            per_class,
            "coverage",
            title=f"Per-class coverage (target = {1 - alpha_slider.value:.2f})",
            hline=1 - alpha_slider.value,
            range_y=[0, 1.02],
            hover=["support", "avg_set_size", "empty_rate"],
        )
    )
    fig_pc_cov
    return


@app.cell
def _(alpha_slider, cpr, per_class):
    fig_pc_size = (
        None
        if per_class.empty
        else cpr.per_class_bar(
            per_class,
            "avg_set_size",
            title=f"Per-class average set size (alpha = {alpha_slider.value:.2f})",
            hover=["support", "coverage", "empty_rate"],
        )
    )
    fig_pc_size
    return


@app.cell
def _(alpha_slider, cpr, per_class):
    fig_pc_empty = (
        None
        if per_class.empty
        else cpr.per_class_bar(
            per_class,
            "empty_rate",
            title=f"Per-class empty-set rate (alpha = {alpha_slider.value:.2f})",
            hover=["support", "coverage", "avg_set_size"],
        )
    )
    fig_pc_empty
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Flag composition (QC view).** A file is *flagged* for manual review whenever its prediction set is not a confident correct singleton. At fixed alpha coverage is held constant, so the lever is the *composition* of the flags, not the raw rate. The stacked bar height is the flag rate (`1 - clean`), split into: **ambiguous** (size ≥ 2 — a hedge among known classes), **singleton disagrees** (a confident single prediction that conflicts with the label — a specific mislabel hypothesis), and **empty** (reject / OOD — the uninformative miss). Prefer the informative types and minimise empties; this is where RAPS (fewer empties) and SAPS/APS (more empties, sharper reject) part ways.
    """
    )
    return


@app.cell
def _(alpha_slider, cpr, per_class):
    fig_flags = (
        None
        if per_class.empty
        else cpr.flag_composition_bar(
            per_class,
            title=f"Per-class flag rate by type (alpha = {alpha_slider.value:.2f})",
        )
    )
    fig_flags
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Class-conditional (Mondrian) calibration

    Mondrian CP uses a **separate threshold per class** for a per-class guarantee, but every class needs enough *calibration* samples on its own. Below the floor `ceil(1/alpha) - 1` a class is forced into every set (trivial coverage); the feasibility table checks this from the per-fold label counts before the marginal-vs-Mondrian comparison is trusted. A genuine lift moves coverage up **without** the set size ballooning toward the class count.
    """
    )
    return


@app.cell
def _(alpha_slider, cp, pred_paths):
    if pred_paths:
        _, _, _classes, _true = cp.load_prediction_csv(pred_paths[0])
        _labels = [_classes[i] for i in _true]
        # n_splits=len(folds): k-fold puts ~1/k of the data in each fold's validation set,
        # and calib_frac=0.5 of that is used for calibration.
        feasibility = cp.mondrian_feasibility(
            _labels,
            alpha=alpha_slider.value,
            n_splits=len(pred_paths),
            calib_frac=0.5,
        )
    else:
        feasibility = None
    feasibility
    return (feasibility,)


@app.cell
def _(alpha_slider, feasibility, mo):
    # Surface the degeneracy the feasibility table predicts: at alpha=0.05 the floor is 19
    # calibration samples per class per fold, which most classes miss -- TorchCP then sets
    # their Mondrian threshold to inf (forced into every set). Flag it so the degenerate
    # Mondrian columns below are not mistaken for a real per-class lift.
    if feasibility is None or feasibility.empty:
        _banner = None
    else:
        _degenerate = feasibility.loc[
            feasibility["status"] == "degenerate", "class"
        ].tolist()
        if _degenerate:
            _banner = mo.callout(
                mo.md(
                    f"**{len(_degenerate)} of {len(feasibility)} classes are below the "
                    f"Mondrian floor at alpha = {alpha_slider.value:.2f}** "
                    f"(`{', '.join(map(str, _degenerate))}`). Their per-class threshold is "
                    f"`inf`, so Mondrian forces them into **every** set — trivial coverage at "
                    f"set size ≈ class count. The Mondrian columns below are degenerate for "
                    f"these classes; raise alpha or pool more calibration to make it usable."
                ),
                kind="warn",
            )
        else:
            _banner = mo.callout(
                mo.md("All classes clear the Mondrian floor at this alpha."),
                kind="success",
            )
    _banner
    return


@app.cell
def _(alpha_slider, cp, methods_ms, pd, per_class_over_folds, pred_paths):
    # Mondrian calibration warns once per degenerate class (quantile > 1 -> threshold inf);
    # that is the documented floor, already reported by the feasibility table above, so
    # suppress the redundant noise here.
    with cp.suppress_quantile_warning():
        _marg = per_class_over_folds(
            pred_paths, alpha_slider.value, tuple(methods_ms.value), mondrian=False
        ).assign(calibration="marginal")
        _mond = per_class_over_folds(
            pred_paths, alpha_slider.value, tuple(methods_ms.value), mondrian=True
        ).assign(calibration="mondrian")
    mondrian_cmp = (
        pd.concat([_marg, _mond], ignore_index=True)
        if not _marg.empty
        else pd.DataFrame()
    )
    return (mondrian_cmp,)


@app.cell
def _(alpha_slider, mondrian_cmp, px):
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
            title=f"Marginal vs Mondrian per-class coverage (alpha = {alpha_slider.value:.2f})",
            template="plotly_white",
        )
        fig_mondrian.add_hline(y=1 - alpha_slider.value, line=dict(dash="dash"))
    fig_mondrian
    return


@app.cell
def _(alpha_slider, mondrian_cmp, px):
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
            title=f"Marginal vs Mondrian per-class set size (alpha = {alpha_slider.value:.2f})",
            template="plotly_white",
        )
    fig_mondrian_size
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. Hyperparameter sensitivity (RAPS / SAPS)

    Sweep the regularization knobs (RAPS `kreg`, `penalty`; SAPS `weight`) over the folds and watch each class's average set size. The guarantee is unaffected — only set size moves — so a class whose set size **swings** across the grid is hyperparam-sensitive (worth tuning), while a flat one is robust to the choice.
    """
    )
    return


@app.cell
def _(RAPS_GRID, SAPS_GRID, alpha_slider, cp, pd, pred_paths):
    if pred_paths:
        raps_sweep = cp.sweep_hparams(
            pred_paths, "RAPS", RAPS_GRID, alpha=alpha_slider.value
        )
        saps_sweep = cp.sweep_hparams(
            pred_paths, "SAPS", SAPS_GRID, alpha=alpha_slider.value
        )
        hp_sweep = pd.concat([raps_sweep, saps_sweep], ignore_index=True)
    else:
        hp_sweep = pd.DataFrame()
    hp_sweep
    return (hp_sweep,)


@app.cell
def _(cpr, hp_sweep):
    # Set size across the grid, per class, faceted by method. Spread = sensitivity.
    fig_hp = (
        None
        if hp_sweep.empty
        else cpr.sensitivity_line(
            hp_sweep,
            title="Average set size across the hyperparameter grid (per class)",
        )
    )
    fig_hp
    return


@app.cell
def _(cpr, hp_sweep):
    # Sensitivity ranking: range (max - min) of avg set size across the grid per class.
    fig_sens = (
        None
        if hp_sweep.empty
        else cpr.sensitivity_range_bar(
            hp_sweep,
            title="Set-size sensitivity to hyperparameters (range across grid)",
        )
    )
    fig_sens
    return


@app.cell
def _(cp, cpr, mo):
    # Quick export for just the selected classifier (the section-5 scan does all of them).
    export_button = mo.ui.run_button(label="Save figures for selected classifier")
    export_alpha_dd = mo.ui.dropdown(
        options={f"{a:.2f}": a for a in cp.REPORT_ALPHAS},
        value="0.05",
        label="Export alpha",
    )
    mo.vstack(
        [
            mo.md(
                f"### Export — write the selected run's summary figures to "
                f"`<run>/{cpr.REPORT_DIR_NAME}/`"
            ),
            mo.hstack([export_button, export_alpha_dd], justify="start"),
        ]
    )
    return export_alpha_dd, export_button


@app.cell
def _(classifier_dd, cpr, export_alpha_dd, export_button, mo, run_dirs):
    if export_button.value:
        _out = cpr.save_summary_figures(
            run_dirs[classifier_dd.value], alpha=export_alpha_dd.value
        )
        _msg = mo.md(f"Saved summary figures for **{classifier_dd.value}** to `{_out}`.")
    else:
        _msg = mo.md("_Press to save this classifier's summary figures._")
    _msg
    return


@app.cell
def _(cp, cpr, mo):
    mo.md(
        f"""
        ## 5. Cross-classifier "hands up" summary

        Scan **every** classifier under the training dir: per focus method, the
        worst-covered class (at its default hyperparameters), whether Mondrian is feasible
        (every class clears the floor), and the worst-class hyperparameter sensitivity
        (set-size range over the **full** RAPS/SAPS grid). The heavy per-fold computation is
        cached to `{cp.FOLD_REPORT_NAME}` inside each split folder, so the first scan is slow
        and every later one is fast — tick **Force recompute** to rebuild the caches (e.g.
        after changing the canonical grids/alphas in the module). The scan alpha is one of
        the cached standard QC values `{cp.REPORT_ALPHAS}`, independent of the slider above.

        With **Save figures** ticked, the scan also writes each run's summary figures
        (per-class coverage / set size / empty rate + hyperparameter sensitivity) to a
        `{cpr.REPORT_DIR_NAME}/` folder **next to that run's split folders**.
        """
    )
    return


@app.cell
def _(cp, cpr, mo):
    scan_button = mo.ui.run_button(label="Run cross-classifier scan")
    scan_alpha_dd = mo.ui.dropdown(
        options={f"{a:.2f}": a for a in cp.REPORT_ALPHAS},
        value="0.05",
        label="Scan alpha",
    )
    force_cb = mo.ui.checkbox(label="Force recompute caches")
    save_figs_cb = mo.ui.checkbox(
        value=True, label=f"Save figures to <run>/{cpr.REPORT_DIR_NAME}/"
    )
    mo.hstack([scan_button, scan_alpha_dd, force_cb, save_figs_cb], justify="start")
    return force_cb, save_figs_cb, scan_alpha_dd, scan_button


@app.cell
def _(
    FOCUS_METHODS,
    cp,
    cpr,
    force_cb,
    mo,
    pd,
    run_dirs,
    save_figs_cb,
    scan_alpha_dd,
    scan_button,
    tqdm,
):
    mo.stop(not scan_button.value, mo.md("_Press the button to run the scan._"))

    _alpha = scan_alpha_dd.value
    rows = []
    for _label, _run in tqdm(run_dirs.items()):
        # Cached per-fold full report for the run, then aggregate across folds on raw counts.
        _rep = cp.run_report(_run, force=force_cb.value)
        if _rep.empty:
            continue

        # Save this run's summary figures next to its split folders (reuse the loaded report).
        if save_figs_cb.value:
            cpr.write_figures(
                cpr.build_summary_figures(_rep, alpha=_alpha),
                _run / cpr.REPORT_DIR_NAME,
                alpha=_alpha,
            )

        _agg = cp.aggregate_per_class(
            _rep, group_cols=["method", "alpha", "combo", "true_class"]
        )
        _at = _agg[_agg["alpha"] == _alpha]

        # Mondrian feasibility from the first fold's label distribution.
        _paths = sorted(_run.glob("split*/validation_prediction.csv"))
        _, _, _cls, _tru = cp.load_prediction_csv(_paths[0])
        _mondrian_ok = bool(
            cp.mondrian_feasibility(
                [_cls[i] for i in _tru],
                alpha=_alpha,
                n_splits=len(_paths),
                calib_frac=0.5,
            )["clears_floor"].all()
        )

        for _method in FOCUS_METHODS:
            _method_rows = _at[_at["method"] == _method]
            # Headline coverage at the method's default hyperparameters.
            _default = _method_rows[
                _method_rows["combo"] == cp.combo_label(cp.REPORT_DEFAULTS[_method])
            ]
            _worst = _default.loc[_default["coverage"].idxmin()]
            # Sensitivity: per-class set-size range across the full grid, worst over classes.
            _sens = (
                _method_rows.groupby("true_class")["avg_set_size"]
                .agg(lambda s: s.max() - s.min())
                .max()
            )
            rows.append(
                {
                    "classifier": _label,
                    "method": _method,
                    "worst_class": _worst["true_class"],
                    "worst_coverage": round(_worst["coverage"], 3),
                    "worst_empty_rate": round(_worst["empty_rate"], 3),
                    "n_classes": _method_rows["true_class"].nunique(),
                    "mondrian_feasible": _mondrian_ok,
                    "max_setsize_sensitivity": round(float(_sens), 3),
                }
            )

    hands_up = (
        pd.DataFrame(rows).sort_values(["worst_coverage", "classifier"])
        if rows
        else pd.DataFrame()
    )
    hands_up
    return (hands_up,)


@app.cell
def _(hands_up, px, scan_alpha_dd):
    if hands_up.empty:
        fig_hands = None
    else:
        _alpha = scan_alpha_dd.value
        fig_hands = px.bar(
            hands_up,
            x="classifier",
            y="worst_coverage",
            color="method",
            barmode="group",
            range_y=[0, 1.02],
            hover_data=[
                "worst_class",
                "worst_empty_rate",
                "mondrian_feasible",
                "max_setsize_sensitivity",
            ],
            title=f"Worst-covered class per classifier (alpha = {_alpha:.2f})",
            template="plotly_white",
        )
        fig_hands.add_hline(y=1 - _alpha, line=dict(dash="dash"))
        fig_hands.update_xaxes(tickangle=45)
    fig_hands
    return


if __name__ == "__main__":
    app.run()
