# Compare RNA-Seq Unique vs UniqueMultiple mapping predictions, per classifier, per fold.
#
# predict_CV.py was run once per classifier (6 classifiers, 10-fold CV runs) on a single
# HDF5 list holding BOTH RNA-Seq mappings together:
#   - "Unique" mapping samples         (part of training)
#   - "UniqueMultiple" mapping samples (never seen by the models)
# So each classifier has ONE concatenated CSV (concatenated_test_prediction_*.csv) stacking
# every fold's predictions, with an `origin` column naming the per-fold source file.
# Each biological sample (EpiRR) usually has both mappings; this notebook pairs them 1-to-1
# and measures how much the prediction changes between the two.
#
# Per classifier, per fold (splitN, read from `origin`):
#   1. Drop Unique predictions whose signal-ID is in that fold's splitN_training_*.md5
#      (keep only samples the fold model did NOT train on). We deliberately do NOT use the
#      validation .md5: the prediction list can hold RNA-Seq samples in neither split.
#   2. Match each surviving Unique sample to its UniqueMultiple counterpart on
#      (EpiRR, strand-track e.g. minusRaw). Unique identity comes from the metadata JSON
#      (md5sum -> epirr_id, track_type); UniqueMultiple identity is parsed from the ID itself.
#   3. Compare the two probability vectors from the SAME fold model: flag argmax
#      disagreement, and measure Euclidean (L2) and total-variation (0.5*L1) distance.
#
# This is a plain (non-reactive) notebook: edit the CONFIG cell, then run top-to-bottom.
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

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from epiclass.core.metadata import Metadata
    from epiclass.utils.conformal.prediction import (
        calibrate_predictor,
        classify_flags,
        predict_sets,
    )
    from epiclass.utils.general_utility import find_signal_id_lists

    return (
        Counter,
        Metadata,
        Path,
        calibrate_predictor,
        classify_flags,
        find_signal_id_lists,
        go,
        make_subplots,
        mo,
        np,
        pd,
        predict_sets,
        px,
        re,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # RNA-Seq Unique vs UniqueMultiple — prediction stability

    Both mappings were predicted together, so each classifier has **one** concatenated prediction CSV. For every CV fold (read from `origin`), pair each held-out Unique sample with its UniqueMultiple counterpart (same EpiRR + strand track) and compare the two probability vectors produced by the **same fold model**.

    Edit the **CONFIG** cell below, then run all cells.
    """
    )
    return


@app.cell
def _(Path):
    # ----------------------------- CONFIG (edit me) -----------------------------
    # Root holding the 6 classifier CV-run directories (mirrors predict_CV_6classifiers.sh).
    BASE_DIR = (
        Path.home()
        / "mounts/narval-mount"
        / "projects/rrg-jacquesp-ab/rabyj"
        / "epiclass-project/output/epiclass-logs/epiatlas-dfreeze-v2.1/hg38_100kb_all_none"
    )
    CV_SUBDIR = "10fold-oversampling"  # CV run dir holding split*/

    # classifier name -> its CV-run directory (the folder containing split0/, split1/, ...).
    MODELS = {
        "assay": BASE_DIR / "assay_epiclass_1l_3000n" / "11c" / CV_SUBDIR,
        "cell_type": BASE_DIR
        / "harmonized_sample_ontology_intermediate_1l_3000n"
        / CV_SUBDIR,
        "life_stage": BASE_DIR / "harmonized_donor_life_stage_1l_3000n" / CV_SUBDIR,
        "sex": BASE_DIR / "harmonized_donor_sex_1l_3000n" / "w-mixed" / CV_SUBDIR,
        "cancer": BASE_DIR / "harmonized_sample_cancer_high_1l_3000n" / CV_SUBDIR,
        "biomaterial": BASE_DIR / "harmonized_biomaterial_type_1l_3000n" / CV_SUBDIR,
    }

    # The single concatenated CSV lives in <cv_root>/<PREDICTIONS_SUBDIR>/<RUN_SUBDIR>/.
    PREDICTIONS_SUBDIR = "predictionsCV"
    RUN_SUBDIR = "RNA_UniqueMultiple"

    # Unstranded RNA (plusRaw + minusRaw summed) predictions live in a sibling run
    # dir; the Unique baseline is re-read from RUN_SUBDIR above (no re-run needed).
    UNSTRANDED_RUN_SUBDIR = "RNA_Unstranded"

    # Mapping TSV written by utils/preprocessing/sum_stranded_rna_hdf5.py: bridges each
    # summed sample's new_id -> its two source md5s (-> EpiRR via the metadata JSON).
    # One shared file for all classifiers (the summed inputs don't depend on the model).
    UNSTRANDED_PAIR_MAPPING = (
        Path.home()
        / "Projects/epiclass/output/paper/data/hdf5/rna_unstranded/pair_mapping.tsv"
    )

    # Metadata mapping Unique md5sum -> epirr_id / track_type.
    METADATA_JSON = (
        Path.home()
        / "Projects/epiclass/output/paper/data/metadata/epiatlas/hg38_2023-epiatlas-dfreeze_v2.1_w_encode_noncore_2.json"
    )
    return (
        METADATA_JSON,
        MODELS,
        PREDICTIONS_SUBDIR,
        RUN_SUBDIR,
        UNSTRANDED_PAIR_MAPPING,
        UNSTRANDED_RUN_SUBDIR,
    )


@app.cell
def _(METADATA_JSON, MODELS):
    if not METADATA_JSON.exists():
        raise FileNotFoundError(METADATA_JSON)

    for path in MODELS.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return


@app.cell
def _(Path, find_signal_id_lists, np, pd, re):
    # ----------------------------- helpers -----------------------------
    EPIRR_RE = re.compile(r"IHECRE\d+(?:\.\d+)?")
    STRANDS = ("minusRaw", "plusRaw")
    META_COLS = ("origin", "ID", "Predicted class", "True class")

    def parse_um_identity(signal_id):
        """(EpiRR, strand) for a UniqueMultiple signal-ID, or None.

        UniqueMultiple IDs are the full hdf5 stem, e.g.
        'ihec.rna-seq...IHECRE00000717.1...UniqueMultiple.minusRaw_100kb_can'
        -> ('IHECRE00000717.1', 'minusRaw'). Unique md5sums have no IHECRE -> None.
        """
        m = EPIRR_RE.search(signal_id)
        if not m:
            return None
        strand = next((s for s in STRANDS if s in signal_id), None)
        return (m.group(0), strand) if strand else None

    def unique_track_to_strand(track_type):
        """Normalize a Unique track_type ('Unique_minusRaw') to a strand ('minusRaw')."""
        if not track_type:
            return None
        stripped = (
            track_type[len("Unique_") :]
            if track_type.startswith("Unique_")
            else track_type
        )
        return stripped if stripped in STRANDS else None

    def unique_identity(signal_id, metadata):
        """(EpiRR, strand) for a Unique md5sum via metadata, or None."""
        meta = metadata.get(signal_id)
        if meta is None:
            return None
        strand = unique_track_to_strand(meta.get("track_type"))
        epirr = meta.get("epirr_id") or meta.get("reference_registry_id")
        if not epirr or strand is None:
            return None
        m = EPIRR_RE.search(str(epirr))
        return (m.group(0) if m else str(epirr), strand)

    def prob_distances(p, q):
        """(Euclidean L2, total variation = 0.5*L1) between two probability vectors."""
        diff = p - q
        return float(np.linalg.norm(diff)), float(0.5 * np.abs(diff).sum())

    def split_from_origin(origin):
        """Fold name ('split3' / 'fold_3') parsed from an `origin` filename, or None."""
        m = re.match(r"(split\d+|fold_\d+)", str(origin))
        return m.group(1) if m else None

    def find_concatenated(pred_dir):
        """The concatenated_test_prediction_*.csv in pred_dir, or None."""
        if not pred_dir.is_dir():
            return None
        hits = sorted(pred_dir.glob("concatenated_test_prediction*.csv"))
        return hits[0] if hits else None

    def load_concatenated(path):
        """Return (df, classes). Adds a 'split' column parsed from 'origin'.

        Concatenated columns are: origin, ID, Predicted class, <one column per class>.
        """
        df = pd.read_csv(path)
        classes = [c for c in df.columns if c not in META_COLS]
        df = df.copy()
        df["split"] = df["origin"].map(split_from_origin)
        df["ID"] = df["ID"].astype(str)
        return df, classes

    def read_training_ids(cv_root, split_name):
        """Set of signal-IDs in <cv_root>/<split>/<split>_training_*.md5, or None if absent."""
        lists = find_signal_id_lists(cv_root / split_name, f"{split_name}_training_*")
        if not lists:
            return None
        chosen = sorted(lists)[-1]  # newest timestamp if several
        return {ln.strip() for ln in chosen.read_text().splitlines() if ln.strip()}

    def build_comparisons(models, metadata, predictions_subdir, run_subdir):
        """Per-pair comparison DataFrame across all models/folds. Returns (df, notes)."""
        rows, notes = [], []
        for model_name, cv_root in models.items():
            cv_root = Path(cv_root)
            csv = find_concatenated(cv_root / predictions_subdir / run_subdir)
            if csv is None:
                notes.append(f"{model_name}: no concatenated CSV under {run_subdir}/")
                continue
            df, classes = load_concatenated(csv)

            for split_name, sub in df.groupby("split"):
                if split_name is None:
                    notes.append(f"{model_name}: {len(sub)} rows with unparsable origin")
                    continue
                train_ids = read_training_ids(cv_root, split_name)
                if train_ids is None:
                    notes.append(f"{model_name}/{split_name}: no training .md5; skipped")
                    continue

                ids = sub["ID"].tolist()
                probs = sub[classes].to_numpy(dtype=float)

                # Route every row to one side: UniqueMultiple (parsed from the ID) or
                # Unique (resolved via metadata, and only if NOT in this fold's training).
                u_map, m_map = {}, {}
                for sid, prob in zip(ids, probs):
                    um_key = parse_um_identity(sid)
                    if um_key is not None:
                        m_map[um_key] = (sid, prob)
                        continue
                    if sid in train_ids:
                        continue
                    u_key = unique_identity(sid, metadata)
                    if u_key is not None:
                        u_map[u_key] = (sid, prob)

                for key in u_map.keys() & m_map.keys():
                    epirr, strand = key
                    u_sid, u_prob = u_map[key]
                    m_sid, m_prob = m_map[key]
                    l2, tv = prob_distances(u_prob, m_prob)
                    u_arg = classes[int(np.argmax(u_prob))]
                    m_arg = classes[int(np.argmax(m_prob))]
                    rows.append(
                        dict(
                            model=model_name,
                            split=split_name,
                            epirr=epirr,
                            strand=strand,
                            unique_id=u_sid,
                            multiple_id=m_sid,
                            unique_pred=u_arg,
                            multiple_pred=m_arg,
                            argmax_agree=bool(u_arg == m_arg),
                            l2=l2,
                            tv=tv,
                        )
                    )
        return pd.DataFrame(rows), notes

    return (
        build_comparisons,
        find_concatenated,
        load_concatenated,
        parse_um_identity,
        unique_identity,
    )


@app.cell
def _(METADATA_JSON, Metadata, mo):
    metadata = Metadata(METADATA_JSON) if METADATA_JSON.is_file() else None
    mo.md(
        f"Loaded metadata: **{len(list(metadata.signal_ids))} samples** from `{METADATA_JSON}`"
        if metadata is not None
        else f"⚠️ Metadata JSON not found at `{METADATA_JSON}` — fix the CONFIG cell."
    )
    return (metadata,)


@app.cell
def _(MODELS):
    subset_models = {k: v for k, v in MODELS.items() if k not in ["assay", "cell_type"]}
    return (subset_models,)


@app.cell
def _(subset_models):
    subset_models
    return


@app.cell
def _(
    PREDICTIONS_SUBDIR,
    RUN_SUBDIR,
    build_comparisons,
    metadata,
    mo,
    subset_models,
):
    comparisons, _notes = (
        build_comparisons(subset_models, metadata, PREDICTIONS_SUBDIR, RUN_SUBDIR)
        if metadata is not None
        else (None, ["Metadata not loaded; cannot build comparisons."])
    )

    _msg = (
        f"Matched **{len(comparisons)} pairs** across {comparisons['model'].nunique()} models.\n\n"
        if comparisons is not None and not comparisons.empty
        else "No pairs matched.\n\n"
    )
    if _notes:
        _msg += "Notes:\n\n" + "\n".join(f"- {n}" for n in _notes)
    mo.md(_msg)
    return (comparisons,)


@app.cell
def _(comparisons):
    # Raw per-pair table (one row per matched Unique/UniqueMultiple pair, per fold).
    comparisons
    return


@app.cell
def _(comparisons, mo, pd):
    # Per-model summary: pair counts, argmax-disagreement rate, mean distances.
    if comparisons is None or comparisons.empty:
        _summary = mo.md("No data to summarize.")
    else:
        _g = comparisons.groupby("model")
        _summary = pd.DataFrame(
            {
                "n_pairs": _g.size(),
                "n_disagree": _g["argmax_agree"].apply(lambda s: int((~s).sum())),
                "disagree_rate": _g["argmax_agree"].apply(lambda s: float((~s).mean())),
                "mean_l2": _g["l2"].mean(),
                "mean_tv": _g["tv"].mean(),
            }
        ).reset_index()
    _summary
    return


@app.cell
def _(comparisons, mo):
    # Samples whose argmax class CHANGES between Unique and UniqueMultiple — the flags.
    if comparisons is None or comparisons.empty:
        _flagged = mo.md("No data.")
    else:
        _df = comparisons[~comparisons["argmax_agree"]][
            [
                "model",
                "split",
                "epirr",
                "strand",
                "unique_pred",
                "multiple_pred",
                "l2",
                "tv",
            ]
        ].sort_values(["model", "l2"], ascending=[True, False])
        _flagged = mo.vstack(
            [mo.md(f"### Argmax disagreements: {len(_df)} pair(s)"), mo.ui.table(_df)]
        )
    _flagged
    return


@app.cell
def _(comparisons, mo, px):
    # Distance histograms (per model), one for Euclidean L2 and one for total variation.
    if comparisons is None or comparisons.empty:
        _plots = mo.md("No data to plot.")
    else:
        _fig_l2 = px.histogram(
            comparisons,
            x="l2",
            color="model",
            barmode="overlay",
            nbins=60,
            marginal="box",
            title="Euclidean (L2) distance between Unique and UniqueMultiple prob. vectors",
            template="plotly_white",
        )
        _fig_tv = px.histogram(
            comparisons,
            x="tv",
            color="model",
            barmode="overlay",
            nbins=60,
            marginal="box",
            title="Total variation (½·L1) distance — fraction of probability mass moved",
            template="plotly_white",
        )
        _plots = mo.vstack([_fig_l2, _fig_tv])
    _plots
    return


@app.cell
def _(mo):
    mo.md(
        """
    # Accuracy / F1 vs ground-truth — Unique vs UniqueMultiple, per task

    The stability section above only compares the two mappings *to each other*. Here we score each mapping against the **true label** (from the metadata JSON) for **every** classification task, so we can see which mapping the models classify more correctly.

    Pairs are restricted to each fold's **validation set** (`splitN_validation_*` list). This is deliberate: the RNA-Seq prediction list also holds samples whose true label is **not** one of the classifier's output classes (e.g. cell-type ontology terms the model was never trained on) — scoring those would unfairly tank accuracy/F1. A sample is in a fold's validation set only if its label is in that classifier's class space, so the metric is well-defined. Both mappings of a pair share the same true label, so Unique vs UniqueMultiple is scored on identical samples.
    """
    )
    return


@app.cell
def _():
    # Task -> metadata category holding that task's ground-truth label.
    # (Keys mirror MODELS; values are columns in the metadata JSON.)
    TASK_CATEGORY = {
        "assay": "assay_epiclass",
        "cell_type": "harmonized_sample_ontology_intermediate",
        "life_stage": "harmonized_donor_life_stage",
        "sex": "harmonized_donor_sex",
        "cancer": "harmonized_sample_cancer_high",
        "biomaterial": "harmonized_biomaterial_type",
    }
    return (TASK_CATEGORY,)


@app.cell
def _(
    Path,
    find_concatenated,
    find_signal_id_lists,
    load_concatenated,
    np,
    parse_um_identity,
    pd,
    unique_identity,
):
    def build_task_metrics(
        models, metadata, task_category, predictions_subdir, run_subdir
    ):
        """Per-task, per-fold Accuracy / macro-F1 / Brier per mapping. Returns (df, notes).

        For each (task, fold): keep Unique samples in that fold's validation list, pair
        them with their UniqueMultiple counterpart on (EpiRR, strand), then score each
        mapping against the metadata true label -- Accuracy and macro-F1 on the argmax,
        and the multiclass Brier score on the full probability vector. Long-form rows:
        (model, split, mapping, Accuracy, F1_macro, Brier, n).
        """
        from sklearn.metrics import accuracy_score, brier_score_loss, f1_score

        def read_validation_ids(cv_root, split_name):
            lists = find_signal_id_lists(
                cv_root / split_name, f"{split_name}_validation_*"
            )
            if not lists:
                return None
            chosen = sorted(lists)[-1]  # newest timestamp if several
            return {ln.strip() for ln in chosen.read_text().splitlines() if ln.strip()}

        rows, notes = [], []
        for model_name, cv_root in models.items():
            cv_root = Path(cv_root)
            category = task_category.get(model_name)
            if category is None:
                notes.append(f"{model_name}: no metadata category mapping; skipped")
                continue
            csv = find_concatenated(cv_root / predictions_subdir / run_subdir)
            if csv is None:
                notes.append(f"{model_name}: no concatenated CSV under {run_subdir}/")
                continue
            df, classes = load_concatenated(csv)

            for split_name, sub in df.groupby("split"):
                if split_name is None:
                    continue
                valid_ids = read_validation_ids(cv_root, split_name)
                if valid_ids is None:
                    notes.append(
                        f"{model_name}/{split_name}: no validation list; skipped"
                    )
                    continue

                ids = sub["ID"].tolist()
                probs = sub[classes].to_numpy(dtype=float)

                # UniqueMultiple from the ID; Unique only if it is in this fold's
                # validation set (guarantees its true label is in the class space).
                u_map, m_map = {}, {}
                for sid, prob in zip(ids, probs):
                    um_key = parse_um_identity(sid)
                    if um_key is not None:
                        m_map[um_key] = (sid, prob)
                        continue
                    if sid not in valid_ids:
                        continue
                    u_key = unique_identity(sid, metadata)
                    if u_key is not None:
                        u_map[u_key] = (sid, prob)

                pairs = sorted(u_map.keys() & m_map.keys())
                class_idx = {c: i for i, c in enumerate(classes)}
                y_true, u_args, m_args, u_probs, m_probs = [], [], [], [], []
                for key in pairs:
                    u_sid, u_prob = u_map[key]
                    _, m_prob = m_map[key]
                    rec = metadata.get(u_sid)
                    true_label = rec.get(category) if rec else None
                    if not true_label or true_label not in class_idx:
                        continue
                    y_true.append(true_label)
                    u_args.append(classes[int(np.argmax(u_prob))])
                    m_args.append(classes[int(np.argmax(m_prob))])
                    u_probs.append(u_prob)
                    m_probs.append(m_prob)

                if not y_true:
                    if not pairs:
                        notes.append(
                            f"{model_name}/{split_name}: 0 validation pairs matched"
                        )
                    else:
                        notes.append(
                            f"{model_name}/{split_name}: {len(pairs)} pairs matched but "
                            f"none had a '{category}' label in the class space -- check "
                            "METADATA_JSON / re-run the metadata cell"
                        )
                    continue

                # Renormalize each row to sum to 1 (stored softmax probs drift slightly
                # off 1.0; brier_score_loss warns otherwise) -- argmax is unaffected.
                def _norm(probs):
                    mat = np.asarray(probs)
                    return mat / mat.sum(axis=1, keepdims=True)

                # macro-F1 over the true classes present this fold (same label set for
                # both mappings, so the comparison is apples-to-apples).
                labels = sorted(set(y_true))
                for mapping, y_pred, p_mat in (
                    ("Unique", u_args, _norm(u_probs)),
                    ("UniqueMultiple", m_args, _norm(m_probs)),
                ):
                    rows.append(
                        dict(
                            model=model_name,
                            split=split_name,
                            mapping=mapping,
                            Accuracy=accuracy_score(y_true, y_pred),
                            F1_macro=f1_score(
                                y_true,
                                y_pred,
                                labels=labels,
                                average="macro",
                                zero_division=0,
                            ),
                            # Multiclass Brier (a proper score on the full prob vector,
                            # lower=better) -- argmax-blind, unlike Accuracy/F1. labels=
                            # classes maps p_mat columns (model-output order) to labels;
                            # scale_by_half=False keeps the standard sum-form range [0, 2].
                            Brier=brier_score_loss(
                                y_true, p_mat, labels=classes, scale_by_half=False
                            ),
                            n=len(y_true),
                        )
                    )
        return pd.DataFrame(rows), notes

    return (build_task_metrics,)


@app.cell
def _(
    MODELS,
    PREDICTIONS_SUBDIR,
    RUN_SUBDIR,
    TASK_CATEGORY,
    build_task_metrics,
    metadata,
    mo,
):
    task_metrics, _notes = (
        build_task_metrics(
            MODELS, metadata, TASK_CATEGORY, PREDICTIONS_SUBDIR, RUN_SUBDIR
        )
        if metadata is not None
        else (None, ["Metadata not loaded; cannot compute metrics."])
    )

    _msg = (
        f"Computed metrics for **{task_metrics['model'].nunique()} tasks** "
        f"across {task_metrics['split'].nunique()} folds.\n\n"
        if task_metrics is not None and not task_metrics.empty
        else "No metrics computed.\n\n"
    )
    if _notes:
        _msg += "Notes:\n\n" + "\n".join(f"- {n}" for n in _notes)
    mo.md(_msg)
    return (task_metrics,)


@app.cell
def _(task_metrics):
    # Long-form metrics: one row per (task, fold, mapping).
    task_metrics
    return


@app.cell
def _(go, make_subplots, mo, task_metrics):
    # Grouped box plots: Accuracy, macro-F1 and Brier per task, Unique vs
    # UniqueMultiple, each box a distribution across the 10 CV folds (fig2 styling).
    # Accuracy/F1: higher is better (y in [0, 1]). Brier: lower is better.
    if task_metrics is None or task_metrics.empty:
        _out = mo.md("No metrics to plot.")
    else:
        _metric_names = ["Accuracy", "F1_macro", "Brier"]
        _subplot_titles = ["Accuracy (↑)", "F1_macro (↑)", "Brier (↓)"]
        _mappings = ["Unique", "UniqueMultiple"]
        _colors = {"Unique": "#636EFA", "UniqueMultiple": "#EF553B"}

        # Order tasks by mean Unique accuracy (most accurate first).
        _order = (
            task_metrics[task_metrics["mapping"] == "Unique"]
            .groupby("model")["Accuracy"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        _fig = make_subplots(
            rows=1,
            cols=len(_metric_names),
            subplot_titles=_subplot_titles,
            horizontal_spacing=0.06,
        )
        for _col, _metric in enumerate(_metric_names, start=1):
            for _mapping in _mappings:
                _s = task_metrics[task_metrics["mapping"] == _mapping]
                _fig.add_trace(
                    go.Box(
                        x=_s["model"],
                        y=_s[_metric],
                        name=_mapping,
                        fillcolor=_colors[_mapping],
                        line=dict(color="black", width=1.2),
                        marker=dict(size=3, color="white", line_width=1),
                        boxmean=True,
                        boxpoints="all",
                        pointpos=0,
                        legendgroup=_mapping,
                        showlegend=_col == 1,
                    ),
                    row=1,
                    col=_col,
                )
        _fig.update_xaxes(categoryorder="array", categoryarray=_order)
        # Accuracy/F1 share a [0, 1] scale; Brier auto-scales (small, lower=better).
        _fig.update_yaxes(range=[0, 1.02], row=1, col=1)
        _fig.update_yaxes(range=[0, 1.02], row=1, col=2)
        _fig.update_yaxes(rangemode="tozero", row=1, col=3)
        _fig.update_layout(
            boxmode="group",
            template="plotly_white",
            title_text="RNA-Seq Unique vs UniqueMultiple — per-task metrics (validation-set pairs, across 10 folds)",
            yaxis_title="Value",
            height=550,
            width=1500,
        )
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Does conformal prediction catch the new Unique→Multiple errors?

    The metrics above show UniqueMultiple loses accuracy/F1 and worsens Brier vs Unique. Here we ask whether **post-hoc conformal prediction** (the `utils/conformal/` layer, SAPS score) would *catch* those new errors instead of letting them pass as confident-but-wrong predictions.

    Per fold: calibrate a SAPS conformal threshold on that fold's **Unique** RNA validation predictions (true labels from metadata, same restriction as the metrics section), then build prediction **sets** for the paired **UniqueMultiple** samples under the same fold model. Counts are summed across folds, then rates derived (sum-then-divide).

    Important caveat (see `utils/conformal/methods.md`): calibrating on Unique and testing on Multiple **breaks exchangeability** — Multiple is a near-OOD shift — so the marginal coverage guarantee is *not* assured and can degrade. The actionable signal under shift is the **prediction-set geometry**, read here via the per-sample flag (`utils/conformal/prediction.py:classify_flags`):

    - **clean** — singleton = true class (confident & correct, no review)
    - **hedge** — set of ≥2 classes (ambiguous; may still contain the true class)
    - **disagree** — singleton ≠ true class (confident **wrong** — the failure CP should avoid)
    - **empty** — no class fits (abstain → manual review)

    The key view is the **new-error breakdown**: of the samples Unique got right but Multiple's argmax got wrong, how many does CP turn into a hedge (ideally still containing the truth) or an empty (abstain), versus let slip through as a confident-wrong singleton (`disagree`).
    """
    )
    return


@app.cell
def _():
    # Conformal config. SAPS is methods.md's default for near-OOD/uncalibrated nets.
    # alpha = target miscoverage (coverage target 1 - alpha); the headline alpha drives
    # the new-error composition plot. Calibration uses RNA Unique validation only.
    CP_METHOD = "SAPS"
    CP_ALPHAS = (0.05, 0.1, 0.2)
    CP_HEADLINE_ALPHA = 0.05
    return CP_ALPHAS, CP_HEADLINE_ALPHA, CP_METHOD


@app.cell
def _(
    Counter,
    Path,
    calibrate_predictor,
    classify_flags,
    find_concatenated,
    find_signal_id_lists,
    load_concatenated,
    np,
    parse_um_identity,
    pd,
    predict_sets,
    unique_identity,
):
    def build_cp_unique_to_multiple(
        models, metadata, task_category, predictions_subdir, run_subdir, method, alphas
    ):
        """Calibrate CP on Unique RNA validation, apply to paired UniqueMultiple.

        Per (task, fold): calibrate ``method`` on the fold's Unique-validation RNA
        predictions, build sets for the paired UniqueMultiple samples, and tally the
        flag composition + the new-error breakdown (Unique argmax right, Multiple
        argmax wrong). Counts summed across folds.

        Returns ``(summary_df, samples_df, notes)``: ``summary_df`` has one row per
        (model, alpha); ``samples_df`` has one row per (model, alpha, paired sample)
        with the true/unique/multiple labels, the prediction set, its flag, and an
        ``is_new_error`` marker -- the table to inspect confident-wrong cases.
        """

        def read_validation_ids(cv_root, split_name):
            lists = find_signal_id_lists(
                cv_root / split_name, f"{split_name}_validation_*"
            )
            if not lists:
                return None
            chosen = sorted(lists)[-1]
            return {ln.strip() for ln in chosen.read_text().splitlines() if ln.strip()}

        def new_err_bucket(row, size, true_i):
            """Which new-error outcome a prediction set falls into."""
            if size == 0:
                return "empty"  # abstain
            if size == 1:
                return "disagree"  # argmax-wrong singleton (slipped through)
            return "hedge_in" if row[true_i] == 1 else "hedge_out"

        rows, samples, notes = [], [], []
        for model_name, cv_root in models.items():
            cv_root = Path(cv_root)
            category = task_category.get(model_name)
            if category is None:
                notes.append(f"{model_name}: no metadata category mapping; skipped")
                continue
            csv = find_concatenated(cv_root / predictions_subdir / run_subdir)
            if csv is None:
                notes.append(f"{model_name}: no concatenated CSV under {run_subdir}/")
                continue
            df, classes = load_concatenated(csv)
            cidx = {c: i for i, c in enumerate(classes)}

            cov = {a: [0, 0] for a in alphas}  # [n_covered, n]
            size_sum = {a: 0 for a in alphas}
            flag_counts = {a: Counter() for a in alphas}
            new_err_counts = {a: Counter() for a in alphas}
            n_new_err = 0

            for split_name, sub in df.groupby("split"):
                if split_name is None:
                    continue
                valid = read_validation_ids(cv_root, split_name)
                if valid is None:
                    notes.append(
                        f"{model_name}/{split_name}: no validation list; skipped"
                    )
                    continue

                probs = sub[classes].to_numpy(float)
                cal_p, cal_t, u_map, m_map = [], [], {}, {}
                for sid, p in zip(sub["ID"], probs):
                    um_key = parse_um_identity(sid)
                    if um_key is not None:
                        m_map[um_key] = (sid, p)
                        continue
                    if sid not in valid:
                        continue
                    rec = metadata.get(sid)
                    lab = rec.get(category) if rec else None
                    if not lab or lab not in cidx:
                        continue
                    cal_p.append(p)
                    cal_t.append(cidx[lab])
                    u_key = unique_identity(sid, metadata)
                    if u_key is not None:
                        u_map[u_key] = (sid, p, cidx[lab])

                keys = sorted(u_map.keys() & m_map.keys())
                if len(cal_p) < 20 or not keys:
                    if not keys:
                        notes.append(f"{model_name}/{split_name}: no eval pairs")
                    continue
                cal_p = np.asarray(cal_p)
                cal_t = np.asarray(cal_t)
                m_prob = np.array([m_map[k][1] for k in keys])
                y = np.array([u_map[k][2] for k in keys])
                u_arg = np.array([u_map[k][1] for k in keys]).argmax(1)
                m_arg = m_prob.argmax(1)
                new_err = (u_arg == y) & (m_arg != y)
                n_new_err += int(new_err.sum())

                for a in alphas:
                    predictor = calibrate_predictor(cal_p, cal_t, method, a)
                    mem = predict_sets(predictor, m_prob)
                    flags = classify_flags(mem, y)
                    flag_counts[a].update(flags)
                    cov[a][0] += int(mem[np.arange(len(y)), y].sum())
                    cov[a][1] += len(y)
                    size_sum[a] += int(mem.sum())
                    for i, key in enumerate(keys):
                        row = mem[i]
                        sz = int(row.sum())
                        if new_err[i]:
                            new_err_counts[a][new_err_bucket(row, sz, y[i])] += 1
                        samples.append(
                            dict(
                                model=model_name,
                                alpha=a,
                                split=split_name,
                                epirr=key[0],
                                strand=key[1],
                                unique_id=u_map[key][0],
                                multiple_id=m_map[key][0],
                                true_class=classes[y[i]],
                                unique_pred=classes[u_arg[i]],
                                multiple_pred=classes[m_arg[i]],
                                prediction_set=";".join(
                                    classes[j] for j in np.flatnonzero(row)
                                ),
                                set_size=sz,
                                flag=flags[i],
                                is_new_error=bool(new_err[i]),
                            )
                        )

            for a in alphas:
                n = cov[a][1]
                if n == 0:
                    continue
                fc, ec = flag_counts[a], new_err_counts[a]
                caught = ec["empty"] + ec["hedge_in"] + ec["hedge_out"]
                rows.append(
                    dict(
                        model=model_name,
                        alpha=a,
                        n_classes=len(classes),
                        n_pairs=n,
                        target_coverage=1 - a,
                        coverage=cov[a][0] / n,
                        avg_set_size=size_sum[a] / n,
                        clean=fc["clean"] / n,
                        hedge=fc["hedge"] / n,
                        disagree=fc["disagree"] / n,
                        empty=fc["empty"] / n,
                        n_new_err=n_new_err,
                        ne_recovered=ec["hedge_in"],
                        ne_hedge_out=ec["hedge_out"],
                        ne_empty=ec["empty"],
                        ne_disagree=ec["disagree"],
                        catch_rate=caught / n_new_err if n_new_err else float("nan"),
                        recover_rate=(
                            ec["hedge_in"] / n_new_err if n_new_err else float("nan")
                        ),
                    )
                )
        return pd.DataFrame(rows), pd.DataFrame(samples), notes

    return (build_cp_unique_to_multiple,)


@app.cell
def _(
    CP_ALPHAS,
    CP_METHOD,
    MODELS,
    PREDICTIONS_SUBDIR,
    RUN_SUBDIR,
    TASK_CATEGORY,
    build_cp_unique_to_multiple,
    metadata,
    mo,
):
    cp_summary, cp_samples, _notes = (
        build_cp_unique_to_multiple(
            MODELS,
            metadata,
            TASK_CATEGORY,
            PREDICTIONS_SUBDIR,
            RUN_SUBDIR,
            CP_METHOD,
            CP_ALPHAS,
        )
        if metadata is not None
        else (None, None, ["Metadata not loaded; cannot run conformal prediction."])
    )

    _msg = (
        f"Conformal ({CP_METHOD}) calibrated on Unique, applied to UniqueMultiple for "
        f"**{cp_summary['model'].nunique()} tasks** × {len(CP_ALPHAS)} alphas "
        f"({len(cp_samples)} per-sample rows).\n\n"
        if cp_summary is not None and not cp_summary.empty
        else "No conformal results.\n\n"
    )
    if _notes:
        _msg += "Notes:\n\n" + "\n".join(f"- {n}" for n in _notes)
    mo.md(_msg)
    return cp_samples, cp_summary


@app.cell
def _(cp_summary):
    # Per-(task, alpha) conformal summary: coverage vs target, set size, flag mix, and
    # the new-error breakdown (recovered / hedge_out / empty / confident-wrong disagree).
    cp_summary
    return


@app.cell
def _(CP_HEADLINE_ALPHA, CP_METHOD, cp_summary, go, mo):
    # Headline plot: of the NEW errors (Unique right, Multiple argmax wrong), what does
    # CP do with them at the QC alpha? Stacked fractions per task. "recovered" = hedge
    # still containing the true class; "disagree" = confident-wrong singleton (slipped).
    _cp_summary = cp_summary[cp_summary["model"] != "assay"]
    if _cp_summary is None or _cp_summary.empty:
        _out = mo.md("No conformal results to plot.")
    else:
        _d = _cp_summary[_cp_summary["alpha"] == CP_HEADLINE_ALPHA].copy()
        _d = _d[_d["n_new_err"] > 0]
        _d["recovered"] = _d["ne_recovered"] / _d["n_new_err"]
        _d["abstain (empty)"] = _d["ne_empty"] / _d["n_new_err"]
        _d["hedge (no true)"] = _d["ne_hedge_out"] / _d["n_new_err"]
        _d["confident-wrong"] = _d["ne_disagree"] / _d["n_new_err"]
        _d = _d.sort_values("recovered", ascending=False)
        _outcomes = {
            "recovered": "#2ca02c",
            "abstain (empty)": "#1f77b4",
            "hedge (no true)": "#ff7f0e",
            "confident-wrong": "#d62728",
        }
        _fig = go.Figure()
        for _label, _color in _outcomes.items():
            _fig.add_trace(
                go.Bar(
                    x=_d["model"],
                    y=_d[_label],
                    name=_label,
                    marker_color=_color,
                    text=_d[_label].map(lambda v: f"{v:.0%}"),
                    textposition="inside",
                    hovertemplate="%{x}: %{y:.1%}<extra>" + _label + "</extra>",
                )
            )
        _fig.update_layout(
            barmode="stack",
            template="plotly_white",
            title_text=(
                f"Fate of the new Unique→Multiple errors under {CP_METHOD} "
                f"(α={CP_HEADLINE_ALPHA}) — green+blue = caught, red = slipped through"
            ),
            yaxis_title="fraction of new errors",
            xaxis_title="task",
            height=500,
            width=950,
            legend=dict(orientation="h", y=-0.18),
        )
        _fig.update_yaxes(range=[0, 1.0])
        _out = _fig
    _out
    return


@app.cell
def _(CP_METHOD, cp_summary, go, mo):
    # Coverage vs target across alphas: where does the Unique→Multiple shift break the
    # marginal guarantee? Bars = empirical coverage on Multiple; dashed lines = targets.
    if cp_summary is None or cp_summary.empty:
        _out = mo.md("No conformal results to plot.")
    else:
        _alphas = sorted(cp_summary["alpha"].unique())
        _order = (
            cp_summary[cp_summary["alpha"] == _alphas[0]]
            .sort_values("coverage", ascending=False)["model"]
            .tolist()
        )
        _fig = go.Figure()
        for _a in _alphas:
            _s = cp_summary[cp_summary["alpha"] == _a]
            _fig.add_trace(
                go.Bar(
                    x=_s["model"],
                    y=_s["coverage"],
                    name=f"α={_a} (cov on Multiple)",
                    hovertemplate="%{x}: cov=%{y:.3f}<extra>α=" + str(_a) + "</extra>",
                )
            )
            _fig.add_hline(
                y=1 - _a,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"target {1 - _a:.2f}",
                annotation_position="right",
            )
        _fig.update_xaxes(categoryorder="array", categoryarray=_order)
        _fig.update_yaxes(range=[0.5, 1.01])
        _fig.update_layout(
            barmode="group",
            template="plotly_white",
            title_text=(
                f"{CP_METHOD} empirical coverage on UniqueMultiple vs target "
                "(bars below the dashed line = shift broke the guarantee)"
            ),
            yaxis_title="empirical coverage",
            xaxis_title="task",
            height=500,
            width=1000,
        )
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Per-sample conformal results — inspect confident-wrong individually

    `cp_samples` is the full per-(task, alpha, sample) table: true / unique-pred / multiple-pred labels, the prediction set + its flag, and `is_new_error`. Use it to drill into specific failures. The table below isolates the **confident-wrong** cases (`flag == "disagree"` and `is_new_error`) at the headline α, and the transition summary shows which true→predicted class flips dominate (e.g. sex is overwhelmingly male→female, not `mixed`).
    """
    )
    return


@app.cell
def _(cp_samples):
    # Full per-sample CP table (every paired sample, every alpha) for separate analysis.
    cp_samples
    return


@app.cell
def _(CP_HEADLINE_ALPHA, cp_samples, mo, pd):
    # Confident-wrong NEW errors at the headline alpha: CP returned a single wrong class
    # (the dangerous slip-through). Interactive table + true->predicted transition counts.
    if cp_samples is None or cp_samples.empty:
        cp_confident_wrong = pd.DataFrame()
        _view = mo.md("No conformal samples.")
    else:
        cp_confident_wrong = cp_samples[
            (cp_samples["alpha"] == CP_HEADLINE_ALPHA)
            & (cp_samples["flag"] == "disagree")
            & (cp_samples["is_new_error"])
        ].sort_values(["model", "true_class", "multiple_pred"])
        _transitions = (
            cp_confident_wrong.groupby(
                ["model", "true_class", "multiple_pred"], as_index=False
            )
            .size()
            .sort_values(["model", "size"], ascending=[True, False])
        )
        _view = mo.vstack(
            [
                mo.md(
                    f"### {len(cp_confident_wrong)} confident-wrong new errors "
                    f"(α={CP_HEADLINE_ALPHA})"
                ),
                mo.md("**Dominant true→predicted flips:**"),
                mo.ui.table(_transitions, selection=None),
                mo.md("**Individual samples:**"),
                mo.ui.table(cp_confident_wrong, selection=None),
            ]
        )
    _view
    return (cp_confident_wrong,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Unstranded RNA (plusRaw + minusRaw summed) — 2-to-1 vs Unique

    A third representation: the two stranded RNA tracks are **summed into one
    unstranded signal per EpiRR** (see `utils/preprocessing/sum_stranded_rna_hdf5.py`)
    and predicted with the same fold models. Because there is now **one** unstranded
    sample per EpiRR but **two** Unique stranded samples (plusRaw + minusRaw), the
    match is **2-to-1**: the Unique reference is the **average** of the plusRaw and
    minusRaw probability vectors (over the strands the fold held out), compared to the
    single unstranded prediction from the same fold model.

    The Unique baseline is re-read from the existing `RNA_UniqueMultiple` run (its rows
    already hold the Unique predictions), so only the unstranded predictions need to be
    generated. A **new error** here is a pair the averaged-Unique argmax got right but
    the unstranded argmax got wrong.

    A summed sample's prediction `ID` is a content-free md5 of its two source filenames,
    so identity is recovered through the **mapping TSV** written by the summing utility
    (`new_id → source md5 → EpiRR` via the metadata JSON) — see the next cell.
    """
    )
    return


@app.cell
def _(
    Path,
    find_concatenated,
    find_signal_id_lists,
    load_concatenated,
    np,
    pd,
    re,
    unique_identity,
):
    def read_fold_id_list(cv_root, split_name, kind):
        """Set of signal-IDs in <cv_root>/<split>/<split>_<kind>_*, or None if absent."""
        lists = find_signal_id_lists(cv_root / split_name, f"{split_name}_{kind}_*")
        if not lists:
            return None
        chosen = sorted(lists)[-1]  # newest timestamp if several
        return {ln.strip() for ln in chosen.read_text().splitlines() if ln.strip()}

    def unstranded_by_epirr(sub, classes, id_to_epirr):
        """{EpiRR: (id, prob)} for a fold's unstranded rows (one per EpiRR).

        The summed sample's ``ID`` is a content-free md5 of its two source
        filenames; ``id_to_epirr`` (built from the utility's mapping TSV +
        metadata) resolves it to an EpiRR. A bare IHECRE regex is kept as a
        fallback for hand-named files that already carry the EpiRR.
        """
        out = {}
        for sid, prob in zip(sub["ID"], sub[classes].to_numpy(dtype=float)):
            epirr = id_to_epirr.get(str(sid))
            if epirr is None:
                m = re.search(r"IHECRE\d+(?:\.\d+)?", str(sid))
                epirr = m.group(0) if m else None
            if epirr is not None:
                out[epirr] = (sid, prob)
        return out

    def unique_avg_by_epirr(sub, classes, metadata, keep_ids):
        """{EpiRR: (rep_id, mean_prob, n_strands)} averaging Unique strands in keep_ids."""
        acc = {}
        for sid, prob in zip(sub["ID"], sub[classes].to_numpy(dtype=float)):
            if sid not in keep_ids:
                continue
            key = unique_identity(sid, metadata)  # (EpiRR, strand) via metadata
            if key is None:
                continue
            epirr = key[0]
            rep_id, probs = acc.get(epirr, (sid, []))
            probs.append(prob)
            acc[epirr] = (rep_id, probs)
        return {e: (rid, np.mean(ps, axis=0), len(ps)) for e, (rid, ps) in acc.items()}

    def build_unstranded_comparisons(
        models,
        metadata,
        id_to_epirr,
        predictions_subdir,
        unique_subdir,
        unstranded_subdir,
    ):
        """Per-EpiRR averaged-Unique vs unstranded comparison. Returns (df, notes).

        Per (model, fold): read the unstranded prediction for each EpiRR (identity via
        ``id_to_epirr``), average the held-out Unique strand predictions (dropping IDs
        in the fold's training list) into one reference per EpiRR, match on EpiRR, and
        measure argmax (dis)agreement + L2 / total-variation distance between the two
        probability vectors.
        """
        rows, notes = [], []
        for model_name, cv_root in models.items():
            cv_root = Path(cv_root)
            u_csv = find_concatenated(cv_root / predictions_subdir / unique_subdir)
            s_csv = find_concatenated(cv_root / predictions_subdir / unstranded_subdir)
            if u_csv is None:
                notes.append(f"{model_name}: no Unique CSV under {unique_subdir}/")
                continue
            if s_csv is None:
                notes.append(
                    f"{model_name}: no unstranded CSV under {unstranded_subdir}/"
                )
                continue
            u_df, u_classes = load_concatenated(u_csv)
            s_df, s_classes = load_concatenated(s_csv)
            if u_classes != s_classes:
                notes.append(f"{model_name}: class columns differ Unique vs unstranded")
                continue
            classes = u_classes

            u_by_split = dict(tuple(g) for g in u_df.groupby("split"))
            for split_name, s_sub in s_df.groupby("split"):
                if split_name is None or split_name not in u_by_split:
                    continue
                train_ids = read_fold_id_list(cv_root, split_name, "training")
                if train_ids is None:
                    notes.append(f"{model_name}/{split_name}: no training list; skipped")
                    continue
                u_sub = u_by_split[split_name]
                keep = set(u_sub["ID"]) - train_ids  # Unique rows not trained on
                u_map = unique_avg_by_epirr(u_sub, classes, metadata, keep)
                s_map = unstranded_by_epirr(s_sub, classes, id_to_epirr)

                for epirr in u_map.keys() & s_map.keys():
                    rep_id, u_prob, n_strands = u_map[epirr]
                    s_id, s_prob = s_map[epirr]
                    diff = u_prob - s_prob
                    rows.append(
                        dict(
                            model=model_name,
                            split=split_name,
                            epirr=epirr,
                            n_unique_strands=n_strands,
                            unique_id=rep_id,
                            unstranded_id=s_id,
                            unique_pred=classes[int(np.argmax(u_prob))],
                            unstranded_pred=classes[int(np.argmax(s_prob))],
                            argmax_agree=bool(np.argmax(u_prob) == np.argmax(s_prob)),
                            l2=float(np.linalg.norm(diff)),
                            tv=float(0.5 * np.abs(diff).sum()),
                        )
                    )
        return pd.DataFrame(rows), notes

    return (
        build_unstranded_comparisons,
        read_fold_id_list,
        unique_avg_by_epirr,
        unstranded_by_epirr,
    )


@app.cell
def _(UNSTRANDED_PAIR_MAPPING, metadata, mo, pd, unique_identity):
    # Build {new_id -> EpiRR} from the utility's mapping TSV: each summed sample's
    # md5 ID -> its source md5s -> EpiRR via metadata. Either source strand resolves
    # to the same EpiRR, so the first that hits wins.
    def _load_unstranded_id_map(mapping_path, meta):
        if meta is None or not mapping_path.is_file():
            return {}, mapping_path.is_file()
        table = pd.read_csv(mapping_path, sep="\t", dtype=str)
        mapping = {}
        for _, row in table.iterrows():
            for src in (row.get("id_a"), row.get("id_b")):
                key = unique_identity(src, meta) if src else None
                if key is not None:
                    mapping[str(row["new_id"])] = key[0]
                    break
        return mapping, True

    uns_id_to_epirr, _mapping_found = _load_unstranded_id_map(
        UNSTRANDED_PAIR_MAPPING, metadata
    )
    mo.md(
        f"Resolved **{len(uns_id_to_epirr)} summed IDs → EpiRR** from `{UNSTRANDED_PAIR_MAPPING}`."
        if _mapping_found
        else f"⚠️ Pair-mapping TSV not found at `{UNSTRANDED_PAIR_MAPPING}` — "
        "summed IDs will fall back to an IHECRE regex (only works for EpiRR-named files)."
    )
    return (uns_id_to_epirr,)


@app.cell
def _(
    MODELS,
    PREDICTIONS_SUBDIR,
    RUN_SUBDIR,
    UNSTRANDED_RUN_SUBDIR,
    build_unstranded_comparisons,
    metadata,
    mo,
    uns_id_to_epirr,
):
    uns_comparisons, _notes = (
        build_unstranded_comparisons(
            MODELS,
            metadata,
            uns_id_to_epirr,
            PREDICTIONS_SUBDIR,
            RUN_SUBDIR,
            UNSTRANDED_RUN_SUBDIR,
        )
        if metadata is not None
        else (None, ["Metadata not loaded; cannot build comparisons."])
    )

    _msg = (
        f"Matched **{len(uns_comparisons)} EpiRR pairs** across "
        f"{uns_comparisons['model'].nunique()} models.\n\n"
        if uns_comparisons is not None and not uns_comparisons.empty
        else "No unstranded pairs matched.\n\n"
    )
    if _notes:
        _msg += "Notes:\n\n" + "\n".join(f"- {n}" for n in _notes)
    mo.md(_msg)
    return (uns_comparisons,)


@app.cell
def _(uns_comparisons):
    # Raw per-EpiRR table (averaged-Unique vs unstranded, per fold).
    uns_comparisons
    return


@app.cell
def _(mo, pd, uns_comparisons):
    # Per-model summary: pair counts, argmax-disagreement rate, mean distances.
    if uns_comparisons is None or uns_comparisons.empty:
        _summary = mo.md("No data to summarize.")
    else:
        _g = uns_comparisons.groupby("model")
        _summary = pd.DataFrame(
            {
                "n_pairs": _g.size(),
                "n_disagree": _g["argmax_agree"].apply(lambda s: int((~s).sum())),
                "disagree_rate": _g["argmax_agree"].apply(lambda s: float((~s).mean())),
                "mean_l2": _g["l2"].mean(),
                "mean_tv": _g["tv"].mean(),
            }
        ).reset_index()
    _summary
    return


@app.cell
def _(mo, uns_comparisons):
    # New errors: EpiRRs whose argmax class CHANGES from averaged-Unique to unstranded.
    if uns_comparisons is None or uns_comparisons.empty:
        _flagged = mo.md("No data.")
    else:
        _df = uns_comparisons[~uns_comparisons["argmax_agree"]][
            ["model", "split", "epirr", "unique_pred", "unstranded_pred", "l2", "tv"]
        ].sort_values(["model", "l2"], ascending=[True, False])
        _flagged = mo.vstack(
            [mo.md(f"### Argmax disagreements: {len(_df)} EpiRR(s)"), mo.ui.table(_df)]
        )
    _flagged
    return


@app.cell
def _(
    Path,
    find_concatenated,
    load_concatenated,
    np,
    pd,
    read_fold_id_list,
    unique_avg_by_epirr,
    unstranded_by_epirr,
):
    def build_unstranded_task_metrics(
        models,
        metadata,
        id_to_epirr,
        task_category,
        predictions_subdir,
        unique_subdir,
        unstranded_subdir,
    ):
        """Per-task, per-fold Accuracy / macro-F1 / Brier for Unique(avg) vs Unstranded.

        Restricts Unique rows to the fold's validation list (guarantees the true label
        is in the class space), averages the strands per EpiRR, pairs with the unstranded
        prediction, and scores both against the metadata true label. Long-form rows:
        (model, split, mapping, Accuracy, F1_macro, Brier, n).
        """
        from sklearn.metrics import accuracy_score, brier_score_loss, f1_score

        def _norm(probs):
            mat = np.asarray(probs)
            return mat / mat.sum(axis=1, keepdims=True)

        rows, notes = [], []
        for model_name, cv_root in models.items():
            cv_root = Path(cv_root)
            category = task_category.get(model_name)
            if category is None:
                notes.append(f"{model_name}: no metadata category mapping; skipped")
                continue
            u_csv = find_concatenated(cv_root / predictions_subdir / unique_subdir)
            s_csv = find_concatenated(cv_root / predictions_subdir / unstranded_subdir)
            if u_csv is None or s_csv is None:
                notes.append(f"{model_name}: missing Unique or unstranded CSV")
                continue
            u_df, classes = load_concatenated(u_csv)
            s_df, s_classes = load_concatenated(s_csv)
            if classes != s_classes:
                notes.append(f"{model_name}: class columns differ; skipped")
                continue
            cidx = {c: i for i, c in enumerate(classes)}

            u_by_split = dict(tuple(g) for g in u_df.groupby("split"))
            for split_name, s_sub in s_df.groupby("split"):
                if split_name is None or split_name not in u_by_split:
                    continue
                valid = read_fold_id_list(cv_root, split_name, "validation")
                if valid is None:
                    notes.append(f"{model_name}/{split_name}: no validation list")
                    continue
                u_sub = u_by_split[split_name]
                u_map = unique_avg_by_epirr(
                    u_sub, classes, metadata, set(u_sub["ID"]) & valid
                )
                s_map = unstranded_by_epirr(s_sub, classes, id_to_epirr)

                y_true, u_args, s_args, u_probs, s_probs = [], [], [], [], []
                for epirr in sorted(u_map.keys() & s_map.keys()):
                    rep_id, u_prob, _ = u_map[epirr]
                    _, s_prob = s_map[epirr]
                    rec = metadata.get(rep_id)
                    true_label = rec.get(category) if rec else None
                    if not true_label or true_label not in cidx:
                        continue
                    y_true.append(true_label)
                    u_args.append(classes[int(np.argmax(u_prob))])
                    s_args.append(classes[int(np.argmax(s_prob))])
                    u_probs.append(u_prob)
                    s_probs.append(s_prob)

                if not y_true:
                    continue

                labels = sorted(set(y_true))
                for mapping, y_pred, p_mat in (
                    ("Unique (avg)", u_args, _norm(u_probs)),
                    ("Unstranded", s_args, _norm(s_probs)),
                ):
                    rows.append(
                        dict(
                            model=model_name,
                            split=split_name,
                            mapping=mapping,
                            Accuracy=accuracy_score(y_true, y_pred),
                            F1_macro=f1_score(
                                y_true,
                                y_pred,
                                labels=labels,
                                average="macro",
                                zero_division=0,
                            ),
                            Brier=brier_score_loss(
                                y_true, p_mat, labels=classes, scale_by_half=False
                            ),
                            n=len(y_true),
                        )
                    )
        return pd.DataFrame(rows), notes

    return (build_unstranded_task_metrics,)


@app.cell
def _(
    MODELS,
    PREDICTIONS_SUBDIR,
    RUN_SUBDIR,
    TASK_CATEGORY,
    UNSTRANDED_RUN_SUBDIR,
    build_unstranded_task_metrics,
    metadata,
    mo,
    uns_id_to_epirr,
):
    uns_task_metrics, _notes = (
        build_unstranded_task_metrics(
            MODELS,
            metadata,
            uns_id_to_epirr,
            TASK_CATEGORY,
            PREDICTIONS_SUBDIR,
            RUN_SUBDIR,
            UNSTRANDED_RUN_SUBDIR,
        )
        if metadata is not None
        else (None, ["Metadata not loaded; cannot compute metrics."])
    )

    _msg = (
        f"Computed metrics for **{uns_task_metrics['model'].nunique()} tasks** "
        f"across {uns_task_metrics['split'].nunique()} folds.\n\n"
        if uns_task_metrics is not None and not uns_task_metrics.empty
        else "No metrics computed.\n\n"
    )
    if _notes:
        _msg += "Notes:\n\n" + "\n".join(f"- {n}" for n in _notes)
    mo.md(_msg)
    return (uns_task_metrics,)


@app.cell
def _(uns_task_metrics):
    # Long-form metrics: one row per (task, fold, mapping).
    uns_task_metrics
    return


@app.cell
def _(go, make_subplots, mo, uns_task_metrics):
    # Grouped box plots: Accuracy, macro-F1 and Brier per task, Unique(avg) vs Unstranded,
    # each box a distribution across the CV folds. Accuracy/F1 higher=better; Brier lower.
    if uns_task_metrics is None or uns_task_metrics.empty:
        _out = mo.md("No metrics to plot.")
    else:
        _metric_names = ["Accuracy", "F1_macro", "Brier"]
        _subplot_titles = ["Accuracy (↑)", "F1_macro (↑)", "Brier (↓)"]
        _mappings = ["Unique (avg)", "Unstranded"]
        _colors = {"Unique (avg)": "#636EFA", "Unstranded": "#00CC96"}

        _order = (
            uns_task_metrics[uns_task_metrics["mapping"] == "Unique (avg)"]
            .groupby("model")["Accuracy"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        _fig = make_subplots(
            rows=1,
            cols=len(_metric_names),
            subplot_titles=_subplot_titles,
            horizontal_spacing=0.06,
        )
        for _col, _metric in enumerate(_metric_names, start=1):
            for _mapping in _mappings:
                _s = uns_task_metrics[uns_task_metrics["mapping"] == _mapping]
                _fig.add_trace(
                    go.Box(
                        x=_s["model"],
                        y=_s[_metric],
                        name=_mapping,
                        fillcolor=_colors[_mapping],
                        line=dict(color="black", width=1.2),
                        marker=dict(size=3, color="white", line_width=1),
                        boxmean=True,
                        boxpoints="all",
                        pointpos=0,
                        legendgroup=_mapping,
                        showlegend=_col == 1,
                    ),
                    row=1,
                    col=_col,
                )
        _fig.update_xaxes(categoryorder="array", categoryarray=_order)
        _fig.update_yaxes(range=[0, 1.02], row=1, col=1)
        _fig.update_yaxes(range=[0, 1.02], row=1, col=2)
        _fig.update_yaxes(rangemode="tozero", row=1, col=3)
        _fig.update_layout(
            boxmode="group",
            template="plotly_white",
            title_text="RNA-Seq Unique (avg) vs Unstranded — per-task metrics (validation-set pairs, across folds)",
            yaxis_title="Value",
            height=550,
            width=1500,
        )
        _out = _fig
    _out
    return


if __name__ == "__main__":
    app.run()
