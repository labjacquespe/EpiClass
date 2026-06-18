"""Create similarity-stratified cross-validation folds for AVE training.

Produces a fold-definition JSON consumable by ``ave_general_training.py --folds``
(and ``general_training.py --folds``): ``{"fold0": {"md5sum": [ids...]}, ...}``.

Folds are **representative** (balanced strata), stratified primarily by **assay**
and secondarily by **similar cell type**. "Similar cell type" is derived from the
ontology Lin semantic-similarity matrix produced by
``utils/notebooks/metadata/evaluate_biospecimen_similarity.ipynb`` (a square
``term x term`` CSV of Lin similarities over the cohort's
``harmonized_sample_ontology_curie`` terms): close curies are agglomeratively
clustered into cell-type groups, shrinking the otherwise-huge curie cardinality
into something stratifiable.

Stratification label per sample = ``f"{assay}|{celltype_group}"``. Assay has
priority: any composite stratum with fewer than ``n_folds`` members collapses to
an assay-only label, so assay balance always holds and cell-type balance is
best-effort.

By default samples are grouped by **EpiRR** before splitting (one EpiRR = one
source biological sample profiled across many assays/tracks): all md5sums of an
EpiRR stay in the same fold, mirroring
``LazyEpiAtlasFoldFactory._split_dataset`` (``StratifiedGroupKFold`` with
``groups=epirr``). This avoids leaking a biological sample across folds.

This module depends only on pandas / numpy / scipy / scikit-learn — it does NOT
import the notebook or the ontology libraries (``nxontology``/``obonet``); those
stay confined to the matrix-building notebook.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.core.metadata import Metadata

ASSAY_FALLBACKS = ["assay", "assay_epiclass"]
EPIRR_FALLBACKS = ["epirr_id", "reference_registry_id"]
REMAINDER_GROUP = "*"  # cell-type group placeholder for assay-only collapse
RARE_STRATUM = "__rare__"
DEFAULT_SWEEP = [0.4, 0.5, 0.6, 0.7, 0.8]

FoldDefinitions = Dict[str, Dict[str, List[str]]]


# ---------------------------------------------------------------------------
# Cell-type clustering
# ---------------------------------------------------------------------------


def cluster_terms(sim_matrix: pd.DataFrame, sim_threshold: float) -> Dict[str, int]:
    """Cluster ontology terms by Lin similarity into cell-type groups.

    ``sim_matrix`` is a square ``term x term`` DataFrame of Lin similarities
    (1.0 on the diagonal, NaN for unmapped pairs). Terms are agglomeratively
    clustered (average linkage) on the distance ``1 - similarity``; the tree is
    cut so members of a group are at least ``sim_threshold`` similar.

    Returns ``{term: group_id}``.
    """
    terms = list(sim_matrix.index)
    if list(sim_matrix.columns) != terms:
        # Align columns to the row order so the matrix is truly square/symmetric.
        sim_matrix = sim_matrix.reindex(index=terms, columns=terms)

    if len(terms) == 1:
        return {terms[0]: 0}

    sim = sim_matrix.to_numpy(dtype=float)
    # Symmetrize (the notebook fills only computed pairs) and turn into distance.
    sim = np.fmax(sim, sim.T)
    dist = 1.0 - sim
    dist[np.isnan(dist)] = 1.0  # unmapped pairs -> maximally distant
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry for squareform

    condensed = squareform(dist, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    labels = fcluster(linkage_matrix, t=1.0 - sim_threshold, criterion="distance")
    return dict(zip(terms, (int(label) for label in labels)))


def name_groups(
    term_to_group: Dict[str, int], samples: pd.DataFrame
) -> Tuple[Dict[str, str], "OrderedDict[str, List[str]]"]:
    """Assign stable generic names (``group1``, ``group2``, ...) to clusters.

    Only clusters with at least one cohort sample are named, ordered by
    descending sample count (``group1`` is the largest), ties broken by the
    smallest member curie. ``samples`` must have a ``curie`` column.

    Returns ``(curie -> group_name, {group_name: [curies]})``, the latter
    ordered ``group1, group2, ...``.
    """
    curie_counts = samples["curie"].value_counts()
    gid_to_curies: Dict[int, List[str]] = defaultdict(list)
    for curie in samples["curie"].unique():
        gid_to_curies[term_to_group[curie]].append(curie)

    ordered = sorted(
        gid_to_curies.items(),
        key=lambda kv: (-int(curie_counts[kv[1]].sum()), min(kv[1])),
    )

    term_to_name: Dict[str, str] = {}
    name_to_curies: "OrderedDict[str, List[str]]" = OrderedDict()
    for i, (_, curies) in enumerate(ordered, start=1):
        name = f"group{i}"
        name_to_curies[name] = sorted(curies)
        for curie in curies:
            term_to_name[curie] = name
    return term_to_name, name_to_curies


def _within_group_lin(
    sim_matrix: pd.DataFrame, name_to_curies: "OrderedDict[str, List[str]]"
) -> Dict[str, "Tuple[float, float]"]:
    """Return ``{term: (mean_lin, median_lin)}`` vs other members of its group.

    Singletons have no in-group neighbour, so their value is ``(nan, nan)``.
    """
    result: Dict[str, Tuple[float, float]] = {}
    for curies in name_to_curies.values():
        for term in curies:
            others = [other for other in curies if other != term]
            if not others:
                result[term] = (float("nan"), float("nan"))
                continue
            vals = sim_matrix.loc[term, others].to_numpy(dtype=float)
            result[term] = (float(np.nanmean(vals)), float(np.nanmedian(vals)))
    return result


def group_content_table(
    name_to_curies: "OrderedDict[str, List[str]]",
    samples: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    celltype_field: str,
    summary_field: str,
) -> pd.DataFrame:
    """Return the per-curie group membership table.

    Columns: ``group``, ``<celltype_field>`` (the curie), ``<summary_field>``
    (its summary-category value(s)), ``n_samples``, and the mean/median Lin
    similarity of the curie to the other members of its own group.
    """
    curie_counts = samples["curie"].value_counts()
    summary_of = (
        samples.groupby("curie")["summary"]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .to_dict()
    )
    within = _within_group_lin(sim_matrix, name_to_curies)

    rows = []
    for name, curies in name_to_curies.items():
        for curie in curies:
            mean_lin, median_lin = within[curie]
            rows.append(
                {
                    "group": name,
                    celltype_field: curie,
                    summary_field: summary_of.get(curie, "--empty--"),
                    "n_samples": int(curie_counts[curie]),
                    "mean_lin_within_group": mean_lin,
                    "median_lin_within_group": median_lin,
                }
            )
    columns = [
        "group",
        celltype_field,
        summary_field,
        "n_samples",
        "mean_lin_within_group",
        "median_lin_within_group",
    ]
    return pd.DataFrame(rows, columns=columns)


def sweep_thresholds(
    sim_matrix: pd.DataFrame, samples: pd.DataFrame, thresholds: List[float]
) -> pd.DataFrame:
    """Report cell-type grouping stats across similarity thresholds.

    For each threshold, clusters the cohort's curies and records how many groups
    result and how concentrated they are — useful for picking ``--sim-threshold``
    before committing to a split. Returns one row per threshold.
    """
    curie_counts = samples["curie"].value_counts()
    rows = []
    for threshold in thresholds:
        _, name_to_curies = name_groups(cluster_terms(sim_matrix, threshold), samples)
        n_curies = [len(curies) for curies in name_to_curies.values()]
        n_samples = [
            int(curie_counts[curies].sum()) for curies in name_to_curies.values()
        ]
        rows.append(
            {
                "sim_threshold": threshold,
                "n_groups": len(name_to_curies),
                "n_singleton_groups": sum(1 for n in n_curies if n == 1),
                "largest_group_curies": max(n_curies, default=0),
                "largest_group_samples": max(n_samples, default=0),
            }
        )
    return pd.DataFrame(rows)


def sweep_group_contents(
    sim_matrix: pd.DataFrame,
    samples: pd.DataFrame,
    thresholds: List[float],
    celltype_field: str,
    summary_field: str,
) -> pd.DataFrame:
    """Return group definitions at each threshold.

    Long format: ``sim_threshold`` prepended to the :func:`group_content_table`
    columns, so a sweep records exactly which curies each group held — and how
    cohesive they were (within-group Lin) — at every threshold.
    """
    frames = []
    for threshold in thresholds:
        _, name_to_curies = name_groups(cluster_terms(sim_matrix, threshold), samples)
        content = group_content_table(
            name_to_curies, samples, sim_matrix, celltype_field, summary_field
        )
        content.insert(0, "sim_threshold", threshold)
        frames.append(content)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Stratum assignment (assay priority)
# ---------------------------------------------------------------------------


def assign_strata(
    samples: pd.DataFrame, term_to_group: Dict[str, int], n_folds: int
) -> pd.Series:
    """Return the stratification label per sample, with assay priority.

    ``samples`` must have columns ``assay`` and ``curie``. Composite strata
    (``assay|group``) with fewer than ``n_folds`` members collapse to assay-only
    (``assay|*``); assay-only strata still below ``n_folds`` collapse to a single
    shared rare bucket. Raises if the rare bucket itself cannot be split.
    """
    groups = samples["curie"].map(term_to_group)
    composite = samples["assay"].astype(str) + "|" + groups.astype(str)

    composite_counts = composite.value_counts()
    assay_only = samples["assay"].astype(str) + "|" + REMAINDER_GROUP
    strata = composite.where(composite.map(composite_counts) >= n_folds, assay_only)

    assay_counts = strata.value_counts()
    strata = strata.where(strata.map(assay_counts) >= n_folds, RARE_STRATUM)

    rare_total = int((strata == RARE_STRATUM).sum())
    if 0 < rare_total < n_folds:
        raise ValueError(
            f"{rare_total} sample(s) fall into a residual stratum too small to "
            f"split into {n_folds} folds. Lower --n-folds or provide more data."
        )
    return strata


# ---------------------------------------------------------------------------
# Fold assignment
# ---------------------------------------------------------------------------


def _assign_folds(
    md5s: List[str],
    strata: List[str],
    groups: List[str] | None,
    n_folds: int,
    seed: int,
) -> Dict[str, int]:
    """Map each md5sum to a validation-fold index via (grouped) stratified k-fold."""
    y = np.asarray(strata)
    x_placeholder = np.zeros((len(md5s), 1))

    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(x_placeholder, y, groups=np.asarray(groups))
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(x_placeholder, y)

    fold_of: Dict[str, int] = {}
    for fold_i, (_, valid_idxs) in enumerate(split_iter):
        for idx in valid_idxs:
            fold_of[md5s[idx]] = fold_i
    return fold_of


def build_folds(
    samples: pd.DataFrame,
    term_to_group: Dict[str, int],
    n_folds: int,
    seed: int,
    group_by_epirr: bool,
) -> FoldDefinitions:
    """Build the ``{foldN: {"md5sum": [...]}}`` definition from per-sample records.

    ``samples`` must have columns ``md5sum``, ``assay``, ``curie`` and, when
    ``group_by_epirr`` is True, ``epirr``.
    """
    strata = assign_strata(samples, term_to_group, n_folds)

    md5s = samples["md5sum"].tolist()
    groups = samples["epirr"].astype(str).tolist() if group_by_epirr else None
    fold_of = _assign_folds(md5s, strata.tolist(), groups, n_folds, seed)

    folds: FoldDefinitions = {f"fold{i}": {"md5sum": []} for i in range(n_folds)}
    for md5 in md5s:
        folds[f"fold{fold_of[md5]}"]["md5sum"].append(md5)
    return folds


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _fold_label_table(
    folds: FoldDefinitions,
    label_of: Dict[str, str],
    label_name: str,
    col_order: List[str] | None = None,
) -> pd.DataFrame:
    """Return a ``fold x label`` count table from the fold JSON.

    Reading membership back from ``folds`` (rather than the in-memory split)
    doubles as a sanity check that the definition is complete and well-formed.
    """
    rows = [
        (fold_name, label_of[md5])
        for fold_name, id_dict in folds.items()
        for md5 in id_dict["md5sum"]
    ]
    table = pd.crosstab(
        index=pd.Series([r[0] for r in rows], name="fold"),
        columns=pd.Series([r[1] for r in rows], name=label_name),
    )
    table = table.reindex(sorted(table.index, key=lambda n: int(n.removeprefix("fold"))))
    if col_order is not None:
        table = table.reindex(columns=[c for c in col_order if c in table.columns])
    return table


def summarize_folds(
    metadata: Metadata,
    folds: FoldDefinitions,
    summary_category: str,
) -> pd.DataFrame:
    """Return a ``fold x summary_category`` count table built from the fold JSON."""
    label_of = {
        md5: metadata[md5].get(summary_category, "--empty--")
        for id_dict in folds.values()
        for md5 in id_dict["md5sum"]
    }
    return _fold_label_table(folds, label_of, summary_category)


def summarize_groups(
    folds: FoldDefinitions,
    md5_to_group: Dict[str, str],
    group_order: List[str],
) -> pd.DataFrame:
    """Return a ``fold x group`` count table (generic group names) from the JSON."""
    return _fold_label_table(folds, md5_to_group, "group", col_order=group_order)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_category(metadata: Metadata, requested: str, fallbacks: List[str]) -> str:
    """Return ``requested`` if present, else the first available fallback."""
    categories = set(metadata.get_categories())
    if requested in categories:
        return requested
    for candidate in fallbacks:
        if candidate in categories:
            return candidate
    raise ValueError(f"None of {[requested, *fallbacks]} found in metadata categories.")


def collect_samples(
    metadata: Metadata,
    assay_category: str,
    celltype_field: str,
    epirr_field: str | None,
    sim_terms: set,
    *,
    summary_field: str | None = None,
) -> pd.DataFrame:
    """Build the per-sample record table, erroring on unresolvable curies.

    Keeps only samples carrying both an assay and a cell-type curie. Any curie
    absent from the similarity matrix is an error (no silent drop), mirroring
    ``GeneralFoldFactory._resolve_folds``. When ``summary_field`` is given, its
    per-sample value is recorded in a ``summary`` column (used to annotate the
    group-content tables).
    """
    records = []
    missing_curie: List[str] = []
    for md5, dset in metadata.items:
        assay = dset.get(assay_category)
        curie = dset.get(celltype_field)
        if not assay or not curie:
            continue
        if curie not in sim_terms:
            missing_curie.append(curie)
            continue
        epirr = dset.get(epirr_field) if epirr_field else None
        record = {"md5sum": md5, "assay": assay, "curie": curie, "epirr": epirr or md5}
        if summary_field is not None:
            record["summary"] = dset.get(summary_field, "--empty--")
        records.append(record)

    if missing_curie:
        offenders = sorted(set(missing_curie))
        raise ValueError(
            f"{len(offenders)} cell-type curie(s) are absent from the similarity "
            f"matrix (recompute it over this cohort): {offenders[:10]}"
            + (" ..." if len(offenders) > 10 else "")
        )
    if not records:
        raise ValueError("No samples carry both an assay and a cell-type curie.")
    return pd.DataFrame.from_records(records)


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument(
        "metadata", type=Path, help="Training metadata JSON file.",
    )
    parser.add_argument(
        "similarity_matrix", type=Path,
        help="Square term x term Lin-similarity CSV from "
             "evaluate_biospecimen_similarity.ipynb.",
    )
    parser.add_argument(
        "out", type=Path, help="Output fold-definition JSON path.",
    )
    parser.add_argument(
        "--assay-category", type=str, default="assay",
        help="Metadata category for assay (default: assay; falls back to "
             "assay_epiclass).",
    )
    parser.add_argument(
        "--celltype-field", type=str, default="harmonized_sample_ontology_curie",
        help="Metadata field holding the cell-type ontology curie used for "
             "similarity clustering.",
    )
    parser.add_argument(
        "--summary-category", type=str,
        default="harmonized_sample_ontology_intermediate",
        help="Metadata category used only for the per-fold count summary.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=10, help="Number of CV folds (default: 10).",
    )
    parser.add_argument(
        "--sim-threshold", type=float, default=0.4,
        help="Lin-similarity cut for cell-type clustering (default: 0.4).",
    )
    parser.add_argument(
        "--sweep", type=float, nargs="*", default=None, metavar="T",
        help="Sweep mode: report cell-type group stats across similarity "
             "thresholds and write <out>_sweep.csv instead of building folds. "
             "Pass values to override the default "
             f"({' '.join(str(t) for t in DEFAULT_SWEEP)}), e.g. --sweep 0.3 0.5.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--group-by", choices=["epirr", "none"], default="epirr",
        help="Keep all md5sums of the same EpiRR in one fold (default: epirr). "
             "Use 'none' for a plain per-sample stratified split.",
    )
    # fmt: on
    return parser.parse_args()


def main():
    """Create similarity-stratified folds from the command line."""
    cli = parse_arguments()

    metadata = Metadata(cli.metadata)
    assay_category = _resolve_category(metadata, cli.assay_category, ASSAY_FALLBACKS)

    epirr_field: str | None = None
    if cli.group_by == "epirr":
        epirr_field = _resolve_category(metadata, EPIRR_FALLBACKS[0], EPIRR_FALLBACKS)

    sim_matrix = pd.read_csv(cli.similarity_matrix, index_col=0)
    samples = collect_samples(
        metadata,
        assay_category=assay_category,
        celltype_field=cli.celltype_field,
        epirr_field=epirr_field,
        sim_terms=set(sim_matrix.index),
        summary_field=cli.summary_category,
    )

    if cli.sweep is not None:
        thresholds = cli.sweep or DEFAULT_SWEEP
        cli.out.parent.mkdir(parents=True, exist_ok=True)

        table = sweep_thresholds(sim_matrix, samples, thresholds)
        sweep_path = cli.out.with_name(f"{cli.out.stem}_sweep.csv")
        table.to_csv(sweep_path, index=False)

        contents = sweep_group_contents(
            sim_matrix, samples, thresholds, cli.celltype_field, cli.summary_category
        )
        contents_path = cli.out.with_name(f"{cli.out.stem}_sweep_groups.csv")
        contents.to_csv(contents_path, index=False)

        print(
            f"Threshold sweep over {len(samples)} samples / "
            f"{samples['curie'].nunique()} curies"
        )
        print(table.to_string(index=False))
        print(f"\nStats -> {sweep_path}")
        print(f"Group definitions per threshold -> {contents_path}")
        return

    term_to_group = cluster_terms(sim_matrix, cli.sim_threshold)
    term_to_name, name_to_curies = name_groups(term_to_group, samples)
    print(
        f"{len(samples)} samples, {samples['curie'].nunique()} curies -> "
        f"{len(name_to_curies)} cell-type groups (Lin >= {cli.sim_threshold}); "
        f"assay category '{assay_category}'; "
        f"grouping by {'EpiRR (' + str(epirr_field) + ')' if epirr_field else 'none'}."
    )

    folds = build_folds(
        samples,
        term_to_group=term_to_group,
        n_folds=cli.n_folds,
        seed=cli.seed,
        group_by_epirr=epirr_field is not None,
    )

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    print(f"Wrote {cli.n_folds} folds to {cli.out}")

    print("\nPer-fold assay counts:")
    _print_assay_balance(samples, folds)

    # --- Cell-type group definitions + per-fold group counts ---
    groups_table = group_content_table(
        name_to_curies, samples, sim_matrix, cli.celltype_field, cli.summary_category
    )
    groups_path = cli.out.with_name(f"{cli.out.stem}_groups.csv")
    groups_table.to_csv(groups_path, index=False)
    print(f"\nCell-type group definitions -> {groups_path}")
    print(groups_table.to_string(index=False))

    md5_to_group = dict(zip(samples["md5sum"], samples["curie"].map(term_to_name)))
    group_summary = summarize_groups(folds, md5_to_group, list(name_to_curies))
    group_summary_path = cli.out.with_name(f"{cli.out.stem}_group_summary.csv")
    group_summary.to_csv(group_summary_path)
    print(f"\nPer-fold cell-type group counts -> {group_summary_path}")
    print(group_summary)

    # --- Per-fold counts by the summary metadata category ---
    summary = summarize_folds(metadata, folds, cli.summary_category)
    summary_path = cli.out.with_name(f"{cli.out.stem}_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nPer-fold counts by '{cli.summary_category}' -> {summary_path}")
    print(summary)


def _print_assay_balance(samples: pd.DataFrame, folds: FoldDefinitions) -> None:
    """Print fold sizes and per-fold assay counts for a quick balance check."""
    assay_of = dict(zip(samples["md5sum"], samples["assay"]))
    for fold_name in sorted(folds, key=lambda n: int(n.removeprefix("fold"))):
        md5s = folds[fold_name]["md5sum"]
        counts = Counter(assay_of[m] for m in md5s)
        breakdown = ", ".join(f"{a}:{c}" for a, c in sorted(counts.items()))
        print(f"  {fold_name}: {len(md5s)} samples ({breakdown})")


if __name__ == "__main__":
    main()
