"""Unit tests for create_similar_celltype_folds (no ontology deps needed)."""
# pylint: disable=protected-access
from __future__ import annotations

import pandas as pd

from epiclass.core.lazy.general_fold_factory import GeneralFoldFactory
from epiclass.core.metadata import Metadata
from epiclass.utils.create_similar_celltype_folds import (
    build_folds,
    cluster_terms,
    collect_samples,
    group_content_table,
    name_groups,
    summarize_folds,
    summarize_groups,
    sweep_group_contents,
    sweep_thresholds,
)

# Two similar curies (ca/cb) and two distinct ones (cx/cy).
CURIES = ["CL:aa", "CL:bb", "CL:xx", "CL:yy"]
ASSAYS = ["h3k4me3", "input"]
N_FOLDS = 2


def _md5(i: int) -> str:
    """Return a 32-char id (Metadata.from_dict requires length 32)."""
    return f"{i:032d}"


def _sim_matrix() -> pd.DataFrame:
    """Lin-similarity matrix: ca~cb (0.9), everything else distinct (0.1)."""
    data = {}
    for a in CURIES:
        row = []
        for b in CURIES:
            if a == b:
                row.append(1.0)
            elif {a, b} == {"CL:aa", "CL:bb"}:
                row.append(0.9)
            else:
                row.append(0.1)
        data[a] = row
    return pd.DataFrame(data, index=CURIES, columns=CURIES)


def _metadata() -> Metadata:
    """Two samples per (assay, curie); intermediate label mirrors the curie."""
    intermediate = {
        "CL:aa": "lymphocyte",
        "CL:bb": "lymphocyte",
        "CL:xx": "neuron",
        "CL:yy": "hepatocyte",
    }
    meta: dict = {}
    i = 0
    for assay in ASSAYS:
        for curie in CURIES:
            for _ in range(2):
                md5 = _md5(i)
                meta[md5] = {
                    "md5sum": md5,
                    "assay": assay,
                    "harmonized_sample_ontology_curie": curie,
                    "harmonized_sample_ontology_intermediate": intermediate[curie],
                    "epirr_id": f"EPIRR{i}",
                }
                i += 1
    return Metadata.from_dict(meta)


def test_cluster_terms_groups_similar_curies():
    """ca and cb cluster together; cx is separate (Lin >= 0.4 cut)."""
    groups = cluster_terms(_sim_matrix(), sim_threshold=0.4)
    assert groups["CL:aa"] == groups["CL:bb"]
    assert groups["CL:aa"] != groups["CL:xx"]
    assert groups["CL:xx"] != groups["CL:yy"]


def test_build_folds_partition_and_balance():
    """Every md5 lands in exactly one fold; assay counts are balanced."""
    meta = _metadata()
    groups = cluster_terms(_sim_matrix(), sim_threshold=0.4)
    samples = collect_samples(
        meta,
        assay_category="assay",
        celltype_field="harmonized_sample_ontology_curie",
        epirr_field=None,
        sim_terms=set(CURIES),
    )

    folds = build_folds(samples, groups, n_folds=N_FOLDS, seed=42, group_by_epirr=False)

    assert len(folds) == N_FOLDS
    all_ids = [md5 for f in folds.values() for md5 in f["md5sum"]]
    assert sorted(all_ids) == sorted(meta.signal_ids)  # partition, no dup/loss

    assay_of = dict(zip(samples["md5sum"], samples["assay"]))
    for assay in ASSAYS:
        per_fold = [
            sum(assay_of[m] == assay for m in f["md5sum"]) for f in folds.values()
        ]
        assert max(per_fold) - min(per_fold) <= 1


def test_folds_roundtrip_through_resolve_folds():
    """Emitted JSON is consumable by GeneralFoldFactory._resolve_folds."""
    meta = _metadata()
    groups = cluster_terms(_sim_matrix(), sim_threshold=0.4)
    samples = collect_samples(
        meta, "assay", "harmonized_sample_ontology_curie", None, set(CURIES)
    )
    folds = build_folds(samples, groups, n_folds=N_FOLDS, seed=42, group_by_epirr=False)

    id_key, resolved = GeneralFoldFactory._resolve_folds(meta, folds)

    assert id_key == "md5sum"
    assert sorted(m for ids in resolved.values() for m in ids) == sorted(meta.signal_ids)


def test_epirr_grouping_keeps_group_together():
    """Samples sharing an EpiRR are never split across folds."""
    meta = _metadata()
    # Force two samples (different assays) to share one EpiRR.
    meta[_md5(0)]["epirr_id"] = "SHARED"
    meta[_md5(8)]["epirr_id"] = "SHARED"

    groups = cluster_terms(_sim_matrix(), sim_threshold=0.4)
    samples = collect_samples(
        meta, "assay", "harmonized_sample_ontology_curie", "epirr_id", set(CURIES)
    )
    folds = build_folds(samples, groups, n_folds=N_FOLDS, seed=42, group_by_epirr=True)

    fold_of = {m: name for name, f in folds.items() for m in f["md5sum"]}
    assert fold_of[_md5(0)] == fold_of[_md5(8)]


CURIE_COL = "harmonized_sample_ontology_curie"
SUMMARY_COL = "harmonized_sample_ontology_intermediate"


def _samples(meta: Metadata):
    """Collect the per-sample table (no EpiRR grouping), with summary column."""
    return collect_samples(
        meta, "assay", CURIE_COL, None, set(CURIES), summary_field=SUMMARY_COL
    )


def test_name_groups_orders_by_size_and_content():
    """group1 is the largest cluster; group content lists its curies + counts."""
    meta = _metadata()
    groups = cluster_terms(_sim_matrix(), sim_threshold=0.4)
    samples = _samples(meta)

    term_to_name, name_to_curies = name_groups(groups, samples)

    # lymphocyte cluster {ca, cb} = 8 samples -> the largest -> group1.
    assert list(name_to_curies) == ["group1", "group2", "group3"]
    assert name_to_curies["group1"] == ["CL:aa", "CL:bb"]
    assert term_to_name["CL:aa"] == term_to_name["CL:bb"] == "group1"
    # Singletons cx/cy (4 samples each) tie-break by curie string.
    assert name_to_curies["group2"] == ["CL:xx"]
    assert name_to_curies["group3"] == ["CL:yy"]

    content = group_content_table(
        name_to_curies, samples, _sim_matrix(), CURIE_COL, SUMMARY_COL
    )
    assert list(content.columns) == [
        "group",
        CURIE_COL,
        SUMMARY_COL,
        "n_samples",
        "mean_lin_within_group",
        "median_lin_within_group",
    ]
    g1 = content[content["group"] == "group1"]
    assert set(g1[CURIE_COL]) == {"CL:aa", "CL:bb"}
    assert (g1["n_samples"] == 4).all()  # 2 assays x 2 samples per curie
    assert (g1[SUMMARY_COL] == "lymphocyte").all()
    # ca and cb are each 0.9 similar to the only other member of group1.
    assert (g1["mean_lin_within_group"] == 0.9).all()
    # A singleton group has no in-group neighbour -> NaN.
    g2 = content[content["group"] == "group2"]
    assert g2["mean_lin_within_group"].isna().all()


def test_summarize_groups_totals_match_group_sizes():
    """Per-fold group counts sum to each group's total sample count."""
    meta = _metadata()
    groups = cluster_terms(_sim_matrix(), sim_threshold=0.4)
    samples = _samples(meta)
    term_to_name, name_to_curies = name_groups(groups, samples)
    folds = build_folds(samples, groups, n_folds=N_FOLDS, seed=42, group_by_epirr=False)

    md5_to_group = dict(zip(samples["md5sum"], samples["curie"].map(term_to_name)))
    group_summary = summarize_groups(folds, md5_to_group, list(name_to_curies))

    assert list(group_summary.columns) == ["group1", "group2", "group3"]
    assert len(group_summary) == N_FOLDS
    assert group_summary["group1"].sum() == 8
    assert group_summary["group2"].sum() == 4
    assert group_summary["group3"].sum() == 4


def test_sweep_thresholds_reports_group_counts():
    """A higher Lin cut splits the ca/cb cluster apart (more groups)."""
    meta = _metadata()
    samples = _samples(meta)

    table = sweep_thresholds(_sim_matrix(), samples, [0.4, 0.95]).set_index(
        "sim_threshold"
    )

    # At 0.4, ca~cb (0.9) cluster together -> 3 groups; at 0.95 they split -> 4.
    assert table.loc[0.4, "n_groups"] == 3
    assert table.loc[0.4, "largest_group_curies"] == 2
    assert table.loc[0.95, "n_groups"] == 4
    assert table.loc[0.95, "n_singleton_groups"] == 4


def test_sweep_group_contents_records_each_threshold():
    """Sweep group contents list every curie per group at each threshold."""
    meta = _metadata()
    samples = _samples(meta)

    contents = sweep_group_contents(
        _sim_matrix(), samples, [0.4, 0.95], CURIE_COL, SUMMARY_COL
    )

    assert list(contents.columns) == [
        "sim_threshold",
        "group",
        CURIE_COL,
        SUMMARY_COL,
        "n_samples",
        "mean_lin_within_group",
        "median_lin_within_group",
    ]
    # Every threshold lists all four curies exactly once.
    for threshold in (0.4, 0.95):
        rows = contents[contents["sim_threshold"] == threshold]
        assert sorted(rows[CURIE_COL]) == sorted(CURIES)
    # At 0.4 ca/cb share one group; at 0.95 every curie is its own group.
    at_04 = contents[contents["sim_threshold"] == 0.4]
    assert at_04[at_04[CURIE_COL].isin(["CL:aa", "CL:bb"])]["group"].nunique() == 1
    assert contents[contents["sim_threshold"] == 0.95]["group"].nunique() == 4


def test_summary_totals_match_global_counts():
    """The per-fold summary's column totals equal the global label counts."""
    meta = _metadata()
    groups = cluster_terms(_sim_matrix(), sim_threshold=0.4)
    samples = collect_samples(
        meta, "assay", "harmonized_sample_ontology_curie", None, set(CURIES)
    )
    folds = build_folds(samples, groups, n_folds=N_FOLDS, seed=42, group_by_epirr=False)

    summary = summarize_folds(meta, folds, "harmonized_sample_ontology_intermediate")

    assert len(summary) == N_FOLDS
    global_counts = pd.Series(
        [d["harmonized_sample_ontology_intermediate"] for d in meta.datasets]
    ).value_counts()
    for label, total in global_counts.items():
        assert summary[label].sum() == total
