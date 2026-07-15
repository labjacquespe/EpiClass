"""Build the pre-filtered RNA read-handling predictions artifact for the paper.

The Supplementary Fig. 5G figure compares three RNA-Seq read-handling variants on
every classification task:

* **Unique**            -- multi-mapped reads excluded (what the models trained on).
* **UniqueMultiple**    -- multi-mapped reads included.
* **Unstranded (summed)** -- the two Unique stranded tracks summed into one signal
  per EpiRR.

The honest end-to-end chain (run models -> pair -> filter -> score) depends on code
not yet on ``master`` (the conformal layer and the lazy ``predict.py``), so the
Quarto figure cannot reproduce it inline. Instead this module owns everything up to
"here are the validation-set model predictions" and writes a small, tidy long-form
CSV; the Quarto doc reads it and scores Accuracy / macro-F1 in a few pandas lines.

Because 5G plots only Accuracy and macro-F1 (no Brier), the artifact needs only the
true label and the argmax predicted label per sample -- no per-class probability
columns -- so a single combined table across all classifiers is enough.

Each row also carries the **Comet experiment key** of the classifier that produced
it (parsed from the prediction ``origin`` filename, see ``build_prediction_tag`` in
``core/prediction_files.py``) plus the model directory, so the exact models can be
cross-checked against ``complete_epiclass_models_table_data.csv``.

This module is intentionally free of conformal / lazy imports: it reads only data
artifacts (concatenated prediction CSVs, the metadata JSON, fold ``.md5`` lists and
the unstranded ``pair_mapping.tsv``). The marimo notebook
``mo_compare_rna_mapping_predictions.py`` imports these helpers rather than
duplicating them.

Run as a script to (re)generate the CSV::

    python -m epiclass.utils.notebooks.predictions.make_rna_read_handling_predictions
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from epiclass.core.metadata import Metadata
from epiclass.utils.general_utility import find_signal_id_lists

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EPIRR_RE = re.compile(r"IHECRE\d+(?:\.\d+)?")
STRANDS = ("minusRaw", "plusRaw")
META_COLS = ("origin", "ID", "Predicted class", "True class")
# A comet-ml experiment key is a 32-char lowercase hex string; the per-fold
# prediction filename (and hence `origin`) is `split{N}_<cometkey>_<ckptstem>_...`.
_EXP_KEY_RE = re.compile(r"split\d+_([0-9a-f]{32})_")


# --------------------------------------------------------------------------- #
# Identity helpers
# --------------------------------------------------------------------------- #
def parse_um_identity(signal_id: str) -> Optional[Tuple[str, str]]:
    """(EpiRR, strand) for a UniqueMultiple signal-ID, or None.

    UniqueMultiple IDs are the full hdf5 stem, e.g.
    'ihec.rna-seq...IHECRE00000717.1...UniqueMultiple.minusRaw_100kb_can'
    -> ('IHECRE00000717.1', 'minusRaw'). Unique md5sums have no IHECRE -> None.
    """
    match = EPIRR_RE.search(signal_id)
    if not match:
        return None
    strand = next((s for s in STRANDS if s in signal_id), None)
    return (match.group(0), strand) if strand else None


def unique_track_to_strand(track_type: Optional[str]) -> Optional[str]:
    """Normalize a Unique track_type ('Unique_minusRaw') to a strand ('minusRaw')."""
    if not track_type:
        return None
    stripped = (
        track_type[len("Unique_") :] if track_type.startswith("Unique_") else track_type
    )
    return stripped if stripped in STRANDS else None


def unique_identity(signal_id: str, metadata) -> Optional[Tuple[str, str]]:
    """(EpiRR, strand) for a Unique md5sum via metadata, or None."""
    meta = metadata.get(signal_id)
    if meta is None:
        return None
    strand = unique_track_to_strand(meta.get("track_type"))
    epirr = meta.get("epirr_id") or meta.get("reference_registry_id")
    if not epirr or strand is None:
        return None
    match = EPIRR_RE.search(str(epirr))
    return (match.group(0) if match else str(epirr), strand)


def split_from_origin(origin: str) -> Optional[str]:
    """Fold name ('split3' / 'fold_3') parsed from an `origin` filename, or None."""
    match = re.match(r"(split\d+|fold_\d+)", str(origin))
    return match.group(1) if match else None


def experiment_key_from_origin(origin: str) -> Optional[str]:
    """Comet experiment key parsed from a per-fold prediction `origin`, or None.

    The training pipeline tags each fold's prediction file as
    ``split{N}_<cometkey>_<ckptstem>_test_prediction_<list>.csv`` (see
    ``core/prediction_files.build_prediction_tag``); that filename is what lands in
    the concatenated CSV's ``origin`` column, so the producing classifier's key is
    recoverable without touching checkpoints or SLURM logs.
    """
    match = _EXP_KEY_RE.search(str(origin))
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
# Prediction loading / fold-list helpers
# --------------------------------------------------------------------------- #
def find_concatenated(pred_dir: Path) -> Optional[Path]:
    """The concatenated_test_prediction_*.csv in pred_dir, or None."""
    if not pred_dir.is_dir():
        return None
    hits = sorted(pred_dir.glob("concatenated_test_prediction*.csv"))
    return hits[0] if hits else None


def load_concatenated(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Return (df, classes). Adds 'split' and 'experiment_key' columns from 'origin'.

    Concatenated columns are: origin, ID, Predicted class, <one column per class>.
    """
    df = pd.read_csv(path)
    classes = [c for c in df.columns if c not in META_COLS]
    df = df.copy()
    df["split"] = df["origin"].map(split_from_origin)
    df["experiment_key"] = df["origin"].map(experiment_key_from_origin)
    df["ID"] = df["ID"].astype(str)
    return df, classes


def read_fold_id_list(cv_root: Path, split_name: str, kind: str) -> Optional[set]:
    """Set of signal-IDs in <cv_root>/<split>/<split>_<kind>_*, or None if absent."""
    lists = find_signal_id_lists(cv_root / split_name, f"{split_name}_{kind}_*")
    if not lists:
        return None
    chosen = sorted(lists)[-1]  # newest timestamp if several
    return {ln.strip() for ln in chosen.read_text().splitlines() if ln.strip()}


# --------------------------------------------------------------------------- #
# Unstranded (summed) helpers
# --------------------------------------------------------------------------- #
def unstranded_by_epirr(sub: pd.DataFrame, classes, id_to_epirr) -> dict:
    """{EpiRR: (id, prob)} for a fold's unstranded rows (one per EpiRR).

    The summed sample's ``ID`` is a content-free md5 of its two source filenames;
    ``id_to_epirr`` (built from the utility's mapping TSV + metadata) resolves it to
    an EpiRR. A bare IHECRE regex is kept as a fallback for hand-named files.
    """
    out = {}
    for sid, prob in zip(sub["ID"], sub[classes].to_numpy(dtype=float)):
        epirr = id_to_epirr.get(str(sid))
        if epirr is None:
            match = EPIRR_RE.search(str(sid))
            epirr = match.group(0) if match else None
        if epirr is not None:
            out[epirr] = (sid, prob)
    return out


def unique_avg_by_epirr(sub: pd.DataFrame, classes, metadata, keep_ids) -> dict:
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


def load_unstranded_id_map(mapping_path: Path, metadata) -> dict:
    """{summed new_id -> EpiRR} from the summing utility's pair-mapping TSV.

    Each summed sample's md5 ``new_id`` -> its two source md5s -> EpiRR via metadata.
    Either source strand resolves to the same EpiRR, so the first hit wins. Returns
    an empty dict if the metadata or mapping file is unavailable.
    """
    if metadata is None or not Path(mapping_path).is_file():
        return {}
    table = pd.read_csv(mapping_path, sep="\t", dtype=str)
    mapping = {}
    for _, row in table.iterrows():
        for src in (row.get("id_a"), row.get("id_b")):
            key = unique_identity(src, metadata) if src else None
            if key is not None:
                mapping[str(row["new_id"])] = key[0]
                break
    return mapping


# --------------------------------------------------------------------------- #
# Per-sample prediction builders
# --------------------------------------------------------------------------- #
def build_per_file_predictions(  # pylint: disable=too-many-branches
    models, metadata, task_category, predictions_subdir, run_subdir
) -> pd.DataFrame:
    """Per-sample Unique + UniqueMultiple predictions on each fold's validation set.

    For each (task, fold): keep Unique samples in that fold's validation list, pair
    them with their UniqueMultiple counterpart on (EpiRR, strand), attach the
    metadata true label and the argmax predicted label. Both mappings are scored by
    the same fold model, so they share the fold's experiment key. Long-form rows:
    (model, model_dir, experiment_key, split, mapping, epirr, strand, true_class,
    predicted_class).
    """
    rows = []
    for model_name, cv_root in models.items():
        cv_root = Path(cv_root)
        category = task_category.get(model_name)
        if category is None:
            continue
        csv = find_concatenated(cv_root / predictions_subdir / run_subdir)
        if csv is None:
            continue
        df, classes = load_concatenated(csv)
        class_idx = {c: i for i, c in enumerate(classes)}

        for split_name, sub in df.groupby("split"):
            if split_name is None:
                continue
            valid_ids = read_fold_id_list(cv_root, split_name, "validation")
            if valid_ids is None:
                continue
            exp_key = sub["experiment_key"].iloc[0]

            probs = sub[classes].to_numpy(dtype=float)
            # UniqueMultiple from the ID; Unique only if in this fold's validation
            # set (guarantees its true label is in the class space).
            u_map, m_map = {}, {}
            for sid, prob in zip(sub["ID"], probs):
                um_key = parse_um_identity(sid)
                if um_key is not None:
                    m_map[um_key] = (sid, prob)
                    continue
                if sid not in valid_ids:
                    continue
                u_key = unique_identity(sid, metadata)
                if u_key is not None:
                    u_map[u_key] = (sid, prob)

            for key in sorted(u_map.keys() & m_map.keys()):
                epirr, strand = key
                u_sid, u_prob = u_map[key]
                m_sid, m_prob = m_map[key]
                rec = metadata.get(u_sid)
                true_label = rec.get(category) if rec else None
                if not true_label or true_label not in class_idx:
                    continue
                # signal_id is the prediction 'ID': the md5sum for Unique, the full
                # hdf5 stem for UniqueMultiple -- kept for per-file traceability.
                for mapping, sid, prob in (
                    ("Unique", u_sid, u_prob),
                    ("UniqueMultiple", m_sid, m_prob),
                ):
                    rows.append(
                        {
                            "model": model_name,
                            "model_dir": str(cv_root),
                            "experiment_key": exp_key,
                            "split": split_name,
                            "mapping": mapping,
                            "epirr": epirr,
                            "strand": strand,
                            "signal_id": sid,
                            "label_category": category,
                            "true_class": true_label,
                            "predicted_class": classes[int(np.argmax(prob))],
                        }
                    )
    return pd.DataFrame(rows)


def build_unstranded_predictions(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    models,
    metadata,
    id_to_epirr,
    task_category,
    predictions_subdir,
    unique_subdir,
    unstranded_subdir,
) -> pd.DataFrame:
    """Per-sample 'Unstranded (summed)' predictions on each fold's validation set.

    The summed-per-EpiRR prediction is matched to a validation Unique strand (which
    fixes the true label in the class space), and scored by the same fold model. The
    averaged-Unique companion is intentionally not emitted here: the combined figure
    keeps Unique from the per-file section so it stays the paper's Unique numbers.
    Same columns as ``build_per_file_predictions`` (strand is None).
    """
    rows = []
    for model_name, cv_root in models.items():
        cv_root = Path(cv_root)
        category = task_category.get(model_name)
        if category is None:
            continue
        u_csv = find_concatenated(cv_root / predictions_subdir / unique_subdir)
        s_csv = find_concatenated(cv_root / predictions_subdir / unstranded_subdir)
        if u_csv is None or s_csv is None:
            continue
        u_df, classes = load_concatenated(u_csv)
        s_df, s_classes = load_concatenated(s_csv)
        if classes != s_classes:
            continue
        cidx = {c: i for i, c in enumerate(classes)}

        u_by_split = dict(tuple(g) for g in u_df.groupby("split"))
        for split_name, s_sub in s_df.groupby("split"):
            if split_name is None or split_name not in u_by_split:
                continue
            valid = read_fold_id_list(cv_root, split_name, "validation")
            if valid is None:
                continue
            exp_key = s_sub["experiment_key"].iloc[0]
            u_sub = u_by_split[split_name]
            u_map = unique_avg_by_epirr(
                u_sub, classes, metadata, set(u_sub["ID"]) & valid
            )
            s_map = unstranded_by_epirr(s_sub, classes, id_to_epirr)

            for epirr in sorted(u_map.keys() & s_map.keys()):
                rep_id, _, _ = u_map[epirr]
                s_id, s_prob = s_map[epirr]
                rec = metadata.get(rep_id)
                true_label = rec.get(category) if rec else None
                if not true_label or true_label not in cidx:
                    continue
                # signal_id is the summed prediction's 'ID' (a content-free md5 of the
                # two source filenames); epirr is the resolved biological identity.
                rows.append(
                    {
                        "model": model_name,
                        "model_dir": str(cv_root),
                        "experiment_key": exp_key,
                        "split": split_name,
                        "mapping": "Unstranded (summed)",
                        "epirr": epirr,
                        "strand": None,
                        "signal_id": s_id,
                        "label_category": category,
                        "true_class": true_label,
                        "predicted_class": classes[int(np.argmax(s_prob))],
                    }
                )
    return pd.DataFrame(rows)


def build_read_handling_predictions(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    models,
    metadata,
    id_to_epirr,
    task_category,
    predictions_subdir,
    run_subdir,
    unstranded_subdir,
) -> pd.DataFrame:
    """The full pre-filtered artifact: Unique + UniqueMultiple + Unstranded (summed)."""
    per_file = build_per_file_predictions(
        models, metadata, task_category, predictions_subdir, run_subdir
    )
    unstranded = build_unstranded_predictions(
        models,
        metadata,
        id_to_epirr,
        task_category,
        predictions_subdir,
        run_subdir,
        unstranded_subdir,
    )
    return pd.concat([per_file, unstranded], ignore_index=True)


# --------------------------------------------------------------------------- #
# Default config + CLI
# --------------------------------------------------------------------------- #
def default_config() -> dict:
    """Default paths/config mirroring the notebook CONFIG (edit here or via argv)."""
    base_dir = (
        Path.home()
        / "mounts/narval-mount"
        / "projects/rrg-jacquesp-ab/rabyj"
        / "epiclass-project/output/epiclass-logs/epiatlas-dfreeze-v2.1/hg38_100kb_all_none"
    )
    cv_subdir = "10fold-oversampling"
    paper_data = Path.home() / "Projects/epiclass/output/paper/data"
    return {
        "models": {
            "assay": base_dir / "assay_epiclass_1l_3000n" / "11c" / cv_subdir,
            "cell_type": base_dir
            / "harmonized_sample_ontology_intermediate_1l_3000n"
            / cv_subdir,
            "life_stage": base_dir / "harmonized_donor_life_stage_1l_3000n" / cv_subdir,
            "sex": base_dir / "harmonized_donor_sex_1l_3000n" / "w-mixed" / cv_subdir,
            "cancer": base_dir / "harmonized_sample_cancer_high_1l_3000n" / cv_subdir,
            "biomaterial": base_dir / "harmonized_biomaterial_type_1l_3000n" / cv_subdir,
        },
        "task_category": {
            "assay": "assay_epiclass",
            "cell_type": "harmonized_sample_ontology_intermediate",
            "life_stage": "harmonized_donor_life_stage",
            "sex": "harmonized_donor_sex",
            "cancer": "harmonized_sample_cancer_high",
            "biomaterial": "harmonized_biomaterial_type",
        },
        "predictions_subdir": "predictionsCV",
        "run_subdir": "RNA_UniqueMultiple",
        "unstranded_subdir": "RNA_Unstranded",
        "unstranded_pair_mapping": paper_data / "hdf5/rna_unstranded/pair_mapping.tsv",
        "metadata_json": paper_data
        / "metadata/epiatlas/hg38_2023-epiatlas-dfreeze_v2.1_w_encode_noncore_2.json",
        "output": paper_data / "rna_variations/rna_read_handling_predictions.csv",
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Build the pre-filtered RNA read-handling predictions CSV from the CLI."""
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=cfg["output"], help="Destination CSV path."
    )
    parser.add_argument(
        "--metadata-json", type=Path, default=cfg["metadata_json"], help="Metadata JSON."
    )
    args = parser.parse_args(argv)

    metadata = Metadata(args.metadata_json)
    id_to_epirr = load_unstranded_id_map(cfg["unstranded_pair_mapping"], metadata)

    preds = build_read_handling_predictions(
        cfg["models"],
        metadata,
        id_to_epirr,
        cfg["task_category"],
        cfg["predictions_subdir"],
        cfg["run_subdir"],
        cfg["unstranded_subdir"],
    )

    # Provenance header ('#'-commented so readers can pd.read_csv(comment="#")):
    # true_class is not free-standing ground truth -- it is metadata[label_category]
    # read from this specific metadata JSON, so its source is stated in the file.
    header = (
        "# RNA read-handling predictions -- Supplementary Fig. 5G.\n"
        "# Produced by make_rna_read_handling_predictions.py (see that module).\n"
        "# predicted_class is the classifier argmax; experiment_key / model_dir\n"
        "#   identify the model (cross-check complete_epiclass_models_table_data.csv).\n"
        f"# true_class = metadata[label_category], from: {args.metadata_json.name}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(header)
        preds.to_csv(handle, index=False)
    counts = preds.groupby(["model", "mapping"]).size().unstack(fill_value=0)
    print(f"Wrote {len(preds)} rows -> {args.output}\n{counts}")


if __name__ == "__main__":
    main()
