"""Check that k-fold runs keep each epigenome in a single validation split.

An epigenome is an EpiRR without its version. When an epigenome has files in two
validation splits, the training set of each of these splits holds files of the epigenome
it validates on.
"""
from __future__ import annotations

from typing import Collection, Dict, Mapping

import pandas as pd


def epigenome_id(epirr: str) -> str:
    """Return the EpiRR without its version, which identifies the epigenome."""
    return str(epirr).split(".", maxsplit=1)[0]


def splits_per_epigenome(
    split_md5s: Mapping[str, Collection[str]], epirr: pd.Series
) -> pd.Series:
    """Return the number of validation splits holding files of each epigenome.

    Args:
        split_md5s: {split name: md5sums of its validation files}
        epirr: EpiRR of each file (index: md5sum), e.g. metadata_df["EpiRR"].

    Raises:
        ValueError: If a file has no EpiRR.
    """
    files = pd.DataFrame(
        [(split, md5) for split, md5s in split_md5s.items() for md5 in md5s],
        columns=["split", "md5sum"],
    )
    missing = set(files["md5sum"]) - set(epirr.index)
    if missing:
        raise ValueError(f"{len(missing)} file(s) without an EpiRR.")
    epigenomes = epirr.loc[files["md5sum"]].map(epigenome_id).to_numpy()
    return files.groupby(epigenomes)["split"].nunique()


def leakage_summary(
    split_md5s: Mapping[str, Collection[str]], epirr: pd.Series
) -> Dict[str, int]:
    """Return the number of files, epigenomes and epigenomes in several validation splits."""
    n_splits = splits_per_epigenome(split_md5s, epirr)
    return {
        "files": sum(len(md5s) for md5s in split_md5s.values()),
        "epigenomes": len(n_splits),
        "epigenomes_in_several_splits": int((n_splits > 1).sum()),
    }


def check_no_epigenome_leakage(
    split_md5s: Mapping[str, Collection[str]], epirr: pd.Series
) -> None:
    """Raise if an epigenome has files in more than one validation split.

    Args: as splits_per_epigenome.
    """
    n_splits = splits_per_epigenome(split_md5s, epirr)
    leaking = n_splits[n_splits > 1]
    if not leaking.empty:
        raise ValueError(
            f"{len(leaking)} epigenome(s) in more than one validation split "
            f"(e.g. {leaking.head(3).to_dict()})."
        )
