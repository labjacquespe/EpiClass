"""Tests for the paper's k-fold epigenome leakage check (utils/notebooks/paper/fold_leakage.py).

Uses the files of the 36 epiatlas epigenomes with two "epirr_id" versions: grouping
folds by "epirr_id" put such epigenomes in two validation splits.
"""
# pylint: disable=missing-function-docstring
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import pytest

from epiclass.utils.notebooks.paper.fold_leakage import (
    check_no_epigenome_leakage,
    leakage_summary,
    splits_per_epigenome,
)
from tests.epilap_test_data import FIXTURES_DIR

METADATA_PATH = FIXTURES_DIR / "epiatlas_v2_multiversion_epirr-metadata.json"


@pytest.fixture(name="metadata_df", scope="module")
def fixture_metadata_df() -> pd.DataFrame:
    with open(METADATA_PATH, "r", encoding="utf8") as file:
        datasets = json.load(file)["datasets"]
    return pd.DataFrame(datasets).set_index("md5sum")


def splits_by(metadata_df: pd.DataFrame, field: str) -> Dict[str, List[str]]:
    """Return {split: md5sums}, one validation split per value of field."""
    return {
        f"split{i}": list(group.index)
        for i, (_, group) in enumerate(metadata_df.groupby(field))
    }


def test_epirr_grouping_passes(metadata_df):
    split_md5s = splits_by(metadata_df, "EpiRR")
    check_no_epigenome_leakage(split_md5s, metadata_df["EpiRR"])
    assert leakage_summary(split_md5s, metadata_df["EpiRR"]) == {
        "files": len(metadata_df),
        "epigenomes": 36,
        "epigenomes_in_several_splits": 0,
    }


def test_epirr_id_grouping_leaks(metadata_df):
    split_md5s = splits_by(metadata_df, "epirr_id")
    assert (splits_per_epigenome(split_md5s, metadata_df["EpiRR"]) == 2).all()
    with pytest.raises(ValueError, match="36 epigenome"):
        check_no_epigenome_leakage(split_md5s, metadata_df["EpiRR"])


def test_file_in_two_splits_leaks(metadata_df):
    split_md5s = splits_by(metadata_df, "EpiRR")
    split_md5s["split0"].append(split_md5s["split1"][0])
    with pytest.raises(ValueError, match="1 epigenome"):
        check_no_epigenome_leakage(split_md5s, metadata_df["EpiRR"])


def test_file_without_epirr_raises(metadata_df):
    split_md5s = {"split0": [*metadata_df.index[:2], "unknown_md5"]}
    with pytest.raises(ValueError, match="1 file"):
        splits_per_epigenome(split_md5s, metadata_df["EpiRR"])
