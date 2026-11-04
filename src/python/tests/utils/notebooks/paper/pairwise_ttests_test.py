"""Tests for the paper's per-fold t-tests (pairwise_ttests and hedges_g in paper_utilities)."""
# pylint: disable=missing-function-docstring
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, t as t_dist

from epiclass.utils.notebooks.paper.paper_utilities import hedges_g, pairwise_ttests

VALS_A = [0.96, 0.97, 0.95, 0.96, 0.98, 0.97, 0.96, 0.95, 0.97, 0.96]
VALS_B = [0.95, 0.96, 0.94, 0.97, 0.95, 0.96, 0.93, 0.95, 0.96, 0.94]


@pytest.fixture(name="row")
def fixture_row():
    all_metrics = {
        fs: {"cat": {f"split{i}": {"Accuracy": val} for i, val in enumerate(vals)}}
        for fs, vals in [("fs_a", VALS_A), ("fs_b", VALS_B)]
    }
    return pairwise_ttests(all_metrics, metrics_to_compare=["Accuracy"]).iloc[0]


def test_welch_values(row):
    var_a, var_b, n = np.var(VALS_A, ddof=1), np.var(VALS_B, ddof=1), len(VALS_A)
    diff = np.mean(VALS_A) - np.mean(VALS_B)
    se = np.sqrt(var_a / n + var_b / n)
    df = (var_a / n + var_b / n) ** 2 / (
        (var_a / n) ** 2 / (n - 1) + (var_b / n) ** 2 / (n - 1)
    )
    margin = t_dist.ppf(0.975, df) * se

    assert row["n_per_group"] == n
    assert row["std_a"] == pytest.approx(np.sqrt(var_a))
    assert row["diff_a_minus_b"] == pytest.approx(diff) and diff > 0
    assert row["t_stat"] == pytest.approx(diff / se)
    assert row["df"] == pytest.approx(df)
    assert row["diff_ci95_low"] == pytest.approx(diff - margin)
    assert row["diff_ci95_high"] == pytest.approx(diff + margin)
    assert row["p_value"] == pytest.approx(2 * t_dist.sf(diff / se, df))


def test_hedges_g(row):
    """With equal group sizes, d = t * sqrt(2 / n), and g = J * d. The CI is as documented."""
    n = len(VALS_A)
    correction = 1 - 3 / (4 * 2 * n - 9)
    g = correction * row["t_stat"] * np.sqrt(2 / n)
    margin = norm.ppf(0.975) * np.sqrt(2 * n / n**2 + g**2 / (2 * (2 * n - 3.94)))
    assert row["hedges_g"] == pytest.approx(g)
    assert row["hedges_g_ci95_low"] == pytest.approx(g - margin)
    assert row["hedges_g_ci95_high"] == pytest.approx(g + margin)
    g, low, high = hedges_g(VALS_B, VALS_A)
    assert (g, low, high) == pytest.approx(
        (-row["hedges_g"], -row["hedges_g_ci95_high"], -row["hedges_g_ci95_low"])
    )
