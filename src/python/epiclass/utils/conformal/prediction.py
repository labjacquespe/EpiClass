"""Post-hoc split conformal prediction on EpiClass prediction CSVs.

Turns the per-class softmax probabilities already written by
``analysis.write_pred_table`` (``ID, True class, Predicted class, <one column per
class>``) into *prediction sets* carrying a marginal coverage guarantee
``P(true label in set) >= 1 - alpha``.

The model and training pipeline are never touched: calibration and set
construction run entirely on the prediction CSVs. Calibration uses a labelled
prediction file (typically a fold's ``validation_prediction.csv``).

Two workflows are exposed (see ``main`` / ``--mode``):

* ``evaluate`` -- split one labelled CSV into calibration/evaluation halves,
  calibrate on the first and report empirical coverage + average set size on the
  second, for every method x alpha. This is how the coverage guarantee is checked
  (test CSVs are unlabelled).
* ``apply`` -- calibrate on a full labelled CSV, then emit prediction-set CSVs for
  an (unlabelled) test CSV.

Implementation note: the EpiClass CSVs already store softmax *probabilities*, while
TorchCP score functions default to applying a softmax to their input. We therefore
build every score with ``score_type="identity"`` and feed the probabilities directly.
APS/RAPS/SAPS keep their ``randomized=True`` term (it is what makes them exactly
valid); reproducibility comes from seeding the torch RNG (see ``RNG_SEED``).

See ``methods.md`` (same directory) for a description of LAC/APS/RAPS/SAPS.
"""
# This module gathers the whole post-hoc conformal layer (split + per-class + Mondrian +
# hyperparameter sweep + cached report + CV+); it is cohesive enough to live in one file.
# pylint: disable=too-many-lines
from __future__ import annotations

import argparse
import itertools
import math
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torchcp.classification.predictor import ClassConditionalPredictor, SplitPredictor
from torchcp.classification.score import APS, LAC, RAPS, SAPS
from torchcp.classification.utils.metrics import Metrics

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.core.prediction_files import resolve_split_prediction_csvs

# A calibrated TorchCP predictor: marginal (SplitPredictor) or class-conditional /
# Mondrian (ClassConditionalPredictor). Both share the model-free
# calculate_threshold / predict_with_logits interface used here.
ConformalPredictor = Union[SplitPredictor, ClassConditionalPredictor]

# Columns written by analysis.write_pred_table that are NOT class-probability columns.
NON_CLASS_COLUMNS = ("True class", "Predicted class")

DEFAULT_METHODS: Tuple[str, ...] = ("LAC", "APS", "RAPS", "SAPS")
DEFAULT_ALPHAS: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)

# Torch RNG seed used before every scoring call. APS/RAPS/SAPS rely on a uniform
# randomization term for their coverage guarantee (it breaks ties in the discrete
# cumulative score); seeding makes the resulting sets reproducible run to run.
RNG_SEED = 0

# Literature-default regularization hyperparameters for the regularized scores.
# RAPS (Angelopoulos et al. 2021): ``kreg`` top classes are penalty-free, ``penalty``
# is the per-rank penalty beyond them. SAPS (Huang et al. 2023): ``weight`` is the
# constant rank weight for every non-top class. Both leave the coverage guarantee
# intact and only change set size, so they can be tuned/swept on a calibration split.
RAPS_DEFAULTS: Dict[str, float] = {"kreg": 1, "penalty": 0.1}
SAPS_DEFAULTS: Dict[str, float] = {"weight": 0.2}


def build_score(method: str, **hparams):
    """Return a TorchCP score function for ``method`` configured for probabilities.

    ``score_type="identity"`` because the CSVs already hold softmax probabilities
    (the default "softmax" would double-apply it). APS/RAPS/SAPS keep
    ``randomized=True`` -- the randomization term is what makes them exactly valid;
    reproducibility is handled separately by seeding the torch RNG (see ``RNG_SEED``).

    ``hparams`` overrides the regularization defaults (``RAPS_DEFAULTS`` /
    ``SAPS_DEFAULTS``): pass ``kreg`` / ``penalty`` for RAPS, ``weight`` for SAPS to
    sweep set size on a calibration split. LAC/APS take no hyperparameters. Unknown
    keys for a method are ignored so a single grid dict can be reused across methods.
    """
    method = method.upper()
    if method == "LAC":
        return LAC(score_type="identity")
    if method == "APS":
        return APS(score_type="identity", randomized=True)
    if method == "RAPS":
        params = {
            **RAPS_DEFAULTS,
            **{k: hparams[k] for k in RAPS_DEFAULTS if k in hparams},
        }
        return RAPS(score_type="identity", randomized=True, **params)
    if method == "SAPS":
        params = {
            **SAPS_DEFAULTS,
            **{k: hparams[k] for k in SAPS_DEFAULTS if k in hparams},
        }
        return SAPS(score_type="identity", randomized=True, **params)
    raise ValueError(
        f"Unknown conformal method '{method}'. Options: {', '.join(DEFAULT_METHODS)}."
    )


def load_prediction_csv(
    path: str | Path,
) -> Tuple[List[str], np.ndarray, List[str], Optional[np.ndarray]]:
    """Load an EpiClass prediction CSV.

    Returns ``(ids, probs, classes, true_idx)`` where

    * ``ids`` -- list of sample IDs (the ``ID`` index column),
    * ``probs`` -- ``(n, k)`` float array of per-class probabilities,
    * ``classes`` -- ordered class names (the probability columns, in file order;
      this order is the integer encoding, matching ``training_mapping.tsv``),
    * ``true_idx`` -- ``(n,)`` int array of true-class indices, or ``None`` when the
      ``True class`` column is absent (test CSVs drop it).

    The first column is always the sample ID (``write_pred_table`` writes the index
    first); it is used as the index whether or not it carries an ``ID`` header, so
    files with an unnamed index column load correctly.

    The label columns are forced to ``str``: a paired-end CSV uses ``TRUE``/``FALSE``
    class names, which pandas would otherwise infer as booleans -- a ``True class`` value
    of ``True`` then no longer matches the string class-column headers ``"TRUE"``/``"FALSE"``
    (and ``str(True)`` is ``"True"``, not ``"TRUE"``). ``dtype`` keys absent from the file
    (e.g. ``True class`` in an unlabelled test CSV) are ignored by pandas.
    """
    df = pd.read_csv(path, index_col=0, dtype={c: str for c in NON_CLASS_COLUMNS})
    classes = [c for c in df.columns if c not in NON_CLASS_COLUMNS]
    if not classes:
        raise ValueError(f"No class-probability columns found in '{path}'.")

    probs = df[classes].to_numpy(dtype=np.float64)
    ids = [str(i) for i in df.index.tolist()]

    true_idx: Optional[np.ndarray] = None
    if "True class" in df.columns:
        class_to_idx = {name: i for i, name in enumerate(classes)}
        try:
            true_idx = np.array(
                [class_to_idx[label] for label in df["True class"]], dtype=np.int64
            )
        except KeyError as err:
            raise ValueError(
                f"'True class' value {err} in '{path}' is not among the probability "
                "columns; CSV is malformed."
            ) from err

    return ids, probs, classes, true_idx


@contextmanager
def suppress_quantile_warning():
    """Silence TorchCP's benign "quantile exceeds 1 ... threshold set as inf" warning.

    Mondrian calibration raises this once per class whose per-fold calibration count is
    below the degeneracy floor ``ceil(1/alpha) - 1``: no finite threshold exists, so the
    class is forced into every set (trivial coverage). ``mondrian_feasibility`` predicts
    exactly these classes ahead of time, so the warning is redundant noise -- wrap Mondrian
    calibration in this to keep output clean while the feasibility table reports the
    degeneracy explicitly.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantile exceeds 1.*", category=UserWarning
        )
        yield


def calibrate_predictor(
    cal_probs: np.ndarray,
    cal_true_idx: np.ndarray,
    method: str,
    alpha: float,
    *,
    mondrian: bool = False,
    hparams: Optional[Mapping[str, float]] = None,
) -> ConformalPredictor:
    """Calibrate a TorchCP predictor on probabilities + true-class indices.

    With ``mondrian=False`` (default) a marginal ``SplitPredictor`` is used (one global
    threshold). With ``mondrian=True`` a ``ClassConditionalPredictor`` computes a
    separate threshold per class, giving a per-class coverage guarantee -- but every
    class needs enough calibration samples on its own (see ``mondrian_feasibility``);
    a class with too few collapses to being included in every set.

    ``hparams`` (optional) overrides the score's regularization defaults; see
    ``build_score``. The returned predictor has ``q_hat`` set and can build sets via
    ``predict_sets``.
    """
    predictor_cls = ClassConditionalPredictor if mondrian else SplitPredictor
    predictor = predictor_cls(
        score_function=build_score(method, **(hparams or {})),
        model=None,
        alpha=alpha,
        device="cpu",
    )
    torch.manual_seed(RNG_SEED)
    predictor.calculate_threshold(
        torch.as_tensor(cal_probs, dtype=torch.float32),
        torch.as_tensor(cal_true_idx, dtype=torch.long),
        alpha,
    )
    return predictor


def predict_sets(predictor: ConformalPredictor, probs: np.ndarray) -> np.ndarray:
    """Return an ``(n, k)`` 0/1 membership array of prediction sets for ``probs``."""
    torch.manual_seed(RNG_SEED)
    membership = predictor.predict_with_logits(
        torch.as_tensor(probs, dtype=torch.float32)
    )
    return membership.cpu().numpy().astype(np.int64)


def evaluate_sets(membership: np.ndarray, true_idx: np.ndarray) -> Dict[str, float]:
    """Return empirical coverage and average set size for a membership matrix."""
    metrics = Metrics()
    sets = torch.as_tensor(membership, dtype=torch.int64)
    labels = torch.as_tensor(true_idx, dtype=torch.int64)
    return {
        "empirical_coverage": float(metrics("coverage_rate")(sets, labels)),
        "avg_set_size": float(metrics("average_size")(sets, labels)),
    }


def membership_to_labels(
    membership: np.ndarray, classes: Sequence[str]
) -> List[List[str]]:
    """Convert a 0/1 membership matrix into per-row lists of class names."""
    classes = list(classes)
    return [[classes[j] for j in np.flatnonzero(row)] for row in membership]


# Per-sample QC flag categories, derived from a prediction set's shape vs the true
# class. They partition the labelled samples (every row is exactly one of these):
#   empty    -- size 0: the model rejects every class (out-of-distribution candidate);
#   clean    -- singleton that *is* the true class (confident & correct);
#   disagree -- singleton that is *not* the true class (confident & wrong -> mislabel?);
#   hedge    -- 2+ classes (the model is unsure, true class may or may not be inside).
FLAG_EMPTY, FLAG_CLEAN, FLAG_DISAGREE, FLAG_HEDGE = (
    "empty",
    "clean",
    "disagree",
    "hedge",
)
FLAG_CATEGORIES: Tuple[str, ...] = (FLAG_CLEAN, FLAG_HEDGE, FLAG_DISAGREE, FLAG_EMPTY)


def classify_flags(membership: np.ndarray, true_idx: np.ndarray) -> List[str]:
    """Label each sample with its QC flag category from set shape vs the true class.

    ``membership`` is an ``(n, k)`` 0/1 matrix; ``true_idx`` an ``(n,)`` int array of
    true-class indices. Returns a length-``n`` list drawn from ``FLAG_CATEGORIES``. Needs
    the true class, so it only applies to labelled (cross-validation) samples, not to
    unlabelled deployment sets.
    """
    sizes = membership.sum(axis=1)
    covered = membership[np.arange(len(true_idx)), true_idx] == 1
    flags = np.empty(len(true_idx), dtype=object)
    flags[sizes == 0] = FLAG_EMPTY
    singleton = sizes == 1
    flags[singleton & covered] = FLAG_CLEAN
    flags[singleton & ~covered] = FLAG_DISAGREE
    flags[sizes >= 2] = FLAG_HEDGE
    return flags.tolist()


def evaluate_sets_per_class(
    membership: np.ndarray, true_idx: np.ndarray, classes: Sequence[str]
) -> pd.DataFrame:
    """Per-true-class coverage, set-size and prediction-set *shape* counts.

    Marginal coverage can hide a rare class being systematically under-covered;
    this stratifies coverage by the true label. Returns one row per class with the
    raw counts so results from several folds can be summed before deriving rates.

    Counts (each per true class):

    * ``support`` -- number of samples of the class;
    * ``n_covered`` -- sets containing the true class;
    * ``set_size_sum`` -- summed set size;
    * ``n_empty`` -- empty sets (an automatic miss; the mechanism behind a hard class
      being under-covered, and disambiguates a low ``avg_set_size``);
    * ``n_singleton_correct`` -- singletons equal to the true class (the "clean pass":
      a confident, correct prediction needing no review);
    * ``n_singleton_wrong`` -- singletons of a *different* class (a confident
      disagreement with the label -> a specific mislabel hypothesis);
    * ``n_multi`` -- sets of size >= 2 (a hedge: ambiguous among known classes).

    The four shape counts partition the support: ``n_empty + n_singleton_correct +
    n_singleton_wrong + n_multi == support``. ``n_singleton_correct`` is the only
    non-flagged outcome; the other three make up the QC "flag" (route to review).
    """
    classes = list(classes)
    set_sizes = membership.sum(axis=1)
    rows: List[Dict[str, float | str]] = []
    for idx, name in enumerate(classes):
        mask = true_idx == idx
        support = int(mask.sum())
        sizes = set_sizes[mask]
        covered = membership[mask, idx] == 1 if support else np.zeros(0, dtype=bool)
        is_singleton = sizes == 1
        n_covered = int(membership[mask, idx].sum()) if support else 0
        size_sum = int(sizes.sum()) if support else 0
        n_empty = int((sizes == 0).sum()) if support else 0
        n_singleton_correct = int((is_singleton & covered).sum()) if support else 0
        n_singleton_wrong = int((is_singleton & ~covered).sum()) if support else 0
        n_multi = int((sizes >= 2).sum()) if support else 0
        rows.append(
            {
                "true_class": name,
                "support": support,
                "n_covered": n_covered,
                "set_size_sum": size_sum,
                "n_empty": n_empty,
                "n_singleton_correct": n_singleton_correct,
                "n_singleton_wrong": n_singleton_wrong,
                "n_multi": n_multi,
                "coverage": n_covered / support if support else float("nan"),
                "avg_set_size": size_sum / support if support else float("nan"),
                "empty_rate": n_empty / support if support else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# Raw per-class count columns produced by evaluate_sets_per_class; summed across
# folds/grid points before rates are re-derived (sum-then-divide, not mean-of-means).
# The last three are the prediction-set shape counts behind the QC flag composition.
PER_CLASS_COUNT_COLS = (
    "support",
    "n_covered",
    "set_size_sum",
    "n_empty",
    "n_singleton_correct",
    "n_singleton_wrong",
    "n_multi",
)


def aggregate_per_class(
    frames: pd.DataFrame | Sequence[pd.DataFrame],
    group_cols: Sequence[str] = ("method", "true_class"),
) -> pd.DataFrame:
    """Sum the raw per-class counts across folds/frames and re-derive the rates.

    Both the CV-fold drivers and the hyperparameter sweep need the same
    sum-then-divide aggregation (never mean-of-means) on the counts from
    ``evaluate_sets_per_class``. Concatenates ``frames`` (a single DataFrame or a
    sequence), groups by ``group_cols``, sums ``PER_CLASS_COUNT_COLS`` and assigns
    ``coverage`` / ``avg_set_size`` / ``empty_rate``. An empty input yields an empty
    DataFrame.
    """
    if isinstance(frames, pd.DataFrame):
        combined = frames
    else:
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(list(group_cols), as_index=False)[list(PER_CLASS_COUNT_COLS)]
        .sum()
        .assign(
            coverage=lambda d: d["n_covered"] / d["support"],
            avg_set_size=lambda d: d["set_size_sum"] / d["support"],
            empty_rate=lambda d: d["n_empty"] / d["support"],
        )
    )


def _calib_eval_split(
    n: int, calib_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return shuffled ``(calibration, evaluation)`` index arrays."""
    perm = np.random.default_rng(seed).permutation(n)
    n_cal = int(round(calib_frac * n))
    if not 0 < n_cal < n:
        raise ValueError(
            f"calib_frac={calib_frac} yields {n_cal}/{n} calibration samples; "
            "need a non-empty split on both sides."
        )
    return perm[:n_cal], perm[n_cal:]


def _q_hat_scalar(predictor: ConformalPredictor) -> float:
    """A single q_hat for reporting; NaN for Mondrian (it has one per class)."""
    q_hat = torch.as_tensor(predictor.q_hat)
    return float(q_hat) if q_hat.numel() == 1 else float("nan")


def run_evaluate(
    pred_csv: str | Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    calib_frac: float = 0.5,
    seed: int = 42,
    *,
    mondrian: bool = False,
) -> pd.DataFrame:
    """Calib/eval split a labelled CSV; report coverage + set size per method x alpha.

    ``mondrian=True`` switches to class-conditional calibration. Returns a tidy
    DataFrame with one row per ``(method, alpha)``.
    """
    ids, probs, _classes, true_idx = load_prediction_csv(pred_csv)
    if true_idx is None:
        raise ValueError(
            f"'{pred_csv}' has no 'True class' column; evaluate mode needs labels."
        )

    cal_sel, eval_sel = _calib_eval_split(len(ids), calib_frac, seed)

    rows: List[Dict[str, float | str]] = []
    for method in methods:
        for alpha in alphas:
            predictor = calibrate_predictor(
                probs[cal_sel], true_idx[cal_sel], method, alpha, mondrian=mondrian
            )
            membership = predict_sets(predictor, probs[eval_sel])
            scores = evaluate_sets(membership, true_idx[eval_sel])
            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "mondrian": mondrian,
                    "target_coverage": 1 - alpha,
                    "q_hat": _q_hat_scalar(predictor),
                    "n_calib": len(cal_sel),
                    "n_eval": len(eval_sel),
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def run_evaluate_per_class(
    pred_csv: str | Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    alpha: float = 0.1,
    calib_frac: float = 0.5,
    seed: int = 42,
    *,
    mondrian: bool = False,
    hparams: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Per-class coverage / set size at a single ``alpha``, via a calib/eval split.

    ``mondrian=True`` switches to class-conditional calibration. ``hparams`` (optional)
    overrides the score's regularization defaults (see ``build_score``) -- the hook the
    hyperparameter sweep drives. Returns a tidy DataFrame with one row per
    ``(method, true_class)``, carrying both rates (``coverage``, ``avg_set_size``) and
    the raw counts (``support``, ``n_covered``, ``set_size_sum``) needed to aggregate
    correctly across folds.
    """
    ids, probs, classes, true_idx = load_prediction_csv(pred_csv)
    if true_idx is None:
        raise ValueError(
            f"'{pred_csv}' has no 'True class' column; per-class eval needs labels."
        )

    cal_sel, eval_sel = _calib_eval_split(len(ids), calib_frac, seed)

    frames: List[pd.DataFrame] = []
    for method in methods:
        predictor = calibrate_predictor(
            probs[cal_sel],
            true_idx[cal_sel],
            method,
            alpha,
            mondrian=mondrian,
            hparams=hparams,
        )
        membership = predict_sets(predictor, probs[eval_sel])
        per_class = evaluate_sets_per_class(membership, true_idx[eval_sel], classes)
        per_class.insert(0, "method", method)
        per_class.insert(1, "alpha", alpha)
        frames.append(per_class)
    return pd.concat(frames, ignore_index=True)


def expand_grid(grid: Mapping[str, Sequence]) -> List[Dict[str, float]]:
    """Expand a ``{param: [values]}`` grid into the list of hyperparameter combos.

    ``expand_grid({"kreg": [1, 2], "penalty": [0, 0.1]})`` yields the four dicts of the
    Cartesian product. An empty grid yields ``[{}]`` (a single default combo).
    """
    keys = list(grid)
    if not keys:
        return [{}]
    return [
        dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))
    ]


def combo_label(hparams: Mapping[str, float]) -> str:
    """Stable label for a hyperparameter combo: sorted ``k=v`` pairs, ``"default"`` if empty.

    Shared by ``sweep_hparams`` and the report so the *same* combo always gets the same
    label (e.g. ``combo_label(RAPS_DEFAULTS)`` identifies the default-configured rows).
    """
    return ", ".join(f"{k}={hparams[k]}" for k in sorted(hparams)) or "default"


def sweep_hparams(
    pred_csvs: str | Path | Sequence[str | Path],
    method: str,
    grid: Mapping[str, Sequence] | Sequence[Mapping[str, float]],
    *,
    alpha: float = 0.05,
    calib_frac: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep a regularized score's hyperparameters and report per-class coverage/size.

    For each hyperparameter combo and each fold CSV, calibrate+evaluate ``method`` via
    ``run_evaluate_per_class``, then aggregate across folds on the **raw counts**
    (sum-then-divide, never mean-of-means). The result surfaces which classes are
    sensitive to the regularization (coverage / set size swings across the grid) vs
    robust (flat).

    Parameters
    ----------
    pred_csvs:
        One labelled prediction CSV or a sequence of them (the folds of a CV run).
    method:
        A regularized score (``"RAPS"`` or ``"SAPS"``). ``LAC``/``APS`` take no
        hyperparameters, so a grid over them collapses to a single combo.
    grid:
        Either a ``{param: [values]}`` mapping (expanded to its Cartesian product) or an
        explicit list of combo dicts. Keys irrelevant to ``method`` are ignored by
        ``build_score``, so a shared grid can be reused across methods.

    Returns a tidy DataFrame with one row per ``(combo, true_class)``: one column per
    swept hyperparameter, a ``combo`` label string, and the aggregated ``support``,
    ``n_covered``, ``set_size_sum``, ``n_empty`` counts plus the derived ``coverage``,
    ``avg_set_size`` and ``empty_rate`` rates.
    """
    if isinstance(pred_csvs, (str, Path)):
        pred_csvs = [pred_csvs]
    combos = expand_grid(grid) if isinstance(grid, Mapping) else [dict(c) for c in grid]
    hparam_keys = sorted({k for combo in combos for k in combo})

    frames: List[pd.DataFrame] = []
    for combo in combos:
        per_fold = [
            run_evaluate_per_class(
                csv,
                methods=[method],
                alpha=alpha,
                calib_frac=calib_frac,
                seed=seed,
                hparams=combo,
            )
            for csv in pred_csvs
        ]
        agg = aggregate_per_class(per_fold, group_cols=["true_class"]).assign(
            method=method,
            combo=combo_label(combo),
        )
        for key in hparam_keys:
            agg[key] = combo.get(key, float("nan"))
        frames.append(agg)

    return pd.concat(frames, ignore_index=True)


# Canonical configuration of the cached cross-classifier report (section 5 of the report
# notebook): marginal per-class results at the standard QC alphas, over the full RAPS/SAPS
# grids; LAC/APS are unregularized (a single "default" combo). Changing any of these means
# the on-disk caches are stale -- rebuild with force=True.
REPORT_ALPHAS: Tuple[float, ...] = (0.05, 0.1, 0.2)
REPORT_GRIDS: Dict[str, dict] = {
    "RAPS": {"kreg": [1, 2, 5], "penalty": [0.0, 0.01, 0.1, 0.5]},
    "SAPS": {"weight": [0.05, 0.1, 0.2, 0.5]},
}
REPORT_DEFAULTS: Dict[str, dict] = {
    "LAC": {},
    "APS": {},
    "RAPS": RAPS_DEFAULTS,
    "SAPS": SAPS_DEFAULTS,
}
# Cache file written beside each fold's prediction CSV (i.e. in its split folder).
FOLD_REPORT_NAME = "conformal_report.csv"


def compute_fold_report(
    pred_csv: str | Path, calib_frac: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    """Full marginal per-class report for **one** fold's prediction CSV.

    Stacks ``sweep_hparams`` over every method (the unregularized LAC/APS run with an empty
    grid, RAPS/SAPS over ``REPORT_GRIDS``) at every ``REPORT_ALPHAS`` value, tagging each
    block with its ``alpha``. This is the unit cached per split folder by ``fold_report``;
    aggregate the per-fold reports across folds with ``aggregate_per_class``.
    """
    grids = {"LAC": {}, "APS": {}, **REPORT_GRIDS}
    blocks: List[pd.DataFrame] = []
    for alpha in REPORT_ALPHAS:
        for method, grid in grids.items():
            block = sweep_hparams(
                pred_csv, method, grid, alpha=alpha, calib_frac=calib_frac, seed=seed
            )
            block.insert(0, "alpha", alpha)
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _fold_report_complete(df: pd.DataFrame) -> bool:
    """Whether a cached fold report covers the current canonical methods x alphas.

    Also requires the current count columns, so a cache written before a schema change
    (e.g. the flag-composition counts were added) is treated as stale and rebuilt.
    """
    required_cols = {"alpha", "method", "combo", *PER_CLASS_COUNT_COLS}
    if df.empty or not required_cols <= set(df.columns):
        return False
    if not set(REPORT_ALPHAS) <= set(df["alpha"].unique()):
        return False
    return {"LAC", "APS", *REPORT_GRIDS} <= set(df["method"].unique())


def fold_report(
    pred_csv: str | Path,
    *,
    force: bool = False,
    calib_frac: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-fold full report, cached as ``conformal_report.csv`` in the fold's split folder.

    Reuses the cached CSV unless ``force=True`` or the cache is missing / incomplete (its
    alphas or methods no longer match the canonical config). Otherwise recomputes via
    ``compute_fold_report`` and writes the cache. Returns the report DataFrame either way.
    """
    pred_csv = Path(pred_csv)
    cache = pred_csv.parent / FOLD_REPORT_NAME
    if not force and cache.exists():
        cached = pd.read_csv(cache)
        if _fold_report_complete(cached):
            return cached
    report = compute_fold_report(pred_csv, calib_frac=calib_frac, seed=seed)
    report.to_csv(cache, index=False)
    return report


def run_report(
    run_dir: str | Path,
    *,
    pattern: Optional[str] = None,
    force: bool = False,
    calib_frac: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Stack the cached per-fold reports of a CV run, tagging each row with its fold.

    By default resolves one validation prediction CSV per fold under ``run_dir`` (newest
    tagged file per ``split*``/``fold_*`` wins) via ``resolve_split_prediction_csvs``. Pass an
    explicit ``pattern`` to glob a custom layout instead. Concatenates each fold's
    ``fold_report`` (read from or written to that split folder's cache). Aggregate across folds
    downstream with
    ``aggregate_per_class(..., group_cols=["method", "alpha", "combo", "true_class"])``.
    """
    if pattern is None:
        paths = list(resolve_split_prediction_csvs(run_dir, "validation").values())
    else:
        paths = sorted(Path(run_dir).glob(pattern))
    blocks = [
        fold_report(p, force=force, calib_frac=calib_frac, seed=seed).assign(
            fold=p.parent.name
        )
        for p in paths
    ]
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def mondrian_feasibility(
    labels: Iterable | Mapping,
    alpha: float = 0.1,
    target_delta: float = 0.05,
    n_splits: Optional[int] = None,
    calib_frac: float = 1.0,
) -> pd.DataFrame:
    """Judge class-conditional (Mondrian) calibration feasibility from label counts.

    Mondrian conformal needs a separate threshold per class, so every class needs
    enough *calibration* samples on its own. This checks a label distribution against
    two thresholds at the chosen ``alpha``:

    * **floor** ``ceil(1/alpha) - 1`` -- below it no finite threshold exists and the
      class is forced into every prediction set (trivial coverage);
    * **reliability** ``alpha*(1-alpha)/target_delta**2`` -- the count needed for the
      per-class coverage to sit within +/- ``target_delta`` of the target.

    Parameters
    ----------
    labels:
        Either an iterable of per-sample labels (e.g. a ``True class`` column) or a
        ``{class: count}`` mapping.
    n_splits:
        If given, divide counts by it to estimate the *per-fold* calibration count --
        k-fold puts ~``1/n_splits`` of the data in each fold's validation set, and
        Mondrian calibrates per fold (calibration cannot be pooled across folds).
    calib_frac:
        Fraction of each (per-fold) set actually used for calibration.

    Returns one row per class (sorted by the projected calibration count) with the
    projected ``n_calib``, the two thresholds, boolean ``clears_floor`` / ``reliable``
    flags, and a ``status`` of ``degenerate`` / ``noisy`` / ``ok``. Mondrian is viable
    overall iff ``clears_floor`` is True for every class.
    """
    if isinstance(labels, Mapping):
        counts = pd.Series(dict(labels), dtype="int64")
    else:
        counts = pd.Series(list(labels)).value_counts()

    scale = calib_frac / (n_splits or 1)
    # Floor the projection: stratified k-fold makes per-fold counts approximate, and
    # a class is only safe if it clears the floor in its *smallest* fold.
    n_calib = np.floor(counts.to_numpy() * scale).astype(int)

    floor = math.ceil(1 / alpha) - 1
    reliable_n = math.ceil(alpha * (1 - alpha) / target_delta**2)
    clears_floor = n_calib >= floor
    reliable = n_calib >= reliable_n

    df = pd.DataFrame(
        {
            "class": counts.index.astype(str),
            "n_total": counts.to_numpy().astype(int),
            "n_calib": n_calib,
            "floor": floor,
            "clears_floor": clears_floor,
            "reliable_n": reliable_n,
            "reliable": reliable,
            "status": np.select(
                [~clears_floor, ~reliable], ["degenerate", "noisy"], default="ok"
            ),
        }
    )
    return df.sort_values("n_calib").reset_index(drop=True)


def write_set_csv(
    path: str | Path,
    ids: Sequence[str],
    classes: Sequence[str],
    probs: np.ndarray,
    membership: np.ndarray,
) -> Path:
    """Write a prediction-set CSV mirroring the style of ``write_pred_table``.

    Columns: ``ID, Predicted class, Prediction set`` (``;``-joined class names),
    ``Set size``, then one 0/1 column per class (membership indicator).
    """
    path = Path(path)
    classes = list(classes)
    set_labels = membership_to_labels(membership, classes)
    predicted = [classes[j] for j in probs.argmax(axis=1)]

    df = pd.DataFrame(membership, index=list(ids), columns=classes)
    df.insert(0, "Predicted class", predicted)
    df.insert(1, "Prediction set", [";".join(labels) for labels in set_labels])
    df.insert(2, "Set size", membership.sum(axis=1))
    df.to_csv(path, encoding="utf8", index_label="ID")
    return path


def run_apply(
    calib_csv: str | Path,
    test_csv: str | Path,
    out_dir: str | Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    alpha: float = 0.1,
    *,
    mondrian: bool = False,
) -> List[Path]:
    """Calibrate on a labelled CSV; write prediction-set CSVs for an unlabelled CSV.

    ``mondrian=True`` switches to class-conditional calibration. Returns the list of
    written paths (one per method).
    """
    _, cal_probs, cal_classes, cal_true_idx = load_prediction_csv(calib_csv)
    if cal_true_idx is None:
        raise ValueError(f"Calibration file '{calib_csv}' has no 'True class' column.")
    test_ids, test_probs, test_classes, _ = load_prediction_csv(test_csv)
    if cal_classes != test_classes:
        raise ValueError(
            "Calibration and test CSVs have different class columns/order; "
            f"{cal_classes} != {test_classes}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_mondrian" if mondrian else ""
    written: List[Path] = []
    for method in methods:
        predictor = calibrate_predictor(
            cal_probs, cal_true_idx, method, alpha, mondrian=mondrian
        )
        membership = predict_sets(predictor, test_probs)
        out_path = out_dir / f"prediction_sets_{method}{tag}_alpha{alpha}.csv"
        write_set_csv(out_path, test_ids, test_classes, test_probs, membership)
        written.append(out_path)
        print(f"'{out_path.name}' written to '{out_path.parent}'")
    return written


# --------------------------------------------------------------------------- #
# CV+ (cross-conformal) -- use every fold's out-of-fold scores as calibration.
# --------------------------------------------------------------------------- #
def true_class_scores(score_fn, probs: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Nonconformity score of each sample's *true* class: an ``(n,)`` array."""
    torch.manual_seed(RNG_SEED)
    scores = score_fn(
        torch.as_tensor(probs, dtype=torch.float32),
        torch.as_tensor(true_idx, dtype=torch.long),
    )
    return scores.cpu().numpy().astype(np.float64)


def all_class_scores(score_fn, probs: np.ndarray) -> np.ndarray:
    """Nonconformity score of *every* class: an ``(n, k)`` array."""
    torch.manual_seed(RNG_SEED)
    scores = score_fn(torch.as_tensor(probs, dtype=torch.float32))
    return scores.cpu().numpy().astype(np.float64)


def cv_plus_membership(
    cal_scores: Sequence[np.ndarray],
    test_score_matrices: Sequence[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """CV+ prediction-set membership from per-fold calibration + test scores.

    The CV+ / jackknife+ rule (Barber et al. 2021) compares each calibration point's
    *out-of-fold* score against the test point's score **under the same fold's model**,
    then pools the comparison over all folds into a conformal p-value

        ``p(x, y) = (1 + #{i : s_i >= s_test_{k(i)}(x, y)}) / (n + 1)``

    and includes label ``y`` when ``p(x, y) > alpha``. This uses *all* the data for
    calibration (every sample appears once as an out-of-fold score) rather than a single
    fold's validation set, at the cost of the slightly looser ``>= 1 - 2*alpha``
    worst-case guarantee (empirically close to ``1 - alpha``).

    Parameters
    ----------
    cal_scores:
        One ``(n_k,)`` array per fold ``k`` -- the true-class nonconformity scores of
        fold ``k``'s validation samples under model ``k`` (out-of-fold scores).
    test_score_matrices:
        One ``(m, num_classes)`` array per fold ``k`` -- the all-class nonconformity
        scores of the **same** ``m`` test samples under model ``k``. Same length/order as
        ``cal_scores``.

    Returns an ``(m, num_classes)`` 0/1 membership array.
    """
    if len(cal_scores) != len(test_score_matrices):
        raise ValueError("Need one test score matrix per calibration fold.")
    n_total = int(sum(len(s) for s in cal_scores))
    if n_total == 0:
        raise ValueError("No calibration scores provided.")

    total_ge = np.zeros_like(test_score_matrices[0], dtype=np.int64)
    for s_cal, s_test in zip(cal_scores, test_score_matrices):
        # #{i in fold k : s_cal_i >= s_test} == n_k - searchsorted(sorted, s_test, left)
        sorted_cal = np.sort(s_cal)
        total_ge += len(s_cal) - np.searchsorted(sorted_cal, s_test, side="left")

    pvals = (1 + total_ge) / (n_total + 1)
    return (pvals > alpha).astype(np.int64)


def cv_plus_sets(
    calib_csvs: Sequence[str | Path],
    test_csvs: Sequence[str | Path],
    method: str,
    alpha: float = 0.05,
    hparams: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, Optional[float]]:
    """Build CV+ prediction sets for test samples from per-fold calib + test CSVs.

    ``calib_csvs`` are the ``K`` labelled fold ``validation_prediction.csv`` (out-of-fold
    scores). ``test_csvs`` are the **same** test samples scored under each of the ``K``
    fold models (same IDs, same order, same class columns) -- this is what CV+ needs at
    deployment: every new sample is scored by all ``K`` models. ``test_csvs`` may be
    unlabelled (apply) or labelled (to also report coverage).

    Returns ``(ids, classes, membership, mean_probs, coverage)`` where ``mean_probs`` is
    the ensemble-mean probability matrix (its argmax is the point prediction) and
    ``coverage`` is ``None`` unless the test CSVs carry a ``True class`` column.
    """
    if len(calib_csvs) != len(test_csvs):
        raise ValueError(
            f"Need one test CSV per calibration fold: got {len(calib_csvs)} calib, "
            f"{len(test_csvs)} test."
        )
    if not calib_csvs:
        raise ValueError("No calibration/test CSVs provided.")
    score_fn = build_score(method, **(hparams or {}))

    classes_ref: Optional[List[str]] = None
    cal_scores: List[np.ndarray] = []
    for csv in calib_csvs:
        _, probs, classes, true_idx = load_prediction_csv(csv)
        if true_idx is None:
            raise ValueError(f"Calibration CSV '{csv}' has no 'True class' column.")
        if classes_ref is None:
            classes_ref = classes
        elif classes != classes_ref:
            raise ValueError("Calibration CSVs have mismatched class columns/order.")
        cal_scores.append(true_class_scores(score_fn, probs, true_idx))

    ids_ref: Optional[List[str]] = None
    test_true: Optional[np.ndarray] = None
    test_mats: List[np.ndarray] = []
    prob_sum: Optional[np.ndarray] = None
    for csv in test_csvs:
        ids, probs, classes, true_idx = load_prediction_csv(csv)
        if classes != classes_ref:
            raise ValueError("Test CSV class columns/order differ from calibration.")
        if ids_ref is None:
            ids_ref, test_true, prob_sum = ids, true_idx, probs.copy()
        else:
            if ids != ids_ref:
                raise ValueError(
                    "Test CSVs must list the same samples in the same order."
                )
            prob_sum = prob_sum + probs
        test_mats.append(all_class_scores(score_fn, probs))

    membership = cv_plus_membership(cal_scores, test_mats, alpha)
    mean_probs = prob_sum / len(test_csvs)

    coverage: Optional[float] = None
    if test_true is not None:
        covered = membership[np.arange(len(ids_ref)), test_true]
        coverage = float((covered == 1).mean())

    return ids_ref, classes_ref, membership, mean_probs, coverage


def run_cv_plus_apply(
    calib_csvs: Sequence[str | Path],
    test_csvs: Sequence[str | Path],
    out_dir: str | Path,
    methods: Sequence[str] = DEFAULT_METHODS,
    alpha: float = 0.05,
) -> List[Path]:
    """Write CV+ prediction-set CSVs for the test samples, one per method.

    Calibrates on the ``K`` fold validation CSVs and scores the test samples under all
    ``K`` fold models (``test_csvs``). The point-prediction column uses the ensemble-mean
    probabilities. Prints the empirical coverage when the test CSVs are labelled.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for method in methods:
        ids, classes, membership, mean_probs, coverage = cv_plus_sets(
            calib_csvs, test_csvs, method, alpha
        )
        out_path = out_dir / f"cv_plus_sets_{method}_alpha{alpha}.csv"
        write_set_csv(out_path, ids, classes, mean_probs, membership)
        written.append(out_path)
        msg = f"'{out_path.name}' written to '{out_path.parent}'"
        if coverage is not None:
            msg += f" (empirical coverage {coverage:.3f})"
        print(msg)
    return written


def parse_arguments() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--mode", choices=["evaluate", "apply", "cv-plus"], required=True, help="evaluate: split one labelled CSV to measure coverage/size. apply: calibrate then emit sets for a test CSV. cv-plus: cross-conformal sets using all folds' out-of-fold scores.")
    parser.add_argument("--pred-csv", type=Path, help="[evaluate] Labelled prediction CSV (e.g. validation_prediction.csv).")
    parser.add_argument("--calib-csv", type=Path, help="[apply] Labelled calibration CSV.")
    parser.add_argument("--test-csv", type=Path, help="[apply] Test CSV to build sets for.")
    parser.add_argument("--calib-csvs", type=Path, nargs="+", help="[cv-plus] The K labelled fold validation CSVs (out-of-fold scores).")
    parser.add_argument("--test-csvs", type=Path, nargs="+", help="[cv-plus] The same test samples scored under each of the K fold models (same order as --calib-csvs).")
    parser.add_argument("--out-dir", type=Path, help="[apply/cv-plus] Directory for prediction-set CSVs.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="Conformal score methods to run.")
    parser.add_argument("--alpha", type=float, default=0.1, help="[apply/cv-plus] Target miscoverage.")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS), help="[evaluate] Target miscoverage values to sweep.")
    parser.add_argument("--calib-frac", type=float, default=0.5, help="[evaluate] Fraction used for calibration.")
    parser.add_argument("--seed", type=int, default=42, help="[evaluate] Calib/eval split seed.")
    parser.add_argument("--mondrian", action="store_true", help="Use class-conditional (Mondrian) calibration: one threshold per class. Needs enough calibration samples per class.")
    # fmt: on
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_arguments()
    if args.mode == "evaluate":
        if args.pred_csv is None:
            raise SystemExit("--mode evaluate requires --pred-csv")
        results = run_evaluate(
            args.pred_csv,
            methods=args.methods,
            alphas=args.alphas,
            calib_frac=args.calib_frac,
            seed=args.seed,
            mondrian=args.mondrian,
        )
        print(results.to_string(index=False))
    elif args.mode == "cv-plus":
        missing = [
            name
            for name, val in (
                ("--calib-csvs", args.calib_csvs),
                ("--test-csvs", args.test_csvs),
                ("--out-dir", args.out_dir),
            )
            if not val
        ]
        if missing:
            raise SystemExit(f"--mode cv-plus requires {', '.join(missing)}")
        run_cv_plus_apply(
            args.calib_csvs,
            args.test_csvs,
            args.out_dir,
            methods=args.methods,
            alpha=args.alpha,
        )
    else:  # apply
        missing = [
            name
            for name, val in (
                ("--calib-csv", args.calib_csv),
                ("--test-csv", args.test_csv),
                ("--out-dir", args.out_dir),
            )
            if val is None
        ]
        if missing:
            raise SystemExit(f"--mode apply requires {', '.join(missing)}")
        run_apply(
            args.calib_csv,
            args.test_csv,
            args.out_dir,
            methods=args.methods,
            alpha=args.alpha,
            mondrian=args.mondrian,
        )


if __name__ == "__main__":
    main()
