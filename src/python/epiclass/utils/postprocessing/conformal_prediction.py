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

See ``conformal_methods.md`` (same directory) for a description of LAC/APS/RAPS/SAPS.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torchcp.classification.predictor import ClassConditionalPredictor, SplitPredictor
from torchcp.classification.score import APS, LAC, RAPS, SAPS
from torchcp.classification.utils.metrics import Metrics

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser

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


def build_score(method: str):
    """Return a TorchCP score function for ``method`` configured for probabilities.

    ``score_type="identity"`` because the CSVs already hold softmax probabilities
    (the default "softmax" would double-apply it). APS/RAPS/SAPS keep
    ``randomized=True`` -- the randomization term is what makes them exactly valid;
    reproducibility is handled separately by seeding the torch RNG (see ``RNG_SEED``).
    RAPS/SAPS penalties use common literature defaults; tune on a calibration split.
    """
    method = method.upper()
    if method == "LAC":
        return LAC(score_type="identity")
    if method == "APS":
        return APS(score_type="identity", randomized=True)
    if method == "RAPS":
        return RAPS(score_type="identity", randomized=True, penalty=0.1, kreg=1)
    if method == "SAPS":
        return SAPS(score_type="identity", randomized=True, weight=0.2)
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
    """
    df = pd.read_csv(path, index_col=0)
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


def calibrate_predictor(
    cal_probs: np.ndarray,
    cal_true_idx: np.ndarray,
    method: str,
    alpha: float,
    mondrian: bool = False,
) -> ConformalPredictor:
    """Calibrate a TorchCP predictor on probabilities + true-class indices.

    With ``mondrian=False`` (default) a marginal ``SplitPredictor`` is used (one global
    threshold). With ``mondrian=True`` a ``ClassConditionalPredictor`` computes a
    separate threshold per class, giving a per-class coverage guarantee -- but every
    class needs enough calibration samples on its own (see ``mondrian_feasibility``);
    a class with too few collapses to being included in every set.

    The returned predictor has ``q_hat`` set and can build sets via ``predict_sets``.
    """
    predictor_cls = ClassConditionalPredictor if mondrian else SplitPredictor
    predictor = predictor_cls(
        score_function=build_score(method), model=None, alpha=alpha, device="cpu"
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


def evaluate_sets_per_class(
    membership: np.ndarray, true_idx: np.ndarray, classes: Sequence[str]
) -> pd.DataFrame:
    """Per-true-class coverage and set-size counts for one membership matrix.

    Marginal coverage can hide a rare class being systematically under-covered;
    this stratifies coverage by the true label. Returns one row per class with the
    raw counts (``support``, ``n_covered``, ``set_size_sum``, ``n_empty``) so results
    from several folds can be summed before deriving rates.

    ``empty_rate`` (fraction of samples that got an *empty* set) disambiguates a low
    ``avg_set_size``: a class can average below 1 either via many small non-empty sets
    or via outright empties, and an empty set is an automatic miss -- the mechanism
    behind a hard class being under-covered.
    """
    classes = list(classes)
    set_sizes = membership.sum(axis=1)
    rows: List[Dict[str, float | str]] = []
    for idx, name in enumerate(classes):
        mask = true_idx == idx
        support = int(mask.sum())
        n_covered = int(membership[mask, idx].sum()) if support else 0
        size_sum = int(set_sizes[mask].sum()) if support else 0
        n_empty = int((set_sizes[mask] == 0).sum()) if support else 0
        rows.append(
            {
                "true_class": name,
                "support": support,
                "n_covered": n_covered,
                "set_size_sum": size_sum,
                "n_empty": n_empty,
                "coverage": n_covered / support if support else float("nan"),
                "avg_set_size": size_sum / support if support else float("nan"),
                "empty_rate": n_empty / support if support else float("nan"),
            }
        )
    return pd.DataFrame(rows)


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
) -> pd.DataFrame:
    """Per-class coverage / set size at a single ``alpha``, via a calib/eval split.

    ``mondrian=True`` switches to class-conditional calibration. Returns a tidy
    DataFrame with one row per ``(method, true_class)``, carrying both rates
    (``coverage``, ``avg_set_size``) and the raw counts (``support``, ``n_covered``,
    ``set_size_sum``) needed to aggregate correctly across folds.
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
            probs[cal_sel], true_idx[cal_sel], method, alpha, mondrian=mondrian
        )
        membership = predict_sets(predictor, probs[eval_sel])
        per_class = evaluate_sets_per_class(membership, true_idx[eval_sel], classes)
        per_class.insert(0, "method", method)
        per_class.insert(1, "alpha", alpha)
        frames.append(per_class)
    return pd.concat(frames, ignore_index=True)


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


def parse_arguments() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--mode", choices=["evaluate", "apply"], required=True, help="evaluate: split one labelled CSV to measure coverage/size. apply: calibrate then emit sets for a test CSV.")
    parser.add_argument("--pred-csv", type=Path, help="[evaluate] Labelled prediction CSV (e.g. validation_prediction.csv).")
    parser.add_argument("--calib-csv", type=Path, help="[apply] Labelled calibration CSV.")
    parser.add_argument("--test-csv", type=Path, help="[apply] Test CSV to build sets for.")
    parser.add_argument("--out-dir", type=Path, help="[apply] Directory for prediction-set CSVs.")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="Conformal score methods to run.")
    parser.add_argument("--alpha", type=float, default=0.1, help="[apply] Target miscoverage.")
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
