"""Precompute conformal prediction-set artifacts for the interpretation apps.

The two marimo apps (CV examination, deployment) are thin READ-ONLY consumers; this
driver does the heavy computation once and writes per-sample CSVs the apps glob.

Two modes:

* ``cv-examine`` -- give every *training* validation sample an honest prediction set, to
  validate the dataset and flag imperfect/mislabelled samples. Uses **strict within-fold
  leave-one-out (LOO)**: each fold's model was trained on the other 9 folds, so the only
  honest, same-model, out-of-sample data for fold ``i`` is fold ``i`` itself. For every
  sample ``j`` in fold ``i`` we calibrate on ``fold_i \\ {j}`` and predict ``j`` -- maximal
  use of the ~10% validation fold, rigorously valid within each fold with no cross-model
  assumption. Computed in closed form from nonconformity scores (no per-sample refit; same
  ``searchsorted`` core as ``prediction.cv_plus_membership``), so it is deterministic.

* ``deploy`` -- thin path-resolver over ``prediction.run_cv_plus_apply`` (the opposite,
  pooled case: a new sample is scored by all K fold models and compared to each model's
  out-of-fold calibration).

Run as ``python -m epiclass.utils.conformal.precompute --mode cv-examine --run-dir ...``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from epiclass.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epiclass.utils.conformal import prediction as cp

# Output folder, a sibling of ``conformal_report/`` next to the split folders.
SETS_DIR_NAME = "conformal_sets"
VALIDATION_GLOB = "split*/validation_prediction.csv"
DEFAULT_TEST_GLOB = "split*/test_prediction.csv"


# --------------------------------------------------------------------------- #
# Within-fold leave-one-out membership (closed form, deterministic).
# --------------------------------------------------------------------------- #
def _loo_membership(t: np.ndarray, scores: np.ndarray, alpha: float) -> np.ndarray:
    """Marginal within-fold LOO prediction-set membership.

    ``t`` is the ``(n,)`` true-class nonconformity score of each fold sample (the
    calibration scores); ``scores`` the ``(n, k)`` all-class score matrix. For sample
    ``j`` and class ``c`` the LOO conformal p-value is

        ``p_j(c) = (1 + #{m != j : t[m] >= scores[j, c]}) / n``

    (calibration is ``fold \\ {j}``, size ``n - 1``, so the denominator is ``n``); class
    ``c`` is included iff ``p_j(c) > alpha``. The self term subtracts sample ``j``'s own
    calibration score ``t[j]`` -- this is exact regardless of score randomization, because
    it removes the actual value being excluded from the pool. Returns an ``(n, k)`` 0/1
    array.
    """
    n = len(t)
    sorted_t = np.sort(t)
    ge_all = n - np.searchsorted(sorted_t, scores, side="left")  # #{t >= scores[j,c]}
    self_ge = (t[:, None] >= scores).astype(np.int64)  # remove sample j's own t[j]
    pvals = (1 + (ge_all - self_ge)) / n
    return (pvals > alpha).astype(np.int64)


def _loo_membership_mondrian(
    t: np.ndarray,
    scores: np.ndarray,
    true_idx: np.ndarray,
    alpha: float,
    feasible: np.ndarray,
) -> np.ndarray:
    """Class-conditional (Mondrian) within-fold LOO membership.

    For a *feasible* class ``c`` the calibration pool is the fold's same-true-class members
    (one threshold per class); an *infeasible* class (``feasible[c]`` is False) falls back
    to the marginal pool so it is never trivially forced into every set. ``feasible`` is a
    ``(k,)`` bool mask. Returns an ``(n, k)`` 0/1 array.
    """
    n, k = scores.shape
    membership = np.zeros((n, k), dtype=np.int64)
    sorted_t_global = np.sort(t)
    for c in range(k):
        s_col = scores[:, c]
        if feasible[c]:
            is_c = true_idx == c
            pool = np.sort(t[is_c])
            n_pool = len(pool)
            ge = n_pool - np.searchsorted(pool, s_col, side="left")
            self_ge = np.where(is_c, (t >= s_col).astype(np.int64), 0)
            n_eff = n_pool - is_c.astype(np.int64)  # LOO only for class-c samples
            pvals = (1 + (ge - self_ge)) / (n_eff + 1)
        else:
            ge = n - np.searchsorted(sorted_t_global, s_col, side="left")
            self_ge = (t >= s_col).astype(np.int64)
            pvals = (1 + (ge - self_ge)) / n
        membership[:, c] = (pvals > alpha).astype(np.int64)
    return membership


# --------------------------------------------------------------------------- #
# Per-fold worker (picklable: top-level, simple args) + aggregation.
# --------------------------------------------------------------------------- #
def _examine_fold(task: Tuple[str, str, Optional[Mapping], Tuple[float, ...]]) -> Dict:
    """Score one fold and build its LOO sets at every alpha (the parallel unit).

    ``task`` is ``(csv_path, method, hparams, alphas)``. Loads + scores the fold once, then
    loops alphas in-process (LOO membership is cheap). Returns a result dict consumed by
    ``_write_examination``.
    """
    csv_path, method, hparams, alphas = task
    ids, probs, classes, true_idx = cp.load_prediction_csv(csv_path)
    if true_idx is None:
        raise ValueError(
            f"'{csv_path}' has no 'True class' column; cv-examine needs labelled folds."
        )
    score_fn = cp.build_score(method, **(hparams or {}))
    # Shared scoring helpers, the same machinery behind cv_plus_membership.
    true_scores = cp.true_class_scores(score_fn, probs, true_idx)
    all_scores = cp.all_class_scores(score_fn, probs)
    predicted = [classes[j] for j in probs.argmax(axis=1)]
    fold = Path(csv_path).parent.name

    per_alpha: Dict[float, Dict] = {}
    for alpha in alphas:
        marginal = _loo_membership(true_scores, all_scores, alpha)
        feas = cp.mondrian_feasibility(
            [classes[i] for i in true_idx], alpha=alpha, calib_frac=1.0
        )
        feasible_classes = set(feas.loc[feas["clears_floor"], "class"])
        feasible_mask = np.array([c in feasible_classes for c in classes], dtype=bool)
        mondrian = None
        if feasible_mask.any():
            mondrian = _loo_membership_mondrian(
                true_scores, all_scores, true_idx, alpha, feasible_mask
            )
        per_alpha[alpha] = {
            "marginal": marginal,
            "mondrian": mondrian,
            "feasibility": feas.assign(fold=fold, alpha=alpha),
        }

    return {
        "fold": fold,
        "ids": ids,
        "classes": classes,
        "true_idx": true_idx,
        "predicted": predicted,
        "alphas": per_alpha,
    }


def _examination_df(
    fold_result: Dict, classes: Sequence[str], membership: np.ndarray
) -> pd.DataFrame:
    """Build the per-sample examination frame for one fold (the on-disk schema)."""
    classes = list(classes)
    true_idx = fold_result["true_idx"]
    set_labels = cp.membership_to_labels(membership, classes)
    flags = cp.classify_flags(membership, true_idx)
    df = pd.DataFrame(membership, index=list(fold_result["ids"]), columns=classes)
    df.insert(0, "fold", fold_result["fold"])
    df.insert(1, "True class", [classes[i] for i in true_idx])
    df.insert(2, "Predicted class", list(fold_result["predicted"]))
    df.insert(3, "Prediction set", [";".join(labels) for labels in set_labels])
    df.insert(4, "Set size", membership.sum(axis=1))
    df.insert(5, "flag_category", flags)
    df.index.name = "ID"
    return df


def _stack_variant(fold_results: List[Dict], classes: Sequence[str], alpha, variant):
    """Concatenate one variant's per-fold examination frames into a single frame.

    A fold with ``mondrian=None`` (no class cleared the floor) falls back to its marginal
    membership for the Mondrian file, so every sample is still represented.
    """
    frames = []
    for fr in fold_results:
        membership = fr["alphas"][alpha][variant]
        if membership is None:
            membership = fr["alphas"][alpha]["marginal"]
        frames.append(_examination_df(fr, classes, membership))
    return pd.concat(frames)


def _write_examination(
    run_dir: Path, method: str, fold_results: List[Dict], alphas: Sequence[float]
) -> List[Path]:
    """Write the marginal + (when feasible) Mondrian set CSVs and feasibility sidecars."""
    out_dir = Path(run_dir) / SETS_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = fold_results[0]["classes"]
    for fr in fold_results:
        if fr["classes"] != classes:
            raise ValueError("Folds have mismatched class columns/order.")

    written: List[Path] = []
    for alpha in alphas:
        marg_path = out_dir / f"cv_examination_{method}_alpha{alpha}.csv"
        _stack_variant(fold_results, classes, alpha, "marginal").to_csv(
            marg_path, index_label="ID"
        )
        written.append(marg_path)

        if any(fr["alphas"][alpha]["mondrian"] is not None for fr in fold_results):
            mond_path = out_dir / f"cv_examination_{method}_mondrian_alpha{alpha}.csv"
            _stack_variant(fold_results, classes, alpha, "mondrian").to_csv(
                mond_path, index_label="ID"
            )
            written.append(mond_path)

        feas_path = out_dir / f"mondrian_feasibility_alpha{alpha}.csv"
        pd.concat(
            [fr["alphas"][alpha]["feasibility"] for fr in fold_results], ignore_index=True
        ).to_csv(feas_path, index=False)
        written.append(feas_path)
    return written


def _examination_complete(run_dir: Path, method: str, alphas: Sequence[float]) -> bool:
    """True iff every alpha's marginal examination CSV already exists."""
    out_dir = Path(run_dir) / SETS_DIR_NAME
    return all(
        (out_dir / f"cv_examination_{method}_alpha{a}.csv").exists() for a in alphas
    )


# --------------------------------------------------------------------------- #
# Drivers.
# --------------------------------------------------------------------------- #
def run_cv_examine(
    run_dir: str | Path,
    *,
    method: str = "SAPS",
    hparams: Optional[Mapping[str, float]] = None,
    alphas: Sequence[float] = cp.REPORT_ALPHAS,
    jobs: int = 1,
    force: bool = False,
) -> List[Path]:
    """Precompute honest within-fold LOO prediction sets for a 10-fold run directory.

    Globs ``split*/validation_prediction.csv`` under ``run_dir``, computes LOO sets per fold
    at each alpha (marginal ``method`` + Mondrian where per-fold-per-class feasible), and
    writes them to ``<run_dir>/conformal_sets/``. Skips work when the marginal CSVs already
    exist unless ``force``. Folds are scored in parallel across ``jobs`` processes.
    """
    run_dir = Path(run_dir)
    alphas = tuple(alphas)
    fold_csvs = sorted(run_dir.glob(VALIDATION_GLOB))
    if not fold_csvs:
        raise ValueError(f"No '{VALIDATION_GLOB}' found under '{run_dir}'.")
    if not force and _examination_complete(run_dir, method, alphas):
        print(
            f"cv-examine: '{run_dir / SETS_DIR_NAME}' already complete for {method} "
            f"at alphas {list(alphas)} (use --force to recompute)."
        )
        return []

    tasks = [(str(csv), method, hparams, alphas) for csv in fold_csvs]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_examine_fold, tasks))
    else:
        results = [_examine_fold(task) for task in tasks]

    written = _write_examination(run_dir, method, results, alphas)
    for path in written:
        print(f"wrote '{path}'")
    return written


def run_deploy(
    run_dir: str | Path,
    new_data_dir: str | Path,
    *,
    methods: Sequence[str] = ("SAPS",),
    alpha: float = 0.05,
    calib_glob: str = VALIDATION_GLOB,
    test_glob: str = DEFAULT_TEST_GLOB,
) -> List[Path]:
    """Resolve the K fold calibration + K test CSVs and run CV+ on new data.

    ``run_dir`` holds the fold ``validation_prediction.csv`` (calibration); ``new_data_dir``
    holds the same new samples scored under each fold model (``test_glob``, sorted to match
    the calibration folds). Writes ``cv_plus_sets_{METHOD}_alpha{ALPHA}.csv`` into
    ``<new_data_dir>/conformal_sets/``.
    """
    run_dir, new_data_dir = Path(run_dir), Path(new_data_dir)
    calib_csvs = sorted(run_dir.glob(calib_glob))
    test_csvs = sorted(new_data_dir.glob(test_glob))
    if not calib_csvs:
        raise ValueError(f"No calibration CSVs ('{calib_glob}') under '{run_dir}'.")
    if len(calib_csvs) != len(test_csvs):
        raise ValueError(
            f"Fold count mismatch: {len(calib_csvs)} calibration CSVs vs "
            f"{len(test_csvs)} test CSVs ('{test_glob}' under '{new_data_dir}'). "
            "CV+ needs the test samples scored under every fold model."
        )
    out_dir = new_data_dir / SETS_DIR_NAME
    return cp.run_cv_plus_apply(
        calib_csvs, test_csvs, out_dir, methods=list(methods), alpha=alpha
    )


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["cv-examine", "deploy"],
        required=True,
        help="cv-examine: honest within-fold LOO sets for a 10-fold training run. "
        "deploy: CV+ sets for new data scored under all K fold models.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Classifier run directory containing split*/validation_prediction.csv.",
    )
    parser.add_argument(
        "--new-data-dir",
        type=Path,
        help="[deploy] Directory with the new samples scored under each fold model.",
    )
    parser.add_argument(
        "--method", default="SAPS", help="[cv-examine] Conformal score (default SAPS)."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["SAPS"],
        help="[deploy] Conformal scores to emit.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(cp.REPORT_ALPHAS),
        help="[cv-examine] Target miscoverage values to emit.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="[deploy] Target miscoverage."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="[cv-examine] Worker processes over the per-fold scoring tasks.",
    )
    parser.add_argument(
        "--test-glob",
        default=DEFAULT_TEST_GLOB,
        help="[deploy] Glob (under --new-data-dir) for the per-fold test CSVs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="[cv-examine] Recompute even if the output CSVs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_arguments()
    if args.mode == "cv-examine":
        run_cv_examine(
            args.run_dir,
            method=args.method,
            alphas=args.alphas,
            jobs=args.jobs,
            force=args.force,
        )
    else:  # deploy
        if args.new_data_dir is None:
            raise SystemExit("--mode deploy requires --new-data-dir")
        run_deploy(
            args.run_dir,
            args.new_data_dir,
            methods=args.methods,
            alpha=args.alpha,
            test_glob=args.test_glob,
        )


if __name__ == "__main__":
    main()
