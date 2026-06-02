# Conformal prediction scores for classification

Short reference for the four split-conformal score functions we use (all available in
TorchCP `torchcp.classification.score`). Notation:

- `k` classes, softmax probabilities `p(x) = (p_1, …, p_k)`, true label `y`.
- Calibration set of `n` labelled points held out from training.
- Target miscoverage `α` (e.g. `α = 0.1` → 90 % coverage).

**Split-conformal recipe (common to all four):**

1. Pick a *nonconformity score* `s(x, y)` — higher means "label `y` fits `x` worse".
2. On calibration data compute scores `s_i = s(x_i, y_i)` and take the conformal quantile
   `q̂ = the ⌈(n+1)(1−α)⌉ / n empirical quantile` of `{s_i}`.
3. For a test point, the prediction set is every label whose score is below threshold:
   `C(x) = { y : s(x, y) ≤ q̂ }`.

This guarantees **marginal coverage** `P(y ∈ C(x)) ≥ 1 − α` under exchangeability, *regardless
of the method*. The methods differ only in the score `s`, which changes set **size** and how
well coverage holds **conditionally** (per class / per region).

---

## LAC — Least Ambiguous set-valued Classifier

Also called THR / "softmax score".

- **Score:** `s(x, y) = 1 − p_y(x)`.
- **Set:** keep every class whose softmax prob is ≥ `1 − q̂`.
- **Pros:** provably the **smallest** average set size for a given marginal coverage; trivially
  simple; great baseline.
- **Cons:** marginal only — tends to **under-cover hard classes and over-cover easy ones**
  (poor class-conditional coverage). Sensitive to miscalibrated softmax.
- *Sadinle, Lei & Wasserman, 2019.*

## APS — Adaptive Prediction Sets

- **Score:** sort the probabilities descending and accumulate until you reach the true class:
  `s(x, y) = Σ_{j : p_j ≥ p_y} p_j` (the total probability mass of all classes at least as
  likely as `y`, often with a randomized tie-break term `u·p_y`).
- **Set:** greedily add classes from most to least probable until the cumulative mass exceeds
  `q̂`.
- **Pros:** much better **adaptivity / conditional coverage** — sets grow on genuinely
  ambiguous inputs and shrink on easy ones.
- **Cons:** **larger average sets** than LAC, especially with many classes; the long
  low-probability tail inflates sets.
- *Romano, Sesia & Candès, 2020.*

## RAPS — Regularized APS

APS plus a penalty that discourages large sets.

- **Score:** APS cumulative mass **plus** a regularization term on the rank of `y`:
  `+ λ · max(0, rank(y) − k_reg)`. Two hyperparameters: `k_reg` (no penalty for the top
  `k_reg` classes) and `λ` (penalty strength).
- **Set:** like APS but the penalty makes it stop adding tail classes sooner.
- **Pros:** keeps most of APS's adaptivity while cutting the tail → **smaller, more stable
  sets**; the headline method for large-class problems (e.g. ImageNet).
- **Cons:** needs tuning of `k_reg`, `λ` (can tune on a calibration split); coverage guarantee
  is unaffected, only set size.
- *Angelopoulos, Bates, Malik & Jordan, 2021 — the [tutorial you linked](https://arxiv.org/pdf/2107.07511).*

## SAPS — Sorted Adaptive Prediction Sets

A simplification of RAPS that discards all probability information except the top-1.

- **Score:** keep the maximum probability `p_(1)`, but replace every non-top class's
  contribution with a constant ranking weight `λ`. So the cumulative score depends only on the
  **rank** of `y` and the single largest probability, not the full tail.
- **Set:** add classes by rank; each step beyond the top adds a fixed `λ`.
- **Pros:** even **smaller sets** than RAPS in practice and **only one hyperparameter** (`λ`);
  robust when the softmax tail is unreliable.
- **Cons:** throws away tail probability detail, so slightly less adaptive than full RAPS in
  some regimes.
- *Huang et al., 2023.*

---

## Choosing between them

| Method | Set size       | Conditional coverage | Hyperparams      | Use when                               |
|--------|----------------|----------------------|------------------|----------------------------------------|
| LAC    | smallest       | weakest              | none             | baseline / well-calibrated probs       |
| APS    | largest        | strong               | none             | want adaptivity, few classes           |
| RAPS   | small          | strong               | `k_reg`, `λ`     | many classes, tame APS's tail          |
| SAPS   | smallest-ish   | strong               | `λ`              | many classes, simpler than RAPS        |

All four give the **same marginal coverage** `≥ 1 − α`; pick based on average set size and how
much you care about per-class coverage. For an exploratory sweep we run all four and compare
empirical coverage vs `1 − α` and average set size vs `α`.

**Reference:** Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*,
[arXiv:2107.07511](https://arxiv.org/pdf/2107.07511).

---

## General findings (method & implementation)

Reusable for any EpiClass classifier, independent of the dataset.

### Implementation gotchas (TorchCP)

- **Feed probabilities, not logits.** EpiClass CSVs already store softmax probabilities, and
  TorchCP score functions default to applying softmax again. Build every score with
  `score_type="identity"` and pass the probabilities directly.
- **Keep `randomized=True` for APS/RAPS/SAPS.** The uniform tie-break term is what makes them
  exactly valid. With `randomized=False`, SAPS scores the top class inconsistently between
  calibration and prediction, collapsing `q̂` to 0 and producing **all-empty sets** (0 % coverage).
  Reproducibility comes from seeding the torch RNG instead (`RNG_SEED`), not from disabling
  randomization.

### Marginal coverage is necessary but not sufficient

The empirical-vs-target coverage plot lands all four methods on the diagonal **by construction** —
the guarantee forces it, so this plot can *never* distinguish methods or reveal a problem. A rare
class can be badly under-covered while the marginal line looks perfect, because the common classes
over-cover to compensate. **Always stratify coverage by true class.**

### How the methods differ on imbalanced data

- **LAC shrinks sets on the hard class.** Its single global threshold is tuned to the majority
  classes, so on a rare/hard class it produces **empty sets** rather than larger ones. An empty
  set is an automatic miss, so the per-class empty-set rate ≈ the per-class coverage shortfall —
  the mechanism behind LAC under-covering rare classes.
- **APS/RAPS/SAPS enlarge sets adaptively**, so they put their largest sets on the ambiguous rare
  class and cover it better. SAPS tends to cover a rare class best because it is the **most
  conservative** (the same property that gives it the highest marginal over-coverage), at the cost
  of slightly larger sets.
- **Takeaway:** at small `k` with a rare class, prefer **SAPS or APS**, not LAC. LAC's "smallest
  sets" advantage is worthless when those small sets are empties that miss the class you care about.
  Regularization in RAPS/SAPS only pays off at large `k`; at `k = 3` it is near-irrelevant.

### Diagnostics that actually decide it

Marginal coverage validates the machinery; these three pick the method:

1. **Per-class coverage** — is the rare class below `1 − α`?
2. **Per-class average set size** — is coverage bought by discrimination (size ≈ 1–2) or by
   abstaining (size → `k`)?
3. **Per-class empty-set rate** — disambiguates a sub-1 average set size (many small non-empty
   sets vs. outright empties); empties are guaranteed misses and a useful "route to manual review"
   flag.

Use enough eval samples per class to trust the numbers: SE of a coverage estimate ≈ `√(α(1−α)/n)`.

### Class-conditional (Mondrian) calibration: sample-size requirements

Mondrian CP (a separate `q̂` per class) would *guarantee* per-class coverage, but needs enough
calibration samples **per class, per fold** (you cannot pool calibration across folds — different
model each fold). Floor for a finite threshold: `n_c ≥ ⌈1/α⌉ − 1` (9 at `α = 0.1`); for stable
coverage within ±δ, `n_c ≈ α(1−α)/δ²` (~35 for ±5 %, ~225 for ±2 %). Below the floor the class is
forced into every set (trivial coverage). Clustered CP (`ClusteredPredictor`/`RC3PPredictor`) pools
similar classes to borrow calibration strength, but has little to pool at small `k`. With a rare
class and too few labels, marginal-only conformal (pick SAPS/APS) is the honest option.

---

## Data-specific findings (donor-sex classifier)

Concrete numbers from running `conformal_prediction.py` on a 3-class donor-sex classifier
(`male` / `female` / rare `mixed`), calibrating on per-fold `validation_prediction.csv`, aggregated
over the 10 CV folds at `α = 0.1`. These are specific to this run, not general conclusions.

- **`female`/`male` covered on target (~0.90) by all four methods**; the rare `mixed` class is where
  they diverge.
- **LAC under-covers `mixed`: ~0.81** (target 0.90). Its empty-set rate on `mixed` spiked to **~0.15**
  (vs ~0.07 on `male`/`female`) — almost exactly the coverage shortfall.
- **Rare-class coverage ranking: SAPS (~0.91) > APS ≈ RAPS (~0.86) > LAC (~0.81).** Cost was modest:
  no method approached set size 3, so coverage came from discrimination, not abstention — SAPS
  `mixed` set size ≈ 1.23 (mostly singletons, ~¼ pairs).
- **Estimates are trustworthy**: `mixed` had 156 eval samples pooled over folds (SE ≈ 0.024), so the
  LAC under-coverage and the method ranking are real, not noise.
- **Mondrian is feasible but noisy here**: `mixed` had ~13 calibration samples per fold (per-fold validation ~26, half for calibration) — above the `α = 0.1` floor of 9, so class-conditional calibration is *not* degenerate (mixed is genuinely thresholded, not forced into every set), but below the ~36 reliability count, so per-fold per-class coverage has SD ≈ ±8%. Mondrian did lift `mixed` onto target (~0.95 for SAPS), and the lift is real, but high-variance — indicative, not a tight guarantee. Conclusion: marginal conformal with **SAPS or APS** is the safe default; reaching ~36 mixed calibration samples/fold (more labels or fewer/larger folds) would make Mondrian bankable.
- **Set-size check confirms the Mondrian lift is genuine, not forcing**: marginal→Mondrian `mixed` average set size only nudged up (LAC 0.84→0.95, APS 1.13→1.24, RAPS 1.04→1.12, SAPS 1.24→1.28) — none ballooned toward the class count of 3, so coverage came from real per-class thresholds, not from forcing `mixed` into every set. LAC's increase reflects Mondrian filling in some of its empty sets; `female`/`male` set sizes barely moved (their per-class threshold ≈ the global one), so the cost is targeted only where needed.
