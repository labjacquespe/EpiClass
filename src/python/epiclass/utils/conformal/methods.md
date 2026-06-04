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

### How RAPS `kreg` depends on the number of classes `k`

RAPS penalizes a class only when its **rank exceeds `kreg`** (ranks are 1 = most probable … `k` = least). The score adds `λ · max(0, rank(y) − kreg)`, so the top `kreg` classes are always penalty-free and only the **`k − kreg` lowest-ranked classes** can ever be penalized. That single fact explains everything about how `kreg` behaves as `k` changes:

- **`kreg ≥ k` ⇒ penalty never fires ⇒ RAPS ≡ APS.** No rank exceeds `kreg`, so `λ` does nothing and set size is independent of the penalty. In the sweep these rows are **flat** — not a bug, just an inert setting.
- **`kreg = k − 1` ⇒ only the single last class can be penalized (once).** Negligible effect at small `k`.
- **The smaller `kreg` is relative to `k`, the more of the tail the penalty controls**, so the more set size responds to `λ`. `kreg = 1` (only the top class free) gives the penalty the most to act on and is the strongest regularization.
- **SAPS has no `kreg`** — it behaves like an implicit `kreg = 1` (keep the top-1, charge a constant weight for every other class), which is why SAPS regularization stays active even when RAPS's does not.

Reading the swept grid `kreg ∈ {1, 2, 5}` by class count (✓ = penalty active, ≈ = barely active, — = inert, i.e. `≡ APS`):

| `k` | Example task       | `kreg=1` | `kreg=2` | `kreg=5` |
| --- | ------------------ | -------- | -------- | -------- |
| 2   | paired-end, cancer | ✓        | —        | —        |
| 3   | donor sex          | ✓        | ≈        | —        |
| 4   | biomaterial type   | ✓        | ✓        | —        |
| 5   | donor life stage   | ✓        | ✓        | —        |

- `k = 2`: `kreg = 1` is the **sole** live knob — it just decides singleton vs the full pair; `kreg ≥ 2` ≡ APS.
- `k = 3`: `kreg = 2` (≈) can penalize only the single last (rank-3) class, so it barely moves set size.
- `kreg = 5` is `≥ k` for every row here, hence inert; for `k = 5` it sits exactly at `kreg = k` (the disabling boundary).

**Practical reading:** for `k ≤ 3` only `kreg = 1` is a live setting — tune `λ` (`penalty`) there and treat `kreg ≥ 2` as "RAPS = APS"; sweeping `kreg` above 1 just wastes grid points. For `k = 4–5`, `kreg ∈ {1, 2}` are both meaningful (1 stronger, 2 gentler) and any `kreg ≥ k` is inert. In general keep `kreg < k`; `kreg = k` disables RAPS's regularization entirely, leaving plain APS.

### Choosing a score when the classifier is uncalibrated (why SAPS)

EpiClass's classifiers are **not calibrated** — like most neural nets they are overconfident, so the softmax *values* are not trustworthy probabilities; only the *ranking* (argmax, top-k order) is reliable. That single fact drives the score choice:

- **APS/RAPS score on the cumulative softmax *mass*** (the summed sorted probabilities up to the true class), so they trust the magnitudes across the whole ranking — exactly the thing miscalibration corrupts, which distorts their set sizes.
- **LAC is the most exposed**: it thresholds directly on `1 − p_top`, entirely at the mercy of the inflated top value, which is why it produces the most confident-*wrong* singletons (high `n_singleton_wrong`).
- **SAPS keeps only the top-1 value plus a constant rank weight, discarding the unreliable tail** — it leans on the rank ordering, the part of an uncalibrated net that survives. It is the robust choice precisely when the numbers are bad.

Two things keep this from being alarming. First, **conformal validity needs only exchangeability, not calibration**: even a wildly miscalibrated net gives valid marginal coverage. Miscalibration costs **efficiency** (larger, less-informative sets and worse per-class behaviour), *not* coverage — and SAPS minimises that efficiency cost. Second, you do **not** need to run both RAPS and SAPS: they are the same underlying scores thresholded two ways (two operating points on one empties↔hedges dial), not two independent detectors. One method plus the always-available softmax **argmax** already gives both signals you want — the *nearest known class* (argmax / ranking) and a *calibrated reject* (an empty set). The flag composition makes the trade concrete: RAPS suppresses empties into informative hedges but trusts the softmax more; SAPS abstains (empty) more but is robust. Note a low flag rate is not automatically good — LAC's low rate is bought by confident-wrong singletons, which is the worst failure mode for QC.

**Temperature scaling does not rescue this for an OOD future.** A single scalar `T` fit on a held-out *in-distribution* split would make APS/RAPS trustworthy *in-distribution*, but: its calibration **does not transfer under distribution shift** (Ovadia et al. 2019); it is a monotone, ranking-preserving transform, so it **does not improve OOD detection**; and it **does not restore the conformal guarantee under shift** (that is broken by exchangeability, which `T` does not touch). So if you expect near-OOD QC data, temperature scaling is an in-distribution *efficiency* tweak, not OOD insurance — it helps least where you need it most. The leverage under shift is elsewhere (set-size / empty-rate as a calibration-free OOD signal, stratified/Mondrian calibration, monitoring, weighted conformal), not in `T`.

**Default:** for EpiClass's uncalibrated classifiers, especially with near-OOD QC data expected, **default to SAPS**; reach for RAPS only if you first temperature-scale *and* your traffic is mostly in-distribution. See the production section below for deployment specifics.

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

## Production deployment & out-of-distribution use

How to actually use this once a classifier is trained on all data and dropped into a public-DB QC pipeline, where the incoming data is wildly out of distribution. The short version: the coverage guarantee is real but **conditional on exchangeability between calibration and test**, and OOD QC data breaks that condition — so plan for the guarantee to degrade and lean on the set geometry, not the nominal number.

### You must hold out a calibration set — training data will not do

Split conformal calibrates `q̂` on labelled scores that are exchangeable with the test point and **not seen during training**. Scores on training data are optimistically biased (the model overfits them), which collapses `q̂` and voids the guarantee. So a model literally "trained on all data" has *no valid calibration set*. Two honest options:

1. **Reserve a dedicated calibration split** the final model never trains on (train on all-but-calibration, calibrate once on the held-out split, deploy). Simplest and what the guarantee is stated for.
2. **Cross-conformal / CV+ / Jackknife+** aggregation if you cannot spare the data — calibrate across CV folds and aggregate, paying a small extra slack in the guarantee but using every sample for both training and calibration.

The 10-fold `validation_prediction.csv` files this module reports on are the evaluation-time stand-in for option 1: each fold's validation predictions are out-of-training for that fold's model, so they give an honest read on the coverage you would get from a held-out calibration set of that size.

### How big should the calibration set be

Marginal coverage: a finite quantile needs `n ≥ ⌈1/α⌉ − 1` (19 at `α = 0.05`); for empirical coverage within ±δ of target, `n ≈ α(1−α)/δ²` (~76 for ±5 %, ~190 for ±2 % at `α = 0.05`). A calibration set of ~1–2k points is a safe marginal default and cheap to hold out.

**Mondrian makes every one of those counts per-class.** So you do *not* reflexively "size it for Mondrian" — you size it to the **rarest class you want a per-class guarantee for**. If that class cannot reach the floor, Mondrian degenerates (the class is forced into every set) and you are better off with marginal conformal (SAPS/APS) or clustered CP. Mondrian only earns its sample cost when the rare classes clear the reliability count (~36 per class at `α = 0.1`, ~190 at `α = 0.05` for ±2 %); otherwise marginal is the honest choice. The `mondrian_feasibility` table makes this call from the label distribution before you trust any Mondrian number.

### When Mondrian is infeasible: stay marginal, do not raise α to unlock it

When the rare class falls below the Mondrian floor at your target `α` (e.g. `α = 0.05` needs 19 calibration samples per class *per fold*, which the rare classes here miss), it is tempting to raise `α` until the floor `⌈1/α⌉−1` becomes reachable. **Don't** — staying marginal at the `α` your QC needs beats lowering `α` to satisfy Mondrian's sample-size math. The two knobs are not substitutes: `α` is a *product decision* (how much coverage the gate requires) and marginal-vs-Mondrian is a *statistical-feasibility* question (do you have the per-class samples). Raising `α` to make Mondrian's arithmetic work trades away the coverage level you decided you need — the tail wagging the dog.

Three reasons it is the wrong trade:

1. **Raising `α` does not fix under-coverage — it lowers the bar.** Mondrian at `α = 0.2` gives a per-class guarantee, but at 80 %. For a 95 %-targeted gate, an 80 % per-class promise is usually the worse deal than a 95 % marginal one that is empirically near-target on the rare class with SAPS/APS.
2. **It is not even a clean statistical win.** The floor only buys a *finite* threshold, not a *reliable* one; you still need ~`α(1−α)/δ²` per class for stability (~36/fold for ±5 % at `α = 0.1`). A rare class with ~13 calib/fold clears the `α = 0.1` floor of 9 but sits well below that — Mondrian there is valid-but-noisy (±8 %), not bankable. Raising `α` lands you in non-degenerate-but-unreliable territory, not safety.
3. **The honest fix is more calibration data, not weaker coverage.** Pool calibration (stop calibrating per-fold; combine folds or hold out a dedicated calibration set) to raise the per-class count enough that Mondrian becomes feasible *at the same `α`*. Or use clustered/group conformal to borrow strength across similar classes. Either way you spend *data* to earn the per-class guarantee, rather than spending *coverage*.

So: keep `α` at the level QC requires, use marginal **SAPS/APS** (best rare-class coverage of the four), and route the rare class to manual review via the set-size / empty-rate signal. Only drop to a higher `α` if you genuinely cannot get more calibration data **and** you would rather promise a lower number honestly than a marginal 95 % that is quietly lower on the class you care about.

### Target coverage

Default operating point: **`α = 0.05` (95 %)**, because these models gate public-DB QC and a missed true label is costly. Report alongside `α ∈ {0.10, 0.20}` so the size/coverage trade-off is visible. But state plainly what 95 % means: it is a **marginal** guarantee under exchangeability, averaged over inputs — not a per-class or per-sample promise, and not a promise on shifted data.

### The out-of-distribution reality

On wildly-OOD public-DB data, calibration and test are **not exchangeable**, so the nominal 95 % is not guaranteed and can degrade sharply (often the true coverage drops while set sizes stay deceptively small). Practical stance for a QC pipeline:

1. **Use set size and empty-set rate as the uncertainty / OOD signal**, not the nominal coverage. A large set says "genuinely ambiguous → route to manual review"; an empty set says "no class fits the calibrated thresholds → likely OOD or mislabel → manual review". This is the part that stays useful under shift.
2. **Stratify where you can.** Mondrian or stratified-by-known-covariate calibration (e.g. per assay) makes exchangeability more plausible *within* a stratum than across the whole distribution, when each stratum clears the sample-size floor.
3. **Monitor.** Track empirical coverage on whatever labelled OOD samples you can get; treat a drop as the signal to recalibrate.
4. **Weighted / shift-robust conformal (Tibshirani et al. 2019)** is the principled fix for *covariate* shift: reweight calibration scores by the test/calibration density ratio. It needs unlabelled target data and a likelihood-ratio estimate, and it does not cover label shift or concept drift. Flagged as **future work**, not yet implemented here.

Bottom line: deploy with a held-out (or CV+) calibration set sized to the rarest class you care about, default `α = 0.05`, report at {0.05, 0.10, 0.20}, and on OOD QC data treat the prediction-set *geometry* (size, emptiness) as the actionable signal while monitoring coverage rather than assuming it.

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
