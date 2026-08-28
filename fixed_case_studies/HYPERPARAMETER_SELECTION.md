# Hyperparameter selection for the Prophet-inherited hyperparameters

> Status: **development-stage procedure**. All selection evidence is computed
> on **training windows only** — the test horizons (smart-home 260-day /
> stocks 365-day) and the confirmation origins are never touched by any step
> below. The result is a *recommendation*; freezing it into `config.py` is a
> deliberate, documented act (see "Freezing" at the end).
>
> This file is the written defence of *how* the Prophet-inherited
> hyperparameters are set.
>(a) a **small, pre-registered candidate set**
> (b) **Bayesian evidence on development data only**
> (c) a **robustness argument** that the chosen values lie in a flat region of
> the selection criterion — so the choice is not an artefact of picking the best of many.

---

## 1. Which hyperparameters, and why they need setting

Vangja inherits the Prophet modelling grammar: a piecewise-linear trend
(`LinearTrend` / `FlatTrend`) and Fourier seasonality
(`FourierSeasonality`), each with a Normal/Laplace prior whose **standard
deviation controls the amount of regularisation**. Two further
Prophet-inherited choices are the **number of Fourier terms**
(`series_order`) and the **number of trend changepoints**
(`n_changepoints`).

| Case study | Hyperparameter | Current frozen value | Why it needs a Bayesian workflow |
|---|---|---|---|
| Smart home | `FlatTrend.intercept_sd` | 0.5 | Prophet default is 5; on 91 daily observations a wide intercept prior invites overfitting |
| Smart home | `FourierSeasonality.beta_sd` | 1.5 | ditto; wider than needed priors let the seasonality absorb noise |
| Smart home | `FourierSeasonality.series_order` (yearly / weekly) | 5 / 3 | too many Fourier terms overfit short series (parameter count grows 2×order) |
| Stocks | `LinearTrend.n_changepoints` | 25 | the Prophet default, but with only ≈63 trading days 25 changepoints is one every ~2.5 days — a parameter explosion that slows NUTS and invites overfitting |
| Stocks | `LinearTrend.slope_sd`, `intercept_sd` | 5 / 5 | Prophet defaults, wide for a 63-day window |
| Stocks | `FourierSeasonality.beta_sd` | 5 | Prophet default |
| Stocks | `FourierSeasonality.series_order` (yearly / weekly) | 6 / 3 | see smart home |

Everything else (means, uniform constant, pooling, shrinkage) is kept
fixed: the workflow is **coordinate-wise** — one hyperparameter is varied at
a time around the frozen configuration, which keeps the number of fits
small (≈4 candidates × 4–6 hyperparameters) and makes each recommendation
attributable to that hyperparameter.

---

## 2. The Bayesian workflow (the primary procedure)

Four steps, in order. Every step runs on the **development training window
of the primary origin** (smart home: 2016-01-01..2016-03-31; stocks: a
frozen subset of the 24 retrospective origins, see §2.4).

### 2.1 Step 1 — Prior-predictive calibration of the prior scales

For each candidate value of a prior-scale hyperparameter (`intercept_sd`,
`beta_sd`), build the model with **plain priors** (transfer off, so the
prior is exactly the candidate's Normal), fit the graph cheaply
(`method="mapx"`), and draw from the prior predictive. Because vangja
scales `y` to ≈ `[-1, 1]` and `t` to `[0, 1]`, the fraction of prior
predictive draws inside `[-2, 2]` is a quantitative plausibility check
(`utils.prior_predictive_coverage`):

- < 5 % coverage → priors too wide; the model spends probability mass in
  physically impossible regions;
- 30–60 % → the target range for a flexible Prophet-like model;
- > 95 % → priors too tight.

This directly attacks the "Prophet priors are too wide" problem: the
Prophet-default `beta_sd=10` typically gives near-0 % coverage on the
scaled data, which is evidence that the default is inappropriate for a 
91-day window.

### 2.2 Step 2 — PSIS-LOO model comparison (selection criterion)

For each hyperparameter and each candidate value, fit the **full study
model** (transfer on, partial pooling, as frozen in `config.py`) with the
posterior-sampling method appropriate to the stage:

- **screening:** ADVI (fast; allowed for development screening);
- **verification:** the top-2 candidates are refit with a small **NUTS**
  run so the final comparison is not a variational artefact.

Then compute the expected log pointwise predictive density via **PSIS-LOO**
(`model.loo()`; Vehtari, Gelman & Gabry 2017). LOO is computed on the
**training data** — it is the Bayesian estimate of out-of-sample
performance *without holding data out*, so the test horizon is untouched by
construction. Candidates are compared by `elpd_loo` difference with its
standard error `dse` (computed from the pointwise differences), and the
selection rule is deliberately conservative:

1. if the best candidate beats the frozen value by **less than 1 `dse`**,
   keep the frozen value ("no evidence to change");
2. if the top candidates are within **1 `dse`** of each other, prefer the
   **smaller** value (smaller sd / fewer terms / fewer changepoints) — the
   parsimony tie-break that also serves the speed goal;
3. otherwise select the best candidate.

The per-candidate **fit wall-time is recorded** and reported, because
reducing `n_changepoints` is also a sampling-cost decision: the report
shows the parameter-count / wall-time trade-off explicitly.

### 2.3 Step 3 — Verify the top-2 with NUTS

The selected and runner-up values are refit with a small NUTS run
(`nuts_sampler="nutpie"`, seeded, `target_accept=0.9`) on the same
development data and re-scored with PSIS-LOO. The report records both
screening (ADVI) and verification (NUTS) tables; the recommendation is
taken from the NUTS table when available.

### 2.4 Step 4 — Aggregate across development units

- **Smart home:** the primary origin (4 appliances, 91 days) plus the
  `origin_jul` rolling origin where available. Pointwise elpd is pooled
  across appliances × origins; the comparison is descriptive, the table
  is the evidence, not a significance test.
- **Stocks:** three frozen development origins (2013-01-01, 2013-07-01,
  2014-01-01) × a seeded 10-stock subset. Pointwise elpd is pooled across
  stocks × origins. The subset and origins are recorded in the report and
  are a documented screening device: the selected values are then used
  for the full study, whose own numbers are the final evidence.

### 2.5 Outputs

Every run writes, under `results_hyperparams/`:

- `*_prior_predictive.csv` — Step 1 coverage per candidate;
- `*_loo_screening.csv`, `*_loo_verification.csv` — Step 2/3 tables
  (elpd, se, elpd_diff, dse, weight, wall-time, parameter count);
- `*_recommendations.json` + `*_selection_report.md` — the chosen values,
  the evidence, and the prose justification;
- provenance (commit SHA, versions, candidates, data hash) in the report.

---

## 3. Better strategies (alternatives to hard selection)

The Bayesian workflow above is still a *selection* procedure, and any
selection — even on development data — carries a (smaller) selection-bias
risk. The following strategies are **strictly better defences** and are
implemented alongside the workflow. They are complementary: the paper can
report the LOO table (workflow) *and* one or more of these as the
robustness argument.

### 3.1 Strategy B — Leave-future-out time-series CV with CRPS (`ts_cv.py`)

**Why it is better.** PSIS-LOO is known to be optimistic for autocorrelated
time series (the LOO posterior overstates the leave-one-out fit when
observations are dependent; Vehtari et al. 2017 note that for time series
k-fold or leave-future-out is preferable). Strategy B evaluates the
candidates by **expanding-window forecasts inside the training window**:
train on the first `k` days, predict the next `h` days, score the
predictive distribution, extend the window, repeat. The scoring rule is
**CRPS** (continuous ranked probability score) computed from the full
posterior-predictive ensemble (posterior draws propagated through the
model), which is a *proper* scoring rule — the same ensemble the paper's
uncertainty claims rest on. The test horizon is still untouched: the
held-out blocks are segments of the training window itself.

**Cost.** ~5 ADVI fits per candidate (one per fold); used as a second,
more trustworthy ranking, and as the headline when the LOO and CV rankings
disagree (report both).

### 3.2 Strategy C — Model averaging instead of selection (`stacking.py`)

**Why it is better.** Selecting one hyperparameter value discards the
uncertainty about the hyperparameter. Strategy C computes **pseudo-BMA
weights with a Bayesian bootstrap** (Vehtari et al. 2017) from the
pointwise elpd of the candidate fits, and reports both (a) the weight
distribution — "the data cannot distinguish values 0.5 and 1.0, weights
0.4/0.6" — and (b) a **model-averaged forecast** (predictions combined with
the weights). If the weights are concentrated, selection and averaging
coincide; if they are diffuse, the paper can report the averaged model and
say the choice is immaterial. This is the practical approximation of the
full-Bayes ideal (strategy E) at a fraction of the cost.

### 3.3 Strategy D — Robustness / flatness sweep (`robustness.py`)

**Why it is better.** The design goal is models "resilient to changes in 
these parameters". Strategy D perturbs each chosen value by ±50 % (or ±1
changepoint / ±1 order for discrete parameters), refits, and reports the
range of `elpd_loo` over the perturbation. If the chosen value sits in a
flat region (range < 1 `dse`, or smaller than the difference to the next
hyperparameter's effect), the value is *robust*: a slightly different
choice would not change the results, so the exact number is not
over-interpreted. The answer is "the objective is flat; any value in this
interval behaves identically".

### 3.4 Strategy E — Full-Bayes hyperpriors (`full_bayes.py`)

**Why it is the theoretical ideal.** Instead of selecting `intercept_sd` /
`beta_sd`, put a prior on them (`HalfNormal`) and sample them jointly with
the model parameters; the posterior of the sd's *is* the data-supported
value, with uncertainty. Implemented as a demonstration on the
single-series structure (flat trend + seasonality) with NUTS: it shows, for
one appliance, the posterior of the sd hyperparameters, which the LOO /
stacking procedures approximate. It does **not** cover the discrete
`n_changepoints` (that would need trans-dimensional sampling, out of
scope) — for discrete hyperparameters strategy C (BMA over candidates) is
the practical equivalent.

### 3.5 How to defend the chosen values in the paper

Recommended phrasing, one paragraph per case study:

> "The Prophet-inherited hyperparameters were set with a Bayesian workflow
> on development data only. For each candidate value we computed the
> prior-predictive coverage on the scaled target (target 30–60 %), the
> PSIS-LOO expected log predictive density on the training window, and —
> for the top two candidates — a small NUTS refit. Values within one
> standard error of the best were resolved toward the smaller, more
> regularising value. As a robustness check we verified that the selection
> criterion is flat within ±50 % of the chosen values, and we report a
> leave-future-out CRPS evaluation and model-averaged weights, neither of
> which changes the conclusion. No test-horizon or confirmation data were
> used at any step."

---

## 4. Pre-registration and freezing

- The candidate grids live in `hyperparams/candidates.py` and are **frozen
  before results are inspected**: changing them requires amending this
  document, exactly like changing `config.py` (§"Frozen configurations" in
  `README.md`).
- The scripts **never modify `config.py`**. They write
  `results_hyperparams/*_recommendations.json`; freezing the recommended
  values into `config.py` is a manual step that must (a) update the
  per-value justification in `fixed_case_studies/README.md` and (b) re-hash
  the confirmation origins if the configuration affects them (it does not —
  the confirmation origins are data, not configuration — but the
  justification must be amended).
- Selection is on development data; the confirmation stage then runs the
  frozen values **once** (PROTOCOL.md §5).

## 5. Reproducibility

All fits are seeded (`random_seed` threaded through `fit()`); the report
records the commit SHA, package versions, candidate grids, origins, stock
subset and seeds. Fits are cached under `results_hyperparams/cache/` so the
scripts resume after interruption.
