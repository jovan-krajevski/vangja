# Fixed case studies — the revised experimental design

This folder contains the corrected, audit-compliant experiment suite for the
Vangja paper. It is the answer to three problems the audit found in the
original study (see `../audit/` and `../REVIEW/` for the audit and its
correction plan):

1. **Technical defects** — wrong regularisation sign/form, inconsistent
   seasonal regularisation scale, wrong changepoint-transfer dispatch,
   inconsistent slope-transfer definition, interpolation leakage, missing
   seeds / `target_accept`, broken NUTS dispatch. These were fixed in
   `src/` (commit a562629 and the changes in this working tree) and are
   covered by `tests/test_regularization.py`, `tests/test_stocks.py`,
   `tests/test_time_series.py` and `tests/test_utils.py`.
2. **Foundational design problems** — a ~2,500-configuration grid whose best
   entry was selected *on the test horizon*, MAPE as the selection metric,
   retrospective data presented as evidence, context series scored as
   targets, "Prophet"/"TimeSeers" arms that were actually Vangja
   configurations, and an unvalidated covariance-transfer claim. These
   cannot be fixed in a single code change; they require a different
   experimental approach, which is what this folder implements.
3. **Non-technical / manuscript issues** — honest labels, disclosures
   (S&P 500 overlap, survivorship, calibration), and terminology. These are
   implemented here in the form of labels, provenance records and frozen
   analysis plans; the manuscript (`../paper.tex`) must be rewritten against
   the new results once they exist (see "Manuscript corrections" below).

---

## 1. The core idea in one paragraph

**No grid. No test-horizon selection.** Each case study has exactly **one
frozen configuration**, chosen *a priori* from modelling considerations and
from the retrospective study (disclosed as such), plus a **small, frozen
list of single-perturbation ablations** whose only purpose is to see how each
hyperparameter behaves. Everything runs on identical, provenance-recorded
data; the primary metric is **Relative MAE against persistence**; the
headline aggregate is the **median across target × origin units**; and the
stocks study is scored on **new confirmation origins that were never
evaluated before** (frozen and hash-committed *before* their results are
seen). The old 24 stock origins and the 2016 smart-home period are used only
as **retrospective development data**, and are labelled as such everywhere.

This directly addresses the audit's central criticism (F-1, F-2): the
selection bias of "choose the best of 2,592 configurations on the test set"
disappears because there is nothing to select. The remaining hyperparameter
choices are defended explicitly below, value by value.

---

## 2. What was fixed in `src/` (technical findings)

| Finding | Fix | Where |
|---|---|---|
| F-4 / P0-2 trend regularisation sign+form | negative quadratic `−φ·(w−w_transferred)²` in **all three** pooling modes | `src/vangja/components/linear_trend.py`, `tests/test_regularization.py` |
| F-5 / P0-3 seasonal regularisation scale | one per-grid-point normalization `(1/n_t)·min(0, ‖f_old‖²−‖f_new‖²)`, identical across pooling modes, gated per series on `period > 2·n_obs` | `src/vangja/components/fourier_seasonality.py`, `tests/test_regularization.py` |
| F-6 / P0-4 changepoint-transfer dispatch | reads `delta_tune_method` (not `tune_method`) in all pooling modes | `linear_trend.py`, `tests/test_regularization.py` |
| F-7 / P0-5 slope-transfer definition | the transferred slope is the **end-of-history slope** (`slope + Σδ` under `delta_side="left"`; raw slope under `"right"`), implemented and documented | `linear_trend.py`, `tests/test_regularization.py` |
| F-9 / P0-7 interpolation leak | `load_stock_data` now splits **first** and interpolates (only if requested) within each split using only that split's observations; default is trading-day data, no synthetic weekend rows at window edges | `src/vangja/datasets/loaders.py`, `tests/test_stocks.py` |
| F-11 / P0-8 seeds & NUTS dispatch | `fit()` accepts `random_seed` and `target_accept`; seeds are threaded to MAP/VI/MCMC/predictive sampling; `nuts_sampler` no longer collides with an explicit `step`; `fit_info` records optimizer diagnostics and versions | `src/vangja/time_series.py`, `tests/test_time_series.py` |
| P0-10 metric | `relative_mae()` (with denominator threshold) and `persistence_forecast()` added | `src/vangja/utils.py`, `tests/test_utils.py` |
| P0-13 four-arm design support | `fit(..., include_source_in_target=True, source_data=...)` appends the context as a hierarchical group (co-learning arm) | `src/vangja/time_series.py`, `tests/test_time_series.py` |
| P0-16 universe provenance | public `get_sp500_tickers_at_date()` reconstructs membership **at** a historical date | `src/vangja/datasets/stocks.py` |

PyMC is pinned (`pymc[nutpie]~=6.2`, `uv.lock` committed) — F-11's
dependency-pinning item is also closed.

---

## 3. Protocol summary (what the new runs do differently)

### 3.1 Status of the data (F-2, PROTOCOL.md §1–§3)

- **Stocks, origins 2013-01-01 … 2014-12-01 (24 origins):** retrospective
  development data. They were evaluated in the original study, so no number
  computed on them is presented as confirmation. They are used to check code
  behaviour and for retrospective comparability.
- **Stocks, origins 2023-01-01 … 2024-12-01 (24 origins):** the **new
  confirmation origins**. They were never evaluated before; their results
  have not been inspected. They are frozen in
  `protocol/CONFIRMATION_ORIGINS.csv` (SHA-256
  `ab70ebfbf3cefed315785095d4c5c754791621a3b67cf88763689d784fcc29a0`),
  which is committed **before** any confirmation results are produced.
  `04_run_confirmation.py` verifies the hash and refuses to score if the
  file was modified. Horizon overlap between monthly origins is handled by
  block-resampling whole origin blocks in the paired comparison (§3.8).
- **Smart home (2016):** retrospective. There is no independent confirmation
  data (no second household/period). Every output of the smart-home runs is
  labelled `retrospective`; the paper may only use it as an illustrative
  development study with narrowed generalisation claims (PROTOCOL.md §3.3).

### 3.2 Primary metric (F-13, P0-10, PROTOCOL.md §4)

```
Relative MAE(model) = MAE(model) / MAE(persistence)      per target × origin
```

- Persistence = last training value carried forward (the random walk).
  `< 1` beats persistence; `> 1` is worse.
- **Primary aggregate = median** across target × origin units. Mean, IQR and
  the proportion of units below 1 are also reported, together with the full
  distribution (saved in the unit CSVs).
- **Denominator rule:** the Relative MAE is computed in the *scaled* space
  used for fitting; units whose persistence MAE is `< ε = 1e-3` on the
  scaled target are excluded from the relative-MAE aggregate (count
  recorded) and their absolute MAE is reported separately. Never divide by
  a clipped near-zero denominator.
- **Seed aggregation:** per-unit MAEs are averaged over the seeds actually
  run *before* forming Relative MAE.
- **Failed fits:** recorded in `failures.csv`, reported as failure counts;
  never silently dropped, never imputed.
- **Secondary metrics** (continuity only): MAE, RMSE, safeguarded MAPE
  (denominator floored at 1e-6), and the negative-transfer rate.
- **Context exclusion (F-3, P0-12):** `^GSPC` / `source` / the temperature
  series never appear in any target aggregate; the report scripts assert
  this and abort if violated.

### 3.3 Configuration selection (F-1, P0-11 — the big one)

**There is no selection.** One configuration per case study is frozen in
`stocks/config.py` and `smart_home/config.py` *before* any new results are
produced, with a written justification per hyperparameter (§4). The
development origins are scored with the same frozen configuration; they are
diagnostics, not a selection stage. No configuration is ever chosen — or
re-chosen — based on confirmation results. This is strictly stronger than
"select on development data only" and removes the selection bias the audit
identified, at the cost of making the study an honest demonstration rather
than an optimisation contest.

This is exactly the trade-off the audit asked for: instead of reporting the
best of ~2,500 configurations, we report *one sensible configuration* and
use a handful of ablations to *understand* the contribution of each
hyperparameter.

### 3.4 The frozen ablation list (the "couple of ablations")

Per case study, the frozen configuration plus **7 runs per origin**:

| Arm | Stocks | Smart home | What it isolates |
|---|---|---|---|
| `main` | transfer (prior_from_idata) + partial pooling, trend reg φ=1 | transfer + partial pooling + uniform constant | the proposed model |
| `no_transfer` | hierarchical pooling only | hierarchical pooling only | the transfer contribution (honest relabel of the former "TimeSeers" arm — it is a Vangja config, not the TimeSeers package, F-14) |
| `target_only` | individual pooling, no transfer | individual pooling, no transfer | isolated per-series Prophet-*like* Vangja (honest relabel of the former "Prophet" arm, F-14) |
| `context_group` | S&P 500 as an extra hierarchical group, no transfer | temperature as an extra group, no transfer | the hierarchy contribution of the context (F-16: this separates transfer from context-as-group, so the two are no longer confounded) |
| `trend_reg_off` / `regularised` | φ_trend = 0 | φ_seasonal = 1 | the regularisation potentials |
| `seasonal_reg_on` / `uniform_constant_off` | φ_seasonal = 1 | uniform constant off | seasonal cap / sign-flipping factor |
| `tight_fs_shrinkage` / `shrinkage_10` | seasonal shrinkage 10000 | shrinkage 10 | shrinkage strength |

Two further targeted analyses run separately, never on the full matrix:
`combined` (transfer + context-as-group, development origins only — it
reproduces the former headline design but, per F-16, is *not* used for
attribution), and `transfer_gold`/`transfer_gold_smp` (the negative-control
sensitivity of §3.6, confirmation origins, high-weight subset only).

### 3.5 Baselines (P0-14)

- **Persistence** is the primary-metric denominator and the first baseline.
- **Drift** and **seasonal-naïve (7-day)** baselines are computed in
  `common.py`/report scripts on the identical processed data.
- The former "Prophet" and "TimeSeers" arms are **relabelled** as
  Prophet-like / TimeSeers-like *Vangja configurations* (F-14). The official
  Prophet package is **not** claimed as beaten; if such a claim is ever
  wanted, the official package must be run on identical data — it is not
  part of this suite.

### 3.6 Stock universe, S&P overlap, survivorship (F-10, P0-16, P1-9)

- **Universe rule:** S&P 500 membership **at the origin date**,
  reconstructed from the Wikipedia historical changes table
  (`get_sp500_tickers_at_date`, accurate from ~1997). Membership is **not**
  conditioned on survival through the horizon; stocks that delist or lack
  data are recorded as availability outcomes in
  `stocks/data/availability.csv` instead of being silently dropped
  (survivorship bias disclosed, not "corrected").
- Each origin's universe, with membership dates where known, is frozen in
  `stocks/data/universe/universe_<origin>.csv`; every cached price file gets
  a SHA-256 sidecar.
- **S&P 500 mechanically contains the targets** (overlap, not future-data
  leakage — only historical index values at the cutoff are used). This is
  disclosed and probed by the **negative-control sensitivity**: transfer
  from gold futures (`GC=F`, unrelated to the constituents) on a frozen
  high-weight mega-cap subset, compared with S&P-transfer on the same subset
  (`05_run_ablations.py`).
- **Trading days only:** `load_stock_data(..., interpolate=False)`; no
  calendar-day invention, no cross-split interpolation (F-9).

### 3.7 Smart-home specifics (PROTOCOL.md §3)

- Primary split: train = 2016-01-01 … 2016-03-31 (91 days), horizon =
  2016-04-01 … 2016-12-16 (260 days). Two extra rolling origins
  (2016-04-01 and 2016-07-01, 91-day windows, shorter horizons) as
  sensitivities.
- **Context cutoff correction:** the original runner loaded Boston
  temperature through the *end of the test period* (a future leak into the
  transferred seasonality). Here the temperature context always ends at the
  end of each origin's training window.
- Four appliance series → per-series effects and descriptive uncertainty
  only; no population-level claims (PROTOCOL.md §9).

### 3.8 Dependence-aware paired comparison (P0-18, PROTOCOL.md §9)

No pooled Diebold–Mariano and no stock-only bootstrap (overlapping horizons,
repeated stocks, shared market dates violate their assumptions). The
transfer vs no-transfer comparison on the confirmation stage is made with
**two-way block resampling**: stocks are resampled with replacement and
whole **half-year origin blocks** are resampled with replacement (keeping
overlapping horizons grouped), recomputing the median paired difference each
iteration; percentile CIs and the negative-transfer rate are reported. If
too few independent blocks exist, the output is descriptive only — no
formal significance claim. The smart-home comparison is purely descriptive
(4 series).

### 3.9 Covariance transfer (F-8, P0-17, P1-3)

The original study fitted the S&P 500 source with mean-field ADVI, so it
could not demonstrate that posterior *covariance* was preserved. Here:

- All **confirmation** runs fit the source with **NUTS** (nutpie,
  `target_accept=0.9`, seeded), so the transferred `prior_from_idata`
  prior carries genuine covariance. ADVI is allowed for development
  screening only and is never used as covariance evidence.
- `stocks/06_covariance_transfer.py` saves the source posterior, its
  covariance/correlation matrices, and directly compares **joint transfer**
  (`prior_from_idata`) with **marginal moment matching** (`parametric`) on
  forecasting performance. The result determines how the paper presents the
  claim: main contribution / secondary uncertainty contribution / package
  capability only (PROTOCOL.md §8).

### 3.10 Uncertainty & calibration (F-15, P1-2, P1-4)

- MAP is a point estimate; `stocks/07_calibration.py` records multi-start
  optimizer diagnostics (status, objective, gradient norm, parameter spread)
  for the finalists.
- Empirical coverage/width of the MAP residual-based intervals on the test
  set, and CRPS from posterior-predictive draws (NUTS refit of the finalists
  on a subset), are computed and saved.
- The paper must disclose that the MAP interval path is a residual
  heuristic unless the calibration results say otherwise.

### 3.11 Seeds, provenance, artifacts (P0-8, P0-12, P0-15, §10–§13)

- Seeds: `BASE_SEED=42`; per-cell seeds `seed_for(origin_idx, config_idx)`;
  the context fit has one seed per origin so **all configs of an origin
  transfer the identical source posterior**; finalists run across
  `FINALIST_SEEDS = (42, 20240816, 7)`.
- Every artifact bundle (`manifest_*.json` + `units_*.csv` +
  `forecasts_*.csv`) records: commit SHA, package versions, seeds,
  `fit_info` (incl. MAP optimizer diagnostics), the aggregate, availability
  info, and the data hash sidecars.
- Everything is checkpointed: completed cells are skipped on restart, so
  HPC jobs can be resubmitted.

---

## 4. Frozen configurations and why (the a priori choices)

### 4.1 Stocks (`stocks/config.py`)

Data: context window 1460 calendar days (`^GSPC`); target window 91 calendar
days (≈63 trading days); horizon 365 calendar days (≈252 trading days).

| Parameter | Value | Why |
|---|---|---|
| `n_changepoints` | 25 | the Prophet default; unchanged |
| `delta_side` | `"right"` | slope parameter = end-of-history slope, the quantity that extrapolation continues from (P0-5) |
| `slope_sd`/`intercept_sd` | 5 | the Prophet defaults (the retrospective grid had a single value 5 for slopes) |
| `beta_sd` (seasonality) | 5 | middle of the retrospective grid {1,5,10}; the default-5 keeps the weakly informative scale |
| `lt pool` | partial, shrinkage 100 | hierarchical strength across stocks; 100 is the paper's stated setting |
| `fs pool` | partial, shrinkage 100 | same hierarchical structure as the trend (the retrospective "best" of 10000 was test-selected and is *not* frozen; it is run as the `tight_fs_shrinkage` ablation instead) |
| yearly order | 6, weekly order 3 | the original runner's structure |
| transfer | `prior_from_idata` (joint MVN) | the method whose covariance properties this paper is about; NUTS-fitted source on confirmation |
| `lt_loss_factor` | 1 | trend regularisation on, as in the paper's described configuration |
| `fs_loss_factor` | 0 | the transferred prior itself constrains amplitude; the cap is probed in the `seasonal_reg_on` ablation |
| uniform constant | off | stocks do not share a sign-flippable seasonal shape |
| scaler | maxabs; scale_mode complete; σ individual | inherited from the original study (each stock keeps its own noise) |
| no calendar interpolation | — | F-9 |

### 4.2 Smart home (`smart_home/config.py`)

Data: appliances Furnace 1/2, Fridge, Wine cellar (daily kW); Boston
temperature context from 2013-01-01 up to each origin's training end.

| Parameter | Value | Why |
|---|---|---|
| trend | `FlatTrend(0.5, 0.1)` individual | energy use hovers around a stable mean; the level is well identified from 91 days, and the Bayesian selection chose the smallest grid value 0.1 (PSIS-LOO on ADVI + NUTS verification and leave-future-out CRPS all agree; see `smart_home/best_hyperparams.md` §3) |
| yearly | FS(365.25, 5), `beta_sd=0.25`, partial, shrinkage 1 | order 5 is **required by the transfer design** (the temperature context model's order; the `prior_from_idata` dimension is tied to the source order) and the NUTS verification shows no cost vs order 3; `beta_sd=0.25` is the PSIS-LOO winner (>1 dse) with the PPC caveat disclosed (§4.1) |
| weekly | FS(7, 2), partial, shrinkage 1 | household-specific and **not transferred**, so a free choice: NUTS verification and leave-future-out CRPS prefer 2 over the frozen 3, and it is the parsimonious option (§4.2) |
| uniform constant | on, `[-1,1]`, shrinkage 1 | sign-flipping seasonal factor (heating vs cooling appliances); its effect is probed by the `uniform_constant_off` ablation |
| transfer | `prior_from_idata` on the yearly seasonality only | temperature informs the yearly shape; weekly patterns are household-specific |
| `loss_factor` | 0 | the amplitude cap is probed in the `regularised` ablation |
| scaler | minmax, individual scaling, σ individual | inherited from the original runner |

These choices are defensible a priori and match the modelling narrative of
the paper. The prior scales (`intercept_sd`, `beta_sd`) and the weekly order
were then **re-selected by the Bayesian workflow on training windows only**
(`02_select_hyperparams.py`; evidence and defence in
`smart_home/best_hyperparams.md`) and frozen deliberately into
`smart_home/config.py`; the yearly order is fixed at 5 by the transfer
dimension. If any of them is wrong, the ablations will show it — and that is
exactly the point of running a few targeted ablations instead of a grid.

---

## 5. How to run

Setup (same environment as the rest of the repo):

```bash
uv sync --extra test --extra datasets
```

### 5.1 Stocks

```bash
# 1. Fetch universe + data (network; writes data/ with SHA-256 sidecars)
python fixed_case_studies/stocks/01_fetch_data.py

# 2. Bayesian hyperparameter selection on development training windows
#    (recommendations are then frozen deliberately into stocks/config.py)
python fixed_case_studies/stocks/02_select_hyperparams.py [--verify]

# 3. Retrospective development runs (24 origins x 7 configs; ADVI source)
python fixed_case_studies/stocks/03_run_development.py

# 4. Confirmation runs (24 frozen origins x 7 configs; NUTS source;
#    hash-verified origins; repeated seeds for main & no_transfer)
python fixed_case_studies/stocks/04_run_confirmation.py

# 5. Targeted analyses: former-headline reproduction (dev only) and the
#    gold negative-control sensitivity (confirmation, high-weight subset)
python fixed_case_studies/stocks/05_run_ablations.py

# 6. Covariance transfer: joint vs marginal on a NUTS-fitted source
python fixed_case_studies/stocks/06_covariance_transfer.py --origin 2023-01-01

# 7. Calibration + MAP diagnostics for the finalists
python fixed_case_studies/stocks/07_calibration.py --origin 2023-01-01

# 8. Aggregates, tables, block-bootstrap paired comparison
python fixed_case_studies/stocks/08_report.py
```

Every script supports `--origins` / `--configs` subsets for testing, and
all runs checkpoint (resubmit to continue). Outputs go to
`fixed_case_studies/stocks/results/` (gitignored).

### 5.2 Smart home

```bash
python fixed_case_studies/smart_home/01_fetch_data.py
python fixed_case_studies/smart_home/02_select_hyperparams.py
python fixed_case_studies/smart_home/03_run_main.py
python fixed_case_studies/smart_home/04_report.py
```

All outputs are labelled `retrospective`. The hyperparameter selection
(`02_select_hyperparams.py`) runs on training windows only; its chosen
values are then frozen deliberately into `smart_home/config.py` (see
`smart_home/best_hyperparams.md` for the evidence and defence).

### 5.3 Bayesian hyperparameter selection (both studies)

The Prophet-inherited hyperparameters (`intercept_sd`, `beta_sd`,
`series_order`, `n_changepoints`, `slope_sd`) are set by a **Bayesian
workflow on development training windows only** — no test horizon is ever
used (the old grid selected on test MAPE, REVIEW/FINDINGS.md F-1).  See
`HYPERPARAMETER_SELECTION.md` for the full protocol and the alternative
strategies (leave-future-out CRPS, model averaging, robustness sweeps,
full-Bayes hyperpriors); all of them are implemented in
`fixed_case_studies/hyperparams/`.

```bash
# smart home (4 appliances x 91-day training windows; ~10-20 min screening)
python fixed_case_studies/smart_home/02_select_hyperparams.py
# stocks (3 dev origins x 10 stocks; ~30-60 min screening; --verify adds NUTS)
python fixed_case_studies/stocks/02_select_hyperparams.py [--verify]
```

Outputs (gitignored): `results_hyperparams/*_selection.md` (per-hyperparameter
evidence), `*_recommendations.json` (chosen value + justification).  The
scripts **never modify `config.py`**; freezing a recommendation is a
deliberate step that must update the per-value justification in §4.

### 5.4 HPC

The original SLURM/Singularity scripts in `../case_studies/` can be reused
verbatim — the entry points are the `python fixed_case_studies/...` commands
above. Expected scale: the confirmation stage is the heavy part (per origin:
one NUTS context fit, reused across configs via the source cache, plus 7
MAP target fits; finalists × 3 seeds).

### 5.5 What "satisfying the audit" means, concretely

| Audit requirement | Where satisfied |
|---|---|
| no test-horizon selection (F-1) | §3.3 — frozen configuration, no selection anywhere |
| retrospective vs confirmatory split (F-2) | §3.1 — frozen, hash-committed confirmation origins; smart home labelled retrospective |
| context excluded from targets (F-3) | §3.2 — enforced and asserted in report scripts |
| fixed transfer machinery (F-4…F-7) | `src/` fixes + `tests/test_regularization.py` |
| trading-day modelling (F-9) | §3.6 + `load_stock_data` fix + tests |
| seeds/target_accept/NUTS dispatch (F-11) | §3.11 + `fit()` fix + tests |
| Relative MAE primary metric (F-13) | §3.2 + `utils.relative_mae` + tests |
| honest baselines (F-14) | §3.4, §3.5 — relabelled arms; no Prophet claim |
| calibrated uncertainty (F-15) | §3.10 + `07_calibration.py` |
| transfer vs hierarchy attribution (F-16) | §3.4 four-arm design |
| covariance evidence (F-8) | §3.9 + NUTS sources on confirmation + `06_covariance_transfer.py` |
| universe & survivorship provenance (P0-16) | §3.6 + `01_fetch_data.py` |
| dependence-aware inference (P0-18) | §3.8 + `common.two_way_block_bootstrap` |

---

## 6. Manuscript corrections still needed (after the new results exist)

`../paper.tex` must be revised against these runs (PLAN P2-3); nothing in
the paper should be written until the confirmation results are in:

1. Abstract/results recomputed; no "60% better than Prophet / 20% better
   than TimeSeers" unless the official packages are actually run.
2. "Parametric transfer" described as **posterior-moment matching**, not MAP.
3. Document the seasonal regularisation formula and the slope-transfer
   definition (now implemented and tested).
4. Honest horizon framing (the 4× horizon is the stocks study only; smart
   home is ≈2.9×).
5. Calibration disclosure for the MAP interval heuristic.
6. Disclose S&P 500 mechanical overlap and the survivorship limitation.
7. Relabel or remove Prophet/TimeSeers baseline claims.
8. All smart-home numbers labelled retrospective/illustrative; generalisation
   claims narrowed.
9. AI-use disclosure.

## 7. Known limitations of the revised design (stated honestly)

- The frozen configuration may not be the best possible one — that is the
  point: the study now measures *a defensible method*, not an optimised
  configuration, so the numbers are honest but possibly conservative.
- Monthly confirmation origins have overlapping horizons; the block
  bootstrap models this overlap but there are few truly independent origin
  blocks (two years of origins), so paired results should be read as
  descriptive-with-uncertainty, not as a formal significance test.
- The confirmation universe is reconstructed from Wikipedia's changes table
  (accurate from ~1997); historical index weights are not available, so the
  high-weight sensitivity uses a frozen mega-cap list as an approximation.
- Smart-home has no independent confirmation data at all; its role is
  illustrative.
- Data revisions by the provider mean bit-for-bit reproduction of any
  specific downloaded dataset is not guaranteed; the SHA-256 sidecars freeze
  the *used* data so every reported number is traceable to the exact inputs.
