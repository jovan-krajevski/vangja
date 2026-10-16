# Vangja protocol — distilled reference

Source of truth: `Vangja-project/REVIEW/PROTOCOL.md` (read-only archive) and the
implementations under `fixed_case_studies/`. Where the two differ, the
implemented value is named explicitly. This is the working reference for design
and review; changes here do not change the frozen protocol.

## 1. Data and splits

| Study | Train window | Horizon | Origins | Source |
|---|---|---|---|---|
| Stocks | 91 calendar days (`TARGET_WINDOW_DAYS`) | 365 days (`HORIZON_DAYS`) | 24 dev origins `DEV_ORIGINS` 2013-01-01..2014-12-01 + frozen confirmation origins | `fixed_case_studies/stocks/config.py:25` |
| Stocks context | 1460 days (`CONTEXT_WINDOW_DAYS`) | — | one per origin | `config.py:34` |
| Smart home | 91 days, primary cutoff 2016-04-01 | 260 days | primary + 2 rolling origins | `fixed_case_studies/smart_home/config.py:39-45` |

- Stocks run on **trading days only**; no calendar interpolation
  (`load_stock_data(..., interpolate=False)`).
- Smart-home temperature context is cut off at the end of each target training
  window (`fixed_case_studies/smart_home/config.py:8-13`).
- Universe rule: `members_at_origin`, reconstructed from the frozen Wikipedia
  revision (`src/vangja/datasets/stocks.py:23`); missing/delisted series are
  availability outcomes, not silent drops.

## 2. Confirmation freeze

- `fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv` + `.sha256` are hashed
  and committed **before** viewing confirmation results.
- `common.load_confirmation_origins` (`common.py:596`) recomputes the hash and
  raises on mismatch; confirmation scoring must be impossible without it.
- If independent confirmation data cannot be obtained, the study is
  retrospective and must be labelled as such.

## 3. Primary metric

```
Relative MAE(unit) = MAE(model) / MAE(persistence)
```

- Unit = target × origin; persistence = last training value carried forward.
- Aggregate = **median** across units; also mean, IQR, `prop_below_1`,
  distribution, excluded count.
- Denominator rule: `REL_MAE_EPSILON = 1e-3` on the scaled target
  (`common.py:44`). Units below ε are excluded from the aggregate and their
  absolute MAE reported separately. Never clip the denominator.
- Seed aggregation: average per-unit MAE across seeds before forming the ratio.
- Secondary: MAE, RMSE, MASE/RMSSE, safeguarded sMAPE/MAPE, negative-transfer
  rate.
- Failed fits: reported with failure rate; success-only tables must say so.

## 4. Selection and freezing

- One global config per study; selection is by the primary aggregate on
  **development** units only (`fixed_case_studies/stocks/config.py:5`).
- Hyperparameters: coordinate-wise candidates; PSIS-LOO on training windows;
  top-2 NUTS verification; parsimony tie-break within 1 `dse`; ±50 % robustness;
  LFO-CV + CRPS; stacking/BMA; optional full-Bayes hyperpriors
  (`HYPERPARAMETER_SELECTION.md`).
- Freezing is a manual, documented act; scripts never edit `config.py`.

## 5. Ablations and baselines

Required baselines (`PROTOCOL.md` §6): persistence, drift, seasonal naïve
(7-day), non-transfer Vangja, transfer Vangja, hierarchical Vangja
(partial pooling; documented subset if compute-limited), honestly relabelled
Prophet-like/TimeSeers-like Vangja ablations, ARIMA/Auto-ARIMA. Deep learning
is out of scope.

Stocks arm names in `stocks/config.py`: `main` (prior_from_idata + partial
pooling), `transfer_only` (transfer + individual), `no_transfer` (hierarchical
only), plus negative-control (`GC=F`) and high-weight-subset sensitivities.

## 6. Transfer × hierarchy arms (`PROTOCOL.md` §7)

| Arm | Transfer prior | Context as group |
|---|---|---|
| 1 | No | No |
| 2 | Yes | No |
| 3 | No | Yes |
| 4 | Yes | Yes |

Context forecasts are excluded from every target aggregate (§7).

## 7. Joint covariance (`PROTOCOL.md` §8)

- The source posterior must come from NUTS (or another covariance-capable
  method) before any joint-covariance claim.
- Save source draws, diagnostics, covariance/correlation matrices,
  transferred-prior covariance, forecasts, calibration.
- ADVI is a screening method; its mean-field posterior is diagonal and is not
  evidence of covariance preservation.

## 8. Dependence-aware comparison (`PROTOCOL.md` §9)

- No pooled Diebold–Mariano; no stock-only bootstrap.
- Two-way block bootstrap over target series and forecast-origin blocks
  (`common.two_way_block_bootstrap`, `common.py:400`); origin blocks group
  overlapping horizons (`add_origin_block`, `common.py:374`).
- Small numbers of independent origins ⇒ descriptive paired effects, no formal
  significance claims.

## 9. Seeds and artifacts (`PROTOCOL.md` §10-§13)

- `BASE_SEED = 42`; `FINALIST_SEEDS = (42, 20240816, 7)`; one source seed per
  origin; per-cell `seed_for` (`common.py:44-49`,
  `fixed_case_studies/stocks/runner.py:397-413`).
- Keep: configs; splits/origins; seeds; environment; aligned observations and
  forecasts; source draws; transferred prior parameters and covariance;
  diagnostics; posterior-predictive draws for finalists; per-horizon errors;
  warnings; failures.
- Provenance per artifact: commit SHA, branch, seed, PyMC/sampler versions,
  data hash, command, runtime, failure state.

## 10. Amendment record

A protocol amendment changes a frozen value and must name: file, old → new
value, justification, and the results it invalidates. Record with
`amendment_record`; a human gate opens automatically.
