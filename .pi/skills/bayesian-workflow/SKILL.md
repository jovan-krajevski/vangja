---
name: bayesian-workflow
description: Fit, diagnose, compare, or transfer Vangja/PyMC models with a defensible Bayesian workflow. Use for priors, prior-predictive checks, NUTS/ADVI/MAP sampling choices, R-hat/ESS/divergence/BFMI/tree-depth checks, PSIS-LOO or LFO-CV comparison, MAP multi-start diagnostics, and transfer-learning priors. Contains the project's numeric thresholds and the known ADVI/MAP traps.
---

# Bayesian workflow

House rules: `AGENTS.md` § Vangja/PyMC gotchas. Diagnostics are a gate, not a
report. A model whose fit fails the thresholds
below cannot produce a number. Thresholds, sources and the scripts that enforce
them are in `references/thresholds.md`.

## When to use

- Building or changing a model, prior, sampler or transfer method.
- Reading `fit_info` or an InferenceData object before downstream evaluation.
- Choosing between candidates (LOO/LFO-CV) or diagnosing a failed fit.
- Any claim involving posterior covariance, calibration, or uncertainty.

## Workflow order

1. **Prior predictive first.** Draw from the prior predictive on the scaled
   target and compute `prior_predictive_coverage`
   (`src/vangja/utils.py:549`). Target **30–60 %** inside `[-2, 2]`;
   `< 5 %` = priors too loose, `> 95 %` = too tight
   (`fixed_case_studies/HYPERPARAMETER_SELECTION.md` §2.1).
2. **Fit with recorded seeds.** `random_seed` always; NUTS also
   `target_accept` (`src/vangja/time_series.py:306`). Never pass an explicit
   `step` together with `nuts_sampler` (`time_series.py:531-544`).
3. **Gate convergence.** R̂ < 1.01, ESS_bulk > 400, ESS_tail > 400, BFMI > 0.3,
   divergences = 0, max-tree-depth hits = 0 (notebook
   `docs/source/notebooks/10_bayesian_workflow.ipynb` §3; WS-S-009). Run
   `scripts/check_convergence.py`; a failing fit is re-fit or reported as
   failed — never silently dropped.
4. **Posterior predictive check.** Outcomes must cover the observed data;
   uncalibrated intervals are not posterior-predictive evidence (WS-S-010).
5. **Compare honestly.** PSIS-LOO via `model.loo()` (`time_series.py:1303`) on
   **training** data; `elpd` differences with `dse` and a parsimony tie-break
   (`fixed_case_studies/hyperparams/scoring.py:56`, `bayesian.py:175`). Use
   LFO-CV + CRPS for autocorrelated series
   (`fixed_case_studies/hyperparams/ts_cv.py:22`,
   `hyperparams/scoring.py:201`).

## Sampler choices

| Method | Use | Check |
|---|---|---|
| `nuts` + `nuts_sampler="nutpie"` | confirmation, joint-covariance claims | full diagnostics; `target_accept ≥ 0.9` |
| `advi` | development screening only | mean-field / diagonal; no covariance claim |
| `mapx` | cheap point fits | `fit_info["map_diagnostics"]`: status, objective, gradient norm, parameter spread; multi-start to detect local optima |

`src/vangja/time_series.py:404-513` records `fit_info` (method, seed,
`target_accept`, package versions, MAP diagnostics). Persist it in every
manifest.

## Transfer learning

- No `tune()` method: construct the target model with the transfer
  configuration and call `fit(..., idata=base.trace)` after fitting the source
  (`fixed_case_studies/stocks/runner.py:250`).
- `tune_method="parametric"` matches **posterior mean/sd**, not the MAP/mode
  (`src/vangja/components/linear_trend.py:70`; WS-S-001). Do not describe it as
  MAP unless the code changes.
- `tune_method="prior_from_idata"` consumes MCMC/VI draws and builds an MVN
  prior. Mean-field ADVI draws ⇒ diagonal prior ⇒ **no covariance
  preservation** (WS-S-008, FINDINGS F-8). Covariance claims require a NUTS
  source posterior.
- Trend regularization is `-loss·Σ(slope_g − slope_transferred)²`
  (`left` side adds `Σδ`); the seasonal potential is the one-sided cap
  `-loss·Σ min(0, (1/n_t)(‖f_old‖²−‖f_new‖²))`, gated on `period > 2·n_obs`;
  changepoints dispatch reads `delta_tune_method` (see `tests/test_regularization.py`).

## Gotchas

- ArviZ 1.x is DataTree-based; there is no `az.waic`, and `az.compare` has no
  `ic` argument. Build fixtures as
  `az.InferenceData(xr.Dataset(), children={group: xr.DataTree(ds)})`.
- pytensor 3: `pt.tensor.as_tensor` (no `pt.constant`).
- PyMC names: `{type}_{i} - {param}` only — no commas, brackets or dots; never
  encode priors in names.
- Non-complete pooling always has a per-group dim even when `n_groups == 1`;
  initvals are arrays and predictions index `[group_code]`.
- `FourierSeasonality`/`LinearTrend` reject fractional orders
  (`series_order=3.0` raises); pass ints.
- Dependencies are pinned deliberately (`pymc[nutpie]~=6.2`,
  `numpy>=2.0,<2.5`, `numba>=0.66`); do not relax.

## Scripts

```bash
python .pi/skills/bayesian-workflow/scripts/diagnose_idata.py \
  --json fit_diagnostics.json --out results/diag --label stocks-main
python .pi/skills/bayesian-workflow/scripts/check_convergence.py \
  --idata trace.nc --label stocks-main        # exit 1 if any threshold fails
```

Reading an `.nc` file needs the project environment: prefix the commands with
`uv run` (the diagnostics JSON path is stdlib-only).

`diagnose_idata.py` reports per-parameter failures and writes
`diagnostics_<label>.json` plus a Markdown report. `check_convergence.py` is the
gate; pass
`--waiver-reason TEXT` only with a recorded human waiver — the waiver text is
written into the output.

## Do not

- Report MAP residual intervals as posterior predictive intervals (WS-S-006,
  WS-S-010).
- Select a configuration on LOO computed over test rows.
- Claim convergence from a table that only shows means (the historical notebook
  claimed "all converged" while R̂/ESS failed — WS-S-007).
