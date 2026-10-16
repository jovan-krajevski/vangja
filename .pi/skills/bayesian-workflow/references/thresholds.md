# Bayesian diagnostics — thresholds and sources

## Convergence (posterior sampling)

| Diagnostic | Threshold | Meaning | Source |
|---|---|---|---|
| R̂ (max over parameters) | `< 1.01` | chains mixed | `docs/source/notebooks/10_bayesian_workflow.ipynb` §3 |
| ESS bulk (min) | `> 400` | enough effective samples, bulk | same |
| ESS tail (min) | `> 400` | enough effective samples, tails | same |
| BFMI (min over chains) | `> 0.3` | no energy-transition problems | same |
| Divergences | `= 0` | no pathological geometry | same |
| Max-tree-depth hits | `= 0` | no saturating trajectories | WS-S-005; the failing notebook baseline hit on all 4 chains |
| `target_accept` (NUTS) | `>= 0.9` project policy; raise to 0.95 on divergences | `fit()` accepts and forwards it | `src/vangja/time_series.py:306,531-544` |
| Chains × draws | recorded per fit; no minimum in the protocol, but ESS thresholds are per-parameter | provenance | `fit_info` |

The historical workflow notebook reported "All models converged" while its own
checks printed `R-hat < 1.01 for all parameters: False` and
`ESS > 400 for all parameters: False` (WS-S-007). Never re-derive a verdict from
a summary table; recompute from the diagnostics.

## Prior predictive (before fitting)

`prior_predictive_coverage` (`src/vangja/utils.py:549`) measures the fraction of
prior-predictive draws inside `[-2, 2]` on the scaled target:

| Coverage | Verdict |
|---|---|
| `< 5 %` | priors too loose (mass in impossible regions) |
| `30–60 %` | target range for a flexible Prophet-like model |
| `> 95 %` | priors too tight |

Source: `fixed_case_studies/HYPERPARAMETER_SELECTION.md` §2.1 (Step 1).

## Model comparison (training data only)

- PSIS-LOO `elpd_loo` with `elpd_diff` and `dse`
  (`fixed_case_studies/hyperparams/scoring.py:40-72`).
- Selection rule: keep the frozen value if the best beats it by `< 1 dse`;
  within `1 dse` prefer the smaller/more regularising value; else take the best
  (`HYPERPARAMETER_SELECTION.md` §2.2).
- LFO-CV with CRPS for autocorrelated series
  (`fixed_case_studies/hyperparams/ts_cv.py:22`,
  `hyperparams/scoring.py:201`).
- Stacking / pseudo-BMA weights when candidates are indistinguishable
  (`hyperparams/scoring.py:117,148`).
- The test horizon is never a selection input.

## MAP fits

- `method="mapx"` returns a point estimate; there is no posterior.
- Check `fit_info["map_diagnostics"]` (`src/vangja/time_series.py:511`):
  optimizer status (converged), objective value, gradient norm (near zero),
  parameter spread. Run multi-start to detect local optima.
- MAP uncertainty from `predict_uncertainty` is a Prophet-style residual
  heuristic (`src/vangja/time_series.py:642`; WS-S-010). Label it as such.

## Transfer-specific

- `parametric` = posterior mean/sd moment matching; **not** the MAP mode
  (WS-S-001).
- `prior_from_idata` over mean-field ADVI draws yields a diagonal MVN; it
  preserves no correlations (WS-S-008). Covariance claims require NUTS or
  another covariance-capable source fit, plus saved covariance matrices
  (`PROTOCOL.md` §8).
- Compose the transfer with `fit(..., idata=source.trace)`; there is no
  `tune()` method.

## Diagnostics file format for the scripts

`diagnose_idata.py` / `check_convergence.py` accept either an ArviZ
InferenceData (`--idata trace.nc`) or this JSON (all keys optional except
`method`):

```json
{
  "label": "stocks-main-nuts",
  "method": "nuts",
  "target_accept": 0.9,
  "chains": 2,
  "draws": 500,
  "r_hat": {"lt_0 - slope": 1.003},
  "ess_bulk": {"lt_0 - slope": 620},
  "ess_tail": {"lt_0 - slope": 510},
  "divergences": 0,
  "max_tree_depth_hits": 0,
  "bfmi": [0.61, 0.58],
  "map_diagnostics": {"status": "converged", "gradient_norm": 1e-4}
}
```

Verdicts: `pass` (all checks), `fail` (any check failed), `warn` (only
non-blocking notes). `check_convergence.py` exits non-zero on `fail` and treats
`warn` as pass unless `--strict`.
