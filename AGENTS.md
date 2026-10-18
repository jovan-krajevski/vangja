# Vangja — agent instructions

Bayesian time-series forecasting package (PyMC) that generalises Facebook Prophet:
piecewise-linear trend, Fourier seasonality, hierarchical pooling, and transfer
learning (posterior of a long *context* series → priors for short *target* series).
Current work: the `update_pymc` branch. `Vangja-project/` is a read-only audit
archive — never modify it. `research_agent/` is the multi-agent research
environment (see its README).

## Commands

```bash
uv sync --extra test --extra datasets   # env (uv.lock is authoritative)
uv run pytest                           # full suite
uv run pytest tests/test_regularization.py -v
uv run black . && uv run ruff check .   # formatting
cd docs && make html                    # docs
```

Entry points for the revised study (no grid, frozen configs, checkpointed):
`fixed_case_studies/stocks/0{1..9}_*.py`, `fixed_case_studies/smart_home/0{1..5}_*.py`.
HPC: `case_studies/singularity*.def`, `case_studies/submit_ablation*.slurm`.

## Map

- `src/vangja/time_series.py` — base class: preprocessing, scaling, `fit()`, `predict()`, `predict_uncertainty()`, operator composition (`+`, `*`, `**`).
- `src/vangja/components/` — `LinearTrend`, `FlatTrend`, `FourierSeasonality`, `NormalConstant`, `UniformConstant`, `BetaConstant`.
- `src/vangja/datasets/` — all data access (`load_*`, `generate_*`, S&P helpers). Use these, never inline data loading.
- `src/vangja/utils.py` — metrics (`relative_mae`, `persistence_forecast`), gaps, model comparison.
- `fixed_case_studies/` — audit-compliant experiment suite (protocol, frozen configs, runners, reports).
- `paper/paper.tex` + `paper/case_studies/tables/` — the manuscript. Claims may only cite artifacts produced on `update_pymc`.
- `docs/` notebooks are part of the deliverable; keep them runnable.

## Non-negotiables

- **Reproducibility:** pass `random_seed` and (for NUTS) `target_accept` to `fit()`. Never pass an explicit `step` together with `nuts_sampler`. Record `fit_info`; every reported number must be recomputable from committed artifacts + data hashes.
- **Study protocol:** no configuration selection on the test horizon; one frozen global config per study; development vs confirmation data kept separate; `CONFIRMATION_ORIGINS.csv` (and its `.sha256`) is frozen before any confirmation results are viewed; context series (`^GSPC`, temperature) never enter target aggregates.
- **Metric:** Relative MAE vs persistence per target×origin unit; aggregate = median; exclude units with persistence MAE `< 1e-3` on the scaled target.
- **No silent drops:** failed fits, excluded units, unavailable data all go to failure/availability files and are reported.
- **Tests:** every code change adds/updates tests. Never weaken or delete an existing test to make a change pass; if a test is wrong, say so explicitly and fix it deliberately.
- **Data:** stocks run on trading days only; `load_stock_data(..., interpolate=False)` is the default; if interpolation is used it is split-first and within-split only.
- **Never commit secrets, data dumps, or result directories** (see `.gitignore`); commit manifests/hashes instead.

## Vangja/PyMC gotchas

- Dependencies are pinned deliberately: `pymc[nutpie]~=6.2`, `numpy>=2.0,<2.5`, `numba>=0.66`. Do not relax without checking `pytensor.link.numba.dispatch` imports.
- ArviZ 1.x is DataTree-based: build fixtures with `az.InferenceData(xr.Dataset(), children={group: xr.DataTree(ds)})`; no `az.waic`, `az.compare` has no `ic`.
- pytensor 3: use `pt.tensor.as_tensor` (no `pt.constant`/`pt.as_tensor_variable`).
- PyMC variable names must not contain commas, brackets or dots: `{type}_{i} - {param}` only. Never encode priors/config in names.
- Non-complete pooling always has a per-group dim, even with `n_groups == 1`; initvals are arrays and predictions index `[group_code]`.
- Transfer learning: no `tune()` method — build a new model and `fit(..., idata=base.trace)`. `parametric` = posterior-moment matching (mean/sd), not the MAP mode. `prior_from_idata` requires MCMC/VI source draws; mean-field ADVI gives a diagonal prior and is not evidence of covariance preservation.
- Transfer regularisation (see `tests/test_regularization.py`): trend potential is `-loss·Σ(slope_g − slope_transferred)²` in all pooling modes (end-of-history slope; `left` side adds `Σδ`); seasonal potential is the one-sided per-grid-point cap `-loss·Σ min(0, (1/n_t)(‖f_old‖²−‖f_new‖²))`, identical across pooling modes, gated on `period > 2·n_obs`; changepoint dispatch reads `delta_tune_method`, never `tune_method`.
- `FourierSeasonality`/`LinearTrend` coerce integral floats to int and raise on fractional orders (`series_order=3.0` is otherwise a `TypeError` deep in numpy).

## Working style

- Prefer small, verifiable steps; run the narrowest relevant test before the full suite.
- Numbers in prose must name the artifact that produced them (path + config + seed).
- When you learn something durable about this codebase, add it here in one line — keep this file terse; move long procedures into a skill or a doc under `docs/`.
