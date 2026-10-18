# Vangja architecture — the component contract

All line numbers refer to the `update_pymc` branch at the revision where this
skill was written; re-check with `grep -n` before citing them in a review.
Public surface: `src/vangja/__init__.py` re-exports
`LinearTrend`, `FlatTrend`, `FourierSeasonality`, `NormalConstant`,
`UniformConstant`, `BetaConstant`, `TimeSeriesModel`, `CombinedTimeSeries`.

## 1. The component lifecycle

`TimeSeriesModel` (`src/vangja/time_series.py:46`) is both base class and
composite. Subclasses implement:

| Method | Purpose | Contract |
|---|---|---|
| `definition(model, data, priors, idata, **kw)` | build RVs + `pm.Deterministic` | must add exactly one prediction column per group; must not write to `model.named_vars` outside PyMC |
| `needs_priors()` | does this component consume a transferred posterior? | default `False`; components with `tune_method` override (`linear_trend.py:935`, `fourier_seasonality.py:631`) |
| `is_individual()` | per-group parameters? | complete pooling → `False`; partial/individual → `True` (`linear_trend.py:938`) |
| `_get_prior_var_names()` | posterior names consumed | `linear_trend.py:751`, `fourier_seasonality.py:626` |
| `_assign_model_idx(model_idxs)` | assign the `_i` in names | `linear_trend.py:746` |
| `_get_initval(initvals, model)` | start values for MAP/ADVI/NUTS | must return `{name: array}` for every per-group RV (`linear_trend.py:761`) |
| `_predict_map(future, map_approx)` | point-forecast path | fills `self._predict_columns[name]` (`linear_trend.py:800`) |
| `_predict_mcmc(future, trace)` | posterior-mean path | same (`linear_trend.py:851`) |
| `_plot(...)` | component plot | optional, behind `plot()` |

`definition` is called once per component inside the composed model built in
`fit()` (`time_series.py:435`). Components are leaves of the operator tree;
`_get_leaf_components()` (`time_series.py:124`) flattens it.

## 2. Operators (composition semantics)

- `a + b` → `AdditiveTimeSeries` (`time_series.py:1471`): `y = a + b`.
- `a ** b` → `MultiplicativeTimeSeries` (`time_series.py:1520`):
  Prophet-style `y = a · (1 + b)`; this is the form the paper uses for
  `g · (1 + s)`.
- `a * b` → `SimpleMultiplicativeTimeSeries` (`time_series.py:1581`):
  `y = a · b` for cases such as `UniformConstant * FourierSeasonality`.
- `__add__/__radd__/__mul__/__rmul__/__pow__/__rpow__` are defined at
  `time_series.py:1355-1371`; `__pow__` maps to `**`, `__rpow__` to `**`.
- The composite `_predict` (`AdditiveTimeSeries:1505`,
  `MultiplicativeTimeSeries:1562`, `SimpleMultiplicativeTimeSeries:1618`)
  folds the leaf `_predict_columns` with the same operator as the definition.
  A new operator must update both `definition` and `_predict` or training and
  forecasting disagree.

## 3. Data flow in `fit()` (`time_series.py:285`)

1. `_process_data(data, scaler, scale_mode, t_scale_params)`
   (`time_series.py:174`) — validates `ds`/`y`/`series`, sorts, computes `t`,
   scales `y`. `scale_mode="complete"` uses one `(y_min, y_max)`; individual
   scaling stores per-series params (`per_series_scale_params` in
   `fixed_case_studies/common.py:138`).
2. Assign `model_idx` per leaf and build the `pm.Model` (`time_series.py:435`).
3. `self.initvals = self._get_model_initvals()` then `get_initval(...)`
   (`time_series.py:242`, `:256`); initvals are data-derived heuristics
   (first/last slope, mean level), not priors.
4. Dispatch on `method` (`types.py` `Method`): `mapx` → `pmx.find_MAP`
   (JAX backend), `map` → `pm.find_MAP`; `advi`/`fullrank_advi`/`svgd`/`asvgd`
   → variational; `nuts` → `pm.sample(nuts_sampler=...)`. `fit_info`
   (`time_series.py:404`) records method, seed, `target_accept`,
   versions, timestamps, and for `mapx` the optimizer diagnostics
   (`fit_info["map_diagnostics"]`, `time_series.py:511`).
5. Store `self.trace` (MCMC/VI), `self.map_approx` (MAP) and
   `self.fit_info`; return `fit_info`.

`predict(horizon, freq)` (`time_series.py:607`) builds the future grid with
`_make_future_df` (`time_series.py:570`; training span + horizon), calls
`_predict` (`time_series.py:859`) and returns one `yhat_{group}` column per
group plus the folded composition. `predict_uncertainty`
(`time_series.py:642`) adds `yhat_lower_{group}`/`yhat_upper_{group}`: posterior
draws for MCMC/VI, and for MAP a residual-calibrated heuristic
(`_get_map_sigma` `time_series.py:820`, `_compute_residual_stds`
`time_series.py:834`) widened by `sqrt(1 + h/n)`. The MAP interval path is a
heuristic, not a posterior predictive — say so whenever it is used for claims.

## 4. Pooling (`pool_type`)

`get_group_definition(data, pool_type)` (`src/vangja/utils.py:121`):

| `pool_type` | group codes | `n_groups` |
|---|---|---|
| `complete` | all rows 0, one shared parameter | 1 (scalar RVs) |
| `partial` | one code per series, hierarchical coupling | `data["series"].nunique()` |
| `individual` | one code per series, free parameters | same |

`partial` uses a non-centred parameterisation with a shared mean, a
`HalfCauchy(σ_scale/shrinkage_strength)` scale and a standardised offset
(e.g. `linear_trend.py:451-480` for the slope). `shrinkage_strength` is the
per-component `λ`: **higher = more shrinkage toward the shared mean**
(`linear_trend.py:164`, `fourier_seasonality.py:143`).

Shapes: `complete` builds scalar RVs; `partial`/`individual` build
`(n_groups,)` / `(n_groups, n_changepoints)` / `(n_groups, 2·order)` RVs. This
holds when `n_groups == 1`. Two documented defects came from forgetting it:
scalar initvals for a vector RV break ADVI's start handling
(`linear_trend.py:781-786` comment), and 2-D prediction columns make the
folded composition broadcast wrongly.

## 5. Transfer (`tune_method` / `delta_tune_method`)

Transfer requires a source `idata`; `fit(..., idata=base.trace)` passes it to
every component with `needs_priors() == True`. There is **no `tune()` method**.

| `tune_method` | What is transferred | Notes |
|---|---|---|
| `None` | nothing | base model |
| `prior_from_idata` | joint posterior → **MVN prior** (mean + covariance) via `pymc-extras` `_get_*_params_from_idata` (`linear_trend.py:202`, `:246`; `fourier_seasonality.py:198`) | preserves correlations; needs MCMC/VI draws. Mean-field ADVI yields a diagonal prior and is **not** covariance evidence. |
| `parametric` | posterior **mean/sd** → same-family prior (`LinearTrend` Laplace `τ=sd/√2`; `FlatTrend` `mu=mean`, `flat_trend.py:141`; `BetaConstant` moment-matched Beta) | posterior-moment matching, **not** the MAP mode |

Changepoints dispatch **separately**: `delta_tune_method` governs the
`delta`/changepoint transfer, `tune_method` the slope
(`linear_trend.py:229`, `:350`, `:662`; verified by
`tests/test_regularization.py::TestTransferDispatch::test_tune_method_differs_from_delta_tune_method`).

### Regularisation potentials (the contract)

`loss_factor_for_tune` adds a `pm.Potential` for both trend and seasonality.
The exact forms, tested in `tests/test_regularization.py`:

- **Trend** (`linear_trend.py:397`, `:572`, `:657`; identical in every
  pooling mode): `-loss · Σ_g (slope_g − slope_transferred_g)²`, where
  `slope_transferred` is the **end-of-history** slope of the source
  (`slope + Σδ` when `delta_side="left"`, the raw `slope` when
  `delta_side="right"`; `_get_slope_params_from_idata` `linear_trend.py:202`,
  `_get_skipped_deltas` `:272`).
- **Seasonality** (`fourier_seasonality.py:288`, `:390`, `:482`; identical
  in every pooling mode): the **one-sided per-grid-point cap**
  `-loss · Σ_g min(0, (1/n_t)·(‖f_old‖² − ‖f_new,g‖²))` — summed over group
  `g` and the `period` grid points. It penalises only amplitude *growth*
  beyond the source, per grid point, so the normalisation is independent of
  `n_obs`; gated on `period > 2·n_obs` (below that the source has not seen a
  full cycle). `loss_factor_for_tune=0` disables it.
- Both potentials are added via `pm.Potential` and therefore are **not**
  part of the prior or the log-likelihood group.

## 6. `fit_info` (what a reported run must carry)

`time_series.py:404` starts with `method`, `random_seed`, `target_accept`,
`include_source_in_target`; then `{vangja,pymc,arviz,numpy,pandas,pytensor}_version`
(each `None` if the package is missing), `fit_started_at`, and for `mapx` the
optimizer diagnostics. `fixed_case_studies` artifacts additionally record
commit SHA, data hashes, `seed_for` values, availability and the aggregate
(`fixed_case_studies/common.py:507 record_artifact`).

## 7. Naming rules

Format: `{type}_{i} - {param}` where `type` is the component prefix, `i` its
`model_idx`, `param` the parameter. Prefix table: `lt` LinearTrend, `fs`
FourierSeasonality, `ft` FlatTrend, `nc` NormalConstant, `uc` UniformConstant,
`bc` BetaConstant. Derived names add suffixes (`lt_0 - slope_shared`,
`lt_0 - delta_sigma`, `fs_1 - beta`); prediction columns are `{type}_{i}_{group}`
and the public future frame uses `yhat_{group}`.

**Forbidden in a PyMC variable name:** `,`, `[`, `]`, `.`. Do not encode
priors or configuration (`fs_0 - beta(p=365.25,n=5)` is a defect: PyMC derives
deterministic names by splitting on those characters, and pytensor/numba
mangling follows). Never rename an existing variable to *hide* a change: an
audit diff will see it. `scripts/lint_pymc_names.py` enforces the rule on a
diff or a tree; `tests/test_time_series.py` asserts the composed model's
names.

## 8. Datasets, metrics, and the fixed case studies

- Loaders live in `src/vangja/datasets/`; `load_stock_data` defaults to
  `interpolate=False` (trading days only) — see `references/datasets.md`.
- Metrics: `relative_mae` (`src/vangja/utils.py:325`), `persistence_forecast`
  (`:383`), `metrics` (`:234`), `prior_predictive_coverage` (`:549`),
  `compare_models` (`:491`). The study's epsilon rule is
  `REL_MAE_EPSILON = 1e-3` on the scaled target
  (`fixed_case_studies/common.py:44`).
- Runners: `fixed_case_studies/stocks/runner.py:285 run_cell` (checkpoint at
  `:307`, `source_seed_for` at `:397`, `seed_for` at `:405`, pkl source cache
  at `:442-453`) and `fixed_case_studies/smart_home/runner.py:249 seed_for`.
  Entry points actually present at this revision are listed in
  `fixed_case_studies/README.md` §5; the stocks study currently ships only
  `01_fetch_data.py`, `config.py`, `runner.py`.
- Never bypass the runner to compute a reported number: the artifact bundle
  (`manifest_*.json` + `units_*.csv` + `forecasts_*.csv`) is the provenance
  contract (`fixed_case_studies/README.md` §3.11).
