# Vangja failure modes — symptom → cause → fix

Each row names the guard that must stay green. If you add a mode, add the
guard with it. This file is deliberately specific to this repo; generic PyMC
advice belongs in `bayesian-workflow`.

## Naming and model graph

| Symptom | Cause | Fix |
|---|---|---|
| `arviz` error "more dims (N) given than existing ones"; nutpie misparses dims | a PyMC variable name contains `,`, `.`, `[` or `]` — e.g. `fs_0 - beta(p=365.25,n=5)` | rename to `fs_0 - beta`; never encode period/order/priors in names. Guard: `tests/test_components.py:430 test_leaf_component_names_are_safe` (UNSAFE regex `[,.[\]]`), plus `scripts/lint_pymc_names.py` |
| Two Fourier components overwrite each other's RVs | `model_idx` not assigned | `_assign_model_idx` (`linear_trend.py:746`, `fourier_seasonality.py:621`) is called from `fit()`; guard `tests/test_components.py:453` |
| `KeyError: "prior_lt_0 - slope"` during transfer | source posterior lacks the variable (`tune_method` on the target but source fit with a different component/order) or name drifted | `needs_priors()`/`_get_prior_var_names()` must agree with the source model; name drift is a defect — never rename to "fix" it |

## Shapes and pooling

| Symptom | Cause | Fix |
|---|---|---|
| ADVI dies at initialisation with a shape/`value` error when `n_groups == 1` | scalar initval for a `(n_groups,)` RV (non-complete pooling always has the group dim) | return an array initval for every per-group RV (`linear_trend.py:761`, comment at `:781-786`); guard `check_pooling_shapes.py` |
| Predictions are 2-D / broadcast wrongly, or `yhat_0` columns vanish | `_predict_columns` written as a 2-D slice instead of a 1-D `(horizon,)` array | index `[group_code]`; guard `tests/test_time_series.py::test_make_future_df_*` and the component `_predict` tests |
| `predict()` raises alignment errors for a single series | consumer assumed a 2-D `predictions[:, group]` matrix | use the `yhat_{group}` column contract (`src/vangja/utils.py:178 filter_predictions_by_series`) |
| Hierarchical coupling is negligible/huge | `shrinkage_strength` misunderstood as "strength of individual variation" | it is `λ` in `HalfCauchy(σ_scale/λ)`: **higher λ = more shrinkage** (`linear_trend.py:455-470`) |

## Transfer learning

| Symptom | Cause | Fix |
|---|---|---|
| Posteriors look identical to the source | `prior_from_idata` under mean-field ADVI gives a diagonal prior; not covariance evidence | for covariance claims fit the source with NUTS/MCMC (`fixed_case_studies/README.md` §3.9) |
| Changepoint transfer ignored, no error | dispatch reads `tune_method` instead of `delta_tune_method` (historical F-6) | read `delta_tune_method` in every pooling mode (`linear_trend.py:229`, `:350`, `:662`); guard `tests/test_regularization.py::TestTransferDispatch::test_tune_method_differs_from_delta_tune_method` |
| Trend regularisation pushes the wrong way / explodes | potential sign or form wrong; or "slope" meant the initial slope rather than the end-of-history slope | potential is `-loss·Σ(slope_g − slope_transferred)²` in all modes; transferred slope = `slope + Σδ` for `delta_side="left"`, raw `slope` for `"right"`; guard `TestTrendPotential*` and `TestSlopeTransfer*` |
| Seasonality regularisation changes with sample size | missing `1/n_t` per-grid-point normalisation, or the `period > 2·n_obs` gate | one-sided cap `-loss·Σ_g min(0, (1/n_t)(‖f_old‖²−‖f_new,g‖²))`; guard `TestSeasonalPotential*` |
| `parametric` behaves like a mode/MAP | docstring or paper says MAP | it is posterior **mean/sd** matching (`flat_trend.py:141`, `fourier_seasonality.py:196`); fix the prose, not the code, unless the decision is deliberate |
| Transfer "works" but target overfits a short series | `loss_factor_for_tune` left at 0 while the transferred prior is weak | raise the loss factor deliberately and record it in the frozen config; never tune it on the test horizon (protocol rule) |

## Sampler / reproducibility

| Symptom | Cause | Fix |
|---|---|---|
| Same seed gives different draws | seed not threaded to every stochastic step (predictive sampling, MAP jitter, uncertainty sub-sampling) | pass `random_seed` to `fit()`; `fit()` forwards it to MAP/ADVI/NUTS and predictive sampling (`tests/test_time_series.py:546-634`) |
| `TypeError`/dispatch error when using NUTS via nutpie | explicit `step` passed together with `nuts_sampler` | never pass both; `tests/test_time_series.py::test_nuts_uses_backend_sampler_without_explicit_step` |
| `import numba` fails or `np.trapz`/`np.row_stack` AttributeError | dependency pins relaxed (`pymc[nutpie]~=6.2`, `numpy>=2.0,<2.5`, `numba>=0.66`) | restore the pins; check `pytensor.link.numba.dispatch` imports before relaxing (`python-packaging-and-environments`) |
| `pt.constant` / `pt.as_tensor_variable` AttributeError on pytensor 3 | legacy API | `pt.tensor.as_tensor` |
| ArviZ 1.x fixture construction fails (`to_dict` shape errors) | old flat-group assumptions | build with `az.InferenceData(xr.Dataset(), children={group: xr.DataTree(ds)})`; no `az.waic`, `az.compare` has no `ic` |
| `series_order=3.0` raises deep in numpy | fractional order accepted then used as a shape | components coerce integral floats to int and raise a clear error on fractional orders (`fourier_seasonality.py:152-158`) |

## Data and evaluation

| Symptom | Cause | Fix |
|---|---|---|
| A stock series is silently absent from results | availability drop not recorded | record every drop in the availability/failure files (`fixed_case_studies/common.py:558`); "no silent drops" is a non-negotiable |
| Forecasts look too smooth for stocks | calendar-day interpolation across the split (historical F-9) | `interpolate=False` by default; if used, split-first and within-split only (`loaders.py:219`) |
| Relative MAE not comparable between arms | one arm scored on prices, another on the scaled target | compute on the fitted scale; ε rule `1e-3` on that scale (`common.py:44`); `forecasting-evaluation` |
| Reported number cannot be traced | no artifact bundle / manifest | every cell writes `manifest_*.json` + `units_*.csv` + `forecasts_*.csv` through `run_cell` (`stocks/runner.py:285`); checkpoint hit at `:307` |
| Repeated seed changes results across finalist runs | wrong seed source (clock/uuid) | seeds come from `seed_for`/`source_seed_for` (`stocks/runner.py:397`, `:405`); `FINALIST_SEEDS = (42, 20240816, 7)` (`common.py:49`) |
