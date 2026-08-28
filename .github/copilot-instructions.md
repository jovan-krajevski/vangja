# Vangja AI Coding Instructions

## Overview

Vangja is a Bayesian time series forecasting package built on PyMC. It extends Facebook Prophet with hierarchical modeling and transfer learning for short time series forecasting.

## Architecture

### Core Components (`src/vangja/`)

- **`time_series.py`** — Base `TimeSeriesModel` class that all components inherit from. Handles data preprocessing, scaling, model fitting (PyMC), and prediction. Models compose via operator overloading (`+`, `*`, `**`).
- **`components/`** — Model building blocks: `LinearTrend`, `FlatTrend`, `FourierSeasonality`, `NormalConstant`, `BetaConstant`, `UniformConstant`. Each implements `definition()`, `_predict_map()`, `_predict_mcmc()`, and `_plot()`.
- **`types.py`** — Type definitions (`PoolType`, `Method`, `Scaler`, `TuneMethod`) with docstrings explaining each literal value.
- **`utils.py`** — Helper functions for group assignment, metrics, and data manipulation (e.g., `remove_random_gaps`).

### Model Composition Pattern

```python
# Additive: y = left + right
model = LinearTrend() + FourierSeasonality(365.25, 10)

# Multiplicative (Prophet-style): y = left * (1 + right)
model = LinearTrend() ** FourierSeasonality(7, 3)

# Simple multiplicative: y = left * right
model = LinearTrend() * FourierSeasonality(7, 3)

# FlatTrend (constant-level baseline, no slope or changepoints):
model = FlatTrend() + FourierSeasonality(365.25, 10)
```

### FlatTrend Component

`FlatTrend` is the simplest possible trend component: a single intercept parameter with no slope and no changepoints. The model is:

```
trend(t) = intercept
```

Useful when the time series has no discernible upward or downward trend, or when the series is too short to reliably estimate a slope. Equivalent to `LinearTrend(n_changepoints=0)` with slope fixed to 0, but more explicit and with fewer parameters.

Key parameters:

- `intercept_mean` (float, default=0): Mean of the Normal prior for the intercept.
- `intercept_sd` (float, default=5): Std dev of the Normal prior for the intercept.

Supports `pool_type="complete"`, `pool_type="partial"`, and `pool_type="individual"`. Transfer learning via `tune_method="parametric"` and `"prior_from_idata"` is supported.

### Multi-Series & Pooling

Data must have columns: `ds` (datetime), `y` (float), optionally `series` (str for multi-series).

**PoolType controls parameter sharing:**

- `"complete"` — All series share same parameters
- `"partial"` — Hierarchical pooling with shared hyperpriors (inspired by [timeseers](https://github.com/MBrouns/timeseers))
- `"individual"` — Each series has independent parameters

**Hierarchical Modeling (Partial Pooling):**

Partial pooling allows series to "borrow strength" from each other while respecting individual differences. The model structure is:

```
slope_shared ~ Normal(0, σ_slope)
slope[i] ~ Normal(slope_shared, σ_individual)
```

Key parameters:

- `shrinkage_strength` — Controls how much series are pulled toward the shared mean. Higher values = more pooling. Start with 10 as default.
- Use partial pooling when series belong to natural groups (products, stores, regions) or have limited data.

See [05_hierarchical_modeling.ipynb](docs/05_hierarchical_modeling.ipynb) for a complete example.

**Simultaneous vs Sequential Fitting:**

Vangja can fit multiple series simultaneously using vectorized computations, which is significantly faster than fitting each series separately. Use `pool_type="individual"` with `scale_mode="individual"` to fit multiple series at once while keeping parameters independent.

- **Same time range**: Sequential and simultaneous fitting produce equivalent results. See [03_multi_series_fitting.ipynb](docs/03_multi_series_fitting.ipynb).
- **Different time ranges**: Results will differ due to changepoint distribution across the combined time range. See [04_multi_series_caveats.ipynb](docs/04_multi_series_caveats.ipynb).

**Changepoint Distribution Caveat:**

When fitting series with different date ranges simultaneously, changepoints (`n_changepoints`) are distributed across the **entire combined time range**. For example, if fitting series from 1949-1960 and 2007-2016 together:

- The combined range spans ~67 years
- Each series only occupies a small portion of normalized time `t = [0, 1]`
- Each series gets far fewer changepoints than if fit separately

For series with non-overlapping date ranges, consider fitting them separately or increasing `n_changepoints`.

**Utilities for multi-series:**

- `remove_random_gaps(df, n_gaps=4, gap_fraction=0.2)` — Remove random contiguous intervals from a time series to simulate missing data. Call this **per-series in notebooks**, not inside data generation functions. The default removes 4 gaps of 20% each.
- `filter_predictions_by_series(future, series_data, yhat_col, horizon)` — Filter predictions to a specific series' date range. **Always use this** when series have different date ranges.
- `metrics(y_true, future, pool_type)` — Calculates metrics by merging on `ds` column (handles different data frequencies)
- `relative_mae(y_true, y_pred, y_persistence, epsilon=None)` — the primary study metric: MAE vs persistence, with a denominator threshold `epsilon` applied on the scaled target (returns `nan` for excluded units)
- `persistence_forecast(train_df, test_df)` — persistence baseline aligned to test dates (long format `ds`, `series`, `yhat`)

### Transfer Learning

Set `tune_method="parametric"` or `"prior_from_idata"` on components, then pass `idata` (ArviZ InferenceData) to `fit()` to transfer knowledge from pre-trained models. There is **no `tune()` method** — transfer learning is done by creating a new model and passing `idata=base_model.trace` to `fit()`:

```python
# Step 1: Fit base model on long series (use MCMC for meaningful posteriors)
base_model = LinearTrend(tune_method="parametric") + FourierSeasonality(365.25, 10, tune_method="parametric")
base_model.fit(long_data, method="nuts", samples=1000, chains=4)

# Step 2: Create target model and fit with transferred priors
target_model = LinearTrend(tune_method="parametric") + FourierSeasonality(365.25, 10, tune_method="parametric")
target_model.fit(short_data, idata=base_model.trace)
```

**Transfer learning method differences:**

- `"parametric"`: Extracts posterior mean/std from idata and uses them as fixed Normal prior parameters. Each group gets its own free parameter. Simpler, more robust, works well in most cases.
- `"prior_from_idata"`: Uses `pymc_extras.utils.prior.prior_from_idata` to create an MvNormal approximation of the joint posterior, preserving inter-parameter correlations. For **complete pooling**, parameters become Deterministic functions of the MvNormal. For **partial pooling**, the MvNormal value becomes the shared hyperprior. For **individual pooling**, each group gets its own free parameter (Normal/Laplace) centered at the MvNormal prior with posterior std as spread.

**Implementation details for `prior_from_idata`:**

- Each component declares needed posterior variable names via `_get_prior_var_names()`. Only these variables (not sigma, not intercepts) are passed to `prior_from_idata`.
- The `_assign_model_idx()` pre-pass assigns model indices before the PyMC model is built, enabling variable name collection.
- `_get_initval()` skips Deterministic variables (checks `model.free_RVs`) to avoid errors during MAP/MCMC.
- The base model **must** be fitted with MCMC or VI (not MAP) to produce a meaningful posterior for transfer.

See [07_transfer_learning.ipynb](docs/07_transfer_learning.ipynb) for a complete example using NYC temperature data to forecast short bike sales time series.

### Datasets Module (`src/vangja/datasets/`)

The `datasets` module provides functions for loading real-world datasets and generating synthetic data. **All notebooks should use these functions instead of inline data generation/loading.**

**Available functions:**

- `load_air_passengers()` — Classic monthly airline passengers (1949-1960)
- `load_peyton_manning()` — Daily Wikipedia page views (2007-2016)
- `load_citi_bike_sales()` — Daily bike rides from NYC Citi Bike station 360 (2013-2014). Requires `pyreadr` (install with `pip install vangja[datasets]`)
- `load_nyc_temperature()` — Daily max temperature for NYC (2012-2017)
- `load_stock_data(tickers, split_date, window_size, horizon_size, cache_path, interpolate)` — Download historical stock OHLCV data, compute typical price `(O+H+L+C)/4`, and split into train/test DataFrames. Requires `yfinance` (install with `pip install vangja[datasets]`)
- `load_kaggle_temperature(city, start_date, end_date, freq)` — Hourly temperature data for 36 cities (2012-2017) from Kaggle. Returns Celsius. Supports temporal aggregation via `freq` (e.g. `"D"`, `"W"`, `"h"`). The `city` parameter is typed as `KaggleTemperatureCity` (a `Literal`). Requires `kagglehub` (install with `pip install vangja[datasets]`)
- `load_smart_home_readings(column, start_date, end_date, freq)` — Smart home appliance energy readings at 1-minute resolution (~2016). Column can be a single `SmartHomeColumn` (returns `ds`/`y`) or a list of them (returns `ds`/`y`/`series` in long format). Supports temporal aggregation via `freq`. Requires `kagglehub` (install with `pip install vangja[datasets]`)
- `get_sp500_tickers_for_range(start_date, end_date, cache_path)` — Return tickers consistently in S&P 500 during a date range by scraping Wikipedia's historical changes table. Accurate from ~1997 onwards.
- `get_sp500_tickers_at_date(target_date, cache_path)` — Return S&P 500 membership **at** a specific historical date (the confirmation-universe rule of the fixed case studies).
- `generate_multi_store_data()` — 5 synthetic store series with same time range
- `generate_hierarchical_products(include_all_year=True)` — 5-6 synthetic product series with opposite seasonality (summer/winter groups). Default time range is 2 years (2018–2019). **Does not introduce gaps** — use `remove_random_gaps()` per-series in notebooks to simulate missing data.

**Adding new datasets:** Create functions in `datasets/loaders.py` (real data), `datasets/synthetic.py` (generated data), or `datasets/stocks.py` (stock/financial data), then export in `datasets/__init__.py`.

**Stock data helpers (`datasets/stocks.py`):** Private functions for downloading OHLCV data via yfinance, parsing S&P 500 constituents/changes from Wikipedia, and reconstructing historical S&P 500 membership. All download functions support a `cache_path: Path | None` parameter for filesystem caching (creates parent directories automatically). Public: `get_sp500_tickers_for_range()` and `get_sp500_tickers_at_date()`.

**Stock loader rule (review F-9):** `load_stock_data(..., interpolate=False)` (default) returns trading days only. With `interpolate=True`, gaps are filled **within each train/test split separately** — never across the split boundary, and never outside each series' own observed range.

**Timeseers modeling pattern:** For series with opposite seasonality (like summer vs winter products), use `UniformConstant(-1, 1)` as a scaling factor:

```python
model = (
    LinearTrend()
    + UniformConstant(-1, 1) * FourierSeasonality(365.25, 5)
    + FourierSeasonality(7, 2)
)
```

This allows the model to learn +1 (peak in summer), -1 (peak in winter), or 0 (no seasonality) for each series. Note: the `UniformConstant` trick is most valuable with **high shrinkage** on the Fourier coefficients. With low/moderate shrinkage, partial pooling on the Fourier coefficients alone can handle opposite seasonality (the shared mean drifts to ~0 and individual deviations compensate). See [06_hierarchical_caveats.ipynb](notebooks/06_hierarchical_caveats.ipynb) for a detailed analysis.

**Shrinkage strength caveat:** `shrinkage_strength` is a hyperparameter that must be tuned per problem. Higher values pull series toward the shared mean more strongly. With opposite seasonality and high shrinkage, the shared Fourier mean is pulled to ~0, weakening seasonal patterns — this is where `UniformConstant` helps by separating seasonal shape from direction.

### Transfer-Learning Regularization Conventions (Potentials)

`loss_factor_for_tune` controls regularization Potentials added during transfer learning (`idata is not None and tune_method is not None`). The forms are fixed and tested in `tests/test_regularization.py`:

- **Trend (`LinearTrend`), all pooling modes**: negative quadratic pulling the slope toward the transferred end-of-history slope (manuscript `-phi (w - w_MAP)^2`):
  `-loss_factor * sum_g (slope_g - slope_transferred)^2` (scalar for complete pooling). Never use `+|slope - mu|` or a positive sign — a regularizer must penalize, not reward, deviation.
- **Seasonality (`FourierSeasonality`), all pooling modes**: one-sided amplitude cap, with a **per-grid-point** normalization that is identical across pooling modes:
  `-loss_factor * sum_g min(0, (1/n_t) * (||f_old||^2 - ||f_new,g||^2))` where `n_t = ceil(period)` grid points over one full period and `f_old` is the curve implied by the transferred coefficients. Gated per series on `period > 2 * n_obs` (for complete pooling use the minimum per-series observation count). The `1/n_t` scale makes the penalty independent of grid density and training size — do not reintroduce `2*period/n_total` or `2*n_group` scaling.
- **Changepoint transfer dispatch**: the changepoint branch must read `delta_tune_method` (never `tune_method`) in all pooling modes. `_get_prior_var_names()` includes `lt_... - delta` only when `delta_tune_method == "prior_from_idata"`. With `delta_tune_method="prior_from_idata"`, delta is a Deterministic for complete/partial pooling and a free per-group Laplace centered at the prior for individual pooling.
- **Slope transfer definition**: the transferred slope quantity is the **end-of-history slope** of the source model. Under `delta_side="right"` the raw `slope` posterior already equals the end slope. Under `delta_side="left"` `_get_slope_params_from_idata` adds `sum(delta)` to the posterior before computing the prior mean/std. Assumes source and target share the same `delta_side` convention.
- Only `LinearTrend` and `FourierSeasonality` have transfer Potentials; the constant components (`FlatTrend`, `NormalConstant`, `UniformConstant`, `BetaConstant`) intentionally have none.

**Testing pattern for Potentials**: compile the Potential expression directly with `pytensor.function([param], loss_expr)` from `model.named_vars` (passing a Deterministic as the input cuts the graph) — this evaluates signs, scaling and gating numerically without sampling. See `tests/test_regularization.py`.

### Prior Predictive Checks (PPC) Workflow

Vangja scales data so that `y ≈ [-1, 1]` and `t ∈ [0, 1]`. This makes prior predictive checks (PPC) especially useful for tuning prior standard deviations. The default Prophet priors (`N(0, 5)` for slope/intercept, `N(0, 10)` for Fourier beta) are intentionally very diffuse — most prior predictive samples will fall far outside the plausible data range.

**Key utility functions:**

- `prior_predictive_coverage(ppc, low=-2, high=2)` — Quantitative check: what fraction of prior predictive samples fall within `[low, high]`. Target 30–60% for a flexible model. Below 5% means priors are too loose; above 95% means too tight.
- `plot_prior_predictive(ppc, show_hdi=True, show_ref_lines=True, t=model.data["t"].values)` — Visual check: spaghetti plot + HDI envelope + reference lines at scaled data bounds.
- `plot_posterior_predictive(...)` — Same enhancements available for posterior predictive.

**Typical workflow:**

```python
model = LinearTrend() + FourierSeasonality(365.25, 10)
model.fit(data, method="mapx")
ppc = model.sample_prior_predictive(samples=500)

# Quantitative check
coverage = prior_predictive_coverage(ppc)
print(f"{coverage*100:.1f}% coverage")

# Visual check with HDI and reference lines
plot_prior_predictive(ppc, show_hdi=True, show_ref_lines=True, t=model.data["t"].values)
```

**Tuning guidelines:**

- Trend (`slope_sd`, `intercept_sd`): default 5 is very wide. Try 1–2 for stable series.
- Seasonality (`beta_sd`): default 10 is extremely wide. Try 0.5–1 to regularize and prevent overfitting high-frequency noise.
- Use `prior_sensitivity_analysis()` to sweep over prior standard deviations and compare forecast metrics.

### Uncertainty Estimation

Vangja supports prediction intervals via `predict_uncertainty()`. Two approaches are used depending on the inference method:

- **MCMC/VI**: Each posterior draw is propagated through the model via `_predict_map` (reusing the existing MAP code path). Percentile-based credible intervals are computed from the resulting trajectory ensemble.
- **MAP/MAPX**: A hybrid residual-calibrated approach: `ŷ(t) ± z * σ̂ * √(1 + h/n)`, where `σ̂ = max(σ_fitted, σ_residual)`, `h` is forecast distance, `n` is training size, and `z` is a Student-t quantile.

See `uncertainty.md` for the full mathematical description and comparison with Prophet's approach.

**Key API:**

```python
# predict_uncertainty returns yhat + yhat_lower_X + yhat_upper_X columns
future = model.predict_uncertainty(horizon=90, interval_width=0.95)

# plot() auto-detects uncertainty columns and shows fill_between bands
model.plot(future)
```

**Design principle**: Uncertainty estimation is implemented entirely in `TimeSeriesModel` — no component-level changes are needed. For MCMC, individual posterior draws are extracted as dicts and fed through `_predict_map`, avoiding modifications to `_predict_mcmc` or any composition operators.

### Plot Clipping

The `plot()` method supports `clip_to_data=True` (default) which clips predictions to the training + test date range. This is essential for transfer learning where the base model's time scale may extend far beyond the target series dates. The clipping ensures plots only show the relevant prediction window.

### Include Source in Target (Ablation Option)

When performing transfer learning ablation studies, the source time series can be included as an additional series in the target model's training data via `fit(..., include_source_in_target=True, source_data=source_df)`. `source_data` (columns `ds`, `y`) is appended as a `"source"` series **before** scaling/processing, and is required when the flag is set (the source idata does not carry the observed data). This transforms the problem into hierarchical co-learning. **Only meaningful with `pool_type="partial"`** since individual pooling means series don't share information.

## Development Workflow

### Environment Setup

```bash
# uv is the package manager for this project (uv.lock is committed)
uv sync --extra test --extra datasets
uv run pytest
```

### Running Tests

```bash
uv run pytest                       # Run all tests
uv run pytest -v --tb=short         # Verbose with short tracebacks (default)
uv run pytest tests/test_components.py  # Test specific module
```

### Key Dependencies

- `pymc~=6.2` — Probabilistic programming
- `pymc-extras~=0.14.0` — Additional PyMC utilities (MAP with JAX)
- `blackjax~=1.6.2` — JAX-based MCMC sampler
- `scikit-learn~=1.9.0` — Metrics

### Dependency Compatibility Pins (important!)

- `numba` is a transitive dependency via preliz. `numba 0.63.0b1` crashes on import with numpy>=2.0 (it overloads the removed `np.trapz`), and `numba<=0.66` crashes with numpy>=2.5 (it overloads the removed `np.row_stack`). pytensor caps numba at `<=0.66`. Therefore pyproject pins `numba>=0.66.0` and `numpy>=2.0,<2.5`. Do not relax these without verifying imports of `pytensor.link.numba.dispatch`.
- `np.row_stack` was removed in numpy 2.5 — do not use it.
- ArviZ 1.x (installed via pymc 6) is DataTree-based:
  - `az.InferenceData(**{group: ds})` no longer works. Build test fixtures with `az.InferenceData(xr.Dataset(), children={group: xr.DataTree(ds)})` or `az.from_dict({"group": {"var": array}})`.
  - `az.compare` has no `ic` argument and `az.waic` is gone. `compare_models()` in `utils.py` handles both arviz generations (manual WAIC fallback for arviz>=1).
- `pt.as_tensor_variable` / `pt.constant` are gone in pytensor 3 — use `pt.tensor.as_tensor`.
- Tests avoid system timezones (no tzdata in CI); use `dateutil.tz.tzoffset`.
- **`nuts_sampler="nutpie"` requires comma-free variable names**: nutpie parses variable names into dims (splitting on `,`/`.`/`[`/`]`), so names like `fs_0 - beta(p=365.25,n=5)` crash arviz with "more dims (N) given than existing ones". All vangja variable names follow the safe `{type}_{idx} - {param}` scheme (no config suffixes), so nutpie and zarr trace round-trips work. The default `nuts_sampler="pymc"` remains the conservative choice for study fits.
- **Wikipedia S&P 500 page**: the live page no longer embeds the historical changes table (a navbox is parsed as table 2). `datasets/stocks.py` fetches a **frozen revision** (`oldid=1306326561`) and selects tables by column content; if the revision ever breaks, pick a newer revision that still has both tables and update the constant + tests.

## Code Conventions

### Adding New Components

1. Create file in `src/vangja/components/` inheriting `TimeSeriesModel`
2. Implement required methods: `definition()`, `_get_initval()`, `_predict_map()`, `_predict_mcmc()`, `_plot()`
3. Export in `components/__init__.py` and main `__init__.py`
4. Follow existing parameter patterns: `pool_type`, `tune_method`, `shrinkage_strength`
5. **Always write tests** for new functions, classes, and components. Tests go in `tests/` and should mock external calls (network, file I/O) where appropriate.

### Parameter Naming

- Priors use `{param}_mean`, `{param}_sd` pattern (e.g., `slope_mean`, `slope_sd`)
- **PyMC variable names must not contain commas, brackets or dots** (nutpie parses variable names as dimension specs — commas/brackets crash it with arviz "more dims (N) given than existing ones", dots are split too). Convention: `{component_type}_{model_idx} - {param_name}` (e.g., `lt_0 - slope`, `fs_0 - beta`, `nc_0 - c`). Do **not** embed configuration values (period, order, prior hyperparameters, bounds) in variable names — the index already disambiguates components, and config lives in `__str__`/plots. Changing a prior must not rename a variable (breaks idata continuity and zarr caches). All samplers (`pymc`, `nutpie`) and zarr round-trips work with this scheme.
- **Non-complete pooling always indexes per group, even when `n_groups == 1`**: parameters then have shape `(n_groups,)`, so initvals must be arrays and `_predict_*` must index `[group_code]`. The old `and self.n_groups > 1` guard produced scalar initvals for shape-(1,) variables (breaks ADVI) and 2-D prediction columns (breaks `predict_uncertainty`). Regression tests: `tests/test_components.py::TestSingleSeriesNonCompletePooling`.

### Inference Methods (`Method` type)

- **Fast**: `"mapx"` (recommended, uses JAX), `"map"`
- **VI**: `"advi"`, `"fullrank_advi"`, `"svgd"`, `"asvgd"`
- **MCMC**: `"nuts"`, `"metropolis"`, `"demetropolisz"`

**Reproducibility (review P0-8):** `fit()` accepts `random_seed` (threaded to `pmx.find_MAP` / `pm.fit` / `pm.sample` / prior & posterior predictive sampling / `predict_uncertainty` sub-sampling) and `target_accept` (only for `method="nuts"`, forwarded through `nuts_sampler_kwargs`). Never pass an explicit `step` alongside `nuts_sampler` — the explicit step silently overrides the advertised backend. `model.fit_info` records seeds, package versions and MAP optimizer diagnostics (`optimizer_result` group: `fun`, `success`, `jac_l2`, `nit`, ...).

## Testing Patterns

### Fixtures in `tests/conftest.py`

- `sample_data` — Single series, 100 days
- `multi_series_data` — Two series for pooling tests
- `linear_data` / `seasonal_data` — Component-specific testing

### Test Structure

Tests are organized by: `test_components.py` (unit), `test_integration.py` (composition), `test_time_series.py` (base class), `test_plotting.py`, `test_types.py`, `test_utils.py`, `test_stocks.py` (stock data & S&P 500).

**Testing guidelines:** Always write tests for new functions added to the codebase. Mock external calls (yfinance, network requests, `pd.read_html`) to keep tests offline and deterministic. Use `tmp_path` for caching tests.

## Documentation

### Documentation Structure

The documentation is built with Sphinx and hosted on GitHub Pages. Structure:

- **README** — Package overview and quick start
- **User Guide** — Jupyter notebooks in `notebooks/` demonstrating features
- **API Reference** — Auto-generated from docstrings

### Documentation Files

- `docs/source/conf.py` — Sphinx configuration
- `docs/source/index.rst` — Main documentation index
- `docs/source/api.rst` — API reference structure
- `docs/source/readme.md` — Symlink to `../../README.md`
- `docs/source/notebooks/` — Symlink to `../../notebooks/`

### Docstring Style

Use NumPy-style docstrings. Key sections:

```python
def function(param1: str, param2: int = 10) -> bool:
    """Short description.

    Longer description if needed.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, default=10
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> function("test", 5)
    True

    See Also
    --------
    other_function : Related function.

    Notes
    -----
    Additional implementation notes.
    """
```

For classes, put detailed parameter docs in the class docstring, not `__init__`:

```python
class MyClass:
    """Short description.

    Parameters
    ----------
    param1 : str
        Description of param1.

    Attributes
    ----------
    attr1 : str
        Description of attr1.

    Examples
    --------
    >>> obj = MyClass("test")
    """

    def __init__(self, param1: str):
        """Create MyClass.

        See the class docstring for full parameter descriptions.
        """
        self.attr1 = param1
```

### Building Documentation Locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build HTML docs
cd docs
make html

# View in browser
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

### Documentation Deployment

Documentation is automatically deployed to GitHub Pages via the `.github/workflows/docs.yml` workflow:

- **On push to main**: Build and deploy automatically
- **On PRs**: Build only (no deployment) to catch errors
- **Manual trigger**: Use "Run workflow" in GitHub Actions

Enable GitHub Pages in repository settings → Pages → Source: "GitHub Actions".

## Case Studies / Ablation Studies

Case studies live in `case_studies/<dataset_name>/`. Each case study directory follows this structure:

- `ablations.md` — Dataset-specific ablation plan
- `train.py` — Original training script
- `ablation_fast.py` — Fast validation version (VI, reduced grid)
- `ablation_full.py` — Full ablation study (NUTS base, comprehensive grid)
- `classical_baselines.py` — Classical model comparisons (ARIMA, Holt-Winters, Seasonal Naive)
- `README.md` — How to run scripts and interpret results
- `results_ablation/` — Output directory (CSV results + plots)
- `results_classical/` — Classical baseline results

A reusable template is available at `case_studies/smart_home/general_ablations.md`.

- **Classical baselines** live in `smart_home/04_run_baselines.py` and `stocks/05_run_baselines.py` (shared model code + `BASELINE_DESCRIPTIONS` in `fixed_case_studies/baselines.py`; `common.baseline_unit_metrics` gives them the identical per-unit schema as the vangja arms, so `05_report.py`/`09_report.py` aggregate models and baselines together). Baselines run on the frozen splits (trading days only for stocks) and use **statsmodels** (not sktime) because sktime requires scikit-learn < 1.6.0, conflicting with vangja's scikit-learn ~= 1.8.0. Install with `uv sync --extra reproducibility`. The legacy temperature-/S&P-informed regressions were dropped — their test-period features would leak the future (context data end at the training cutoff).

### Revised experiment suite (`fixed_case_studies/`)

`fixed_case_studies/` is the audit-compliant rerun suite (read its README first). Conventions:

- **No grid, no test-horizon selection.** Each study has one frozen configuration + a small frozen ablation list in `<study>/config.py`; runners (`02_*`, `03_*`) iterate the registry with checkpointing.
- `common.py` implements the protocol: Relative MAE vs persistence (`REL_MAE_EPSILON = 1e-3` on the scaled target), context-series exclusion (`CONTEXT_SERIES`), provenance recording, seed policy (`BASE_SEED`, `FINALIST_SEEDS`), two-way block bootstrap for the paired comparison, and the hash-verified freeze check of `protocol/CONFIRMATION_ORIGINS.csv`.
- The stocks confirmation stage fits the context with **NUTS** (covariance-capable transfer); ADVI is development-screening only. Source fits are cached per (origin, method, seed) as **zarr + JSON** (`load_or_fit_source` / `load_or_fit_temp_model`) — use `idata.to_zarr`/`az.from_zarr`, not netCDF (netCDF4/h5netcdf are not dependencies).
- Entry scripts bootstrap `sys.path` with the repo root (`sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`) so they run directly from anywhere.
- **Entry-script numbering = pipeline stage, not chronology of writing:** in both studies `01_fetch_data.py` (data) → `02_select_hyperparams.py` (Bayesian hyperparameter selection; its output is frozen deliberately into `config.py`) → the run scripts → `NN_run_baselines.py` (classical baselines, shared model code in `fixed_case_studies/baselines.py`) → `NN_report.py` last. Stocks: `03_run_development`, `04_run_confirmation`, `05_run_baselines`, `06_run_ablations`, `07_covariance_transfer`, `08_calibration`, `09_report`, plus `10_run_fast` as a dev harness. Smart home: `03_run_main`, `04_run_baselines`, `05_report`. The chosen hyperparameters are documented in `smart_home/best_hyperparams.md` (stocks analogue once its selection is run).
- Smart-home tests monkeypatch the loaders on the `vangja.datasets` module (the config module imports them lazily inside functions).

### Bayesian hyperparameter selection (`fixed_case_studies/hyperparams/`)

The Prophet-inherited hyperparameters (`intercept_sd`, `beta_sd`, `series_order`, `n_changepoints`, `slope_sd`) are selected with a **Bayesian workflow on development training windows only** — never on the test horizon (audit F-1). Entry points: `smart_home/02_select_hyperparams.py`, `stocks/02_select_hyperparams.py`; full protocol in `fixed_case_studies/HYPERPARAMETER_SELECTION.md`.

- **Primary (bayesian.py):** prior-predictive calibration of sd's (`prior_predictive_coverage`, target 30–60% in scaled [-2,2]) + coordinate-wise PSIS-LOO (ADVI screening, optional small-NUTS verification of top-2) + a conservative selection rule: keep the frozen value unless beaten by >1 dse; among statistically indistinguishable candidates prefer the smallest (parsimony/speed).
- **Better strategies (all with code):** leave-future-out expanding-window CV with CRPS from posterior draws (`ts_cv.py` — LOO is optimistic for autocorrelated data); pseudo-BMA/stacking model averaging (`stacking.py`); robustness/flatness perturbation sweeps (`robustness.py`); full-Bayes hyperpriors on the sd's (`full_bayes.py`, single-series demo).
- **Gotchas:** varying `series_order` under `prior_from_idata` transfer is invalid (the transferred prior's dimension is tied to the *source* order) — order selection must use the no-transfer arm. `ts_cv.expanding_folds` splits by **date**, not rows (multi-series data). Fit caching is keyed by (param, value, origin, seed, method, cache_tag) where `cache_tag` distinguishes the transfer/plain arms. The scripts never modify `config.py` — they write `results_hyperparams/*_recommendations.json` for a deliberate freeze.
- **Discrete params must stay ints.** `series_order` and `n_changepoints` are counts (used as `np.empty`/`range`/`np.linspace` shapes): `FourierSeasonality(series_order=3.0)` crashes with `TypeError: 'float' object cannot be interpreted as an integer`. Never extract candidate values from LOO-table names with `float(name.split("=")[-1])` — that turns `3` into `3.0` and silently kills every verification fit (the old empty-table `KeyError: 'name'` was exactly this). Use `bayesian.top_values_from_table(table, evidence)` (returns the typed values from the `CandidateEvidence` dict). As a second line of defence, `FourierSeasonality.__init__`/`LinearTrend.__init__` coerce integral floats to int and `ValueError` on genuinely fractional values.
- **`run_loo_selection` surfaces total fit failure.** Per-candidate fit errors are recorded in `CandidateEvidence.error`; if *every* candidate fails, it raises a `RuntimeError` listing the per-origin failures instead of returning an empty table (which previously produced the obscure `KeyError: 'name'`). Scripts wrap `verify_top_k` in try/except and fall back to the ADVI screening table (`select_table = verify_table if ... else screen_table`).
- **Do not pass `random_seed`/`progressbar` inside `fit_kwargs` to the engine** (`bayesian._fit_once` / `ts_cv` / `prior_predictive_sweep` use `setdefault` — a duplicate raises TypeError).

## Self-Updating Instructions

Whenever an LLM is used to generate code for this project, it should consider whether it learned something new during the task that would be useful for future work. If so, it should update this `copilot-instructions.md` file with the new knowledge. Examples of useful updates include:

- New patterns or conventions discovered in the codebase
- Gotchas or non-obvious behaviors of APIs
- Preferred approaches that emerged from discussion with the user
- New utility functions, datasets, or components that were added
- Changes to default parameters or function signatures
