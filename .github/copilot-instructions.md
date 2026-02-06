# Vangja AI Coding Instructions

## Overview

Vangja is a Bayesian time series forecasting package built on PyMC. It extends Facebook Prophet with hierarchical modeling and transfer learning for short time series forecasting.

## Architecture

### Core Components (`src/vangja/`)

- **`time_series.py`** — Base `TimeSeriesModel` class that all components inherit from. Handles data preprocessing, scaling, model fitting (PyMC), and prediction. Models compose via operator overloading (`+`, `*`, `**`).
- **`components/`** — Model building blocks: `LinearTrend`, `FourierSeasonality`, `NormalConstant`, `BetaConstant`, `UniformConstant`. Each implements `definition()`, `_predict_map()`, `_predict_mcmc()`, and `_plot()`.
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
```

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

### Transfer Learning

Set `tune_method="parametric"` or `"prior_from_idata"` on components, then pass `idata` (ArviZ InferenceData) to `fit()` to transfer knowledge from pre-trained models.

### Datasets Module (`src/vangja/datasets/`)

The `datasets` module provides functions for loading real-world datasets and generating synthetic data. **All notebooks should use these functions instead of inline data generation/loading.**

**Available functions:**

- `load_air_passengers()` — Classic monthly airline passengers (1949-1960)
- `load_peyton_manning()` — Daily Wikipedia page views (2007-2016)
- `generate_multi_store_data()` — 5 synthetic store series with same time range
- `generate_hierarchical_products(include_all_year=True)` — 5-6 synthetic product series with opposite seasonality (summer/winter groups). Default time range is 2 years (2018–2019). **Does not introduce gaps** — use `remove_random_gaps()` per-series in notebooks to simulate missing data.

**Adding new datasets:** Create functions in `datasets/loaders.py` (real data) or `datasets/synthetic.py` (generated data), then export in `datasets/__init__.py`.

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

## Development Workflow

### Environment Setup

```bash
conda create -c conda-forge -n pymc_env python=3.13 "pymc>=5.27.1"
conda activate pymc_env
pip install -e ".[test]"
```

### Running Tests

```bash
pytest                    # Run all tests
pytest -v --tb=short      # Verbose with short tracebacks (default)
pytest tests/test_components.py  # Test specific module
```

### Key Dependencies

- `pymc>=5.27.1` — Probabilistic programming
- `pymc-extras==0.7.0` — Additional PyMC utilities (MAP with JAX)
- `blackjax==1.3` — JAX-based MCMC sampler
- `scikit-learn~=1.8.0` — Metrics

## Code Conventions

### Adding New Components

1. Create file in `src/vangja/components/` inheriting `TimeSeriesModel`
2. Implement required methods: `definition()`, `_get_initval()`, `_predict_map()`, `_predict_mcmc()`, `_plot()`
3. Export in `components/__init__.py` and main `__init__.py`
4. Follow existing parameter patterns: `pool_type`, `tune_method`, `shrinkage_strength`

### Parameter Naming

- Priors use `{param}_mean`, `{param}_sd` pattern (e.g., `slope_mean`, `slope_sd`)
- PyMC variable names: `{component_type}_{model_idx} - {param_name}` (e.g., `lt_0 - slope`)

### Inference Methods (`Method` type)

- **Fast**: `"mapx"` (recommended, uses JAX), `"map"`
- **VI**: `"advi"`, `"fullrank_advi"`, `"svgd"`, `"asvgd"`
- **MCMC**: `"nuts"`, `"metropolis"`, `"demetropolisz"`

## Testing Patterns

### Fixtures in `tests/conftest.py`

- `sample_data` — Single series, 100 days
- `multi_series_data` — Two series for pooling tests
- `linear_data` / `seasonal_data` — Component-specific testing

### Test Structure

Tests are organized by: `test_components.py` (unit), `test_integration.py` (composition), `test_time_series.py` (base class), `test_plotting.py`, `test_types.py`, `test_utils.py`.

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

## Self-Updating Instructions

Whenever an LLM is used to generate code for this project, it should consider whether it learned something new during the task that would be useful for future work. If so, it should update this `copilot-instructions.md` file with the new knowledge. Examples of useful updates include:

- New patterns or conventions discovered in the codebase
- Gotchas or non-obvious behaviors of APIs
- Preferred approaches that emerged from discussion with the user
- New utility functions, datasets, or components that were added
- Changes to default parameters or function signatures
