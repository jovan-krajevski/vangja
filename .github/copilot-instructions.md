# Vangja AI Coding Instructions

## Overview

Vangja is a Bayesian time series forecasting package built on PyMC. It extends Facebook Prophet with hierarchical modeling and transfer learning for short time series forecasting.

## Architecture

### Core Components (`src/vangja/`)

- **`time_series.py`** — Base `TimeSeriesModel` class that all components inherit from. Handles data preprocessing, scaling, model fitting (PyMC), and prediction. Models compose via operator overloading (`+`, `*`, `**`).
- **`components/`** — Model building blocks: `LinearTrend`, `FourierSeasonality`, `NormalConstant`, `BetaConstant`, `UniformConstant`. Each implements `definition()`, `_predict_map()`, `_predict_mcmc()`, and `_plot()`.
- **`types.py`** — Type definitions (`PoolType`, `Method`, `Scaler`, `TuneMethod`) with docstrings explaining each literal value.
- **`utils.py`** — Helper functions for group assignment and metrics.

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
- `"partial"` — Hierarchical pooling with shared hyperpriors
- `"individual"` — Each series has independent parameters

### Transfer Learning

Set `tune_method="parametric"` or `"prior_from_idata"` on components, then pass `idata` (ArviZ InferenceData) to `fit()` to transfer knowledge from pre-trained models.

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
