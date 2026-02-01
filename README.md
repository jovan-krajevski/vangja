# vangja

<img src="https://raw.githubusercontent.com/jovan-krajevski/vangja/refs/heads/main/images/logo.webp" width="35%" height="35%" align="right" />

A time-series forecasting package based on Facebook Prophet with an intuitive API capable of modeling short time-series with prior knowledge derived from a similar long time-series.

This package has been inspired by:

* [Facebook Prophet](https://facebook.github.io/prophet/docs/quick_start.html)
* [Facebook Prophet implementation in PyMC3](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/)
* [TimeSeers](https://github.com/MBrouns/timeseers)
* [Modeling short time series with prior knowledge](https://minimizeregret.com/short-time-series-prior-knowledge)
* [Modeling short time series with prior knowledge - PyMC](https://juanitorduz.github.io/short_time_series_pymc/)

# Installation

You need to create a conda PyMC environment before installing `vangja`. The recommended way of installing PyMC is by running:

```bash
conda create -c conda-forge -n pymc_env python=3.13 "pymc>=5.27.1"
```

Install `vangja` with pip:

```bash
pip install vangja
```

# Usage

The data used for fitting the models is expected to be in the same format as the data used for fitting the Facebook Prophet model i.e. it should be a `pandas` dataframe, where the timestamp is stored in column `ds` and the value is stored in column `y`.

The API is heavily inspired by TimeSeers. A simple model consisting of a linear trend, a yearly seasonality and a weekly seasonality can be fitted like this:

```python
from vangja import LinearTrend, FourierSeasonality

model = LinearTrend() + FourierSeasonality(365.25, 10) + FourierSeasonality(7, 10)
model.fit(data)
model.predict(365)
```

## Multiplicative compositions

There are two types of multiplicative compositions that `vangja` supports. The first one supports creating models from components $g(t)$ and $s(t)$ in the form $y(t)=g(t) * (1 + s(t))$. Using `vangja`, this can be written by using the `__pow__` operator:

```python
model = LinearTrend() ** FourierSeasonality(365.25, 10)
```

The second multiplicative composition supports creating models from components $g(t)$ and $s(t)$ in the form $y(t)=g(t) * s(t)$. Using `vangja`, this can be written by using the `__mul__` operator:

```python
model = LinearTrend() * FourierSeasonality(365.25, 10)
```

## Components

Currently, `vangja` supports the following components:

### LinearTrend

A piecewise linear trend with changepoints.

```python
LinearTrend(
    n_changepoints=25,       # Number of potential changepoints
    changepoint_range=0.8,   # Proportion of data for changepoint placement
    slope_mean=0,            # Prior mean for initial slope
    slope_sd=5,              # Prior std for initial slope
    intercept_mean=0,        # Prior mean for intercept
    intercept_sd=5,          # Prior std for intercept
    delta_mean=0,            # Prior mean for changepoint adjustments
    delta_sd=0.05,           # Prior std for changepoint adjustments
    pool_type="complete",    # Pooling: "complete", "partial", or "individual"
    tune_method=None         # Transfer learning: "parametric" or "prior_from_idata"
)
```

### FourierSeasonality

Seasonal patterns modeled using Fourier series.

```python
FourierSeasonality(
    period,                  # Period in days (e.g., 365.25 for yearly)
    series_order,            # Number of Fourier terms (higher = more flexible)
    beta_mean=0,             # Prior mean for Fourier coefficients
    beta_sd=10,              # Prior std for Fourier coefficients
    pool_type="complete",    # Pooling: "complete", "partial", or "individual"
    tune_method=None         # Transfer learning: "parametric" or "prior_from_idata"
)
```

### NormalConstant

A constant term with a Normal prior, useful for baseline offsets.

```python
NormalConstant(
    mu=0,                    # Prior mean
    sigma=1,                 # Prior standard deviation
    pool_type="complete",    # Pooling: "complete", "partial", or "individual"
    tune_method=None         # Transfer learning: "parametric" or "prior_from_idata"
)
```

### UniformConstant

A constant term with a Uniform prior.

```python
UniformConstant(
    lower=0,                 # Lower bound
    upper=1,                 # Upper bound
    pool_type="complete",    # Pooling: "complete", "partial", or "individual"
    tune_method=None         # Transfer learning: "parametric" or "prior_from_idata"
)
```

### BetaConstant

A constant term with a scaled Beta prior, bounded between [lower, upper].

```python
BetaConstant(
    lower=0,                 # Lower bound for scaling
    upper=1,                 # Upper bound for scaling
    alpha=2,                 # Beta distribution alpha parameter
    beta=2,                  # Beta distribution beta parameter
    pool_type="complete",    # Pooling: "complete", "partial", or "individual"
    tune_method=None         # Transfer learning: "parametric" or "prior_from_idata"
)
```

## Pooling Types

When modeling multiple time-series together, you can control how parameters are shared:

- **`"complete"`**: All series share the same parameters (default).
- **`"partial"`**: Hierarchical pooling with shared hyperpriors - parameters are similar but can differ.
- **`"individual"`**: Each series has completely independent parameters.

Note that the pandas dataframe is required to have a `series` column that denotes which rows belong to which time-series.

```python
# Multi-series data with a 'series' column identifying each series
model = LinearTrend(pool_type="partial") + FourierSeasonality(365.25, 10, pool_type="complete")
model.fit(multi_series_data)
```

## Model Tuning (Transfer Learning)

If you have a long time-series and a "similar" short time-series, you can fit a model on the long time-series and then tune it on the short time-series. This is especially useful if you want to model long seasonality on short data (e.g., you have 3 months of data and want to model yearly seasonality).

There are two tuning methods available:

- **`"parametric"`**: Uses posterior mean and standard deviation from the fitted model to set new priors.
- **`"prior_from_idata"`**: Uses the posterior samples directly via multivariate normal approximation.

```python
# Fit on long time-series, transfer knowledge to short time-series
model = LinearTrend() + FourierSeasonality(365.25, 10, tune_method="parametric")
model.fit(long_time_series)
model.tune(short_time_series)
model.predict(365)
```

## Plotting

After fitting, you can visualize the model components:

```python
# Plot the overall model fit
model.plot()

# Make predictions and plot
predictions = model.predict(periods=365)
```

## Metrics

Evaluate forecast accuracy using built-in metrics:

```python
from vangja import metrics

# Compare actual vs predicted values
results = metrics(actual_df, predicted_df, group_col="group")
# Returns MAE, MSE, RMSE, MAPE per group
```

# Contributing

Pull requests and suggestions are always welcome. Please open an issue on the issue list before submitting in order to avoid doing unnecessary work.
