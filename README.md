# vangja

<img src="https://raw.githubusercontent.com/jovan-krajevski/vangja/refs/heads/main/images/logo.webp" width="35%" height="35%" align="right" />

A **Bayesian time series forecasting package** that extends Facebook Prophet with hierarchical modeling and transfer learning capabilities. Vangja enables practitioners to model short time series using prior knowledge derived from similar long time series and is particularly good at forecasting horizons longer than the available data.

## Key Features

- 🚀 **Vectorized Multi-Series Fitting** — Fit multiple time series simultaneously with vectorized computations, significantly faster than fitting sequentially with Facebook Prophet
- 📊 **Hierarchical Bayesian Modeling** — Model multiple related time series with flexible pooling strategies (complete, partial, or individual) for each component
- 🔄 **Bayesian Transfer Learning** — Learn from long time series and transfer knowledge to short time series, enabling accurate long-horizon forecasts from limited data
- ↔️ **Bidirectional Changepoints** — Interpret trend changepoints from right-to-left (in addition to left-to-right), essential for hierarchical modeling of time series with different lengths
- 🎯 **Component-Level Flexibility** — Independently configure pooling strategies and transfer learning methods for each model component (trend, seasonalities, etc.)

## Inspirations

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

model = LinearTrend() + FourierSeasonality(365.25, 10) + FourierSeasonality(7, 3)
model.fit(data)
model.predict(365)
```

## Vectorized Multi-Series Fitting

Unlike Facebook Prophet, which fits time series one at a time, vangja can fit multiple time series **simultaneously** using vectorized computations. This is significantly faster when you have many related time series:

```python
# Data must have a 'series' column identifying each time series
# Example: sales data from multiple stores
multi_series_data = pd.DataFrame({
    'ds': [...],  # timestamps
    'y': [...],   # values
    'series': ['store_A', 'store_A', ..., 'store_B', 'store_B', ...]
})

# Fit all series at once with independent parameters (no pooling)
model = LinearTrend(pool_type="individual") + FourierSeasonality(365.25, 10, pool_type="individual")
model.fit(multi_series_data)
```

## Multiplicative operators

There are two types of multiplicative operators that `vangja` supports. The first one supports creating models from components $g(t)$ and $s(t)$ in the form $y(t)=g(t) * (1 + s(t))$. Using `vangja`, this can be written by using the `__pow__` operator:

```python
model = LinearTrend() ** FourierSeasonality(365.25, 10)
```

The second multiplicative operator supports creating models from components $g(t)$ and $s(t)$ in the form $y(t)=g(t) * s(t)$. Using `vangja`, this can be written by using the `__mul__` operator:

```python
model = LinearTrend() * FourierSeasonality(365.25, 10)
```

## Components

Currently, `vangja` supports the following components:

### LinearTrend

A piecewise linear trend with changepoints. Vangja extends Prophet's trend component with **bidirectional changepoint interpretation**.

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
    delta_side="left",       # "left" or "right" - direction for changepoint interpretation
    pool_type="complete",    # Pooling: "complete", "partial", or "individual"
    delta_pool_type="complete",  # Separate pooling strategy for changepoints
    tune_method=None         # Transfer learning: "parametric" or "prior_from_idata"
)
```

#### Bidirectional Changepoints (`delta_side`)

By default (`delta_side="left"`), the `slope` parameter controls the trend slope at the earliest timestamp, and changepoints modify the slope going forward in time.

Setting `delta_side="right"` reverses this: the `slope` parameter controls the trend slope at the **latest timestamp**, and changepoints modify the slope going backward in time. This is **essential for hierarchical modeling** when you have:

- A long time series spanning many years
- Multiple short time series that only cover the recent period

With `delta_side="right"`, the slope parameter is informed by both the long and short time series (since they overlap at the end), rather than only the long time series (which alone covers the beginning).

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

## Pooling Types (Hierarchical Modeling)

When modeling multiple time series together, you can control how parameters are shared using **hierarchical Bayesian modeling**. This is inspired by TimeSeers but with **greater flexibility** — vangja allows different pooling strategies for each component and even for different parameters within the same component.

- **`"complete"`**: All series share the same parameters. Best when series are very similar.
- **`"partial"`**: Hierarchical pooling with shared hyperpriors — parameters are drawn from a common distribution but can differ between series. This balances information sharing with individual variation.
- **`"individual"`**: Each series has completely independent parameters. Equivalent to fitting each series separately, but vectorized for speed.

Note: The pandas dataframe must have a `series` column that identifies which rows belong to which time series.

```python
# Example: Hierarchical model with different pooling per component
# - Trend slope: partial pooling (similar but not identical across series)
# - Changepoints: complete pooling (shared across all series)
# - Yearly seasonality: complete pooling (same seasonal pattern)
# - Weekly seasonality: partial pooling (similar weekly patterns)

model = (
    LinearTrend(pool_type="partial", delta_pool_type="complete") 
    + FourierSeasonality(365.25, 10, pool_type="complete")
    + FourierSeasonality(7, 3, pool_type="partial")
)
model.fit(multi_series_data)
```

### Why Different Pooling Strategies?

Unlike TimeSeers which applies the same pooling to all components, vangja lets you choose based on domain knowledge:

- **Yearly seasonality**: Often similar across related series → use `"complete"`
- **Weekly seasonality**: May vary by series (e.g., different stores have different weekly patterns) → use `"partial"`
- **Trend slope**: Usually similar for related series → use `"partial"`
- **Changepoints**: When dealing with a long context series and short target series, changepoints are only observable in the long series → use `"complete"`

## Model Tuning (Bayesian Transfer Learning)

A **core feature** of vangja is the ability to transfer knowledge from a long time series to multiple short time series. This is particularly useful when:

- You have only a few months of data but need to model yearly seasonality
- You want to forecast a horizon longer than your available short time series
- You have a "context" time series (e.g., market index) and want to use it to inform forecasts for related series (e.g., individual stocks)

Forecasting short time series is challenging because:
1. Long-period seasonalities (e.g., yearly) cannot be estimated from short data
2. Overfitting is likely when the forecast horizon exceeds the data length
3. Standard methods like Facebook Prophet will produce unreliable forecasts

Vangja implements **Bayesian transfer learning**: fit a model on a long time series, extract the posterior distributions of parameters, and use them as informed priors when fitting short time series.

### Transfer Learning Methods

There are two tuning methods available:

#### 1. Parametric Transfer (`"parametric"`)

Uses the posterior mean (you can also set the mode, or any other value that you need) and standard deviation from the fitted model to set new priors while keeping the same distribution form:

```python
# Step 1: Fit on long time series
model = (
    LinearTrend(tune_method="parametric") 
    + FourierSeasonality(365.25, 10, tune_method="parametric")
)
model.fit(long_time_series)

# Step 2: Transfer to short time series
# The posterior from step 1 becomes the prior for step 2
model.tune(short_time_series)

# Step 3: Forecast with confidence
predictions = model.predict(365)  # Can forecast beyond the short series length!
```

#### 2. Prior from InferenceData (`"prior_from_idata"`)

Uses the full posterior samples via **multivariate normal approximation**, preserving correlations between parameters:

```python
model = (
    LinearTrend(tune_method="prior_from_idata") 
    + FourierSeasonality(365.25, 10, tune_method="prior_from_idata")
)
model.fit(long_time_series)
model.tune(short_time_series)
```

This method captures parameter dependencies (e.g., correlation between trend slope and seasonality amplitude) that the parametric method ignores.

### Combining Hierarchical Modeling with Transfer Learning

Vangja uniquely allows you to combine both approaches:

```python
# Long "context" time series + multiple short "target" time series
all_data = pd.concat([
    long_context_series.assign(series='context'),
    short_series_1.assign(series='target_1'),
    short_series_2.assign(series='target_2'),
])

# Hierarchical model with transfer learning
model = (
    LinearTrend(
        pool_type="partial",           # Hierarchical pooling
        delta_side="right",            # Slope parameter informed by all series
        tune_method="parametric"       # Transfer from context to targets
    ) 
    + FourierSeasonality(365.25, 10, pool_type="complete", tune_method="parametric")
)

# Fit and predict
model.fit(all_data)
predictions = model.predict(365)
```

### Regularization for Transfer Learning

To prevent overfitting when transferring knowledge, vangja supports regularization via the `loss_factor_for_tune` parameter. This adds a penalty term that constrains parameters to stay close to the values learned from the long time series:

```python
FourierSeasonality(
    365.25, 10, 
    tune_method="parametric",
    loss_factor_for_tune=1.0  # Higher = stronger regularization toward context series
)
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

## Vangja vs Facebook Prophet vs TimeSeers

| Feature | Facebook Prophet | TimeSeers | Vangja |
|---------|------------------|-----------|--------|
| Single time series | ✅ | ✅ | ✅ |
| Vectorized multi-series | ❌ | ✅ | ✅ |
| Hierarchical Bayesian | ❌ | ✅ | ✅ |
| Per-component pooling | ❌ | ❌ | ✅ |
| Bidirectional changepoints | ❌ | ❌ | ✅ |
| Transfer learning | ❌ | ❌ | ✅ |
| Parametric prior transfer | ❌ | ❌ | ✅ |
| Multivariate Gaussian prior | ❌ | ❌ | ✅ |
| Regularization for transfer | ❌ | ❌ | ✅ |
| Modern PyMC (5.x) | ❌ | ❌ | ✅ |

## Inference Methods

Vangja supports multiple inference methods:

```python
# MAP estimation (fast, recommended for quick results)
model.fit(data, method="mapx")  # Uses JAX backend via pymc-extras

# Full Bayesian inference with MCMC
model.fit(data, method="nuts", samples=1000, chains=4)

# Variational inference
model.fit(data, method="advi", samples=1000)
```

# Contributing

Pull requests and suggestions are always welcome. Please open an issue on the issue list before submitting in order to avoid doing unnecessary work.