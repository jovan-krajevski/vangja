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
conda create -c conda-forge -n pymc_env python=3.12 "pymc>=5.20.1"
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

* `LinearTrend(n_changepoints=25, changepoint_range=0.8, slope_mean=0, slope_sd=5, intercept_mean=0, intercept_sd=5, delta_mean=0, delta_sd=0.05, allow_tune=False)`
* `FourierSeasonality(period, series_order, beta_mean=0, beta_sd=10, allow_tune=False,tune_method="simple")`
* `UniformConstant(lower, upper, allow_tune=False)`
* `BetaConstant(lower, upper, alpha=0.5, beta=0.5, allow_tune=False)`
* `NormalConstant(mu=0, sd=1, allow_tune=False)`

## Model tuning

If you are given a long time-series and a "similar" short time-series, you can fit a model on the long time-series and then tune it on the short time-series. This is especially useful if you want to model a long seasonality on the short time-series, but you do not have enough data to do it (e.g. you have 3 months of data and want to model the yearly seasonality). In `vangja`, this can be written like this:

```python
model = LinearTrend() + FourierSeasonality(365.25, 10, allow_tune=True)
model.fit(long_time_series)
model.tune(short_time_series)
model.predict(365)
```

# Contributing

Pull requests and suggestions are always welcome. Please open an issue on the issue list before submitting in order to avoid doing unnecessary work.
