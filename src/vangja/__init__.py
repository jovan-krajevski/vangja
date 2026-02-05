"""Vangja: Bayesian time series forecasting with transfer learning.

Vangja is a Bayesian time series forecasting package built on PyMC that
extends Facebook Prophet with hierarchical modeling and transfer learning
capabilities for forecasting short time series.

Main Features
-------------
- **Composable Components**: Build models by combining trend and seasonality
  components using intuitive operators (+, *, **).
- **Hierarchical Modeling**: Partial pooling allows related time series to
  share statistical strength.
- **Transfer Learning**: Pre-train on long time series and transfer knowledge
  to short series forecasts.
- **Multiple Inference Methods**: MAP estimation, variational inference, and
  full MCMC sampling.

Quick Start
-----------
>>> from vangja import LinearTrend, FourierSeasonality
>>> from vangja.datasets import load_air_passengers
>>>
>>> # Load data
>>> data = load_air_passengers()
>>> train = data[:-12]
>>> test = data[-12:]
>>>
>>> # Create and fit a model
>>> model = LinearTrend() ** FourierSeasonality(365.25, 10)
>>> model.fit(train, method="mapx")
>>>
>>> # Predict
>>> predictions = model.predict(horizon=365)

Components
----------
LinearTrend
    Piecewise linear trend with changepoints.
FourierSeasonality
    Seasonal patterns using Fourier series.
NormalConstant
    Constant offset with Normal prior.
BetaConstant
    Bounded constant with Beta prior.
UniformConstant
    Bounded constant with Uniform prior.

Submodules
----------
utils
    Helper functions for metrics and data processing.
datasets
    Functions to load example datasets and generate synthetic data.

See Also
--------
vangja.time_series : Core time series model classes.
vangja.components : Individual model components.
vangja.datasets : Dataset loading and generation.
"""

from vangja import datasets, utils
from vangja.components import (
    BetaConstant,
    FourierSeasonality,
    LinearTrend,
    NormalConstant,
    UniformConstant,
)

__all__ = [
    "LinearTrend",
    "FourierSeasonality",
    "UniformConstant",
    "BetaConstant",
    "NormalConstant",
    "utils",
    "datasets",
]
