"""Model components for vangja time series forecasting.

This module provides the building blocks for constructing time series models.
Components can be combined using arithmetic operators to create complex models.

Trend Components
----------------
LinearTrend
    Piecewise linear trend with optional changepoints.

Seasonal Components
-------------------
FourierSeasonality
    Periodic patterns using Fourier series representation.

Constant Components
-------------------
NormalConstant
    Constant offset with Normal prior.
BetaConstant
    Bounded constant with Beta prior.
UniformConstant
    Bounded constant with Uniform prior.

Examples
--------
>>> from vangja.components import LinearTrend, FourierSeasonality
>>>
>>> # Additive model: y = trend + seasonality
>>> model = LinearTrend() + FourierSeasonality(365.25, 10)
>>>
>>> # Prophet-style multiplicative: y = trend * (1 + seasonality)
>>> model = LinearTrend() ** FourierSeasonality(365.25, 10)
>>>
>>> # Complex model with multiple seasonalities
>>> model = LinearTrend() ** (
...     FourierSeasonality(365.25, 10) +  # yearly
...     FourierSeasonality(7, 3)          # weekly
... )
"""

from vangja.components.beta_constant import BetaConstant
from vangja.components.fourier_seasonality import FourierSeasonality
from vangja.components.linear_trend import LinearTrend
from vangja.components.normal_constant import NormalConstant
from vangja.components.uniform_constant import UniformConstant

__all__ = [
    "LinearTrend",
    "FourierSeasonality",
    "UniformConstant",
    "BetaConstant",
    "NormalConstant",
]
