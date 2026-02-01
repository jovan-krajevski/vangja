"""Type definitions for vangja time series models.

This module provides type aliases and TypedDicts used throughout the vangja
package for type hints and documentation.

Type Aliases
------------
Scaler : Literal["maxabs", "minmax"]
    Methods for scaling the target variable y.
ScaleMode : Literal["individual", "complete"]
    Whether to scale each series separately or all together.
Method : Literal[...]
    Bayesian inference methods for model fitting.
NutsSampler : Literal[...]
    Backend samplers for NUTS inference.
FreqStr : Literal[...]
    Frequency strings for date ranges.
TuneMethod : Literal["parametric", "prior_from_idata"]
    Transfer learning methods.
PoolType : Literal["partial", "complete", "individual"]
    Pooling types for multi-series modeling.

TypedDicts
----------
YScaleParams
    Parameters for y (target) scaling.
TScaleParams
    Parameters for t (time) scaling.
"""

from typing import Literal, TypedDict

Scaler = Literal["maxabs", "minmax"]
"""Scaling method for the target variable y.

- "maxabs": Scale by absolute maximum. y_scaled = y / max(|y|)
- "minmax": Scale to [0, 1]. y_scaled = (y - min(y)) / (max(y) - min(y))
"""

ScaleMode = Literal["individual", "complete"]
"""Scale mode for multi-series data.

- "individual": Scale each series independently.
- "complete": Scale all series together using global min/max.
"""


class YScaleParams(TypedDict):
    """Parameters for scaling the target variable y.

    Attributes
    ----------
    scaler : Scaler
        The scaling method used ("maxabs" or "minmax").
    y_min : float
        The minimum value of y before scaling (0 for maxabs).
    y_max : float
        The maximum value of y (or max(|y|) for maxabs) before scaling.
    """

    scaler: Scaler
    y_min: float
    y_max: float


class TScaleParams(TypedDict):
    """Parameters for scaling the time variable t.

    The time variable is always scaled to [0, 1] using minmax scaling.

    Attributes
    ----------
    ds_min : float
        The minimum datetime value in the dataset.
    ds_max : float
        The maximum datetime value in the dataset.
    """

    ds_min: float
    ds_max: float


Method = Literal[
    "mapx",
    "map",
    "fullrank_advi",
    "advi",
    "svgd",
    "asvgd",
    "nuts",
    "metropolis",
    "demetropolisz",
]
"""Bayesian inference methods.

Point estimates:
- "mapx": Maximum a posteriori using pymc-extras (recommended, uses JAX).
- "map": Maximum a posteriori using PyMC.

Variational inference:
- "advi": Automatic Differentiation Variational Inference.
- "fullrank_advi": Full-rank ADVI.
- "svgd": Stein Variational Gradient Descent.
- "asvgd": Amortized SVGD.

Markov Chain Monte Carlo:
- "nuts": No-U-Turn Sampler (recommended for full posterior).
- "metropolis": Metropolis-Hastings sampler.
- "demetropolisz": Differential Evolution Metropolis-Z.
"""

NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]
"""Backend samplers for NUTS inference.

- "pymc": Default PyMC sampler.
- "nutpie": Fast Rust-based sampler.
- "numpyro": JAX-based sampler from NumPyro.
- "blackjax": JAX-based sampler from BlackJAX.
"""

FreqStr = Literal[
    "Y",
    "M",
    "W",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
    "picosecond",
]
"""Frequency strings for pandas date ranges.

Common values:
- "Y": Year
- "M": Month
- "W": Week
- "D": Day
- "h": Hour
- "m" or "minute": Minute
- "s" or "second": Second
"""

TuneMethod = Literal["parametric", "prior_from_idata"]
"""Transfer learning methods.

- "parametric": Use posterior mean and std from idata to set new priors.
- "prior_from_idata": Use the posterior samples directly as priors via
  multivariate normal approximation.
"""

PoolType = Literal["partial", "complete", "individual"]
"""Pooling types for multi-series modeling.

- "complete": All series share the same parameters.
- "partial": Hierarchical pooling with shared hyperpriors.
- "individual": Each series has completely independent parameters.
"""
