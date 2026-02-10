"""Flat Trend component for vangja time series models.

This module provides a flat (constant-level) trend component as the simplest
possible trend model. Unlike :class:`~vangja.components.linear_trend.LinearTrend`,
it has no slope and no changepoints — just a single intercept parameter.

Classes
-------
FlatTrend
    Constant-level trend (intercept only, no slope).
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from vangja.time_series import TimeSeriesModel
from vangja.types import PoolType, TuneMethod
from vangja.utils import get_group_definition


class FlatTrend(TimeSeriesModel):
    """A flat (constant-level) trend component.

    This is the simplest possible trend component: a single intercept
    parameter with no slope and no changepoints. It models the baseline
    level of the time series as a constant.

    The model is::

        trend(t) = intercept

    This is useful when:

    - The time series has no discernible upward or downward trend.
    - You want a minimal trend component that adds only one parameter.
    - The series is short and estimating a slope would overfit.

    Parameters
    ----------
    intercept_mean : float, default=0
        The mean of the Normal prior for the intercept parameter.
    intercept_sd : float, default=5
        The standard deviation of the Normal prior for the intercept.
    pool_type : PoolType, default="complete"
        Type of pooling for multi-series data. One of:

        - "complete": All series share the same intercept.
        - "partial": Hierarchical pooling with shared hyperpriors.
        - "individual": Each series has an independent intercept.
    tune_method : TuneMethod | None, default=None
        Transfer learning method. One of:

        - "parametric": Use posterior mean/std as new priors.
        - "prior_from_idata": Use posterior samples directly.
        - None: No transfer learning.
    shrinkage_strength : float, default=100
        Controls hierarchical shrinkage for partial pooling. Higher
        values pull individual series intercepts more strongly toward
        the shared mean.

    Attributes
    ----------
    model_idx : int | None
        Index of this component in the model (set during fitting).
    group : np.ndarray
        Array of group codes for each data point.
    n_groups : int
        Number of unique groups/series.
    groups_ : dict[int, str]
        Mapping from group codes to series names.

    Examples
    --------
    >>> from vangja import FourierSeasonality
    >>> from vangja.components import FlatTrend
    >>>
    >>> # Flat trend with seasonal pattern
    >>> model = FlatTrend() + FourierSeasonality(period=365.25, series_order=10)
    >>> model.fit(data, method="mapx")
    >>> predictions = model.predict(horizon=365)

    >>> # With hierarchical pooling for multiple series
    >>> model = FlatTrend(pool_type="partial", shrinkage_strength=50)

    >>> # Transfer learning from a pre-trained model
    >>> target_model = FlatTrend(tune_method="parametric")
    >>> target_model.fit(short_series, idata=source_trace)

    See Also
    --------
    LinearTrend : Piecewise linear trend with changepoints.
    DampedSmooth : Damped dynamic model with AR smoothing.

    Notes
    -----
    ``FlatTrend`` is equivalent to ``LinearTrend(n_changepoints=0)`` with
    the slope fixed to 0, but is more explicit and has fewer parameters
    to estimate. When composing models, it serves as a clean baseline
    that relies on other components (seasonality, GP, etc.) to explain
    temporal variation.
    """

    model_idx: int | None = None

    def __init__(
        self,
        intercept_mean: float = 0,
        intercept_sd: float = 5,
        pool_type: PoolType = "complete",
        tune_method: TuneMethod | None = None,
        shrinkage_strength: float = 100,
    ):
        """Create a FlatTrend model component.

        See the class docstring for full parameter descriptions.
        """
        self.intercept_mean = intercept_mean
        self.intercept_sd = intercept_sd
        self.pool_type = pool_type
        self.tune_method = tune_method
        self.shrinkage_strength = shrinkage_strength

    def _get_params_from_idata(self, idata: az.InferenceData) -> tuple[float, float]:
        """Extract intercept prior parameters from posterior samples.

        Parameters
        ----------
        idata : az.InferenceData
            Sample from a posterior.

        Returns
        -------
        tuple[float, float]
            The mean and standard deviation derived from the posterior.
        """
        key = f"ft_{self.model_idx} - intercept"
        mu = float(idata["posterior"][key].to_numpy().mean())
        sd = float(idata["posterior"][key].to_numpy().std())
        return mu, sd

    def _complete_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add FlatTrend parameters for complete pooling.

        All series share the same intercept parameter.

        Parameters
        ----------
        model : pm.Model
            The PyMC model to add parameters to.
        data : pd.DataFrame
            Processed training data with columns ds, y, t, series.
        priors : dict[str, pt.TensorVariable] | None
            Prior variables from transfer learning.
        idata : az.InferenceData | None
            Posterior samples for transfer learning.

        Returns
        -------
        pt.TensorVariable
            The constant intercept broadcast to all data points.
        """
        with model:
            key = f"ft_{self.model_idx} - intercept"

            if idata is not None and self.tune_method == "parametric":
                mu, sd = self._get_params_from_idata(idata)
                intercept = pm.Normal(key, mu=mu, sigma=sd)
            elif priors is not None and self.tune_method == "prior_from_idata":
                intercept = pm.Deterministic(key, priors[f"prior_{key}"])
            else:
                intercept = pm.Normal(
                    key, mu=self.intercept_mean, sigma=self.intercept_sd
                )

            return pt.ones(len(data)) * intercept

    def _partial_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add FlatTrend parameters for partial pooling.

        Series have individual intercepts with a shared hyperprior.

        Parameters
        ----------
        model : pm.Model
            The PyMC model to add parameters to.
        data : pd.DataFrame
            Processed training data with columns ds, y, t, series.
        priors : dict[str, pt.TensorVariable] | None
            Prior variables from transfer learning.
        idata : az.InferenceData | None
            Posterior samples for transfer learning.

        Returns
        -------
        pt.TensorVariable
            The intercept values indexed by group.
        """
        with model:
            key = f"ft_{self.model_idx} - intercept"
            sd = self.intercept_sd

            if idata is not None and self.tune_method == "parametric":
                mu, sd = self._get_params_from_idata(idata)
                intercept_shared = pm.Normal(
                    f"ft_{self.model_idx} - intercept_shared", mu=mu, sigma=sd
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                intercept_shared = pm.Deterministic(
                    f"ft_{self.model_idx} - intercept_shared",
                    priors[f"prior_{key}"],
                )
            else:
                intercept_shared = pm.Normal(
                    f"ft_{self.model_idx} - intercept_shared",
                    mu=self.intercept_mean,
                    sigma=self.intercept_sd,
                )

            intercept_sigma = pm.HalfNormal(
                f"ft_{self.model_idx} - intercept_sigma",
                sigma=sd / self.shrinkage_strength,
            )
            intercept_offset = pm.Normal(
                f"ft_{self.model_idx} - intercept_offset",
                mu=0,
                sigma=1,
                shape=self.n_groups,
            )
            intercept = pm.Deterministic(
                key, intercept_shared + intercept_offset * intercept_sigma
            )

            return intercept[self.group]

    def _individual_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add FlatTrend parameters for individual pooling.

        Each series gets its own independent intercept parameter.

        Parameters
        ----------
        model : pm.Model
            The PyMC model to add parameters to.
        data : pd.DataFrame
            Processed training data with columns ds, y, t, series.
        priors : dict[str, pt.TensorVariable] | None
            Prior variables from transfer learning.
        idata : az.InferenceData | None
            Posterior samples for transfer learning.

        Returns
        -------
        pt.TensorVariable
            The intercept values indexed by group.
        """
        with model:
            key = f"ft_{self.model_idx} - intercept"

            if idata is not None and self.tune_method == "parametric":
                mu, sd = self._get_params_from_idata(idata)
                intercept = pm.Normal(key, mu=mu, sigma=sd, shape=self.n_groups)
            elif priors is not None and self.tune_method == "prior_from_idata":
                intercept = pm.Deterministic(
                    key, pt.tile(priors[f"prior_{key}"], self.n_groups)
                )
            else:
                intercept = pm.Normal(
                    key,
                    mu=self.intercept_mean,
                    sigma=self.intercept_sd,
                    shape=self.n_groups,
                )

            return intercept[self.group]

    def definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add the FlatTrend parameters to the model.

        Parameters
        ----------
        model : pm.Model
            The PyMC model to add parameters to.
        data : pd.DataFrame
            Processed training data with columns ds, y, t, series.
        model_idxs : dict[str, int]
            Count of the number of components from each type.
        priors : dict[str, pt.TensorVariable] | None
            Prior variables from transfer learning.
        idata : az.InferenceData | None
            Posterior samples for transfer learning.

        Returns
        -------
        pt.TensorVariable
            The flat trend values for all data points.
        """
        model_idxs["ft"] = model_idxs.get("ft", 0)
        self.model_idx = model_idxs["ft"]
        model_idxs["ft"] += 1

        self.group, self.n_groups, self.groups_ = get_group_definition(
            data, self.pool_type
        )

        with model:
            if self.pool_type == "complete":
                return self._complete_definition(model, data, priors, idata)
            elif self.pool_type == "partial":
                return self._partial_definition(model, data, priors, idata)
            elif self.pool_type == "individual":
                return self._individual_definition(model, data, priors, idata)

    def _get_initval(self, initvals: dict[str, float], model: pm.Model) -> dict:
        """Get initial values for the intercept parameter.

        Parameters
        ----------
        initvals : dict[str, float]
            Calculated initvals based on data.
        model : pm.Model
            The model for which the initvals will be set.

        Returns
        -------
        dict
            Initial values mapping model variables to their starting values.
        """
        intercepts = []
        for key in sorted(self.groups_.keys()):
            intercepts.append(initvals.get(f"intercept_{key}", 0))

        if self.pool_type == "complete" or self.n_groups == 1:
            return {
                model.named_vars[f"ft_{self.model_idx} - intercept"]: intercepts[0]
                or 0,
            }

        return {
            model.named_vars[f"ft_{self.model_idx} - intercept"]: np.array(
                [i or 0 for i in intercepts]
            ),
        }

    def _predict_map(
        self, future: pd.DataFrame, map_approx: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Predict using MAP estimates.

        Returns the constant intercept for all timesteps.

        Parameters
        ----------
        future : pd.DataFrame
            DataFrame with timestamps and normalized time ``t``.
        map_approx : dict[str, np.ndarray]
            MAP parameter estimates.

        Returns
        -------
        np.ndarray
            Forecast values, shape ``(n_groups, n_timesteps)``.
        """
        forecasts = []
        for group_code in self.groups_.keys():
            intercept = map_approx[f"ft_{self.model_idx} - intercept"]

            if self.pool_type != "complete" and self.n_groups > 1:
                intercept = intercept[group_code]

            forecast = np.ones(len(future)) * intercept
            forecasts.append(forecast)
            future[f"ft_{self.model_idx}_{group_code}"] = forecast

        return np.vstack(forecasts)

    def _predict_mcmc(
        self, future: pd.DataFrame, trace: az.InferenceData
    ) -> np.ndarray:
        """Predict using MCMC/VI posterior samples.

        Returns the intercept averaged over posterior samples.

        Parameters
        ----------
        future : pd.DataFrame
            DataFrame with timestamps and normalized time ``t``.
        trace : az.InferenceData
            Posterior samples from MCMC or VI inference.

        Returns
        -------
        np.ndarray
            Forecast values, shape ``(n_groups, n_timesteps)``.
        """
        forecasts = []
        for group_code in self.groups_.keys():
            intercept_samples = trace["posterior"][
                f"ft_{self.model_idx} - intercept"
            ].to_numpy()

            if self.pool_type != "complete" and self.n_groups > 1:
                intercept = intercept_samples[:, :, group_code].mean()
            else:
                intercept = intercept_samples.mean()

            forecast = np.ones(len(future)) * intercept
            forecasts.append(forecast)
            future[f"ft_{self.model_idx}_{group_code}"] = forecast

        return np.vstack(forecasts)

    def _plot(self, plot_params, future, data, scale_params, y_true=None, series=""):
        """Plot the FlatTrend component's contribution.

        Parameters
        ----------
        plot_params : dict
            Plotting state with ``idx`` key for subplot indexing.
        future : pd.DataFrame
            DataFrame with predictions.
        data : pd.DataFrame
            Training data.
        scale_params : dict
            Scaling parameters.
        y_true : pd.DataFrame | None
            True values for comparison.
        series : str
            Series identifier for multi-series data.
        """
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"FlatTrend({self.model_idx})")
        plt.grid()

        if series == "":
            series_suffix = "_0"
        else:
            series_suffix = f"_{series}"

        plt.plot(
            future["ds"],
            future[f"ft_{self.model_idx}{series_suffix}"],
            lw=1,
            label=f"ft_{self.model_idx}",
        )
        plt.legend()

    def needs_priors(self, *args, **kwargs):
        """Whether this component needs prior_from_idata transfer learning.

        Returns
        -------
        bool
            True if ``tune_method`` is "prior_from_idata".
        """
        return self.tune_method == "prior_from_idata"

    def is_individual(self, *args, **kwargs):
        """Whether this component uses individual pooling.

        Returns
        -------
        bool
            True if ``pool_type`` is "individual".
        """
        return self.pool_type == "individual"

    def __str__(self):
        return f"FT(pt={self.pool_type},tm={self.tune_method})"
