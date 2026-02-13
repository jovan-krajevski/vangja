"""Uniform Constant component for vangja time series models.

This module provides a constant term with a Uniform prior distribution
for use in time series forecasting models.
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


class UniformConstant(TimeSeriesModel):
    """A constant component with a Uniform prior distribution.

    This component adds a constant term to the model that is sampled from a
    Uniform distribution bounded by lower and upper limits. It's useful for
    modeling parameters that should be constrained to a specific range.

    Parameters
    ----------
    lower : float
        The lower bound of the Uniform prior for the constant parameter.
    upper : float
        The upper bound of the Uniform prior for the constant parameter.
    pool_type : PoolType, default="complete"
        Type of pooling performed when sampling. Options are:

        - "complete": All series share the same constant value.
        - "partial": Series have individual constants with shared hyperpriors.
        - "individual": Each series has a completely independent constant.
    tune_method : TuneMethod | None, default=None
        How the transfer learning is to be performed. Options are:

        - "parametric": Use posterior mean and std from idata to create a
          truncated Normal prior.
        - "prior_from_idata": Use the posterior samples directly as priors.
        - None: This component will not be tuned even if idata is provided.
    shrinkage_strength : float, default=1
        Shrinkage between groups for the hierarchical modeling. Higher values
        result in stronger shrinkage toward the shared mean.

    Attributes
    ----------
    model_idx : int | None
        Index of this component in the model, set during definition.
    group : np.ndarray
        Array of group codes for each data point.
    n_groups : int
        Number of unique groups/series.
    groups_ : dict[int, str]
        Mapping from group codes to group names.

    Examples
    --------
    >>> from vangja import LinearTrend, UniformConstant
    >>> # Add a uniform constant multiplier
    >>> model = LinearTrend() * UniformConstant(lower=0.5, upper=1.5)
    >>> model.fit(data)
    >>> predictions = model.predict(horizon=30)

    >>> # Use partial pooling for multi-series data
    >>> model = LinearTrend() * UniformConstant(lower=0.8, upper=1.2,
    ...                                          pool_type="partial")
    """

    model_idx: int | None = None

    def __init__(
        self,
        lower: float,
        upper: float,
        pool_type: PoolType = "complete",
        tune_method: TuneMethod | None = None,
        shrinkage_strength: float = 1,
    ):
        """Initialize the UniformConstant component.

        Parameters
        ----------
        lower : float
            The lower bound of the Uniform prior for the constant parameter.
        upper : float
            The upper bound of the Uniform prior for the constant parameter.
        pool_type : PoolType, default="complete"
            Type of pooling performed when sampling.
        tune_method : TuneMethod | None, default=None
            How the transfer learning is to be performed.
        shrinkage_strength : float, default=1
            Shrinkage between groups for hierarchical modeling.
        """
        self.lower = lower
        self.upper = upper
        self.pool_type = pool_type
        self.tune_method = tune_method
        self.shrinkage_strength = shrinkage_strength

    def _get_params_from_idata(self, idata: az.InferenceData) -> tuple[float, float]:
        """Extract parameters from posterior samples for transfer learning.

        Parameters
        ----------
        idata : az.InferenceData
            Sample from a posterior.

        Returns
        -------
        tuple[float, float]
            The mean and standard deviation derived from the posterior.
        """
        c_key = f"uc_{self.model_idx} - c(l={self.lower},u={self.upper})"

        mu = float(idata["posterior"][c_key].to_numpy().mean())
        sd = float(idata["posterior"][c_key].to_numpy().std())

        return mu, sd

    def _complete_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add UniformConstant parameters for complete pooling.

        Parameters
        ----------
        model : pm.Model
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor),
            y (target) and series (name of time series).
        priors : dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating
            the posterior sample in idata.
        idata : az.InferenceData | None
            Sample from a posterior for transfer learning.

        Returns
        -------
        pt.TensorVariable
            The constant term to add to the model.
        """
        with model:
            c_key = f"uc_{self.model_idx} - c(l={self.lower},u={self.upper})"

            if idata is not None and self.tune_method == "parametric":
                mu, sd = self._get_params_from_idata(idata)
                # Use truncated normal to stay within bounds
                c = pm.TruncatedNormal(
                    c_key, mu=mu, sigma=sd, lower=self.lower, upper=self.upper
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                c = pm.Deterministic(c_key, priors[f"prior_{c_key}"])
            else:
                c = pm.Uniform(c_key, lower=self.lower, upper=self.upper)

            return c

    def _partial_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add UniformConstant parameters for partial pooling.

        Parameters
        ----------
        model : pm.Model
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor),
            y (target) and series (name of time series).
        priors : dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating
            the posterior sample in idata.
        idata : az.InferenceData | None
            Sample from a posterior for transfer learning.

        Returns
        -------
        pt.TensorVariable
            The constant terms indexed by group.
        """
        with model:
            c_key = f"uc_{self.model_idx} - c(l={self.lower},u={self.upper})"

            if idata is not None and self.tune_method == "parametric":
                mu, sd = self._get_params_from_idata(idata)
                c_shared = pm.TruncatedNormal(
                    f"uc_{self.model_idx} - c_shared",
                    mu=mu,
                    sigma=sd,
                    lower=self.lower,
                    upper=self.upper,
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                c_shared = pm.Deterministic(
                    f"uc_{self.model_idx} - c_shared", priors[f"prior_{c_key}"]
                )
            else:
                c_shared = pm.Uniform(
                    f"uc_{self.model_idx} - c_shared",
                    lower=self.lower,
                    upper=self.upper,
                )

            # For partial pooling, use a hierarchical structure
            range_size = self.upper - self.lower
            c_sigma = pm.HalfNormal(
                f"uc_{self.model_idx} - c_sigma",
                sigma=(range_size / 4) / self.shrinkage_strength,
            )
            c_offset = pm.Normal(
                f"uc_{self.model_idx} - c_offset",
                mu=0,
                sigma=1,
                shape=self.n_groups,
            )
            # Clip to bounds
            c_raw = c_shared + c_offset * c_sigma
            c = pm.Deterministic(
                c_key,
                pm.math.clip(c_raw, self.lower, self.upper),
            )

            return c[self.group]

    def _individual_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add UniformConstant parameters for individual pooling.

        Parameters
        ----------
        model : pm.Model
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor),
            y (target) and series (name of time series).
        priors : dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating
            the posterior sample in idata.
        idata : az.InferenceData | None
            Sample from a posterior for transfer learning.

        Returns
        -------
        pt.TensorVariable
            The constant terms indexed by group.
        """
        with model:
            c_key = f"uc_{self.model_idx} - c(l={self.lower},u={self.upper})"

            if idata is not None and self.tune_method == "parametric":
                mu, sd = self._get_params_from_idata(idata)
                c = pm.TruncatedNormal(
                    c_key,
                    mu=mu,
                    sigma=sd,
                    lower=self.lower,
                    upper=self.upper,
                    shape=self.n_groups,
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                c = pm.Deterministic(
                    c_key, pt.tile(priors[f"prior_{c_key}"], self.n_groups)
                )
            else:
                c = pm.Uniform(
                    c_key,
                    lower=self.lower,
                    upper=self.upper,
                    shape=self.n_groups,
                )

            return c[self.group]

    def definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add the UniformConstant parameters to the model.

        Parameters
        ----------
        model : pm.Model
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor),
            y (target) and series (name of time series).
        model_idxs : dict[str, int]
            Count of the number of components from each type.
        priors : dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating
            the posterior sample in idata.
        idata : az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to
            set the parameters' priors in the model.

        Returns
        -------
        pt.TensorVariable
            The constant term(s) to add to the model.
        """
        model_idxs["uc"] = model_idxs.get("uc", 0)
        self.model_idx = model_idxs["uc"]
        model_idxs["uc"] += 1

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
        """Get the initval of the constant parameter.

        Parameters
        ----------
        initvals : dict[str, float]
            Calculated initvals based on data.
        model : pm.Model
            The model for which the initvals will be set.

        Returns
        -------
        dict
            Empty dictionary as no special initialization is needed.
        """
        return {}

    def _predict_map(
        self, future: pd.DataFrame, map_approx: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Perform MAP prediction for the constant component.

        Parameters
        ----------
        future : pd.DataFrame
            Pandas dataframe containing the timestamps for which inference
            should be performed.
        map_approx : dict[str, np.ndarray]
            The MAP posterior parameter estimate.

        Returns
        -------
        np.ndarray
            Array of shape (n_groups, n_timestamps) with constant values.
        """
        forecasts = []
        c_key = f"uc_{self.model_idx} - c(l={self.lower},u={self.upper})"

        for group_code in self.groups_.keys():
            c_value = map_approx[c_key]
            if self.pool_type != "complete":
                c_value = c_value[group_code]

            forecast = np.ones(len(future)) * c_value
            forecasts.append(forecast)
            future[f"uc_{self.model_idx}_{group_code}"] = forecast

        return np.vstack(forecasts)

    def _predict_mcmc(
        self, future: pd.DataFrame, trace: az.InferenceData
    ) -> np.ndarray:
        """Perform MCMC prediction for the constant component.

        Parameters
        ----------
        future : pd.DataFrame
            Pandas dataframe containing the timestamps for which inference
            should be performed.
        trace : az.InferenceData
            Samples from the posterior.

        Returns
        -------
        np.ndarray
            Array of shape (n_groups, n_timestamps) with constant values.
        """
        forecasts = []
        c_key = f"uc_{self.model_idx} - c(l={self.lower},u={self.upper})"

        for group_code in self.groups_.keys():
            c_samples = trace["posterior"][c_key].to_numpy()

            if self.pool_type != "complete":
                c_value = c_samples[:, :, group_code].mean()
            else:
                c_value = c_samples.mean()

            forecast = np.ones(len(future)) * c_value
            forecasts.append(forecast)
            future[f"uc_{self.model_idx}_{group_code}"] = forecast

        return np.vstack(forecasts)

    def _plot(
        self,
        plot_params: dict,
        future: pd.DataFrame,
        data: pd.DataFrame,
        scale_params: dict,
        y_true: pd.DataFrame | None = None,
        series: int | str = "",
    ) -> None:
        """Plot the constant component.

        Parameters
        ----------
        plot_params : dict
            Dictionary containing plotting parameters, including 'idx' for
            subplot indexing.
        future : pd.DataFrame
            Pandas dataframe containing the predictions.
        data : pd.DataFrame
            The training data.
        scale_params : dict
            Scaling parameters used for the data.
        y_true : pd.DataFrame | None, default=None
            A pandas dataframe containing the true values for comparison.
        series : int | str, default=""
            The series identifier for multi-series plots.
        """
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"UniformConstant({self.model_idx}, l={self.lower}, u={self.upper})")

        # Handle series parameter - convert group_code int to key format
        series_suffix = f"_{series}" if series != "" else "_0"

        col_name = f"uc_{self.model_idx}{series_suffix}"
        if col_name in future.columns:
            plt.bar([0], [future[col_name].iloc[0]])
        else:
            # Fallback for complete pooling
            col_name = f"uc_{self.model_idx}_0"
            if col_name in future.columns:
                plt.bar([0], [future[col_name].iloc[0]])

        plt.axhline(0, c="k", linewidth=1)
        plt.axhline(self.lower, c="r", linewidth=1, linestyle="--", alpha=0.5)
        plt.axhline(self.upper, c="r", linewidth=1, linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)

    def needs_priors(self, *args, **kwargs) -> bool:
        """Check if this component needs priors from idata.

        Returns
        -------
        bool
            True if tune_method is "prior_from_idata", False otherwise.
        """
        return self.tune_method == "prior_from_idata"

    def __str__(self) -> str:
        """Return string representation of the component.

        Returns
        -------
        str
            String representation.
        """
        return f"UC(l={self.lower},u={self.upper},pt={self.pool_type},tm={self.tune_method})"
