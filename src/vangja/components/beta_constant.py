"""Beta Constant component for vangja time series models.

This module provides a constant term with a Beta prior distribution
for use in time series forecasting models, scaled to a specified range.
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


class BetaConstant(TimeSeriesModel):
    """A constant component with a Beta prior distribution scaled to a range.

    This component adds a constant term to the model that is sampled from a
    Beta distribution and then scaled to lie within [lower, upper]. It's useful
    for modeling parameters that should be bounded and have flexible shapes
    controlled by the alpha and beta parameters.

    Parameters
    ----------
    lower : float
        The lower bound for the constant parameter after scaling.
    upper : float
        The upper bound for the constant parameter after scaling.
    alpha : float, default=0.5
        The alpha parameter of the Beta distribution. Controls the shape.
    beta : float, default=0.5
        The beta parameter of the Beta distribution. Controls the shape.
    pool_type : PoolType, default="complete"
        Type of pooling performed when sampling. Options are:

        - "complete": All series share the same constant value.
        - "partial": Series have individual constants with shared hyperpriors.
        - "individual": Each series has a completely independent constant.
    tune_method : TuneMethod | None, default=None
        How the transfer learning is to be performed. Options are:

        - "parametric": Use posterior samples to derive new Beta parameters.
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

    Notes
    -----
    The transformation from Beta to the scaled constant is:
        c = beta_value * (upper - lower) + lower

    Common choices for alpha and beta:
        - alpha=beta=0.5: Jeffrey's prior (U-shaped, more mass at extremes)
        - alpha=beta=1: Uniform distribution
        - alpha=beta=2: Symmetric bell-shaped

    Examples
    --------
    >>> from vangja import LinearTrend, BetaConstant
    >>> # Add a beta-distributed scaling factor between 0.8 and 1.2
    >>> model = LinearTrend() * BetaConstant(lower=0.8, upper=1.2, alpha=2, beta=2)
    >>> model.fit(data)
    >>> predictions = model.predict(horizon=30)

    >>> # Use partial pooling for multi-series data
    >>> model = LinearTrend() * BetaConstant(lower=0.5, upper=1.5,
    ...                                       pool_type="partial")
    """

    model_idx: int | None = None

    def __init__(
        self,
        lower: float,
        upper: float,
        alpha: float = 0.5,
        beta: float = 0.5,
        pool_type: PoolType = "complete",
        tune_method: TuneMethod | None = None,
        shrinkage_strength: float = 1,
    ):
        """Initialize the BetaConstant component.

        Parameters
        ----------
        lower : float
            The lower bound for the constant parameter after scaling.
        upper : float
            The upper bound for the constant parameter after scaling.
        alpha : float, default=0.5
            The alpha parameter of the Beta distribution.
        beta : float, default=0.5
            The beta parameter of the Beta distribution.
        pool_type : PoolType, default="complete"
            Type of pooling performed when sampling.
        tune_method : TuneMethod | None, default=None
            How the transfer learning is to be performed.
        shrinkage_strength : float, default=1
            Shrinkage between groups for hierarchical modeling.
        """
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.beta = beta
        self.pool_type = pool_type
        self.tune_method = tune_method
        self.shrinkage_strength = shrinkage_strength

    def _get_params_from_idata(self, idata: az.InferenceData) -> tuple[float, float]:
        """Extract Beta parameters from posterior samples using method of moments.

        Parameters
        ----------
        idata : az.InferenceData
            Sample from a posterior.

        Returns
        -------
        tuple[float, float]
            The alpha and beta parameters derived from the posterior.
        """
        c_key = f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"

        # Get scaled values and convert back to [0, 1]
        c_samples = idata["posterior"][c_key].to_numpy()
        beta_samples = (c_samples - self.lower) / (self.upper - self.lower)

        # Method of moments estimation for Beta distribution
        mean = float(np.clip(beta_samples.mean(), 0.01, 0.99))
        var = float(np.clip(beta_samples.var(), 1e-6, mean * (1 - mean) - 1e-6))

        # Solve for alpha and beta
        common = mean * (1 - mean) / var - 1
        alpha = max(0.1, mean * common)
        beta = max(0.1, (1 - mean) * common)

        return alpha, beta

    def _complete_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add BetaConstant parameters for complete pooling.

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
            c_key = f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"
            beta_key = f"bc_{self.model_idx} - beta(a={self.alpha},b={self.beta})"

            if idata is not None and self.tune_method == "parametric":
                alpha, beta = self._get_params_from_idata(idata)
                beta_rv = pm.Beta(beta_key, alpha=alpha, beta=beta)
            elif priors is not None and self.tune_method == "prior_from_idata":
                # Convert from scaled back to beta space
                beta_rv = pm.Deterministic(
                    beta_key,
                    (priors[f"prior_{c_key}"] - self.lower) / (self.upper - self.lower),
                )
            else:
                beta_rv = pm.Beta(beta_key, alpha=self.alpha, beta=self.beta)

            c = pm.Deterministic(
                c_key, beta_rv * (self.upper - self.lower) + self.lower
            )

            return c

    def _partial_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add BetaConstant parameters for partial pooling.

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
            c_key = f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"
            beta_key = f"bc_{self.model_idx} - beta(a={self.alpha},b={self.beta})"

            if idata is not None and self.tune_method == "parametric":
                alpha, beta = self._get_params_from_idata(idata)
                mu_beta = pm.Beta(
                    f"bc_{self.model_idx} - mu_beta", alpha=alpha, beta=beta
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                mu_beta = pm.Deterministic(
                    f"bc_{self.model_idx} - mu_beta",
                    (priors[f"prior_{c_key}"] - self.lower) / (self.upper - self.lower),
                )
            else:
                mu_beta = pm.Beta(
                    f"bc_{self.model_idx} - mu_beta", alpha=self.alpha, beta=self.beta
                )

            # Hierarchical structure with offset
            beta_sigma = pm.HalfNormal(
                f"bc_{self.model_idx} - beta_sigma",
                sigma=0.1 / self.shrinkage_strength,
            )
            beta_offset = pm.Normal(
                f"bc_{self.model_idx} - beta_offset",
                mu=0,
                sigma=1,
                shape=self.n_groups,
            )
            # Clip to valid beta range [0, 1]
            beta_raw = mu_beta + beta_offset * beta_sigma
            beta_rv = pm.Deterministic(beta_key, pm.math.clip(beta_raw, 0.001, 0.999))

            c = pm.Deterministic(
                c_key, beta_rv * (self.upper - self.lower) + self.lower
            )

            return c[self.group]

    def _individual_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ) -> pt.TensorVariable:
        """Add BetaConstant parameters for individual pooling.

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
            c_key = f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"
            beta_key = f"bc_{self.model_idx} - beta(a={self.alpha},b={self.beta})"

            if idata is not None and self.tune_method == "parametric":
                alpha, beta = self._get_params_from_idata(idata)
                beta_rv = pm.Beta(beta_key, alpha=alpha, beta=beta, shape=self.n_groups)
            elif priors is not None and self.tune_method == "prior_from_idata":
                beta_rv = pm.Deterministic(
                    beta_key,
                    pt.tile(
                        (priors[f"prior_{c_key}"] - self.lower)
                        / (self.upper - self.lower),
                        self.n_groups,
                    ),
                )
            else:
                beta_rv = pm.Beta(
                    beta_key, alpha=self.alpha, beta=self.beta, shape=self.n_groups
                )

            c = pm.Deterministic(
                c_key, beta_rv * (self.upper - self.lower) + self.lower
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
        """Add the BetaConstant parameters to the model.

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
        model_idxs["bc"] = model_idxs.get("bc", 0)
        self.model_idx = model_idxs["bc"]
        model_idxs["bc"] += 1

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
        c_key = f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"

        for group_code in self.groups_.keys():
            c_value = map_approx[c_key]
            if self.pool_type != "complete":
                c_value = c_value[group_code]

            forecast = np.ones(len(future)) * c_value
            forecasts.append(forecast)
            future[f"bc_{self.model_idx}_{group_code}"] = forecast

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
        c_key = f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"

        for group_code in self.groups_.keys():
            c_samples = trace["posterior"][c_key].to_numpy()

            if self.pool_type != "complete":
                c_value = c_samples[:, :, group_code].mean()
            else:
                c_value = c_samples.mean()

            forecast = np.ones(len(future)) * c_value
            forecasts.append(forecast)
            future[f"bc_{self.model_idx}_{group_code}"] = forecast

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
        plt.title(
            f"BetaConstant({self.model_idx}, l={self.lower}, u={self.upper}, "
            f"a={self.alpha}, b={self.beta})"
        )

        # Handle series parameter
        series_suffix = f"_{series}" if series != "" else "_0"

        col_name = f"bc_{self.model_idx}{series_suffix}"
        if col_name in future.columns:
            plt.bar([0], [future[col_name].iloc[0]])
        else:
            col_name = f"bc_{self.model_idx}_0"
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
        return (
            f"BC(a={self.alpha},b={self.beta},l={self.lower},u={self.upper},"
            f"pt={self.pool_type},tm={self.tune_method})"
        )
